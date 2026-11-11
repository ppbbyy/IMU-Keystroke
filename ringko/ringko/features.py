"""분류용 특징 추출.

isolated_features : 개별 키 시도(F 정지 -> 키). Stage 1 변위 + 궤적 모양 + 회전 + 타격 주변 원시 통계
                    + 윈도우 전체 평균 자세(QwertyRing 행 모델용 tilt) + 자력계 조건부 요 보정 + 정규화용 원시 윈도우.
word_presses      : 연속 타이핑 단어. 단어 시작(F 정지)부터 자세를 계속 적분하고, 자음 키 타격마다 수평 속도를 0으로
                    놓는(ZUPT) 구간 적분으로 '타격 간 변위'를 구해 누적 -> 각 타격의 F 기준 위치 (aLeak식 상대 벡터 연결).
                    타격 간격 ikt(Stage 4 keystroke interval)와 타격별 tilt도 함께 기록.
                    tail = 마지막 자음 타격 → F 복귀 후 정지까지 시간. 끝 자음 뒤에 모음을 더 치는 단어(누나)는
                    끝이 받침인 단어(눈)보다 늦게 멈춘다 → 자음열이 같은 단어를 가르는 단서(decode.TimingModel).
                    타격 구분은 키 로그(키다운) 시각을 쓴다(오라클 분할). IMU만으로 분할하는 성능은 detect_presses로 따로 잰다.
detect_presses    : Stage 2. jerk 피크 + 충격 방향 게이트 + aLeak Algorithm 1(타이핑 리듬 = FFT f_type 또는 중앙 간격).

증강(학습 fold 전용): isolated_features(..., aug=Augment(...))는 시도마다 진폭 배율·잡음·구간 시간이동을 무작위로 넣어
특징을 다시 계산한다. 평가 시에는 학습 fold의 시도에만 증강본을 붙인다(scripts/evaluate.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .hangul import LEFT_KEYS, to_keys
from .motion import G, MotionParams, find_onset, find_press, horizontal_basis, kalman_trend, rotvec_to_mat, session_arrays

N_PATH = 8
N_WIN = 48          # 정규화용 원시 윈도우 길이(리샘플)
RETURN_S = 0.35     # 개별 키: 타격 후 F 복귀까지 포함할 길이(QwertyRing: 윈도우 전체 평균 피치)
TAIL_SKIP_S = 0.05   # 단어 끝: 마지막 타격 충격 구간은 건너뛰고
TAIL_STILL_S = 0.25  # 이 길이만큼 자이로가 정지 임계 아래로 유지되는 첫 시점 = F 복귀 후 정지
TAIL_MAX_S = 2.5     # 이 안에 정지가 없으면 NaN


@dataclass
class Augment:
    """학습 fold 전용 증강. 시도마다 독립 난수."""
    scale: float = 0.06        # 가속도/자이로 진폭 배율 표준편차(1 ± scale)
    noise_g: float = 0.004     # 가속도 가우스 잡음[g]
    noise_dps: float = 0.5     # 자이로 가우스 잡음[dps]
    shift: int = 2             # 시작/끝 인덱스 무작위 이동 ±shift 샘플(시간이동)
    seed: int = 0


def _ref_axis(sd, t, acc):
    rest = sd.trials[(sd.trials.phase == "rest") & (sd.trials.status == "ok")]
    if len(rest):
        r = rest.iloc[0]
        m = (t >= r.cue_ns / 1e9) & (t <= r.end_ns / 1e9)
        g = acc[m].mean(0)
    else:
        g = acc[:200].mean(0)
    return np.eye(3)[int(np.argmin(np.abs(g)))], g / np.linalg.norm(g)


def _quiet_after(t, acc, gyr, ip, max_n=35):
    k0, k1 = ip + 3, min(len(t) - 2, ip + max_n)
    act = np.degrees(np.linalg.norm(gyr[k0:k1], axis=1)) / 10 + \
        np.linalg.norm(acc[k0:k1] - acc[k0 - 1:k1 - 1], axis=1) / G * 20
    return k0 + int(np.argmin(act))


def tail_settle(t, gyr, ip: int, still_dps: float = 8.0) -> float:
    """마지막 타격 ip 뒤 F 복귀를 마치고 손가락이 멈춘 시점까지 시간[s](없으면 NaN). IMU만 쓴다."""
    dt = float(np.median(np.diff(t[max(0, ip - 50):ip + 51])))
    gn = np.degrees(np.linalg.norm(gyr, axis=1)) < still_dps
    w = max(2, int(round(TAIL_STILL_S / dt)))
    j0 = ip + int(round(TAIL_SKIP_S / dt))
    j1 = min(len(t) - w, ip + int(round(TAIL_MAX_S / dt)))
    if j1 <= j0:
        return float("nan")
    # 길이 w 창이 모두 정지인 첫 시작점(누적합으로)
    c = np.concatenate([[0], np.cumsum(~gn[j0:j1 + w])])
    ok = np.where(c[w:w + j1 - j0] - c[:j1 - j0] == 0)[0]
    return float(t[j0 + ok[0]] - t[ip]) if len(ok) else float("nan")


def _rotations(t, gyr, i0, i1, bias):
    """[i0, i1] 각 샘플의 body(t) -> body(i0) 회전."""
    R = np.eye(3)
    Rs = []
    for i in range(i0, i1 + 1):
        Rs.append(R)
        if i < i1:
            R = R @ rotvec_to_mat((gyr[i] - bias) * (t[i + 1] - t[i]))
    return Rs


def _lin_acc(t, acc, gyr, i0, i1, g0, bias, Rs=None):
    """[i0, i1] 선형가속도(기준 = body at i0) 와 회전 누적."""
    if Rs is None:
        Rs = _rotations(t, gyr, i0, i1, bias)
    a = np.einsum("nij,nj->ni", np.asarray(Rs), acc[i0:i1 + 1]) - g0
    return a, Rs


def _segment_disp(tt, a, gz, path_pts=0, kalman_rho=0.0, full_zupt=False):
    """한 구간: (선택) 칼만 추세 제거, 시작 속도 0, 끝 수평 속도 0 보정 후 변위(및 경로 샘플).
    full_zupt: 끝 속도를 수직까지 0으로(키를 누른 순간·F 정지는 3축 모두 정지). 수평 성분 결과는 그대로이고
               수직 성분만 달라진다 → 3차원 위치(F 기준 거리)용."""
    tt = tt - tt[0]
    dt = np.diff(tt)[:, None]
    if kalman_rho > 0:
        tr = kalman_trend(a, kalman_rho)
        a = a - (tr - np.outer(tr @ gz, gz))
    v = np.vstack([np.zeros(3), np.cumsum((a[1:] + a[:-1]) / 2 * dt, axis=0)])
    if tt[-1] > 0:
        ve = v[-1] if full_zupt else v[-1] - (v[-1] @ gz) * gz
        v = v - np.outer(tt / tt[-1], ve)
    s = np.vstack([np.zeros(3), np.cumsum((v[1:] + v[:-1]) / 2 * dt, axis=0)])
    if path_pts:
        idx = np.linspace(0, len(s) - 1, path_pts + 1)[1:].round().astype(int)
        return s[-1], s[idx], v
    return s[-1], None, v


def _raw_stats(t, acc, gyr, ip, pre_s=0.30, post_s=0.10):
    lo, hi = np.searchsorted(t, [t[ip] - pre_s, t[ip] + post_s])
    x = np.hstack([acc[lo:hi] / G, np.degrees(gyr[lo:hi])])
    return np.concatenate([x.mean(0), x.std(0), x.min(0), x.max(0)])


def _tilt(Rs, g0, ax_a, ax_b):
    """윈도우 전체 평균 자세. 중력 방향(body)을 자이로 회전으로 추적: g_body(t) = R(t)^T ĝ0.
    반환 (tilt_a, tilt_b)[deg] = 평균 중력 방향이 세션 정지 자세 기준 두 수평 body 축으로 기운 각도.
    QwertyRing의 '행별 절대 피치' — 정렬(구간 자르기) 없이 접근+입력+복귀 전체 평균."""
    gu = g0 / np.linalg.norm(g0)
    gb = np.einsum("nji,j->ni", np.asarray(Rs), gu)       # R^T ĝ0
    m = gb.mean(0)
    m /= np.linalg.norm(m)
    return float(np.degrees(np.arcsin(np.clip(m @ ax_a, -1, 1)))), float(np.degrees(np.arcsin(np.clip(m @ ax_b, -1, 1))))


def _window(t, a, Rs, gyr, B, i0, i1, bias, n=N_WIN):
    """정규화·윈도우 모델용 원시 윈도우 (n, 6): 기준 프레임 선형가속도[m/s^2] 3 + 각속도[dps] 3, 시간 리샘플."""
    w = np.einsum("nij,nj->ni", np.asarray(Rs), gyr[i0:i1 + 1] - bias)
    x = np.hstack([a @ B.T, np.degrees(w @ B.T)])
    tt = t[i0:i1 + 1]
    tq = np.linspace(tt[0], tt[-1], n)
    return np.stack([np.interp(tq, tt, x[:, k]) for k in range(x.shape[1])], 1).astype(np.float32)


class _Ctx:
    """세션 공통 배열·기준축."""

    def __init__(self, sd, p):
        self.t, self.acc, self.gyr = session_arrays(sd)
        self.ref, self.g_rest = _ref_axis(sd, self.t, self.acc)
        self.ax_a, self.ax_b = horizontal_basis(self.g_rest, self.ref)
        self.n_w = max(3, int(round(p.still_win_s * 233)))
        self.mag = None
        if p.mag_db_max > 0:
            from .magnet import MagTrack
            self.mag = MagTrack(sd, self.t)


def _aug_arrays(ctx, lo, hi, aug: Augment | None, rng):
    """[lo, hi) 지역 배열(증강 시 복사본에 배율·잡음)."""
    t, acc, gyr = ctx.t[lo:hi], ctx.acc[lo:hi], ctx.gyr[lo:hi]
    if aug is None:
        return t, acc, gyr
    sa = 1 + rng.normal(0, aug.scale)
    sg = 1 + rng.normal(0, aug.scale)
    acc = acc * sa + rng.normal(0, aug.noise_g * G, acc.shape)
    gyr = gyr * sg + rng.normal(0, np.radians(aug.noise_dps), gyr.shape)
    return t, acc, gyr


def isolated_features(sd, p: MotionParams | None = None, aug: Augment | None = None, windows: bool = False) -> pd.DataFrame:
    p = p or MotionParams()
    ctx = _Ctx(sd, p)
    T = ctx.t
    tr = sd.trials[(sd.trials.status == "ok") & (sd.trials.phase == "isolated")]
    rng = np.random.default_rng(aug.seed if aug else 0)
    rows = []
    for _, x in tr.iterrows():
        ipg = find_press(T, ctx.acc, x.first_key_ns / 1e9, p.press_search_s)
        if ipg is None:
            continue
        lo = int(max(0, np.searchsorted(T, x.cue_ns / 1e9 - 0.6)))
        hi = int(min(len(T), np.searchsorted(T, T[ipg] + RETURN_S) + 5))
        t, acc, gyr = _aug_arrays(ctx, lo, hi, aug, rng)
        ip = ipg - lo
        i0, src = find_onset(t, acc, gyr, ip, x.cue_ns / 1e9 - 0.05, p)
        ie = _quiet_after(t, acc, gyr, ip)
        ir = len(t) - 1                                         # 복귀 포함 윈도우 끝
        if aug is not None and aug.shift:
            i0 = int(np.clip(i0 + rng.integers(-aug.shift, aug.shift + 1), ctx.n_w, ip - 3))
            ie = int(np.clip(ie + rng.integers(-aug.shift, aug.shift + 1), ip + 1, ir))
        if ie <= i0 + 2:
            continue
        g0 = acc[max(0, i0 - ctx.n_w):i0].mean(0)
        bias = gyr[max(0, i0 - ctx.n_w):i0].mean(0)
        gz = g0 / np.linalg.norm(g0)
        Rs_all = _rotations(t, gyr, i0, ir, bias)
        Rs = Rs_all[:ie - i0 + 1]
        mag = {"dB": np.nan, "yaw_res": np.nan, "mag_used": 0}
        if ctx.mag is not None and ctx.mag.ok:
            mag.update(ctx.mag.trial(lo + i0, lo + ie, Rs, gz))
            if np.isfinite(mag["dB"]) and mag["dB"] <= p.mag_db_max and np.isfinite(mag["yaw_res"]):
                from .magnet import yaw_correction
                Rs = yaw_correction(Rs, gz, mag["yaw_res"], t[i0:ie + 1])
                mag["mag_used"] = 1
        a, _ = _lin_acc(t, acc, gyr, i0, ie, g0, bias, Rs)
        e1, e2 = horizontal_basis(gz, ctx.ref)
        B = np.stack([e1, e2, gz])
        d, path, v = _segment_disp(t[i0:ie + 1], a, gz, N_PATH, p.kalman_rho)
        R = Rs[-1]
        th = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        rv = (np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * (th / (2 * np.sin(th)) if th > 1e-9 else 0.5))
        vh = v - np.outer(v @ gz, gz)
        f = {"trial_id": x.trial_id, "block": x.get("block", 0), "key": x.expect, "onset_src": src,
             "dur": t[ie] - t[i0], "ok_dur": (t[ie] - t[i0]) <= p.max_move_s}
        dd = B @ d * 1000
        f.update({"u": dd[0], "w": dd[1], "vert": dd[2]})
        pp = (path @ B.T) * 1000
        for k in range(N_PATH):
            f[f"pu{k}"], f[f"pw{k}"], f[f"pz{k}"] = pp[k]
        rr = B @ rv
        f.update({"rot_u": np.degrees(rr[0]), "rot_w": np.degrees(rr[1]), "rot_z": np.degrees(rr[2]),
                  "vmax_h": np.linalg.norm(vh, axis=1).max() * 1000, "gyr_peak": np.degrees(np.linalg.norm(gyr[i0:ie], axis=1)).max()})
        for k, val in enumerate(_raw_stats(t, acc, gyr, ip)):
            f[f"raw{k}"] = val
        # 행 모델(QwertyRing): 윈도우 전체(접근+입력+복귀) 평균 자세. 절대(세션 정지 기준) + 시도 시작 대비
        ta, tb = _tilt(Rs_all, g0, ctx.ax_a, ctx.ax_b)
        a0, b0 = _tilt(Rs_all[:1], g0, ctx.ax_a, ctx.ax_b)
        f.update({"tilt_a": ta, "tilt_b": tb, "dtilt_a": ta - a0, "dtilt_b": tb - b0})
        f.update({"mag_dB": mag["dB"], "mag_yaw": mag["yaw_res"], "mag_used": mag["mag_used"]})
        if windows:
            a_all, _ = _lin_acc(t, acc, gyr, i0, ir, g0, bias, Rs_all)
            f["win"] = _window(t, a_all, Rs_all, gyr, B, i0, ir, bias)
        rows.append(f)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # 블록 번호(무작위 블록 설계) — plan.csv에서 trial uid로 연결
    plan = pd.read_csv(sd.path / "plan.csv")
    uid2block = dict(zip(plan.uid, plan.block))
    t2uid = dict(zip(sd.trials.trial_id, sd.trials.uid))
    df["block"] = [uid2block[t2uid[i]] for i in df.trial_id]
    return df


def word_presses(sd, p: MotionParams | None = None, times: dict | None = None, windows: bool = False) -> pd.DataFrame:
    """연속 타이핑 정상 단어의 자음 타격별 특징.
    열: trial_id, word, idx, n, key, u, w(F 기준 누적 위치), du, dw(직전 타격 대비), ikt(직전 타격(첫 타격은 움직임 시작)
        부터의 시간[s] — Stage 4 keystroke interval), nv(정답 단어에서 직전 자음과의 사이 오른손 키 수; 학습용),
        tail(마지막 타격 → F 복귀 정지 시간[s], 단어 안 모든 행에 같은 값), nvt(정답 단어의 끝 자음 뒤 오른손 키 수; 학습용),
        z, dz(수직 누적/구간 변위 mm),
        px, py, pz(F 기준 3차원 위치[mm], 닫힘 보정), r3(= |px,py,pz|, F와의 3차원 거리), closure(보정 전 닫힘 오차 mm)
        — 단어는 F에서 시작해 F로 끝난다. 마지막 타격 → F 정지(tail)까지 복귀 구간도 적분해 그 끝 위치(원래 0이어야 함)를
          닫힘 오차로 보고 시간에 비례해 빼 준다(드리프트 보정). F 정지를 못 찾으면 보정 없이 closure=NaN.
          fdist(기기·센서 독립 3차원 거리 디코더)가 이 열을 쓴다.
        tilt_a/b(타격 전후 중간점까지 윈도우 평균 자세), raw*
    times: {trial_id: 타격 시각 배열[s, host]} 를 주면 키 로그 대신 그 시각(예: detect_presses 결과)으로 분할한다.
           이때 key 열은 ''(타격별 정답을 알 수 없음)."""
    p = p or MotionParams()
    ctx = _Ctx(sd, p)
    t, acc, gyr = ctx.t, ctx.acc, ctx.gyr
    n_w = ctx.n_w
    tr = sd.trials[(sd.trials.status == "ok") & (sd.trials.phase == "continuous")]
    keys = sd.keys[sd.keys.event == "down"]
    rows = []
    for _, x in tr.iterrows():
        if times is not None:
            ts = np.asarray(times.get(x.trial_id, []), float)
            if not len(ts):
                continue
            kd = pd.DataFrame({"host_ns": ts * 1e9, "key": [""] * len(ts)})
            search = (-0.03, 0.03)   # 검출 시각이 이미 jerk 피크
        else:
            kd = keys[(keys.trial_id == x.trial_id) & keys.key.isin(LEFT_KEYS)]
            search = p.press_search_s
        if not len(kd):
            continue
        ips = [find_press(t, acc, s / 1e9, search) for s in kd.host_ns]
        if any(i is None for i in ips):
            continue
        # 같은 타격이 두 키로 잡히는 것 방지(인접 타격 30ms 이내면 다음 샘플로 밀기)
        for k in range(1, len(ips)):
            if ips[k] <= ips[k - 1]:
                ips[k] = ips[k - 1] + 1
        i0, src = find_onset(t, acc, gyr, ips[0], x.cue_ns / 1e9 - 0.05, p)
        ends = [min(_quiet_after(t, acc, gyr, ip, max_n=12), (ips[k + 1] - 1) if k + 1 < len(ips) else 10 ** 9)
                for k, ip in enumerate(ips)]
        if ends[-1] <= i0 + 2:
            continue
        g0 = acc[max(0, i0 - n_w):i0].mean(0)
        bias = gyr[max(0, i0 - n_w):i0].mean(0)
        gz = g0 / np.linalg.norm(g0)
        i_last = min(len(t) - 1, ends[-1] + int(0.15 * 233))
        Rs_all = _rotations(t, gyr, i0, i_last, bias)
        Rs = Rs_all[:ends[-1] - i0 + 1]
        mag_used = 0
        if ctx.mag is not None and ctx.mag.ok:
            m = ctx.mag.trial(i0, ends[-1], Rs, gz)
            if np.isfinite(m["dB"]) and m["dB"] <= p.mag_db_max and np.isfinite(m["yaw_res"]):
                from .magnet import yaw_correction
                Rs = yaw_correction(Rs, gz, m["yaw_res"], t[i0:ends[-1] + 1])
                mag_used = 1
        a, _ = _lin_acc(t, acc, gyr, i0, ends[-1], g0, bias, Rs)
        e1, e2 = horizontal_basis(gz, ctx.ref)
        B = np.stack([e1, e2, gz])
        # 정답 단어의 오른손 키 수(직전 자음 ~ 이 자음 사이): Stage 4 타이밍 모델 학습용
        nv = [0] * len(ips)
        nvt = -1
        if times is None:
            try:
                ks = to_keys(x.target)
                cpos = [i for i, c in enumerate(ks) if c in LEFT_KEYS]
                if len(cpos) == len(ips):
                    nv = [0] + [b - a_ - 1 for a_, b in zip(cpos, cpos[1:])]
                    nvt = len(ks) - 1 - cpos[-1]
            except ValueError:
                pass
        tail = tail_settle(t, gyr, ips[-1], p.still_dps)
        a00, b00 = _tilt(Rs_all[:1], g0, ctx.ax_a, ctx.ax_b)      # 단어 시작(F 정지) 자세
        pos = np.zeros(3)
        start = i0
        P3, T3 = [], []                                     # 타격별 3차원 누적 위치(세계 B 좌표, m)와 시각
        k0 = len(rows)
        for k, (ip, ie) in enumerate(zip(ips, ends)):
            ie = max(ie, start + 2)
            seg = slice(start - i0, ie - i0 + 1)
            d, _, _ = _segment_disp(t[start:ie + 1], a[seg], gz, 0, p.kalman_rho, full_zupt=True)
            pos = pos + d
            uw = B @ pos * 1000
            du = B @ d * 1000
            P3.append(uw)
            T3.append(t[ie])
            f = {"trial_id": x.trial_id, "word": x.target, "idx": k, "n": len(ips), "key": kd.key.iloc[k],
                 "u": uw[0], "w": uw[1], "z": uw[2], "du": du[0], "dw": du[1], "dz": du[2],
                 "seg_dur": t[ie] - t[start],
                 "word_t": t[ip] - t[i0], "onset_src": src,
                 "ikt": t[ip] - (t[ips[k - 1]] if k else t[i0]), "nv": nv[k], "tail": tail, "nvt": nvt,
                 "mag_used": mag_used}
            # 타격 윈도우 = 직전 타격과의 중간 ~ 다음 타격과의 중간(첫/끝은 시작/끝)
            wa = i0 if k == 0 else (ips[k - 1] + ip) // 2
            wb = i_last if k == len(ips) - 1 else (ip + ips[k + 1]) // 2
            ta, tb = _tilt(Rs_all[wa - i0:wb - i0 + 1], g0, ctx.ax_a, ctx.ax_b)
            f.update({"tilt_a": ta, "tilt_b": tb, "dtilt_a": ta - a00, "dtilt_b": tb - b00})
            for j, val in enumerate(_raw_stats(t, acc, gyr, ip)):
                f[f"raw{j}"] = val
            if windows:
                f["win"] = _window(t, a[seg], Rs[seg], gyr, B, start, ie, bias)
            rows.append(f)
            start = ie
        # 단어 끝 F 복귀: 마지막 타격 구간 끝 → F 정지까지 적분. 끝 위치는 F(0)여야 하므로 그 차이(닫힘 오차)를
        # 단어 시작부터의 시간에 비례해 각 타격 위치에서 뺀다.
        P3 = np.array(P3)
        closure = np.full(3, np.nan)
        if np.isfinite(tail):
            i_set = min(len(t) - 1, int(np.searchsorted(t, t[ips[-1]] + tail)))
            if i_set > start + 2:
                Rr = _rotations(t, gyr, i0, i_set, bias)
                ar, _ = _lin_acc(t, acc, gyr, start, i_set, g0, bias, Rr[start - i0:])
                d, _, _ = _segment_disp(t[start:i_set + 1], ar, gz, 0, p.kalman_rho, full_zupt=True)
                closure = B @ (pos + d) * 1000
                frac = (np.array(T3) - t[i0]) / max(t[i_set] - t[i0], 1e-6)
                P3 = P3 - frac[:, None] * closure[None]
        for j, r in enumerate(rows[k0:]):
            r.update({"px": P3[j, 0], "py": P3[j, 1], "pz": P3[j, 2], "r3": float(np.linalg.norm(P3[j])),
                      "closure": float(np.linalg.norm(closure))})
    return pd.DataFrame(rows)


def impact_up(acc: np.ndarray, gz: np.ndarray, i: int, half: int = 6) -> float:
    """i 주변 ±half 샘플에서 중력 반대(위쪽) 방향 선형가속도 최대 [m/s^2].
    키를 누를 때는 손가락이 키에 부딪혀 급정지하므로 큰 위쪽 가속(S01: 하위 10% 10 m/s^2)이 나오고,
    키에서 손가락을 뗄 때(keyup)는 작다(S01: 최대 9, 상위 10% 5.6)."""
    lo, hi = max(0, i - half), min(len(acc), i + half + 1)
    return float((acc[lo:hi] @ gz).max() - G)


def typing_acf(j: np.ndarray, fs: float, max_lag_s: float = 1.6, smooth_s: float = 0.03, pad: int = 8192):
    """aLeak III-A1 타이핑 리듬을 FFT로: 위너-힌친 정리로 펄스열의 자기상관 r(τ) = IFFT(|FFT(x)|²)를 구하고
    겹침 길이로 나눠(불편 추정) 돌려준다. (스펙트럼 최대를 바로 쓰면 짧은 단어에서는 단어 길이(~1Hz) 포락선과
    배음이 이겨 틀린다 — S01에서 확인.) 반환 (τ[s], r(τ))."""
    x = np.asarray(j, float)
    if smooth_s > 0:
        h = int(3 * smooth_s * fs)
        k = np.exp(-0.5 * (np.arange(-h, h + 1) / (smooth_s * fs)) ** 2)
        x = np.convolve(x, k / k.sum(), mode="same")
    n = max(pad, 2 * len(x))
    F = np.fft.rfft(x, n=n)
    r = np.fft.irfft(np.abs(F) ** 2, n=n)
    L = int(max_lag_s * fs)
    lags = np.arange(L)
    r = r[:L] / np.maximum(len(x) - lags, 1)
    return lags / fs, r


def pace_from_acf(tau: np.ndarray, r: np.ndarray, min_s: float = 0.2) -> float:
    """τ ≥ min_s 에서 첫 국소 최대들 중 가장 큰 자기상관의 τ = 평균 타격 간격."""
    m = tau >= min_s
    if m.sum() < 3:
        return np.nan
    rr = r[m]
    pk = [i for i in range(1, len(rr) - 1) if rr[i] >= rr[i - 1] and rr[i] >= rr[i + 1] and rr[i] > 0]
    if not pk:
        return np.nan
    return float(tau[m][max(pk, key=lambda i: rr[i])])


def typing_pace_fft(t: np.ndarray, j: np.ndarray, **kw) -> float:
    """한 구간의 평균 타격 간격[s](FFT 자기상관). 단어 하나는 주기가 몇 개뿐이라 세션 평균(fft_session)보다 부정확."""
    if len(j) < 8:
        return np.nan
    fs = (len(t) - 1) / (t[-1] - t[0])
    return pace_from_acf(*typing_acf(j, fs, **kw))


def detect_presses(sd, k_mad: float = 8.0, min_sep_s: float = 0.18, alpha: float = 0.5,
                   impact_min: float = 6.0, pace: str = "median") -> pd.DataFrame:
    """IMU만으로 연속 타이핑 단어 안의 왼손 타격 검출(Stage 2) — 키 로그와 비교해 정밀도/재현율.
    jerk 국소 최대 + 단어별 적응 임계(중앙값 + k·MAD)
    + 충격 방향 게이트: 위쪽 충격 가속 impact_min 미만 피크 제거(키에서 손 떼는 피크 = 기존 오검출의 96%)
    + 최소 간격 min_sep_s: 키 누름 유지 75~160ms < 타격 간격(S01 최소 268ms) → 누름-뗌 쌍이 남아도 강한 쪽만
    + aLeak Algorithm 1(평균 간격 tpace의 alpha배보다 가까운 약한 피크 제거). tpace 추정:
      'median'      : 그 단어 후보 피크 간격의 중앙값(기본)
      'fft'         : 그 단어 펄스열의 FFT 자기상관 f_type (aLeak 원형, 단어가 짧아 부정확)
      'fft_session' : 세션 전체 단어의 FFT 자기상관을 합친 f_type 하나 — 사용자 고유 리듬
      뗌 피크가 사라진 뒤에는 alpha 0.75가 진짜 타격을 지워서(S01 놓침 38개 중 22개) 0.5로 낮춤.
    기본값은 S01 첫 세션(20261103_171029)에서 고름: 정밀도 98.8% · 재현율 96.0% · 단어별 개수 일치 88%
    (이전 63.9% · 98.3% · 12%). impact_min=0, min_sep_s=0.08, alpha=0.75 이면 이전 방식."""
    from .dataset import jerk
    t = sd.imu.host_ns.to_numpy() / 1e9
    j = jerk(sd.imu)
    acc = sd.imu[["ax", "ay", "az"]].to_numpy() * G
    fs = (len(t) - 1) / (t[-1] - t[0])
    tr = sd.trials[(sd.trials.status == "ok") & (sd.trials.phase == "continuous")]
    keys = sd.keys[sd.keys.event == "down"]
    cands = []
    for _, x in tr.iterrows():   # 1차: 후보 피크(임계·충격 게이트·최소 간격)
        lo, hi = np.searchsorted(t, [x.cue_ns / 1e9, x.end_ns / 1e9 + 0.05])
        jj = j[lo:hi]
        if len(jj) < 10:
            continue
        # 단어 시작(cue = F 위 정지 확인 시점) 직후 0.1초 평균으로 중력 방향
        g0 = acc[lo:min(hi, lo + 23)].mean(0)
        gz = g0 / np.linalg.norm(g0)
        thr = np.median(jj) + k_mad * np.median(np.abs(jj - np.median(jj))) + 1e-6
        pk = [i for i in range(1, len(jj) - 1) if jj[i] > thr and jj[i] >= jj[i - 1] and jj[i] >= jj[i + 1]]
        pk = [i for i in pk if jj[i] > 0.05 * max(jj[pk], default=0)] if pk else []
        if impact_min > 0:
            pk = [i for i in pk if impact_up(acc, gz, lo + i) >= impact_min]
        sel = []
        for i in sorted(pk, key=lambda i: -jj[i]):
            if all(abs(t[lo + i] - t[lo + s]) >= min_sep_s for s in sel):
                sel.append(i)
        sel.sort()
        # 충격 게이트를 통과한 피크만 남긴 펄스열(오른손·뗌 진동의 영향 제거) — FFT 리듬용
        imp = np.zeros(len(jj))
        imp[sel] = jj[sel]
        cands.append((x, lo, hi, jj, sel, imp))
    pace_session = np.nan
    if pace == "fft_session":
        # 단어별 자기상관을 겹침 길이로 가중 합산(세션 전체를 이어 붙인 것과 같지만 단어 경계의 가짜 간격이 없음)
        acc_r, acc_w = None, None
        for _, _, _, _, sel, imp in cands:
            if len(sel) < 2:
                continue
            tau, r = typing_acf(imp / max(imp.max(), 1e-12), fs)
            w = np.maximum(len(imp) - np.arange(len(r)), 0).astype(float)
            acc_r = r * w if acc_r is None else acc_r + r * w
            acc_w = w if acc_w is None else acc_w + w
        if acc_r is not None:
            pace_session = pace_from_acf(tau, acc_r / np.maximum(acc_w, 1))
    out = []
    for x, lo, hi, jj, sel, imp in cands:   # 2차: aLeak Algorithm 1
        tpace = np.nan
        if len(sel) >= 3:
            if pace == "fft":
                tpace = typing_pace_fft(t[lo:hi], imp)
            elif pace == "fft_session":
                tpace = pace_session
            if not np.isfinite(tpace):
                tpace = np.median(np.diff(t[lo + np.array(sel)]))
            changed = True
            while changed and len(sel) > 1:
                changed = False
                for a_, b_ in zip(sel, sel[1:]):
                    if t[lo + b_] - t[lo + a_] < alpha * tpace:
                        sel.remove(a_ if jj[a_] < jj[b_] else b_)
                        changed = True
                        break
        det = t[lo + np.array(sel, dtype=int)] if sel else np.array([])
        kd = keys[keys.trial_id == x.trial_id]
        left = kd[kd.key.isin(LEFT_KEYS)].host_ns.to_numpy() / 1e9
        right = kd[~kd.key.isin(LEFT_KEYS)].host_ns.to_numpy() / 1e9
        out.append({"trial_id": x.trial_id, "word": x.target, "det": det, "left": left, "right": right, "tpace": tpace})
    return pd.DataFrame(out)
