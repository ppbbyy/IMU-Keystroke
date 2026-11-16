"""Stage 2 타격 검출(+ 그 보조 함수). 이 파일의 검출 알고리즘은 3차원 자음 추론 개편에서 수정하지 않았다.

detect_presses    : Stage 2. jerk 피크 + 충격 방향 게이트 + aLeak Algorithm 1(타이핑 리듬 = FFT f_type 또는 중앙 간격).
refine_presses    : detect_presses 뒤 타격 수 보정. 단어가 F로 돌아와 끝나는 '닫힘'으로 약한 후보 타격을 검증해 추가.
tail_settle       : 마지막 타격 → F 복귀 후 정지 시점.
_quiet_after, _segment_disp, _lin_acc, _rotations, _Ctx : 위 함수들이 쓰는 적분 보조(track3d도 그대로 가져다 쓴다).

(이전 버전에 있던 2D 특징 추출 isolated_features / word_presses / Augment 는 2D 자음 추론과 함께 제거됨.
 타격 사이 IMU로 F 기준 3차원 위치를 구하는 일은 track3d.py가 맡는다.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .hangul import LEFT_KEYS
from .motion import G, MotionParams, find_onset, horizontal_basis, kalman_trend, rotvec_to_mat, session_arrays

TAIL_SKIP_S = 0.05   # 단어 끝: 마지막 타격 충격 구간은 건너뛰고
TAIL_STILL_S = 0.25  # 이 길이만큼 자이로가 정지 임계 아래로 유지되는 첫 시점 = F 복귀 후 정지
TAIL_MAX_S = 2.5     # 이 안에 정지가 없으면 NaN


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


# --------------------------------------------------------------------------- F 복귀 닫힘으로 타격 수 보정
def _closure_mm(ctx, p: MotionParams, cue_s: float, ips: list[int]) -> float:
    """타격 인덱스 ips로 단어를 적분(타격마다 3축 ZUPT)하고 마지막 타격 → F 정지까지 복귀까지 이었을 때
    끝 위치와 F(0)의 거리[mm] = 닫힘 오차. F 정지를 못 찾으면 NaN. (word_presses와 같은 적분, 자력계 없음)"""
    t, acc, gyr = ctx.t, ctx.acc, ctx.gyr
    ips = sorted(ips)
    i0, _ = find_onset(t, acc, gyr, ips[0], cue_s - 0.05, p)
    ends = [min(_quiet_after(t, acc, gyr, ip, max_n=12), (ips[k + 1] - 1) if k + 1 < len(ips) else 10 ** 9)
            for k, ip in enumerate(ips)]
    tail = tail_settle(t, gyr, ips[-1], p.still_dps)
    if not np.isfinite(tail) or ends[-1] <= i0 + 2:
        return float("nan")
    i_set = min(len(t) - 1, int(np.searchsorted(t, t[ips[-1]] + tail)))
    g0 = acc[max(0, i0 - ctx.n_w):i0].mean(0)
    bias = gyr[max(0, i0 - ctx.n_w):i0].mean(0)
    gz = g0 / np.linalg.norm(g0)
    a, _ = _lin_acc(t, acc, gyr, i0, max(i_set, ends[-1]), g0, bias)
    pos = np.zeros(3)
    start = i0
    for ie in ends + [i_set]:
        ie = max(ie, start + 2)
        d, _, _ = _segment_disp(t[start:ie + 1], a[start - i0:ie - i0 + 1], gz, 0, p.kalman_rho, full_zupt=True)
        pos = pos + d
        start = ie
    return float(np.linalg.norm(pos) * 1000)


def refine_presses(sd, det: pd.DataFrame, p: MotionParams | None = None, ratio_min: float = 1.0,
                   impact_min: float = 2.5, min_sep_s: float = 0.18, gain: float = 0.6, min_drop_mm: float = 5.0,
                   max_add: int = 2, k_mad: float = 8.0, allow_remove: bool = False) -> pd.DataFrame:
    """Stage 2 검출(detect_presses, 그대로 둠) 뒤에 붙는 타격 수 보정. 단어는 F에서 시작해 F로 끝난다는 제약을 쓴다.

    진짜 타격은 손가락이 키에서 멈추는 순간이라, 그 시각에 속도 0(ZUPT)을 넣으면 적분 드리프트가 줄어
    F로 돌아온 끝 위치가 F에 더 가까워진다(닫힘 오차 감소). 가짜 타격에 속도 0을 넣으면 오히려 커진다.
      추가: 단어 구간(F에서 출발 ~ F 복귀 정지) 안의 약한 후보 — jerk 국소 최대가 단어 임계의 ratio_min배 이상이고
            위쪽 충격이 impact_min 이상(검출의 충격 게이트 6 m/s²에 걸려 빠진 타격; S01 놓침의 절반) —
            를 하나씩 넣어 보고, 닫힘 오차가 gain배 미만 그리고 min_drop_mm 이상 줄면 받아들인다(최대 max_add개).
      제거(allow_remove, 기본 끔): 마지막 검출을 빼 봤을 때 같은 기준으로 닫힘이 좋아지면 뺀다(F 복귀 착지가
            타격으로 잡힌 경우). S01에서는 진짜 마지막 타격도 빼 버려(고침 0~1, 망침 7~14) 끔 — 닫힘만으로는
            '마지막 키 누름'과 'F 착지'를 못 가른다.
    기본값(S01 세 세션 249단어): 개수 일치 88→90%, 96→98%, 97→97% — 고친 단어 3, 망친 단어 0.
    gain을 0.85 이상으로 느슨하게 하면 망치는 단어가 생긴다. 근본적으로 못 찾는 놓침(충격도 jerk도 거의 없는
    부드러운 타격, S01 놓침의 약 절반)은 이 방법으로도 안 된다.
    반환: det 복사본. det = 보정된 타격 시각, n_added, n_removed, closure0/closure(보정 전후 mm) 열 추가."""
    from .dataset import jerk
    p = p or MotionParams()
    ctx = _Ctx(sd, p)
    t, acc = ctx.t, ctx.acc
    j = jerk(sd.imu)
    tr = sd.trials.set_index("trial_id")
    out = []
    for _, r in det.iterrows():
        r = r.to_dict()
        x = tr.loc[r["trial_id"]]
        cue = x.cue_ns / 1e9
        dt_ = np.asarray(r["det"], float)
        r["n_added"], r["n_removed"], r["closure0"], r["closure"] = 0, 0, np.nan, np.nan
        if not len(dt_):
            out.append(r)
            continue
        ips = sorted({int(np.argmin(np.abs(t - s))) for s in dt_})
        c0 = _closure_mm(ctx, p, cue, ips)
        r["closure0"] = r["closure"] = c0
        if not np.isfinite(c0):
            out.append(r)
            continue
        lo, hi = np.searchsorted(t, [cue, x.end_ns / 1e9 + 0.05])
        jj = j[lo:hi]
        thr = np.median(jj) + k_mad * np.median(np.abs(jj - np.median(jj))) + 1e-6
        g0 = acc[lo:min(hi, lo + 23)].mean(0)
        gz = g0 / np.linalg.norm(g0)
        i0, _ = find_onset(t, acc, ctx.gyr, ips[0], cue - 0.05, p)
        tail = tail_settle(t, ctx.gyr, ips[-1], p.still_dps)
        i_end = min(hi - 1, int(np.searchsorted(t, t[ips[-1]] + (tail if np.isfinite(tail) else 0.0))))
        cand = []
        for i in range(max(i0, lo + 1), i_end):
            k = i - lo
            if jj[k] >= ratio_min * thr and jj[k] >= jj[k - 1] and jj[k] >= jj[k + 1] and \
                    impact_up(acc, gz, i) >= impact_min:
                cand.append(i)
        # 0.1초 안의 후보는 가장 센 것 하나만
        cand = sorted(cand, key=lambda i: -j[i])
        keep = []
        for i in cand:
            if all(abs(t[i] - t[q]) >= 0.1 for q in keep):
                keep.append(i)
        c = c0
        for _ in range(max_add):
            trial = [i for i in keep if all(abs(t[i] - t[q]) >= min_sep_s for q in ips)]
            best = None
            for i in trial:
                ci = _closure_mm(ctx, p, cue, ips + [i])
                if np.isfinite(ci) and (best is None or ci < best[1]):
                    best = (i, ci)
            if best is None or not (best[1] < gain * c and c - best[1] >= min_drop_mm):
                break
            ips = sorted(ips + [best[0]])
            c = best[1]
            r["n_added"] += 1
        if allow_remove and len(ips) >= 2 and r["n_added"] == 0:
            cr = _closure_mm(ctx, p, cue, ips[:-1])
            if np.isfinite(cr) and cr < gain * c and c - cr >= min_drop_mm:
                ips = ips[:-1]
                c = cr
                r["n_removed"] = 1
        r["det"] = np.asarray(t[np.array(ips)], float)
        r["closure"] = c
        out.append(r)
    return pd.DataFrame(out)
