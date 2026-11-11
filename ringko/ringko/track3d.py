"""F 기준 3차원 상대 위치 추적 — 타격 검출 결과(수정하지 않은 기존 코드)를 입력으로 받는다.

입력
  · 세션 IMU (가속도 [m/s²], 자이로 [rad/s], 시각 [s])
  · 타격 시각: features.detect_presses / refine_presses 결과(IMU만) 또는 키 로그(오라클 평가용)
  · 단어 시작/끝의 F 정지 — motion.find_onset(시작), features.tail_settle(끝) — 기존 함수 그대로

좌표계 (중력 좌표계 G)
  원점  = 단어 시작 시 F 위 정지 위치 (0,0,0)
  up    = 정지 창에서 잰 중력 반대 방향(가속도계가 재는 비력 +g 방향)
  h1,h2 = up에 수직인 수평면 기저. h1 = 세션 공통 body 기준축을 수평면에 투영(단어마다 yaw가 거의 같게).
  수평면의 '키보드 오른쪽'이 어디인지(yaw)와 거울(손잡이)은 여기서 정하지 않는다 → align3d.

단어 하나의 계산
  1) 시작 정지 창 → 중력 g0, 자이로 바이어스 b. (끝 F 정지 창이 있으면 b는 두 창의 평균)
     가속도 배율 보정: |g0| = 9.80665가 되도록 그 단어의 가속도를 스케일(센서별 감도 차이 흡수)
  2) 자이로 적분 R(t): body(t) → body(t0). 선형가속도 a = R f − g0
  3) 구간 = [시작 → 타격1], [타격1 → 타격2], …, [타격n → F 정지].
     타격 순간 손가락은 키 위에서 멈추므로 각 구간 끝 속도 = 0 (3축 ZUPT, 선형 보정) → 구간 변위 d_j
     (구간 경계 = 타격 직후 가장 조용한 샘플, 기존 word_presses와 같은 규칙)
  4) 루프 닫힘: 단어는 F에서 시작해 F로 끝나므로 Σ d_j = 0 이어야 한다. 남는 닫힘 오차 E를
     '구간 오차 분산 ∝ T_j³'(가속도 백색잡음의 이중적분)에 비례해 나눠 뺀다 = 최소분산(가중 최소제곱) 해.
     (closure='linear'면 기존 방식: 시간 비례)
  5) 키보드 평면(선택, plane=True): 모든 타격의 높이 z가 같은 평면(키 윗면) 위라는 약한 제약을 함께 풀어
     수직 드리프트를 줄인다(높이 z_ref는 미지수로 같이 추정).
  6) 타격별 불확실성: 닫힘을 걸면 위치 오차는 '브라운 다리'(양 끝 0) → σ_k² = q·S_k(S−S_k)/S.
     q[mm²/s³]는 세션의 닫힘 오차들로 라벨 없이 추정(|E|² ≈ 3 q S) — 센서 잡음이 달라도 자동으로 맞는다.

한계(정직하게)
  · 끝점이 F라는 사실은 '오차의 합'만 알려 준다. 중간 경로의 오차 모양은 모델(∝T³) 가정으로 나누는 것이며,
    한 구간에서 크게 틀린 오차(충격, 회전 추정 오차)는 다른 구간으로 번진다.
  · 센서는 손끝이 아니라 중위지골에 있다. 위 행으로 손을 뻗으면 손끝은 ~19mm 움직여도 센서는 덜 움직이고
    높이도 변한다 → 배율 대역 [s_lo, s_hi]와 위치 허용 오차로 흡수(consonant3d).
  · 타격이 빠지거나(검출 누락) 더해지면 구간 분할 자체가 틀린다(기존 검출의 한계를 그대로 받음).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import _segment_disp, tail_settle
from .frames import GYRO_SIGN
from .hangul import LEFT_KEYS, to_consonant_keys
from .motion import G, MotionParams, find_onset, find_press, horizontal_basis, rotvec_to_mat


@dataclass
class TrackParams:
    closure: str = "variance"      # variance(∝T³ 최소분산) | linear(시간 비례, 기존) | none
    plane: bool = True             # 키보드 평면 제약(타격 높이 공통)
    sigma_plane_mm: float = 4.0    # 평면 제약 허용 오차(누름 깊이·센서 위치 차이)
    acc_norm: bool = True          # |g0| = G 가 되도록 가속도 배율 보정
    sigma_floor_mm: float = 3.0    # 위치 σ 하한
    gyro_sign: np.ndarray | None = None   # None = frames.GYRO_SIGN, 'auto' 는 align3d.estimate_gyro_sign 결과를 넣는다
    motion: MotionParams = field(default_factory=MotionParams)


@dataclass
class WordTrack:
    """한 시도(단어 또는 개별 키)의 F 기준 3차원 타격 위치. 좌표 = 중력 좌표계 (h1, h2, up) [mm]."""
    trial_id: int
    phase: str
    target: str                 # 단어(연속) 또는 정답 자음(개별 키)
    truth: str | None           # 정답 자음 키열(평가용, 모르면 None)
    t_strike: np.ndarray        # (n,) 타격 시각 [s]
    P: np.ndarray               # (n,3) 타격 위치 (닫힘·평면 보정 후)
    D: np.ndarray               # (n,3) 직전 타격(첫 타격은 F) → 이 타격 변위 (= diff P)
    P_raw: np.ndarray           # (n,3) 보정 전 누적 위치
    sig: np.ndarray             # (n,) 위치 σ [mm, 축당]
    sig_rel: np.ndarray         # (n,) 구간 변위 σ [mm, 축당]
    closure: np.ndarray         # (3,) 보정 전 닫힘 오차 [mm] (F 복귀를 못 찾으면 NaN)
    seg_T: np.ndarray           # (n+1,) 구간 길이 [s] (마지막 = 복귀 구간, 없으면 NaN)
    z_ref: float = 0.0

    @property
    def n(self) -> int:
        return len(self.P)

    @property
    def has_return(self) -> bool:
        return bool(np.isfinite(self.closure).all())


# --------------------------------------------------------------------------- 센서 배열
def session_arrays(sd, gyro_sign=None):
    """(t[s], acc[m/s²], gyr[rad/s] 오른손). 다른 센서는 이 세 배열만 맞춰 주면 된다(단위·시각)."""
    t = sd.imu.host_ns.to_numpy() / 1e9
    acc = sd.imu[["ax", "ay", "az"]].to_numpy() * G
    s = GYRO_SIGN if gyro_sign is None else np.asarray(gyro_sign, float)
    gyr = np.radians(sd.imu[["gx", "gy", "gz"]].to_numpy()) * s
    return t, acc, gyr


def session_ref_axis(sd, t, acc) -> np.ndarray:
    """정지 중력에 가장 수직인 body 축 = 수평 기준 h1의 원천(단어마다 같은 축을 써서 yaw를 맞춤)."""
    rest = sd.trials[(sd.trials.phase == "rest") & (sd.trials.status == "ok")]
    if len(rest):
        r = rest.iloc[0]
        m = (t >= r.cue_ns / 1e9) & (t <= r.end_ns / 1e9)
        g = acc[m].mean(0) if m.any() else acc[:200].mean(0)
    else:
        g = acc[:200].mean(0)
    return np.eye(3)[int(np.argmin(np.abs(g)))]


# --------------------------------------------------------------------------- 타격 시각 공급
def strikes_from_keylog(sd, phase: str = "continuous") -> dict:
    """오라클(평가용): 키 로그의 왼손 키다운 시각[s]. 개별 키/F 보정은 첫 키다운."""
    tr = sd.trials[(sd.trials.status == "ok") & (sd.trials.phase == phase)]
    kd = sd.keys[sd.keys.event == "down"]
    out = {}
    for _, x in tr.iterrows():
        if phase == "continuous":
            k = kd[(kd.trial_id == x.trial_id) & kd.key.isin(LEFT_KEYS)]
            out[x.trial_id] = k.host_ns.to_numpy() / 1e9
        elif np.isfinite(x.first_key_ns):
            out[x.trial_id] = np.array([x.first_key_ns / 1e9])
    return out


def strikes_from_detection(det: pd.DataFrame) -> dict:
    """features.detect_presses / refine_presses 결과(det 열 = 타격 시각[s])를 그대로 쓴다."""
    return {r.trial_id: np.asarray(r.det, float) for _, r in det.iterrows()}


# --------------------------------------------------------------------------- 닫힘 + 평면 보정
def _correct(d: np.ndarray, T: np.ndarray, has_ret: bool, p: TrackParams, q: float):
    """d (m,3) 구간 변위(마지막 = 복귀, has_ret일 때), T (m,) 구간 길이[s].
    반환 (보정된 구간 변위 (m,3), z_ref). 가중 최소제곱:
      min Σ_j |c_j|²/(q W_j) + Σ_k (P_k,z − z_ref)²/σ_p²   s.t.  Σ_j (d_j + c_j) = 0 (닫힘, has_ret일 때)
    W_j = T_j³ (variance: 가속도 백색잡음 이중적분의 분산) 또는 T_j (linear: 기존 '시간 비례' 보정)."""
    m = len(d)
    n = m - 1 if has_ret else m                     # 타격 수
    W = (T ** 3 if p.closure == "variance" else T) + 1e-4
    close = has_ret and p.closure != "none"
    c = np.zeros_like(d)
    if close:
        c = -np.outer(W / W.sum(), d.sum(0))         # 등식 제약 최소분산 해: 닫힘 오차를 W에 비례해 분배
    zr = 0.0
    if p.plane and n >= 1:                           # z축만 다시: 미지수 c_z (m) + z_ref (1)
        Wv = q * (T ** 3 + 1e-4)
        L = np.tril(np.ones((n, m)))                 # P_k = Σ_{j≤k} (d_j + c_j)
        sp = p.sigma_plane_mm
        rows = [np.hstack([np.diag(1 / np.sqrt(Wv)), np.zeros((m, 1))]),           # 사전 c_j ~ N(0, qT³)
                np.hstack([L / sp, -np.ones((n, 1)) / sp])]                         # 평면 P_z ≈ z_ref
        rhs = [np.zeros(m), -(L @ d[:, 2]) / sp]
        if close:
            big = 1e4 / np.sqrt(Wv.min())
            rows.append(np.hstack([np.ones((1, m)) * big, np.zeros((1, 1))]))      # 닫힘(사실상 등식)
            rhs.append(np.array([-d[:, 2].sum() * big]))
        x, *_ = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)
        c[:, 2] = x[:m]
        zr = float(x[m])
    return d + c, zr


# --------------------------------------------------------------------------- 단어 하나
REF_HZ = 233.0   # 기존 코드의 샘플 수 상수(3, 12)가 맞춰진 S01 실측 출력 주기


def _quiet_after(t, acc, gyr, ip, dt, skip_n=3, max_n=12):
    """타격 직후 가장 조용한 샘플(구간 경계). features._quiet_after와 같은 규칙이지만 샘플 수를 시간으로 환산해
    출력 주기가 다른 센서에서도 같은 시간 창(13ms 건너뛰고 ~50ms 안)을 본다. 233Hz에서는 원래 함수와 동일."""
    k = REF_HZ * dt
    k0 = ip + max(1, int(round(skip_n / k)))
    k1 = min(len(t) - 2, ip + max(k0 - ip + 2, int(round(max_n / k))))
    act = np.degrees(np.linalg.norm(gyr[k0:k1], axis=1)) / 10 + \
        np.linalg.norm(acc[k0:k1] - acc[k0 - 1:k1 - 1], axis=1) / G * 20
    return k0 + int(np.argmin(act))


def _rotations(t, gyr, i0, i1, bias):
    R = np.eye(3)
    Rs = np.empty((i1 - i0 + 1, 3, 3))
    for k, i in enumerate(range(i0, i1 + 1)):
        Rs[k] = R
        if i < i1:
            R = R @ rotvec_to_mat((gyr[i] - bias) * (t[i + 1] - t[i]))
    return Rs


def track_trial(t, acc, gyr, ips, cue_s, ref_axis, p: TrackParams):
    """한 시도의 원시 결과(보정 전). 반환 dict 또는 None.
    ips: 타격 샘플 인덱스(오름차순). cue_s: 수집 화면이 F 정지를 확인한 시각(onset 탐색 하한)."""
    mp = p.motion
    dt = float(np.median(np.diff(t[max(0, ips[0] - 60):ips[0] + 1])))
    n_w = max(3, int(round(mp.still_win_s / dt)))   # 샘플레이트 독립(센서마다 ODR이 달라도)
    i0, src = find_onset(t, acc, gyr, ips[0], cue_s - 0.05, mp)
    if i0 - n_w < 0:
        return None
    ends = [min(_quiet_after(t, acc, gyr, ip, dt), (ips[k + 1] - 1) if k + 1 < len(ips) else 10 ** 9)
            for k, ip in enumerate(ips)]
    tail = tail_settle(t, gyr, ips[-1], mp.still_dps)
    i_set = None
    if np.isfinite(tail):
        i_set = min(len(t) - 1 - n_w, int(np.searchsorted(t, t[ips[-1]] + tail)))
        if i_set <= ends[-1] + 2:
            i_set = None
    g0 = acc[i0 - n_w:i0].mean(0)
    bias = gyr[i0 - n_w:i0].mean(0)
    if i_set is not None:
        bias = 0.5 * (bias + gyr[i_set:i_set + n_w].mean(0))
    a_scale = G / np.linalg.norm(g0) if p.acc_norm else 1.0
    i_last = i_set if i_set is not None else ends[-1]
    if i_last <= i0 + 2:
        return None
    Rs = _rotations(t, gyr, i0, i_last, bias)
    f = acc[i0:i_last + 1] * a_scale
    a = np.einsum("nij,nj->ni", Rs, f) - g0 * a_scale
    up = g0 / np.linalg.norm(g0)
    e1, e2 = horizontal_basis(up, ref_axis)
    B = np.stack([e1, e2, up])
    bounds = [i0] + [max(e, i0 + 2) for e in ends] + ([i_set] if i_set is not None else [])
    for k in range(1, len(bounds)):
        bounds[k] = max(bounds[k], bounds[k - 1] + 2)
    if bounds[-1] > i_last:
        return None
    d, T = [], []
    for s, e in zip(bounds[:-1], bounds[1:]):
        dd, _, _ = _segment_disp(t[s:e + 1], a[s - i0:e - i0 + 1], up, 0, 0.0, full_zupt=True)
        d.append(B @ dd * 1000)
        T.append(t[e] - t[s])
    return {"d": np.array(d), "T": np.array(T), "has_ret": i_set is not None, "onset_src": src,
            "t_strike": t[np.asarray(ips)], "t_bounds": t[np.asarray(bounds)]}


def _finish(raw, x, truth, p: TrackParams, q: float) -> WordTrack:
    d, T, has_ret = raw["d"], raw["T"], raw["has_ret"]
    n = len(raw["t_strike"])
    W = T ** 3 + 1e-4
    P_raw = np.cumsum(d, 0)[:n]
    closure = d.sum(0) if has_ret else np.full(3, np.nan)
    dc, zr = _correct(d, T, has_ret, p, q)
    P = np.cumsum(dc, 0)[:n]
    D = dc[:n]
    S = np.cumsum(W)
    if has_ret and p.closure != "none":
        var = q * S[:n] * (S[-1] - S[:n]) / S[-1]
        var_rel = q * W[:n] * (1 - W[:n] / S[-1])
    else:
        var, var_rel = q * S[:n], q * W[:n]
    fl = p.sigma_floor_mm ** 2
    return WordTrack(int(x.trial_id), x.phase, str(x.target), truth, raw["t_strike"], P, D, P_raw,
                     np.sqrt(var + fl), np.sqrt(var_rel + fl), closure, np.r_[T, [np.nan] * (n + 1 - len(T))], zr)


def track_session(sd, strikes: dict, phase: str = "continuous", p: TrackParams | None = None,
                  source: str = "keylog", q: float | None = None) -> tuple[list[WordTrack], dict]:
    """세션의 모든 정상 시도를 추적. strikes = {trial_id: 타격 시각[s] 배열} (strikes_from_*).
    source: 'keylog'(키다운 → find_press로 jerk 피크에 맞춤) | 'detected'(이미 jerk 피크 시각).
    q: 구간 잡음 강도[mm²/s³]. None이면 이 세션 닫힘 오차로 라벨 없이 추정.
    반환 (WordTrack 목록, 정보 dict{q, n_return, closure_med})."""
    p = p or TrackParams()
    t, acc, gyr = session_arrays(sd, p.gyro_sign)
    ref = session_ref_axis(sd, t, acc)
    search = p.motion.press_search_s if source == "keylog" else (-0.03, 0.03)
    tr = sd.trials[(sd.trials.status == "ok") & (sd.trials.phase == phase)]
    raws = []
    for _, x in tr.iterrows():
        ts = np.asarray(strikes.get(x.trial_id, []), float)
        if not len(ts):
            continue
        ips = [find_press(t, acc, s, search) for s in ts]
        if any(i is None for i in ips):
            continue
        for k in range(1, len(ips)):
            if ips[k] <= ips[k - 1]:
                ips[k] = ips[k - 1] + 1
        raw = track_trial(t, acc, gyr, ips, x.cue_ns / 1e9, ref, p)
        if raw is None:
            continue
        truth = None
        if source == "keylog" or phase != "continuous":
            try:
                truth = to_consonant_keys(x.target) if phase == "continuous" else str(x.expect)
            except ValueError:
                truth = None
        raws.append((raw, x, truth))
    E2 = [(r["d"].sum(0) ** 2).sum() / (3 * (r["T"] ** 3 + 1e-4).sum()) for r, _, _ in raws if r["has_ret"]]
    if q is None:
        q = float(np.median(E2)) if E2 else 2.0e4
    tracks = [_finish(r, x, tr_, p, q) for r, x, tr_ in raws]
    cl = [np.linalg.norm(w.closure) for w in tracks if w.has_return]
    return tracks, {"q": q, "n": len(tracks), "n_return": len(cl),
                    "closure_med_mm": float(np.median(cl)) if cl else float("nan"), "ref_axis": ref}


def tracks_frame(tracks: list[WordTrack]) -> pd.DataFrame:
    """타격별 표(분석·저장용). 열: trial_id, phase, target, idx, n, key(정답 키, 모르면 ''), px,py,pz, dx,dy,dz,
    r3(F 거리), sig, sig_rel, closure(mm)."""
    rows = []
    for w in tracks:
        for k in range(w.n):
            rows.append({"trial_id": w.trial_id, "phase": w.phase, "target": w.target, "idx": k, "n": w.n,
                         "key": w.truth[k] if w.truth and len(w.truth) == w.n else "",
                         "px": w.P[k, 0], "py": w.P[k, 1], "pz": w.P[k, 2],
                         "dx": w.D[k, 0], "dy": w.D[k, 1], "dz": w.D[k, 2],
                         "r3": float(np.linalg.norm(w.P[k])), "sig": w.sig[k], "sig_rel": w.sig_rel[k],
                         "closure": float(np.linalg.norm(w.closure))})
    return pd.DataFrame(rows)
