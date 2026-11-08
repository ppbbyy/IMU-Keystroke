"""파이프라인 Stage 1: 시도별 body -> 정지 기준 world frame 변환과 수평 변위 추정 (MoLe 방식 변형).

한 시도(F 정지 -> 자음 키 타격)마다
 1) 타격 시점: 키다운 시각 근처 가속도 변화량(jerk) 최대 샘플
 2) 움직임 시작: 타격에서 거꾸로 가며 '정지 창'(자이로 < still_dps, |a| 변동 < 0.02g, 60ms)을 찾음.
    못 찾으면 cue 시점(수집 화면이 정지를 확인한 순간)을 사용 -> onset_src='cue'
 3) 정지 창 평균으로 중력(body)과 자이로 바이어스를 잡고(C(0)), 자이로 적분 회전으로 가속도를 기준 프레임으로 옮긴 뒤
    중력을 빼서 선형가속도 -> 이중적분
 4) 끝점(기본: 타격 직전)에서 '수평' 속도만 0으로 선형 보정(평균 제거). 타격 직전 손가락은 아래로 내려오는 중이라
    수직 속도는 0이 아니므로 보정하지 않는다(파일럿에서 전 축 보정보다 나음).
 5) 수평면(중력에 수직) 2D 좌표 (u, w) [mm]로 투영
원시 자이로는 frames.GYRO_SIGN을 곱해 가속도축 오른손 좌표로 맞춘다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .frames import GYRO_SIGN

G = 9.80665


@dataclass
class MotionParams:
    # 기본값 근거: S01 wear1 560회 비교(scripts/stage1_eval.py --sweep). after+8dps가 LOO 키 54%/행 73%,
    # 레이아웃 맞춤 키 45%/행 71%로 최고. (파일럿 42회에서는 before가 나았음 — 표본이 작았음)
    still_dps: float = 8.0
    still_acc_ptp_g: float = 0.02
    still_win_s: float = 0.06
    max_move_s: float = 1.2        # 이보다 긴 적분 구간은 신뢰하지 않음(플래그)
    press_search_s: tuple = (-0.15, 0.10)
    end: str = "after"             # before: 타격 2샘플 전 / after: 타격 후 키 누름 중 가장 조용한 샘플
    zupt: str = "horizontal"       # horizontal | full | none (수평 변위에는 horizontal=full: 수직 보정은 투영에서 사라짐)
    kalman_rho: float = 0.0        # Stage 1-(5) 칼만 스무딩 q/r 비(0=끔). 중력 추정 오차 추세 g'_e(t)를 빼고 적분
    mag_db_max: float = 0.0        # 자력계 조건부 결합: 시도 ΔB[uT]가 이 값 이하일 때만 요 보정(0=끔, 6축만)


def rotvec_to_mat(v: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def kalman_trend(z: np.ndarray, rho: float, r: float = 1.0) -> np.ndarray:
    """MoLe (5): 관측 z(t) = g_e(t) + 잡음, g_e = 랜덤워크(과정 잡음 q = rho·r). RTS 칼만 스무더로 저주파 추세 g_e를 구한다.
    잡음 r을 크게(rho 작게) 잡을수록 추세가 느려진다. rho→0 이면 '구간 평균'(= 기존 선형 ZUPT와 같은 효과),
    rho가 크면 실제 동작까지 지워진다. 축별 독립, z (n, d)."""
    z = np.asarray(z, float)
    n = len(z)
    if n == 0 or rho <= 0:
        return np.zeros_like(z)
    q = rho * r
    xf = np.empty_like(z)
    Pf = np.empty(n)
    Pp = np.empty(n)
    x = z[0].copy()            # 첫 관측으로 시작(정지 기준이라 오차 작음)
    P = r
    for k in range(n):
        if k:
            P = P + q
        Pp[k] = P
        K = P / (P + r)
        x = x + K * (z[k] - x)
        P = (1 - K) * P
        xf[k], Pf[k] = x, P
    xs = xf.copy()
    for k in range(n - 2, -1, -1):
        C = Pf[k] / Pp[k + 1]
        xs[k] = xf[k] + C * (xs[k + 1] - xf[k])
    return xs


def integrate(t: np.ndarray, acc: np.ndarray, gyr: np.ndarray, g_body0: np.ndarray, bias: np.ndarray,
              zupt: str = "horizontal", kalman_rho: float = 0.0) -> dict:
    """t[s], acc[m/s^2](specific force, body), gyr[rad/s](오른손 좌표). 첫 샘플 = 정지 기준.
    반환: 기준 프레임(body at t0) 변위/속도, 최종 회전."""
    n = len(t)
    R = np.eye(3)                                   # body(t) -> body(t0)
    a = np.empty((n, 3))
    for i in range(n):
        a[i] = R @ acc[i] - g_body0
        if i < n - 1:
            R = R @ rotvec_to_mat((gyr[i] - bias) * (t[i + 1] - t[i]))
    tt = t - t[0]
    dt = np.diff(tt)[:, None]
    gz = g_body0 / np.linalg.norm(g_body0)
    if kalman_rho > 0:
        tr = kalman_trend(a, kalman_rho)
        a = a - (tr - np.outer(tr @ gz, gz))       # 수평 성분 추세만 제거(수직은 타격 감속이 커서 추세 추정이 깨짐)
    v = np.vstack([np.zeros(3), np.cumsum((a[1:] + a[:-1]) / 2 * dt, axis=0)])
    if zupt != "none" and tt[-1] > 0:
        ve = v[-1] if zupt == "full" else v[-1] - (v[-1] @ gz) * gz
        v = v - np.outer(tt / tt[-1], ve)
    s = np.vstack([np.zeros(3), np.cumsum((v[1:] + v[:-1]) / 2 * dt, axis=0)])
    return {"disp": s[-1], "path": s, "vel": v, "R": R, "gz": gz}


def horizontal_basis(gz: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """중력 gz에 수직인 평면의 직교 기저. e1 = ref를 평면에 투영한 방향."""
    e1 = ref - (ref @ gz) * gz
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(gz, e1)


def session_arrays(sd):
    t = sd.imu.host_ns.to_numpy() / 1e9
    acc = sd.imu[["ax", "ay", "az"]].to_numpy() * G
    gyr = np.radians(sd.imu[["gx", "gy", "gz"]].to_numpy()) * GYRO_SIGN
    return t, acc, gyr


def find_press(t, acc, key_s: float, search=(-0.15, 0.10)) -> int | None:
    lo, hi = np.searchsorted(t, [key_s + search[0], key_s + search[1]])
    if hi - lo < 3:
        return None
    j = np.linalg.norm(np.diff(acc[lo:hi + 1], axis=0), axis=1)
    return lo + int(np.argmax(j))


def find_onset(t, acc, gyr, ip: int, lower_s: float, p: MotionParams) -> tuple[int, str]:
    """(정지 창 끝 인덱스, 출처). 정지 창은 [i - n_w, i)."""
    n_w = max(3, int(round(p.still_win_s / np.median(np.diff(t[max(0, ip - 50):ip + 1])))))
    gn = np.degrees(np.linalg.norm(gyr, axis=1))
    an = np.linalg.norm(acc, axis=1) / G
    i_low = int(np.searchsorted(t, lower_s))
    for i in range(ip - 3, max(n_w, i_low), -1):
        if gn[i - n_w:i].max() < p.still_dps and np.ptp(an[i - n_w:i]) < p.still_acc_ptp_g:
            return i, "still"
    return max(i_low, n_w), "cue"


def isolated_displacements(sd, p: MotionParams | None = None, phases=("calib", "isolated")) -> pd.DataFrame:
    """정상 개별 키 시도마다 F→키 수평 변위. 열: trial_id, phase, key, u, w [mm], vert[mm], dur[s], rot_deg, onset_src, ok"""
    p = p or MotionParams()
    t, acc, gyr = session_arrays(sd)
    tr = sd.trials[(sd.trials.status == "ok") & sd.trials.phase.isin(phases)]
    # 세션 공통 기준축: 정지 시 중력에 가장 수직인 body 축(수평 기준 e1)
    rest = sd.trials[(sd.trials.phase == "rest") & (sd.trials.status == "ok")]
    if len(rest):
        r = rest.iloc[0]
        m = (t >= r.cue_ns / 1e9) & (t <= r.end_ns / 1e9)
        g_ref = acc[m].mean(0)
    else:
        g_ref = acc[:200].mean(0)
    ref_axis = np.eye(3)[int(np.argmin(np.abs(g_ref)))]
    rows = []
    for _, x in tr.iterrows():
        ip = find_press(t, acc, x.first_key_ns / 1e9, p.press_search_s)
        if ip is None:
            continue
        i0, src = find_onset(t, acc, gyr, ip, x.cue_ns / 1e9 - 0.05, p)
        n_w = max(3, int(round(p.still_win_s * 233)))
        if p.end == "after":
            k0, k1 = ip + 4, min(len(t) - 2, ip + 35)
            act = np.degrees(np.linalg.norm(gyr[k0:k1], axis=1)) / 10 + \
                np.linalg.norm(acc[k0:k1] - acc[k0 - 1:k1 - 1], axis=1) / G * 20
            ie = k0 + int(np.argmin(act))
        else:
            ie = ip - 2
        if ie <= i0 + 2:
            continue
        g0 = acc[max(0, i0 - n_w):i0].mean(0)
        bias = gyr[max(0, i0 - n_w):i0].mean(0)
        res = integrate(t[i0:ie + 1], acc[i0:ie + 1], gyr[i0:ie + 1], g0, bias, p.zupt, p.kalman_rho)
        e1, e2 = horizontal_basis(res["gz"], ref_axis)
        d = res["disp"]
        dur = t[ie] - t[i0]
        rot = np.degrees(np.arccos(np.clip((np.trace(res["R"]) - 1) / 2, -1, 1)))
        rows.append({"trial_id": x.trial_id, "phase": x.phase, "key": x.expect,
                     "u": d @ e1 * 1000, "w": d @ e2 * 1000, "vert": d @ res["gz"] * 1000,
                     "dur": dur, "rot_deg": rot, "onset_src": src, "ok": dur <= p.max_move_s})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 평가 도우미
def loo_centroid_accuracy(df: pd.DataFrame, keys) -> tuple[float, float]:
    """레이아웃 정보 없이: 키별 중앙값 중심, leave-one-out 최근접 중심 분류의 (키, 행) 정확도."""
    from .hangul import KEY_ROW
    V = df[["u", "w"]].to_numpy()
    K = df.key.to_numpy()
    ck = cr = 0
    for i in range(len(V)):
        m = np.ones(len(V), bool)
        m[i] = False
        cents = {k: np.median(V[m & (K == k)], axis=0) for k in keys if (m & (K == k)).any()}
        pred = min(cents, key=lambda k: np.linalg.norm(V[i] - cents[k]))
        ck += pred == K[i]
        cr += KEY_ROW[pred] == KEY_ROW[K[i]]
    return ck / len(V), cr / len(V)


def robust_layout_fit(df: pd.DataFrame, pitch_mm: float = 19.0, iters: int = 10) -> dict:
    """측정 (u,w) -> 키보드 좌표(F 기준 mm) 2D 유사변환(반사 허용)을 반복 재가중 최소제곱으로. 최근접 키 정확도."""
    from .hangul import KEY_ROW, KEY_XY
    keys = list("qwertasdfgzxcv")
    KP = np.array([[(KEY_XY[k][0] - KEY_XY["f"][0]) * pitch_mm, (KEY_XY[k][1] - KEY_XY["f"][1]) * pitch_mm] for k in keys])
    X = df[["u", "w"]].to_numpy()
    Y = np.array([KP[keys.index(k)] for k in df.key])
    wts = np.ones(len(X))
    for _ in range(iters):
        mx, my = np.average(X, 0, wts), np.average(Y, 0, wts)
        Xc, Yc = X - mx, Y - my
        U, S, Vt = np.linalg.svd((Xc * wts[:, None]).T @ Yc)
        Rm = U @ Vt
        sc = S.sum() / (wts * (Xc ** 2).sum(1)).sum()
        F = Xc @ Rm * sc + my
        r = np.linalg.norm(F - Y, axis=1)
        c = 1.5 * max(np.median(r), 1e-6)
        wts = np.where(r < c, 1.0, c / r)  # Huber
    pred = [keys[int(np.argmin(np.linalg.norm(KP - f, axis=1)))] for f in F]
    acc = float(np.mean([p_ == k for p_, k in zip(pred, df.key)]))
    racc = float(np.mean([KEY_ROW[p_] == KEY_ROW[k] for p_, k in zip(pred, df.key)]))
    return {"scale": float(sc), "key_acc": acc, "row_acc": racc, "fitted": F, "pred": pred}
