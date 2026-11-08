"""자력계 조건부 결합(fallback) — docs/pipeline.docx ① 구체화.

노트북 근처에서는 힌지 자석·스피커 때문에 자력계를 믿을 수 없으므로 기본 분석은 6축이고, 자력계는
'시도 구간의 순간 변화량 ΔB가 임계값 이하'일 때만 자이로 적분 회전의 요(yaw, 중력축 회전) 드리프트를 고치는 데 쓴다.

1) hard-iron 보정: 보드 자체 자성은 body 좌표에 고정된 오프셋 h. 축 확인(axes) 단계처럼 손을 여러 방향으로 돌린
   구간의 자력계 샘플에 구(sphere)를 맞춰 h를 구한다(B_body = R^T B_world + h).
2) ΔB: 시도 구간에서 보정된 |B|의 최대-최소[uT]. 환경 자기장이 균일하고 정적이면 회전해도 |B|는 일정하므로
   ΔB가 크다 = 손가락이 자기장 기울기(자석 근처)를 지나가거나 간섭이 있다는 뜻.
3) 요 잔차: 자이로 회전 R(t)(body t -> body t0)로 B를 기준 프레임에 옮기면 원래 일정해야 한다.
   수평면에서 B_ref(끝)과 B_ref(시작)의 각도 = 자이로 요 드리프트 추정치 → ΔB가 작을 때만 보정에 사용.
"""
from __future__ import annotations

import numpy as np


def fit_hard_iron(B: np.ndarray) -> tuple[np.ndarray, float]:
    """구 맞춤 |B - h| = r 의 선형 최소제곱. 반환 (h, r). 회전이 충분하지 않으면 h가 불안정하므로 호출 쪽에서 확인."""
    B = np.asarray(B, float)
    A = np.hstack([2 * B, np.ones((len(B), 1))])
    y = (B ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    h = sol[:3]
    r = float(np.sqrt(max(sol[3] + h @ h, 1e-9)))
    return h, r


def session_hard_iron(sd) -> np.ndarray:
    """axes 단계(없으면 전체)의 자력계로 hard-iron 오프셋. 자력계가 비었으면 0."""
    if sd.mag is None or not len(sd.mag):
        return np.zeros(3)
    B = sd.mag[["mx", "my", "mz"]].to_numpy(float)
    ax = sd.trials[(sd.trials.phase == "axes") & (sd.trials.status == "ok")]
    if len(ax):
        r = ax.iloc[0]
        m = (sd.mag.host_ns >= r.cue_ns) & (sd.mag.host_ns <= r.end_ns)
        if m.sum() > 200:
            B = B[m.to_numpy()]
    h, _ = fit_hard_iron(B)
    return h


class MagTrack:
    """IMU 샘플 시각으로 보간한 hard-iron 보정 자력계(body, uT)."""

    def __init__(self, sd, t_imu: np.ndarray):
        self.ok = sd.mag is not None and len(sd.mag) > 10
        if not self.ok:
            return
        h = session_hard_iron(sd)
        tm = sd.mag.host_ns.to_numpy() / 1e9
        Bm = sd.mag[["mx", "my", "mz"]].to_numpy(float) - h
        self.B = np.stack([np.interp(t_imu, tm, Bm[:, k]) for k in range(3)], 1)
        self.h = h

    def trial(self, i0: int, i1: int, Rs, gz: np.ndarray) -> dict:
        """[i0, i1] 구간(Rs: 각 샘플의 body(t)->body(i0) 회전, 길이 i1-i0+1).
        반환 dB[uT], yaw_res[deg](자이로 적분 요 드리프트 추정, 부호: gz축 오른손)."""
        if not self.ok:
            return {"dB": np.nan, "yaw_res": np.nan}
        B = self.B[i0:i1 + 1]
        n = np.linalg.norm(B, axis=1)
        Bref = np.einsum("nij,nj->ni", np.asarray(Rs), B)
        h0 = Bref[0] - (Bref[0] @ gz) * gz
        h1 = Bref[-1] - (Bref[-1] @ gz) * gz
        if np.linalg.norm(h0) < 1e-6 or np.linalg.norm(h1) < 1e-6:
            return {"dB": float(np.ptp(n)), "yaw_res": np.nan}
        ang = np.arctan2(np.cross(h0, h1) @ gz, h0 @ h1)
        return {"dB": float(np.ptp(n)), "yaw_res": float(np.degrees(ang))}


def yaw_correction(Rs, gz: np.ndarray, yaw_res_deg: float, tt: np.ndarray):
    """자력계가 본 요 잔차를 시간에 선형으로 나눠 회전을 고친다: R'(t) = Rz(-α·t/T) R(t).
    (자이로 바이어스 잔차는 시간에 선형으로 쌓이므로.)"""
    from .motion import rotvec_to_mat
    T = tt[-1] - tt[0]
    if not np.isfinite(yaw_res_deg) or T <= 0:
        return Rs
    a = np.radians(yaw_res_deg)
    return [rotvec_to_mat(-gz * a * (ti - tt[0]) / T) @ R for R, ti in zip(Rs, tt)]
