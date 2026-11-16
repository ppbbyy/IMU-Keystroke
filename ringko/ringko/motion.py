"""IMU 운동 보조 — 타격 시각 맞춤(find_press), 움직임 시작 정지 창(find_onset), 회전, 칼만 추세, 파라미터.

이 함수들은 기존 타격 검출(features.detect_presses / refine_presses)과 3차원 추적(track3d)이 함께 쓴다.
  find_press  : 키 시각 근처 가속도 변화량(jerk) 최대 샘플 = 타격 순간
  find_onset  : 타격에서 거꾸로 가며 '정지 창'(자이로 < still_dps, |a| 변동 < still_acc_ptp_g, still_win_s)을 찾음.
                못 찾으면 cue 시점(수집 화면이 F 정지를 확인한 순간) → onset_src='cue'
원시 자이로는 frames.GYRO_SIGN을 곱해 가속도축 오른손 좌표로 맞춘다(track3d는 align3d.estimate_gyro_sign으로
라벨 없이 다시 정할 수도 있다).
(이전 버전의 2D 수평 변위 isolated_displacements · 2D 레이아웃 맞춤 robust_layout_fit 등은 제거됨 → track3d.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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


