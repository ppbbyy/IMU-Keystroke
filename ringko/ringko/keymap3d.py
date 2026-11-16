"""키보드 물리 기하 — F = (0,0,0) 기준 자음 키의 3차원 위치와 F 거리 min/max 허용 범위.

좌표계 (키보드 좌표계 K, 오른손)
  x = 키보드 오른쪽, y = 사용자에게서 먼 쪽(Q행 방향, '위 행'), z = 키 윗면에 수직 위쪽.
  단위: mm. 내부 계산은 모두 키 간격 pitch_mm로 정해지므로, 키 단위(u = pitch)로 정규화하면 키보드가 달라도
  같은 표가 된다(노트북 18~19mm, 데스크톱 19.05mm — 자로 10키 폭을 재서 pitch만 바꾸면 끝, 학습 없음).

키 하나의 '타격 가능 영역' = 키 칸(cell): 중심 ± pitch/2 (손가락은 키 윗면 어디든 칠 수 있다)
  · 3차원: 높이 z ∈ [−z_env, 0] (누름 깊이 + 센서가 손끝이 아닌 중위지골에 있어 생기는 높이 차)

F 거리 범위 (자음 k)
  d_k      = |c_k|                       칸 중심까지 거리
  r_min,k  = 칸(3차원 상자)에서 원점까지 최소 거리 (F 칸이면 0)
  r_max,k  = 칸 꼭짓점 중 가장 먼 거리
  d_min,k  = max(0, s_lo · r_min,k − m·σ),   d_max,k = s_hi · r_max,k + m·σ
    s_lo, s_hi : 적분·자세에 따른 배율 오차 대역(센서가 손끝보다 덜 움직임 등)
    σ          : 위치 추정 오차(track3d가 타격마다 내는 σ_k; 표에는 대표값)
    m          : 허용 배수(기본 2 → 가우스 가정 시 ~95%)
  임의의 숫자가 아니라 '칸 크기(물리) ⊕ 배율 오차 ⊕ 추정 오차'로 정해진다.

두 키 사이(이전 타격 a → 현재 b)의 이동 거리 범위도 같은 방식: 두 칸 사이 최소·최대 거리.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .hangul import HOME_KEY, KEY_CONSONANT, KEY_XY, LEFT_KEYS

KEYS = list(LEFT_KEYS)
KI = {k: i for i, k in enumerate(KEYS)}


def _rect_min_max(c: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """원점에서 중심 c (..,3), 반폭 h (3,)인 축 정렬 상자까지 최소/최대 거리."""
    a = np.abs(c)
    lo = np.sqrt((np.maximum(a - h, 0) ** 2).sum(-1))
    hi = np.sqrt(((a + h) ** 2).sum(-1))
    return lo, hi


@dataclass
class Keyboard3D:
    pitch_mm: float = 19.0       # 키 간격(노트북 S01 실측 개별 키 중앙값 ≈ 19.5)
    z_env_mm: float = 8.0        # 타격 높이 범위(누름 깊이 1~4mm + 센서 높이 변화)
    scale: tuple = (0.8, 1.2)    # 배율 대역 s_lo, s_hi
    margin: float = 2.0          # 허용 배수 m
    cap_mm: float = 14.5         # 키캡 윗면 폭 → 칸 안 타격 산포 σ_hit = cap/√12 (균등분포 표준편차)
    scale_sd: float = 0.1        # 타격마다 남는 배율 흔들림(대역 안 순위용 가우스 폭)

    def __post_init__(self):
        f = KEY_XY[HOME_KEY]
        p = self.pitch_mm
        # 칸 중심 (z는 칸 높이 범위의 가운데)
        self.C = np.array([[(KEY_XY[k][0] - f[0]) * p, -(KEY_XY[k][1] - f[1]) * p, -self.z_env_mm / 2] for k in KEYS])
        self.C2 = self.C[:, :2]
        self.h = np.array([p / 2, p / 2, self.z_env_mm / 2])
        self.d = np.linalg.norm(self.C2, axis=1)                         # 수평 중심 거리
        self.rmin, self.rmax = _rect_min_max(self.C, self.h)            # 3차원 칸 거리 범위
        V = self.C[None] - self.C[:, None]                               # (a,b,3) 칸 중심 간 벡터
        self.V2 = V[..., :2]
        # 두 칸 사이 거리 범위: 차이 벡터의 상자 반폭 = 2h (z는 같은 평면이므로 한 칸 높이만)
        self.pmin, self.pmax = _rect_min_max(V, np.array([p, p, self.z_env_mm / 2]))
        self.hit_var = self.cap_mm ** 2 / 12.0                           # 키캡 위 균등 타격 위치의 축당 분산

    def band3d(self, z_ref, sigma) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """타격 평면 높이 z_ref(단어별 추정)에서의 3차원 F 거리 (d_k, d_min, d_max), 각 (n,14).
        칸 = (x,y) ± pitch/2, z = z_ref ± z_env/4."""
        z = np.asarray(z_ref, float).reshape(-1, 1)
        C = np.broadcast_to(self.C2[None], (len(z), 14, 2))
        C3 = np.concatenate([C, np.broadcast_to(z[:, :, None], (len(z), 14, 1))], -1)
        lo, hi = _rect_min_max(C3, np.array([self.pitch_mm / 2, self.pitch_mm / 2, self.z_env_mm / 4]))
        s_lo, s_hi = self.scale
        sig = np.asarray(sigma, float).reshape(-1, 1)
        dk = np.linalg.norm(C3, axis=-1)
        return dk, np.maximum(0.0, s_lo * lo - self.margin * sig), s_hi * hi + self.margin * sig

    # 키 단위 정규화 ----------------------------------------------------------
    def to_units(self, x_mm):
        return np.asarray(x_mm, float) / self.pitch_mm

    def band(self, sigma) -> tuple[np.ndarray, np.ndarray]:
        """σ(스칼라 또는 (n,1))에 대한 (d_min, d_max) (…,14)."""
        s_lo, s_hi = self.scale
        sig = np.asarray(sigma, float)
        return (np.maximum(0.0, s_lo * self.rmin - self.margin * sig), s_hi * self.rmax + self.margin * sig)

    def pair_band(self, sigma) -> tuple[np.ndarray, np.ndarray]:
        s_lo, s_hi = self.scale
        sig = np.asarray(sigma, float)
        return (np.maximum(0.0, s_lo * self.pmin - self.margin * sig), s_hi * self.pmax + self.margin * sig)

    def table(self, sigma: float = 8.0) -> pd.DataFrame:
        """자음별 기준 좌표·거리·min/max 표 (mm, 그리고 키 단위 u)."""
        lo, hi = self.band(sigma)
        df = pd.DataFrame({"자음": [KEY_CONSONANT[k] for k in KEYS], "키": KEYS,
                           "x": self.C[:, 0], "y": self.C[:, 1], "z": self.C[:, 2],
                           "d_k": np.linalg.norm(self.C, axis=1), "r_min(칸)": self.rmin, "r_max(칸)": self.rmax,
                           "d_min": lo, "d_max": hi,
                           "방향[deg]": np.where(self.d > 1e-9, np.degrees(np.arctan2(self.C[:, 1], self.C[:, 0])), np.nan)})
        df["d_k[u]"] = df.d_k / self.pitch_mm
        df["d_min[u]"] = df.d_min / self.pitch_mm
        df["d_max[u]"] = df.d_max / self.pitch_mm
        return df.round(2)
