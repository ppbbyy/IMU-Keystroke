"""F 기준 3차원 거리 디코더 — 기기·센서 독립(키 위치 학습 없음).

생각
  단어는 F에서 시작해 F로 끝난다(features.word_presses의 px,py,pz: 복귀 구간까지 적분해 닫힘 오차를 뺀 F 기준 3차원 위치).
  거리는 회전에 불변이다. 그래서 센서를 어떻게 끼웠는지(축 방향), 손 자세가 어떤지(요), 착용마다 달라지는 회전 A를
  몰라도 된다. 키 위치는 학습하지 않고 키보드 물리 배열(키 간격 pitch_mm, 행 엇갈림)과 키 누름 깊이만 쓴다.

  1) 후보: 타격 k의 F 거리 r_k = |p_k| 가 키 c의 [min_c, max_c] 안에 있으면 c는 후보.
       d_c   = |배열상 c 위치 − F|  (3차원: 가로·세로 + 누름 깊이 press_mm)
       min_c = max(0, s_lo · d_c − tol_mm),  max_c = s_hi · d_c + tol_mm
     밖이면 벗어난 만큼 부드럽게 깎는다(-(초과/σ_out)²/2) — 경계 근처 오차로 정답이 아예 사라지지 않게.
  2) 사슬(사전 없이): 1)의 unary + 이웃 타격 간 거리 |p_k − p_{k−1}| ≈ s·|M_a − M_b| (쌍 퍼텐셜) → forward-backward.
  3) 단어(사전): 점 집합 {F, p_1..p_n} 의 모든 쌍 거리 vs 후보 자음열의 배열상 쌍 거리. 배율 s는 후보마다
     최소제곱으로 구하고 [s_lo, s_hi]로 자른다(기기·적분 배율 차이 흡수). + 1)의 대역 점수 + λ·log 빈도.

  한계: 거리만 쓰므로 F에 대해 거울상인 위치(예: d와 g는 둘 다 F에서 1u)는 F 거리로는 못 가른다 —
  다른 타격과의 쌍 거리와 사전이 가른다.

중력 방향 + 세션 yaw (calibrate 후, d/g 구분)
  px,py는 중력에 수직인 수평면, pz는 중력 축이다. 중력만으로는 '위'만 알 뿐 수평면의 방향(키보드 오른쪽이
  어디인지 = yaw)은 모른다 → d(F 왼쪽)/g(F 오른쪽)는 중력만으로도 못 가른다. 그러나
    · 중력을 쓰면 남는 자유도는 수평면 회전(yaw)과 축 손잡이(거울, 센서 축 정의에 따름)뿐이고
    · 한 세션 안에서 손가락 yaw는 거의 일정하다(S01: 단어별 yaw 합성 길이 R = 0.91~0.97).
  그래서 calibrate(정답 없이 그 세션 단어들만으로)가
    1) 손잡이 flip ∈ {+1, −1}과 세션 yaw μ, 집중도 κ(von Mises)를 추정하고
    2) 이후 점수에서 수평 위치를 '방향까지' 맞춘다:
       단어: 후보마다 yaw θ를 Procrustes(회전만, 거울 없음) + 사전 κ·cos(θ−μ)로 구해 |O − s·R(θ)·E|²
       사슬: 타격 위치 ≈ R(μ)·E_c, 이웃 변위 ≈ R(μ)·(E_b − E_a) (yaw 불확실성만큼 σ를 키움)
  키 위치 학습은 여전히 없다(배열만). calibrate는 시험 세션의 라벨 없는 타이핑으로 그때그때 한다.
  calibrate 전(mu=None)에는 거리만 쓰는 위의 동작 그대로.
"""
from __future__ import annotations

import numpy as np

from .decode import KEYS, KI
from .hangul import HOME_KEY, KEY_XY


def key_xyz(pitch_mm: float = 19.0, press_mm: float = 4.0) -> np.ndarray:
    """(14,3) F(정지, 누르지 않음) 기준 키 타격 위치[mm]. x = 오른쪽, y = 아래 행, z = 누름 깊이(아래 −)."""
    f = KEY_XY[HOME_KEY]
    return np.array([[(KEY_XY[k][0] - f[0]) * pitch_mm, (KEY_XY[k][1] - f[1]) * pitch_mm, -press_mm] for k in KEYS])


class FDistModel:
    def __init__(self, pitch_mm: float = 19.0, press_mm: float = 4.0, scale=(0.8, 1.25), tol_mm: float = 12.0,
                 sigma_out: float = 6.0, sigma_pair: float = 10.0):
        self.P = key_xyz(pitch_mm, press_mm)                                  # (14,3)
        self.Px = np.vstack([np.zeros(3), self.P])                            # 0 = F 정지(단어 시작·끝)
        self.d = np.linalg.norm(self.P, axis=1)                               # (14,) F 거리
        self.s_lo, self.s_hi = scale
        self.rmin = np.maximum(0.0, self.s_lo * self.d - tol_mm)
        self.rmax = self.s_hi * self.d + tol_mm
        self.sigma_out, self.sigma_pair = sigma_out, sigma_pair
        self.D = np.linalg.norm(self.P[:, None] - self.P[None], axis=-1)      # (14,14) 키 간 거리
        self.flip, self.mu, self.kappa = 1.0, None, 0.0                       # 중력 + 세션 yaw (calibrate)
        self.press_mm = press_mm

    # ------------------------------------------------------------------ 중력 + yaw
    @property
    def oriented(self) -> bool:
        return self.mu is not None

    def _E(self) -> np.ndarray:
        """(15,2) F(0)와 키의 수평 배열 위치(손잡이 flip 적용)."""
        Eh = self.Px[:, :2].copy()
        Eh[:, 1] *= self.flip
        return Eh

    @staticmethod
    def _rot(th):
        c, s = np.cos(th), np.sin(th)
        return np.array([[c, -s], [s, c]])

    def _sigma_yaw(self, dist):
        """yaw 불확실성(κ)을 거리 dist[mm]의 위치 오차로 바꿔 σ_pair와 합친 표준편차."""
        sd_th = 1.0 / np.sqrt(max(self.kappa, 1e-3))
        return np.sqrt(self.sigma_pair ** 2 + (np.asarray(dist) * sd_th) ** 2)

    def _yaw_fit(self, pos, idx, flip, mu=None, kappa=0.0):
        """후보별 (θ*, s*, 수평 log 점수). 회전만(거울 없음) Procrustes + yaw 사전."""
        O = np.vstack([np.zeros(2), np.asarray(pos, float)[:, :2]])           # (n+1,2) F 포함
        Eh = self.Px[:, :2] * [1.0, flip]
        E = Eh[np.hstack([np.zeros((len(idx), 1), int), idx])]                 # (m,n+1,2)
        A = (E * O[None]).sum((1, 2))                                          # Σ E·O
        B = (E[..., 0] * O[None, :, 1] - E[..., 1] * O[None, :, 0]).sum(1)     # Σ E×O
        ee = (E ** 2).sum((1, 2))
        w = 1.0 / self.sigma_pair ** 2
        prior = (kappa * np.cos(mu), kappa * np.sin(mu)) if mu is not None else (0.0, 0.0)
        th = np.arctan2(w * B + prior[1], w * A + prior[0])
        s = np.clip((A * np.cos(th) + B * np.sin(th)) / np.maximum(ee, 1e-9), self.s_lo, self.s_hi)
        oo = (O ** 2).sum()
        sq = oo - 2 * s * (A * np.cos(th) + B * np.sin(th)) + s ** 2 * ee       # |O − sR(θ)E|²
        ll = -sq * w / 2
        if mu is not None:
            ll = ll + kappa * np.cos(th - mu)
        return th, s, ll

    def calibrate(self, pos_list, lex, iters: int = 2, top: int = 1):
        """정답 없이 손잡이(flip)·세션 yaw(μ)·집중도(κ) 추정. pos_list: 단어별 (n,3) F 기준 위치. self를 바꾸고 반환."""
        best = None
        for flip in (1.0, -1.0):
            mu, kappa = None, 0.0
            for _ in range(iters):
                ths = []
                for pos in pos_list:
                    pos = np.asarray(pos, float)
                    cs, idx = lex.strings(len(pos))
                    if not cs or not np.isfinite(pos).all():
                        continue
                    th, _, ll = self._yaw_fit(pos, idx, flip, mu, kappa)
                    ll = ll + self._band(pos)[np.arange(len(pos))[None], idx - 1].sum(1)
                    for i in np.argsort(-ll)[:top]:
                        ths.append(th[i])
                if not ths:
                    break
                z = np.exp(1j * np.array(ths)).mean()
                R_ = float(min(abs(z), 0.98))
                mu, kappa = float(np.angle(z)), R_ * (2 - R_ ** 2) / (1 - R_ ** 2)   # von Mises κ 근사
            if mu is not None and (best is None or kappa > best[2]):
                best = (flip, mu, kappa)
        if best is not None:
            self.flip, self.mu, self.kappa = best
        return self

    # ------------------------------------------------------------------ 1) F 거리 대역
    def candidates(self, pos: np.ndarray) -> np.ndarray:
        """(n,14) bool: 타격별로 F 거리가 [min, max] 안에 드는 키."""
        r = np.linalg.norm(np.asarray(pos, float), axis=1)[:, None]
        return (r >= self.rmin[None]) & (r <= self.rmax[None])

    def _band(self, pos: np.ndarray) -> np.ndarray:
        """(n,14) log: 대역 안 0, 밖은 벗어난 거리만큼 가우스로 감점."""
        r = np.linalg.norm(np.asarray(pos, float), axis=1)[:, None]
        ex = np.maximum(self.rmin[None] - r, 0) + np.maximum(r - self.rmax[None], 0)
        return -0.5 * (ex / self.sigma_out) ** 2

    def unary(self, pos: np.ndarray) -> np.ndarray:
        """(n,14) log. 대역 점수 + (calibrate 후) 세션 yaw로 돌린 배열 위치와의 수평 방향 일치."""
        u = self._band(pos)
        if self.oriented:
            Eh = self._E()[1:] @ self._rot(self.mu).T                          # (14,2)
            O = np.asarray(pos, float)[:, :2]
            d2 = ((O[:, None] - Eh[None]) ** 2).sum(-1)
            u = u - d2 / (2 * self._sigma_yaw(np.linalg.norm(Eh, axis=1))[None] ** 2)
        return u

    # ------------------------------------------------------------------ 2) 사슬 주변확률
    def _pair(self, dist: float) -> np.ndarray:
        """(14,14) log: 이웃 타격 거리 dist가 배율 대역 안의 s·D[a,b] 와 얼마나 맞는지."""
        lo, hi = self.s_lo * self.D, self.s_hi * self.D
        ex = np.maximum(lo - dist, 0) + np.maximum(dist - hi, 0)
        return -0.5 * (ex / self.sigma_pair) ** 2

    def _pair_vec(self, dv: np.ndarray) -> np.ndarray:
        """(14,14) log: 이웃 수평 변위 dv가 R(μ)·(E_b − E_a)와 맞는지(calibrate 후)."""
        Eh = self._E()[1:] @ self._rot(self.mu).T
        V = Eh[None] - Eh[:, None]                                             # (a,b,2)
        d2 = ((dv[None, None] - V) ** 2).sum(-1)
        return -d2 / (2 * self._sigma_yaw(np.linalg.norm(V, axis=-1)) ** 2)

    def marginals(self, pos: np.ndarray, unary: np.ndarray | None = None) -> np.ndarray:
        """(n,14) 타격별 자음 사후확률(사전 없이). unary: 추가 log 점수(예: 분류기)."""
        pos = np.asarray(pos, float)
        n = len(pos)
        u = self.unary(pos) + (unary if unary is not None else 0)
        if self.oriented:
            psi = [self._pair_vec(pos[k, :2] - pos[k - 1, :2]) for k in range(1, n)]
        else:
            psi = [self._pair(float(np.linalg.norm(pos[k] - pos[k - 1]))) for k in range(1, n)]
        la = np.zeros((n, 14))
        la[0] = u[0]
        for k in range(1, n):
            la[k] = np.logaddexp.reduce(la[k - 1][:, None] + psi[k - 1], axis=0) + u[k]
        lb = np.zeros((n, 14))
        for k in range(n - 2, -1, -1):
            lb[k] = np.logaddexp.reduce(psi[k] + (u[k + 1] + lb[k + 1])[None], axis=1)
        lp = la + lb
        lp -= lp.max(1, keepdims=True)
        p = np.exp(lp)
        return p / p.sum(1, keepdims=True)

    # ------------------------------------------------------------------ 3) 사전 단어 점수
    def score_strings(self, pos: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """pos (n,3), idx (m,n) 키 번호+1(Lexicon.strings 형식). 반환 (m,) log 점수."""
        pos = np.asarray(pos, float)
        n = len(pos)
        if self.oriented:
            _, _, ll = self._yaw_fit(pos, idx, self.flip, self.mu, self.kappa)
            return ll + self._band(pos)[np.arange(n)[None], idx - 1].sum(1)
        Q = np.vstack([np.zeros(3), pos])                                     # 관측 점: F, p_1..p_n
        iu, ju = np.triu_indices(n + 1, 1)
        do = np.linalg.norm(Q[iu] - Q[ju], axis=1)                            # (L,)
        full = np.hstack([np.zeros((len(idx), 1), int), idx])                 # (m, n+1)
        E = self.Px[full]                                                     # (m, n+1, 3)
        de = np.linalg.norm(E[:, iu] - E[:, ju], axis=-1)                     # (m, L)
        s = np.clip((de @ do) / np.maximum((de ** 2).sum(1), 1e-9), self.s_lo, self.s_hi)
        ll = -((do[None] - s[:, None] * de) ** 2).sum(1) / (2 * self.sigma_pair ** 2)
        u = self._band(pos)
        return ll + u[np.arange(n)[None], idx - 1].sum(1)

    def rank_words(self, lex, pos: np.ndarray, lam: float = 0.5, unary: np.ndarray | None = None):
        """[(점수, 단어)] 내림차순. lex: decode.Lexicon."""
        n = len(pos)
        cs, idx = lex.strings(n)
        if not cs:
            return []
        ll = self.score_strings(pos, idx)
        if unary is not None:
            ll = ll + unary[np.arange(n)[None], idx - 1].sum(1)
        out = [(ll[i] + lam * np.log(f), w) for i, s in enumerate(cs) for w, f in lex.by_len[n][s]]
        out.sort(reverse=True)
        return out


def candidate_sets(model: FDistModel, pos: np.ndarray) -> list[str]:
    """사람이 보기 위한 타격별 후보 키 문자열."""
    return ["".join(k for k, ok in zip(KEYS, row) if ok) for row in model.candidates(pos)]


__all__ = ["FDistModel", "key_xyz", "candidate_sets", "KI"]
