"""F 기준 3차원 위치로 자음 후보와 점수를 낸다 — 학습 없음(키보드 물리 배열 + 추정 오차만).

입력: track3d.WordTrack (중력 좌표계 타격 위치 P, 구간 변위 D, 타격별 σ) + Alignment(중력 좌표계 → 키보드 좌표계)
출력: Obs3D (타격별 14자음 log 점수 U, 이웃 타격 쌍 점수 Pair) → 후보 집합, 사슬 주변확률(사전 없이), 사전 단어 점수

점수(타격 k, 자음 c; p = 키보드 좌표 위치, r = |p|, 극좌표로 분해)
  위치가 키 중심 μ_c 주위 등방 가우스라면  log N(p; μ_c, σ²) = −(r − |μ_c|)²/2σ² − r|μ_c|(1 − cos Δθ)/σ² + const
  즉 '거리 항'과 '방향 항'으로 정확히 갈린다. 여기서
   ① 거리 항  → F 거리 min/max 대역(keymap3d.band, 평평한 윗면): 대역 안 0, 밖은 −½(초과/σ)²
                (회전 불변 — 센서 방향·yaw를 몰라도 된다)
   ② 방향 항  → von Mises κ_eff·(cos Δθ − 1),  1/κ_eff = σ_eff²/(r_h·d_c) + σ_ψ²
                σ_eff² = σ_k² + 칸 안 타격 분산,  σ_ψ = yaw 추정 불확실성(정렬이 약하면 방향 항이 저절로 약해짐)
   ③ 높이     → 평면 제약 후 z는 키마다 같으므로 점수에는 안 넣고, 거리 r을 3차원으로 잴 때만 쓴다
   ④ 이전 타격 → 이동 벡터 Δ = p_k − p_{k−1} 를 두 칸 사이 거리 범위(pair_band)와 방향(V_ab)으로 같은 방식
  모드 비교 (허용 범위를 어떤 '영역'으로 볼 것인가)
   'shell'  : ①+④거리 — F 중심 구 껍질(거리 구간만). 정렬 불필요(완전 무보정)
   'sector' : ①②+④ — 구 껍질 ∩ 방향 부채꼴 = 극좌표 3차원 영역 (기본)
   'box'    : 칸 상자 ⊗ 가우스(직교 좌표 3차원 영역): Σ_axis log[Φ((hi−p)/σ) − Φ((lo−p)/σ)]
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .keymap3d import KEYS, KI, Keyboard3D


def _verf(x):
    """erf 근사(Abramowitz–Stegun 7.1.26, 최대 오차 1.5e-7) — scipy 없이 벡터 연산."""
    x = np.asarray(x, float)
    s = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return s * y


def _log_box(p, lo, hi, sig):
    """log P(관측 p | 참값 ~ U[lo,hi], 잡음 N(0,σ²)) (축 하나). 브로드캐스트."""
    z1 = (hi - p) / (np.sqrt(2) * sig)
    z0 = (lo - p) / (np.sqrt(2) * sig)
    v = 0.5 * (_verf(z1) - _verf(z0))
    return np.log(np.maximum(v, 1e-300)) - np.log(hi - lo)


# --------------------------------------------------------------------------- 정렬
@dataclass
class Alignment:
    """중력 좌표계 (h1,h2,up) → 키보드 좌표계 (x,y,z):  K = R(θ)·diag(1, flip)·h / s,  z = up / s.
    θ = yaw, flip = 손잡이(센서 축이 거울상이면 −1), s = 배율(센서 이동/손끝 이동), sigma_yaw = yaw 불확실성[rad]."""
    theta: float = 0.0
    flip: float = 1.0
    scale: float = 1.0
    sigma_yaw: float = np.pi        # 모르면 π: 방향 항이 사실상 꺼짐
    source: str = "none"

    def apply(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        h = X[..., :2] * [1.0, self.flip]
        c, s_ = np.cos(self.theta), np.sin(self.theta)
        out = np.empty_like(X)
        out[..., 0] = c * h[..., 0] - s_ * h[..., 1]
        out[..., 1] = s_ * h[..., 0] + c * h[..., 1]
        out[..., 2] = X[..., 2]
        return out / self.scale

    @property
    def oriented(self) -> bool:
        return self.sigma_yaw < 1.0


@dataclass
class Obs3D:
    U: np.ndarray                 # (n,14) 타격별 자음 log 점수
    Pair: np.ndarray              # (n,14,14) [k, 이전 a, 현재 b] (k=0은 0)
    parts: dict = field(default_factory=dict)   # 구성요소별 (n,14) — 분석용

    @property
    def n(self):
        return len(self.U)


@dataclass
class Scorer3D:
    kb: Keyboard3D = field(default_factory=Keyboard3D)
    mode: str = "sector"          # shell | sector | box
    w_abs: float = 1.0            # 절대 위치(F 기준) 항 가중
    w_rel: float = 1.0            # 이전 타격 대비 이동 항 가중
    w_dir: float = 1.0            # 방향 항 가중(sector)

    # ------------------------------------------------------------------ 타격별 / 쌍
    def unary(self, P: np.ndarray, sig: np.ndarray, al: Alignment, z_ref: float = 0.0) -> dict:
        """타격별 구성요소 log 점수 dict{band, radial, dir | box}, 각 (n,14)."""
        kb = self.kb
        p = al.apply(P)                                                    # (n,3) 키보드 좌표
        n = len(p)
        sg = (np.asarray(sig, float) / al.scale)[:, None]
        zr = z_ref / al.scale
        out = {}
        if self.mode == "box":
            se = np.sqrt(sg ** 2 + 1e-6)
            sc_lo, sc_hi = kb.scale
            L = 0
            for ax in range(2):     # 칸(± pitch/2)을 배율 대역만큼 넓힌 상자 ⊗ 가우스
                lo_, hi_ = kb.C2[:, ax] - kb.pitch_mm / 2, kb.C2[:, ax] + kb.pitch_mm / 2
                L = L + _log_box(p[:, ax:ax + 1], np.minimum(lo_ * sc_hi, lo_ * sc_lo), np.maximum(hi_ * sc_hi, hi_ * sc_lo), se)
            out["box"] = L - 0.5 * ((p[:, 2:3] - zr) / np.sqrt(sg ** 2 + kb.z_env_mm ** 2 / 16)) ** 2
            return out
        r = np.linalg.norm(p - [0, 0, 0], axis=1)[:, None]                 # 3차원 F 거리
        dk, dmin, dmax = kb.band3d(np.full(n, zr), sg)
        ex = np.maximum(dmin - r, 0) + np.maximum(r - dmax, 0)
        out["band"] = -0.5 * (ex / sg) ** 2                                # ① min/max 게이트
        var_r = sg ** 2 + kb.hit_var + (kb.scale_sd * dk) ** 2
        out["radial"] = -0.5 * (r - dk) ** 2 / var_r                       # ① 대역 안 순위(기준 거리 d_k)
        if self.mode == "sector" and al.oriented:                          # ② 방향
            rh = np.linalg.norm(p[:, :2], axis=1)[:, None]
            ang = np.arctan2(p[:, 1], p[:, 0])[:, None] - np.arctan2(kb.C2[:, 1], kb.C2[:, 0])[None]
            inv_k = (sg ** 2 + kb.hit_var) / np.maximum(rh * kb.d[None], 1e-6) + al.sigma_yaw ** 2
            out["dir"] = self.w_dir * (np.cos(ang) - 1) / inv_k
        return out

    def pair(self, D: np.ndarray, sig_rel: np.ndarray, al: Alignment) -> np.ndarray:
        """(n,14,14) 이전 a → 현재 b. k=0(F에서 첫 타격)은 unary와 같은 정보라 0."""
        kb = self.kb
        n = len(D)
        out = np.zeros((n, 14, 14))
        dv = al.apply(D)
        Vn = np.linalg.norm(kb.V2, axis=-1)
        Vang = np.arctan2(kb.V2[..., 1], kb.V2[..., 0])
        for k in range(1, n):
            sg = sig_rel[k] / al.scale
            var = sg ** 2 + 2 * kb.hit_var
            if self.mode == "box":
                d2 = ((dv[k, None, None, :2] - kb.V2) ** 2).sum(-1)
                out[k] = -d2 / (2 * (var + (kb.scale_sd * Vn) ** 2))
                continue
            L = np.linalg.norm(dv[k, :2])                                  # 평면 위 이동 거리
            lo, hi = kb.pair_band(sg)
            ex = np.maximum(lo - L, 0) + np.maximum(L - hi, 0)
            s = -0.5 * (ex / sg) ** 2 - 0.5 * (L - Vn) ** 2 / (var + (kb.scale_sd * Vn) ** 2)
            if self.mode == "sector" and al.oriented:
                ang = np.arctan2(dv[k, 1], dv[k, 0]) - Vang
                inv_k = var / np.maximum(L * Vn, 1e-6) + al.sigma_yaw ** 2
                s = s + self.w_dir * np.where(Vn > 1e-6, (np.cos(ang) - 1) / inv_k, 0.0)
            out[k] = s
        return out

    def observe(self, track, al: Alignment) -> Obs3D:
        parts = self.unary(track.P, track.sig, al, track.z_ref)
        U = self.w_abs * sum(parts.values())
        return Obs3D(U, self.w_rel * self.pair(track.D, track.sig_rel, al), parts)

    # ------------------------------------------------------------------ 후보 / 사슬 / 사전
    def candidates(self, track, al: Alignment) -> np.ndarray:
        """(n,14) bool: F 거리가 [d_min, d_max] 대역 안인 자음(1차 후보, 거리만)."""
        p = al.apply(track.P)
        r = np.linalg.norm(p, axis=1)[:, None]
        _, lo, hi = self.kb.band3d(np.full(track.n, track.z_ref / al.scale), (track.sig / al.scale)[:, None])
        return (r >= lo) & (r <= hi)

    def score_strings(self, obs: Obs3D, idx: np.ndarray) -> np.ndarray:
        """idx (m,n): 키 번호+1 (decode.Lexicon.strings 형식). 반환 (m,) 사슬 에너지 Σ U + Σ Pair."""
        n = obs.n
        c = idx - 1
        ll = obs.U[np.arange(n)[None], c].sum(1)
        if n > 1:
            ll = ll + obs.Pair[np.arange(1, n)[None], c[:, :-1], c[:, 1:]].sum(1)
        return ll


def chain_marginals(obs: Obs3D, lm=None, gamma: float = 0.5, return_logz: bool = False):
    """사전 없이 타격별 자음 사후확률 (n,14) — forward-backward.
    lm: decode.ConsonantLM(사전 텍스트로 만든 자음 트라이그램, IMU 학습 아님), gamma 가중."""
    U, Pr = obs.U, obs.Pair
    n = len(U)
    if lm is None or gamma == 0:
        la = np.zeros((n, 14))
        la[0] = U[0]
        for k in range(1, n):
            la[k] = np.logaddexp.reduce(la[k - 1][:, None] + Pr[k], axis=0) + U[k]
        lb = np.zeros((n, 14))
        for k in range(n - 2, -1, -1):
            lb[k] = np.logaddexp.reduce(Pr[k + 1] + (U[k + 1] + lb[k + 1])[None], axis=1)
        logz = float(np.logaddexp.reduce(la[-1]))
        lp = la + lb
    else:
        T3 = gamma * lm.logp                                     # (15,15,14) [a, b, c], 0 = 단어 시작
        NEG = -1e18
        A = np.full((15, 14), NEG)                               # [이전 상태 a(0..14), 현재 b]
        A[0] = T3[0, 0] + U[0]
        las = [A]
        for k in range(1, n):
            s = np.logaddexp.reduce(A[:, :, None] + T3[:, 1:, :], axis=0)       # (b, c)
            A = np.full((15, 14), NEG)
            A[1:] = s + Pr[k] + U[k][None]
            las.append(A)
        lbs = [None] * n
        lbs[-1] = np.zeros((15, 14))
        for k in range(n - 2, -1, -1):
            term = Pr[k + 1] + U[k + 1][None] + lbs[k + 1][1:]                   # (b, c)
            lbs[k] = np.logaddexp.reduce(T3[:, 1:, :] + term[None], axis=2)
        logz = float(np.logaddexp.reduce(las[-1].ravel()))
        lp = np.stack([np.logaddexp.reduce(a + b, axis=0) for a, b in zip(las, lbs)])
    lp = lp - lp.max(1, keepdims=True)
    P = np.exp(lp)
    P /= P.sum(1, keepdims=True)
    return (P, logz) if return_logz else P


def candidate_strings(scorer: Scorer3D, track, al: Alignment, top: int = 3, lm=None, gamma: float = 0.5):
    """사람이 보기 위한 타격별 상위 후보 [('ㄱ', 확률), …] 목록."""
    from .hangul import KEY_CONSONANT
    M = chain_marginals(scorer.observe(track, al), lm, gamma)
    return [[(KEY_CONSONANT[KEYS[i]], float(M[k, i])) for i in np.argsort(-M[k])[:top]] for k in range(len(M))]


__all__ = ["Alignment", "Obs3D", "Scorer3D", "chain_marginals", "candidate_strings", "KEYS", "KI"]
