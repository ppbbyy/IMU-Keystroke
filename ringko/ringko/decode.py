"""연속 타이핑 디코딩: 기하 매칭(aLeak식 상대 벡터) + 타격별 분류기 + 사전.

모델 (한 단어의 자음 타격 k = 1..n, c_0 = 'F 정지 위치')
  관측 d_k = 타격 k-1 -> k 수평 변위(측정 공간, mm)
  기대    = (M[c_k] - M[c_{k-1}]) @ A (+ b, 첫 이동만)
                                           M: 개별 키 시도에서 잰 키별 F 기준 중앙 위치(사용자 고유 기하)
                                              또는 from_layout(두벌식 물리 배열만)
                                           A: 2x2 선형 보정(회전·배율 — 연속 타이핑 동작이 개별 키보다 작고 비스듬함)
                                           b: 시작 오프셋(F 정지 자세 차이). A, b는 착용마다 다시 맞춘다(fit / adapt)
  log ψ_k(c_{k-1}, c_k) = -|d_k - 기대|^2 / (2σ^2)
  log φ_k(c_k)          = β · log P_cls(c_k | 타격 k 국소 특징)  (+ η · log P(행 | 자세), RowModel)
  단어 점수 = Σ log ψ + Σ log φ + λ · log 빈도(word) (+ κ·cos 내적, + τ·타이밍)
자음 단위 성능은 사전 없이 같은 사슬 모델의 forward-backward 주변확률로 계산.

추가 구성요소(docs/pipeline.docx '문제점' · Stage 4)
  RowModel     (A) QwertyRing식 행 모델: 윈도우 전체 평균 자세(tilt)를 행별 정규분포로 → 키별 unary
  TimingModel  Stage 4 keystroke interval: 타격 간격 ~ 이동거리(Fitts) + 사이 오른손 키 수 → 쌍 퍼텐셜/단어 점수
  inner        (B) aLeak식 기하 매칭: 후보 자음열로 만든 가상 궤적과 관측 궤적의 내적(코사인)
               — rank_words(score=...) 와 kbest_sequences + rerank_inner(사전 없는 자음 후보 재순위)
  착용별 보정   Geometry.from_layout / fit(회전·배율 A + 시작 오프셋 b) / adapt(정답 없이 EM)
  domain_vocab 맥락(사용 분야) 사전으로 범위 좁히기
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .hangul import LEFT_KEYS, to_consonant_keys

KEYS = list(LEFT_KEYS)
KI = {k: i for i, k in enumerate(KEYS)}


class Geometry:
    def __init__(self, key_pos: np.ndarray):
        self.M = np.asarray(key_pos, float)       # (14, 2)
        self.Mx = np.vstack([np.zeros(2), self.M])  # 0 = 시작(F 정지), 1.. = 키
        self.A = np.eye(2)
        self.b = np.zeros(2)                       # 첫 이동(F 정지 -> 첫 키) 오프셋
        self.sigma = 20.0

    @staticmethod
    def from_isolated(iso_df) -> "Geometry":
        return Geometry(iso_df.groupby("key")[["u", "w"]].median().loc[KEYS].to_numpy())

    @staticmethod
    def from_layout(pitch_mm: float = 19.0) -> "Geometry":
        """두벌식 물리 배열만으로(개별 키 데이터 없이). 센서축↔키보드축 회전·배율·반사는 fit/adapt의 A가 맡는다."""
        from .hangul import KEY_XY
        f = KEY_XY["f"]
        return Geometry(np.array([[(KEY_XY[k][0] - f[0]) * pitch_mm, (KEY_XY[k][1] - f[1]) * pitch_mm] for k in KEYS]))

    def copy(self) -> "Geometry":
        g = Geometry(self.M.copy())
        g.A, g.b, g.sigma = self.A.copy(), self.b.copy(), self.sigma
        return g

    def expected(self, cstr: str) -> np.ndarray:
        idx = [0] + [KI[c] + 1 for c in cstr]
        e = np.diff(self.Mx[idx], axis=0) @ self.A
        e[0] += self.b
        return e

    def fit(self, obs_list, cstr_list, offset: bool = True, ridge: float = 0.0, prior=None) -> "Geometry":
        """관측 d = (M[c_k]-M[c_{k-1}]) @ A + [k=1]·b 최소제곱. 착용마다 다시 맞추면 '착용별 회전·오프셋 보정'.
        ridge > 0 (보정 모드): (A, b)를 prior(기본: 현재 값) 쪽으로 당긴다. ridge = '가상 타격 수'로, 데이터 규모
        (열별 평균 제곱)에 맞춰 스케일한다. 보정 단어가 적으면 σ 추정이 작게 나와 과신하므로 σ는 줄이지 않는다."""
        E = np.vstack([np.diff(self.Mx[[0] + [KI[c] + 1 for c in s]], axis=0) for s in cstr_list])
        F = np.vstack([np.eye(len(s))[:, :1] for s in cstr_list])            # 첫 이동 표시
        D = np.vstack(obs_list)
        X = np.hstack([E, F]) if offset else E
        if ridge > 0:
            if prior is None:
                prior = (self.A, self.b)
            P0 = np.vstack([prior[0], prior[1][None]]) if offset else prior[0]
            scale = np.sqrt(np.maximum((X ** 2).mean(0), 1e-9))              # 열별 RMS
            L = np.diag(np.sqrt(ridge) * scale)
            coef, *_ = np.linalg.lstsq(np.vstack([X, L]), np.vstack([D, L @ P0]), rcond=None)
        else:
            coef, *_ = np.linalg.lstsq(X, D, rcond=None)
        sig0 = self.sigma
        self.A = coef[:2]
        self.b = coef[2] if offset else np.zeros(2)
        self.sigma = float(np.sqrt(((D - X @ coef) ** 2).mean()))
        if ridge > 0:
            self.sigma = max(self.sigma, sig0)
        return self

    def adapt(self, obs_list, lexicon=None, iters: int = 5, ridge: float = 50.0, lm=None, gamma: float = 0.5) -> "Geometry":
        """정답 없는 착용별 재보정(hard EM): 현재 A,b로 각 단어를 디코딩(사전 1위, 없으면 사슬 최빈 자음열) →
        그 자음열로 A,b 재적합 반복. 링을 새로 낀 뒤 정답 없이 타이핑만으로 보정.
        ridge는 매번 '처음' 값 쪽으로 당기고(반복하며 표류하지 않게), σ는 처음 값으로 고정(자기 디코딩 과신 방지)."""
        prior = (self.A.copy(), self.b.copy())
        sig0 = self.sigma
        for _ in range(iters):
            cs = []
            for o in obs_list:
                r = lexicon.rank_words(self, o) if lexicon is not None else []
                if r:
                    cs.append(to_consonant_keys(r[0][1]))
                else:
                    P = chain_marginals(self, o, None, lm, gamma)
                    cs.append("".join(KEYS[i] for i in P.argmax(1)))
            self.fit(obs_list, cs, ridge=ridge, prior=prior)
            self.sigma = sig0
        return self

    def expected_pairs(self) -> np.ndarray:
        """(15,14,2) [이전 상태, 현재 키] 기대 변위."""
        exp = (self.Mx[None, 1:, :] - self.Mx[:, None, :]) @ self.A
        exp[0] += self.b
        return exp

    def pair_loglik(self, d: np.ndarray) -> np.ndarray:
        """(15, 14): [이전 상태(0=시작,1..14), 현재 키] -> log ψ."""
        return -((self.expected_pairs() - d) ** 2).sum(-1) / (2 * self.sigma ** 2)


class RowModel:
    """(A) QwertyRing 행 모델: 윈도우 전체 평균 자세 x(tilt)를 행별 정규분포로.
    log P(row(c) | x)(행 균등 사전)를 14키 unary로 돌려준다(같은 행의 키는 같은 값). 공분산은 행 공통(표본이 적어 안정)."""

    def __init__(self, cols=("tilt_a", "tilt_b")):
        self.cols = list(cols)

    def fit(self, df) -> "RowModel":
        from .hangul import KEY_ROW
        X = df[self.cols].to_numpy(float)
        r = df.key.map(KEY_ROW).to_numpy()
        self.mu = np.stack([X[r == k].mean(0) for k in range(3)])
        R = X - self.mu[r]
        self.cov = np.atleast_2d(np.cov(R.T)) + 1e-3 * np.eye(len(self.cols))
        self.icov = np.linalg.inv(self.cov)
        self.row_of = np.array([KEY_ROW[k] for k in KEYS])
        return self

    def row_logpost(self, df) -> np.ndarray:
        X = df[self.cols].to_numpy(float)
        dif = X[:, None, :] - self.mu[None]
        L = -0.5 * np.einsum("nri,ij,nrj->nr", dif, self.icov, dif)               # (n,3)
        return L - np.logaddexp.reduce(L, axis=1, keepdims=True)

    def unary(self, df) -> np.ndarray:
        return self.row_logpost(df)[:, self.row_of]                                 # (n,14)


def right_counts(word: str) -> list[int]:
    """단어의 자음 타격마다 직전 자음과의 사이 오른손(모음) 키 수. 첫 자음은 0."""
    from .hangul import to_keys
    ks = to_keys(word)
    cp = [i for i, c in enumerate(ks) if c in KI]
    return [0] + [b - a - 1 for a, b in zip(cp, cp[1:])]


class TimingModel:
    """Stage 4 keystroke interval.
    ikt_k = c0 + c1·|기대 변위| + c2·[첫 타격] (+ c3·사이 오른손 키 수) + N(0, s²)
    Fitts 법칙(멀수록 오래)과 '모음을 치는 동안 왼손이 기다린다'를 함께 반영. 사이 오른손 키 수는 단어마다 정해져 있으므로
    사전 점수(word_loglik)에서 쓰고, 사전 없는 사슬 모델(pair_loglik)에는 그 항 없이 맞춘 모델(with_nv=False)을 쓴다."""

    def __init__(self, with_nv: bool = True):
        self.with_nv = with_nv

    def _X(self, dist, first, nv):
        cols = [np.ones_like(dist), dist, first.astype(float)]
        if self.with_nv:
            cols.append(np.asarray(nv, float))
        return np.stack(cols, -1)

    def fit(self, geo: Geometry, wp) -> "TimingModel":
        """wp: word_presses(키 로그 분할) — key, idx, ikt, nv 열."""
        dist, first, nv, y = [], [], [], []
        for _, g in wp.groupby("trial_id", sort=False):
            e = geo.expected("".join(g.key))
            dist.extend(np.linalg.norm(e, axis=1))
            first.extend(g.idx.to_numpy() == 0)
            nv.extend(g.nv.to_numpy())
            y.extend(g.ikt.to_numpy())
        X = self._X(np.array(dist), np.array(first), np.array(nv))
        y = np.array(y)
        self.coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ self.coef
        self.s = float(max(1.4826 * np.median(np.abs(r - np.median(r))), 0.03))   # 강건(MAD) 표준편차
        return self

    def pair_loglik(self, geo: Geometry, ikt: float, first: bool) -> np.ndarray:
        dist = np.linalg.norm(geo.expected_pairs(), axis=-1)
        pred = self.coef[0] + self.coef[1] * dist + self.coef[2] * first
        if self.with_nv:
            pred = pred + self.coef[3] * (0 if first else 1)     # 가장 흔한 경우(모음 1개)
        return -((ikt - pred) ** 2) / (2 * self.s ** 2)

    def word_loglik(self, exp: np.ndarray, ikt: np.ndarray, nv) -> float:
        first = np.zeros(len(ikt), bool)
        first[0] = True
        pred = self._X(np.linalg.norm(exp, axis=1), first, nv) @ self.coef
        return float(-((ikt - pred) ** 2).sum() / (2 * self.s ** 2))


def timing_pairs(tm: TimingModel | None, geo: Geometry, ikt, weight: float = 1.0):
    """사슬 모델용 타격별 (15,14) 추가 쌍 퍼텐셜 목록(없으면 None)."""
    if tm is None or ikt is None or weight == 0:
        return None
    return [weight * tm.pair_loglik(geo, float(v), k == 0) for k, v in enumerate(ikt)]


class ConsonantLM:
    """한국어 자음열 트라이그램 log P(c_k | c_{k-2}, c_{k-1}). 상태 0 = 단어 시작, 1..14 = 키.
    어휘의 단어 유형 수로 센다(빈도 가중 없음 — S01에서 빈도 가중보다 나았음). 바이그램과 보간(Witten-Bell 비슷).
    exclude: 평가할 단어를 빼고 만들어 누출을 막는다."""

    def __init__(self, vocab, exclude=(), k_interp: float = 5.0):
        ex = set(exclude)
        C3 = np.zeros((15, 15, 14))
        C2 = np.zeros((15, 14))
        for w, _ in vocab:
            if w in ex:
                continue
            idx = [0, 0] + [KI[c] + 1 for c in to_consonant_keys(w)]
            for a, b, c in zip(idx, idx[1:], idx[2:]):
                C3[a, b, c - 1] += 1
                C2[b, c - 1] += 1
        P2 = (C2 + 0.5) / (C2 + 0.5).sum(1, keepdims=True)
        n3 = C3.sum(2, keepdims=True)
        lam = n3 / (n3 + k_interp)
        self.logp = np.log(lam * (C3 / np.maximum(n3, 1e-9)) + (1 - lam) * P2[None])   # (15,15,14)


def _psis(geo: Geometry, obs, pair_extra=None):
    psis = [geo.pair_loglik(d) for d in obs]
    if pair_extra is not None:
        psis = [a + b for a, b in zip(psis, pair_extra)]
    return psis


def _chain_lm(geo: Geometry, obs, u, T3, gamma, pair_extra=None):
    """상태 (이전 키 a∈0..14, 현재 키 b∈0..13) 위 forward-backward."""
    n = len(obs)
    NEG = -1e18
    psis = _psis(geo, obs, pair_extra)
    la = []
    A = np.full((15, 14), NEG)
    A[0] = psis[0][0] + gamma * T3[0, 0] + u[0]
    la.append(A)
    for k in range(1, n):
        s = np.logaddexp.reduce(A[:, :, None] + gamma * T3[:, 1:, :], axis=0)   # (b, c)
        A = np.full((15, 14), NEG)
        A[1:] = s + psis[k][1:] + u[k][None]
        la.append(A)
    lb = [None] * n
    lb[n - 1] = np.zeros((15, 14))
    for k in range(n - 2, -1, -1):
        term = psis[k + 1][1:] + u[k + 1][None] + lb[k + 1][1:]                 # (b, c)
        lb[k] = np.logaddexp.reduce(gamma * T3[:, 1:, :] + term[None], axis=2)
    out = np.empty((n, 14))
    for k in range(n):
        m = np.logaddexp.reduce(la[k] + lb[k], axis=0)
        m = np.exp(m - m.max())
        out[k] = m / m.sum()
    return out


def chain_marginals(geo: Geometry, obs: np.ndarray, unary: np.ndarray | None = None,
                    lm: ConsonantLM | None = None, gamma: float = 0.5, pair_extra=None) -> np.ndarray:
    """forward-backward. obs (n,2), unary (n,14) log. 반환 (n,14) 타격별 자음 사후확률.
    lm을 주면 자음 트라이그램 사전확률(가중 gamma)을 함께 쓴다(S01: Top-1 CV 46→55%, 시험 세션 44→50%).
    pair_extra: 타격별 (15,14) 추가 쌍 퍼텐셜(예: timing_pairs)."""
    n = len(obs)
    if lm is not None and gamma > 0:
        return _chain_lm(geo, obs, unary if unary is not None else np.zeros((n, 14)), lm.logp, gamma, pair_extra)
    psi = _psis(geo, obs, pair_extra)
    u = unary if unary is not None else np.zeros((n, 14))
    la = np.zeros((n, 14))
    la[0] = psi[0][0] + u[0]
    for k in range(1, n):
        la[k] = np.logaddexp.reduce(la[k - 1][:, None] + psi[k][1:], axis=0) + u[k]
    lb = np.zeros((n, 14))
    for k in range(n - 2, -1, -1):
        lb[k] = np.logaddexp.reduce(psi[k + 1][1:] + (u[k + 1] + lb[k + 1])[None, :], axis=1)
    lp = la + lb
    lp -= lp.max(1, keepdims=True)
    p = np.exp(lp)
    return p / p.sum(1, keepdims=True)


def kbest_sequences(geo: Geometry, obs: np.ndarray, unary=None, lm: ConsonantLM | None = None, gamma: float = 0.5,
                    k: int = 200, pair_extra=None):
    """빔 탐색으로 상위 k 자음열 [(log 점수, 'keys')] (14^n 전체 대신 aLeak식 후보 생성)."""
    n = len(obs)
    psis = _psis(geo, obs, pair_extra)
    u = unary if unary is not None else np.zeros((n, 14))
    beam = [(0.0, 0, 0, ())]          # (점수, 이전이전 상태, 이전 상태, 경로)
    for t_ in range(n):
        cand = []
        for sc, a, b, path in beam:
            add = psis[t_][b] + u[t_]
            if lm is not None and gamma > 0:
                add = add + gamma * lm.logp[a, b]
            for c in np.argsort(-add)[:min(14, k)]:
                cand.append((sc + float(add[c]), b, int(c) + 1, path + (int(c),)))
        cand.sort(key=lambda z: -z[0])
        beam = cand[:k]
    return [(sc, "".join(KEYS[i] for i in path)) for sc, _, _, path in beam]


def inner_score(geo: Geometry, obs: np.ndarray, cstr: str) -> float:
    """(B) aLeak: 가상 궤적(후보 자음열의 기대 변위 연결)과 관측 궤적의 내적 / 노름 = 코사인 유사도 ∈ [-1, 1].
    배율에 무관해 착용·속도에 따라 동작 크기가 달라도 견딘다."""
    e = geo.expected(cstr).ravel()
    d = np.asarray(obs).ravel()
    return float(e @ d / (np.linalg.norm(e) * np.linalg.norm(d) + 1e-9))


def rerank_inner(geo: Geometry, obs, cands, kappa: float = 20.0, mix: float = 1.0):
    """후보 [(점수, 자음열)]을 mix·점수 + κ·코사인으로 재순위. 반환 (순위 목록, 위치별 자음 사후확률 (n,14))."""
    sc = np.array([mix * s + kappa * inner_score(geo, obs, c) for s, c in cands])
    order = np.argsort(-sc)
    ranked = [(float(sc[i]), cands[i][1]) for i in order]
    w = np.exp(sc - sc.max())
    w /= w.sum()
    P = np.full((len(obs), 14), 1e-9)
    for wi, (_, c) in zip(w, cands):
        for k, ch in enumerate(c):
            P[k, KI[ch]] += wi
    return ranked, P / P.sum(1, keepdims=True)


def domain_vocab(vocab, domain_words, floor: float | None = None):
    """맥락(사용 분야) 사전: 분야 단어 목록으로 범위를 좁힌다. 빈도는 일반 어휘 빈도, 일반 어휘에 없으면 floor."""
    freq = dict(vocab)
    fl = floor if floor is not None else (min(freq.values()) if freq else 1.0)
    seen, out = set(), []
    for w in domain_words:
        if w not in seen:
            seen.add(w)
            out.append((w, freq.get(w, fl)))
    return out


class Lexicon:
    def __init__(self, vocab):
        self.freq = dict(vocab)
        self.by_len: dict = defaultdict(lambda: defaultdict(list))
        for w, f in vocab:
            c = to_consonant_keys(w)
            self.by_len[len(c)][c].append((w, f))
        self._idx = {}
        self._nv = {}

    def strings(self, n):
        if n not in self._idx:
            cs = list(self.by_len[n])
            self._idx[n] = (cs, np.array([[KI[ch] + 1 for ch in s] for s in cs]) if cs else np.zeros((0, n), int))
        return self._idx[n]

    def nv(self, w):
        if w not in self._nv:
            self._nv[w] = right_counts(w)
        return self._nv[w]

    def rank_words(self, geo: Geometry, obs: np.ndarray, unary: np.ndarray | None = None, lam: float = 0.0,
                   score: str = "gauss", kappa: float = 20.0, timing: TimingModel | None = None, ikt=None,
                   tau: float = 1.0):
        """[(점수, 단어)] 내림차순.
        score: 'gauss'(기하 가우스 우도) | 'inner'(aLeak 코사인 κ·cos만) | 'both'(가우스 + κ·cos)
        timing/ikt: Stage 4 타이밍 모델(단어별 사이 오른손 키 수 포함), 가중 tau."""
        n = len(obs)
        cs, idx = self.strings(n)
        if not cs:
            return []
        prev = np.hstack([np.zeros((len(cs), 1), int), idx[:, :-1]])
        exp = (geo.Mx[idx] - geo.Mx[prev]) @ geo.A                      # (m,n,2)
        exp[:, 0] += geo.b
        ll = np.zeros(len(cs))
        if score in ("gauss", "both"):
            ll = ll - ((exp - obs[None]) ** 2).sum((1, 2)) / (2 * geo.sigma ** 2)
        if score in ("inner", "both"):
            E = exp.reshape(len(cs), -1)
            d = np.asarray(obs).ravel()
            ll = ll + kappa * (E @ d) / (np.linalg.norm(E, axis=1) * np.linalg.norm(d) + 1e-9)
        if unary is not None:
            ll = ll + unary[np.arange(n)[None, :], idx - 1].sum(1)
        use_t = timing is not None and ikt is not None and tau != 0
        ikt = np.asarray(ikt, float) if use_t else None
        out = []
        for i, s in enumerate(cs):
            for w, f in self.by_len[n][s]:
                sc = ll[i] + lam * np.log(f)
                if use_t:
                    sc += tau * timing.word_loglik(exp[i], ikt, self.nv(w))
                out.append((sc, w))
        out.sort(reverse=True)
        return out
