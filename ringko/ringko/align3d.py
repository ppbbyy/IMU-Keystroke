"""센서 좌표계 → 키보드 좌표계 정렬 — 라벨·학습 없이(또는 몇 번의 탭으로) 한 세션 안에서 추정.

남는 미지수와 정하는 방법
  자이로 부호(축 규약)   estimate_gyro_sign: 정지 구간 쌍 사이 중력 방향 변화(가속도계 실측) vs 자이로 적분 예측.
                         8가지 부호 중 오차 최소. 중력 축 둘레 회전이 부족해 판정이 애매한 축은
                         'F 복귀 닫힘 오차가 작은 쪽'으로 고른다(역시 라벨 없음).
  중력(up)              track3d: 단어마다 F 정지 창의 가속도 평균 → 센서 부착 방향(롤·피치)이 달라도 자동
  yaw θ, 손잡이 flip,    fit_alignment: 세션의 타격 위치 구름을 키보드 배열에 맞추는 EM (라벨 없음)
  배율 s                   E: 현재 정렬로 사슬 주변확률 γ_kc (consonant3d, 선택: 사전 자음 트라이그램)
                           M: 가중 Procrustes — θ = atan2(Σγ (o×c), Σγ (o·c)),  s = Σγ |o|² / Σγ (c·R o)
                           초기값 격자(θ 12개 × flip 2개)마다 돌리고 '주변 우도'(사슬 log Z 합)가 최대인 것을 고른다.
                         yaw 불확실성 σ_ψ: 단어별 최적 θ_w의 원형 분산(von Mises κ) → 점수의 방향 항 가중이 저절로 조절
  (선택) 탭 캘리브레이션  fit_from_taps: F→G, F→D 등 '알려진 키' 몇 번(라벨 수 개) → 바로 Procrustes.
                           이것은 '초기 캘리브레이션'이지 모델 학습이 아니다: 추정하는 것은 θ, flip, s 세 숫자뿐.
"""
from __future__ import annotations

import itertools

import numpy as np

from .consonant3d import Alignment, Scorer3D, chain_marginals
from .keymap3d import KI
from .motion import rotvec_to_mat


# --------------------------------------------------------------------------- 자이로 부호
def _still_segments(t, acc, gyr_raw, win_s=0.1, gyro_dps=5.0, acc_ptp_g=0.02):
    dt = float(np.median(np.diff(t)))
    n = max(3, int(round(win_s / dt)))
    gn = np.degrees(np.linalg.norm(gyr_raw, axis=1))
    an = np.linalg.norm(acc, axis=1) / 9.80665
    pad_g = np.r_[np.full(n, 1e9), gn]
    pad_a = np.r_[np.full(n, an[0]), an]
    sw = np.lib.stride_tricks.sliding_window_view
    still = (sw(pad_g, n + 1).max(1) < gyro_dps) & (np.ptp(sw(pad_a, n + 1), axis=1) < acc_ptp_g)
    segs, i = [], 0
    while i < len(t):
        if still[i]:
            j = i
            while j < len(t) and still[j]:
                j += 1
            segs.append((i, j))
            i = j
        else:
            i += 1
    return segs, still


def estimate_gyro_sign(t, acc, gyr_raw, max_pairs: int = 150) -> dict:
    """반환 {'sign': (3,), 'errors': {sign: 중앙 오차[deg]}, 'ambiguous': [후보 sign...]}.
    gyr_raw: 부호 보정 전 자이로 [rad/s]."""
    segs, still = _still_segments(t, acc, gyr_raw)
    bias = np.median(gyr_raw[still], axis=0) if still.any() else np.zeros(3)
    pairs = []
    for ia, (a0, a1) in enumerate(segs):
        for b0, b1 in segs[ia + 1:ia + 3]:
            ga, gb = acc[a0:a1].mean(0), acc[b0:b1].mean(0)
            ga, gb = ga / np.linalg.norm(ga), gb / np.linalg.norm(gb)
            ang = np.degrees(np.arccos(np.clip(ga @ gb, -1, 1)))
            if ang > 1.0 and t[b0] - t[a1 - 1] < 10.0:
                pairs.append((a1 - 1, b0, ga, gb, ang))
    if len(pairs) > max_pairs:                     # 회전이 큰 쌍 위주로
        pairs = sorted(pairs, key=lambda p: -p[4])[:max_pairs]
    errs = {}
    for signs in itertools.product([1, -1], repeat=3):
        s = np.array(signs, float)
        e = []
        for i0, i1, ga, gb, _ in pairs:
            R = np.eye(3)
            for i in range(i0, i1):
                R = R @ rotvec_to_mat(s * (gyr_raw[i] - bias) * (t[i + 1] - t[i]))
            e.append(np.degrees(np.arccos(np.clip((R.T @ ga) @ gb, -1, 1))))
        errs[signs] = float(np.median(e)) if e else np.nan
    if not pairs:                                   # 판정 불가 → 호출 쪽 기본값(frames.GYRO_SIGN)
        return {"sign": None, "errors": errs, "ambiguous": [], "n_pairs": 0}
    order = sorted(errs, key=lambda k: errs[k])
    best = order[0]
    amb = [k for k in order if errs[k] <= 1.3 * errs[best] + 0.3]     # 중력으로 가르기 애매한 후보(상대 기준)
    return {"sign": np.array(best, float), "errors": errs, "ambiguous": amb, "n_pairs": len(pairs)}


def choose_gyro_sign(sd, phase: str = "continuous", strikes=None, source: str = "keylog") -> dict:
    """세션 하나의 자이로 부호를 라벨 없이 정한다. 1) 중력 예측 오차 2) 애매한 후보끼리는 F 복귀 닫힘 오차(중앙값) 최소.
    strikes/source: 닫힘 계산에 쓸 타격(track3d.track_session과 같은 형식). None이면 키 로그.
    반환 estimate_gyro_sign 결과 + {'sign', 'by'}."""
    from .frames import GYRO_SIGN
    from .motion import G
    from .track3d import TrackParams, strikes_from_keylog, track_session
    t = sd.imu.host_ns.to_numpy() / 1e9
    acc = sd.imu[["ax", "ay", "az"]].to_numpy() * G
    gyr = np.radians(sd.imu[["gx", "gy", "gz"]].to_numpy())
    r = estimate_gyro_sign(t, acc, gyr)
    if r["sign"] is None:
        r.update(sign=GYRO_SIGN.copy(), by="default")
        return r
    r["by"] = "gravity"
    if len(r["ambiguous"]) > 1:
        if strikes is None:
            strikes, source = strikes_from_keylog(sd, phase), "keylog"
        cl = {}
        for s in r["ambiguous"][:4]:
            trs, _ = track_session(sd, strikes, phase, TrackParams(gyro_sign=np.array(s, float)), source=source)
            cl[s] = {w.trial_id: float(np.linalg.norm(w.closure)) for w in trs if w.has_return}
        # 단어별 쌍비교: 후보마다 '그 부호에서 닫힘이 가장 작은 단어' 비율(중앙값 비교보다 단어 몇 개의 큰 오차에 덜 흔들림)
        common = set.intersection(*[set(v) for v in cl.values()])
        wins = {s: 0 for s in cl}
        for tid in common:
            wins[min(cl, key=lambda s: cl[s][tid])] += 1
        r["closure_by_sign"] = {s: (float(np.median(list(v.values()))), wins[s] / max(len(common), 1)) for s, v in cl.items()}
        r["sign"] = np.array(max(wins, key=lambda s: wins[s]), float)
        r["by"] = "gravity+closure"
    return r


# --------------------------------------------------------------------------- yaw / flip / scale EM
def _procrustes(O, C, w):
    """가중: K ≈ R(θ)O/s 를 C에 맞춤. O, C (N,2). 반환 θ, s."""
    a = (w * (O * C).sum(1)).sum()
    b = (w * (O[:, 0] * C[:, 1] - O[:, 1] * C[:, 0])).sum()
    th = np.arctan2(b, a)
    proj = a * np.cos(th) + b * np.sin(th)                      # Σ w C·R O
    s = (w * (O ** 2).sum(1)).sum() / max(proj, 1e-9)
    return float(th), float(s)


def _collect(tracks, scorer, al, lm, gamma):
    O, C, W, logz, per = [], [], [], 0.0, []
    C2 = scorer.kb.C2
    for tr in tracks:
        if not np.isfinite(tr.P).all():
            continue
        M, lz = chain_marginals(scorer.observe(tr, al), lm, gamma, return_logz=True)
        logz += lz
        h = tr.P[:, :2] * [1.0, al.flip]
        Ow = np.repeat(h, 14, 0)
        Cw = np.tile(C2, (len(h), 1))
        ww = M.ravel() / np.repeat(tr.sig ** 2, 14)
        O.append(Ow)
        C.append(Cw)
        W.append(ww)
        per.append((Ow, Cw, ww))
    return np.vstack(O), np.vstack(C), np.concatenate(W), logz, per


def consonant_prior(vocab) -> np.ndarray:
    """사전(텍스트)에서 센 14자음 상대 빈도 π (단어 유형 기준). IMU 학습 아님."""
    from .hangul import to_consonant_keys
    c = np.ones(14)
    for w, _ in vocab:
        for ch in to_consonant_keys(w):
            c[KI[ch]] += 1
    return c / c.sum()


def moment_scale(tracks, prior: np.ndarray, kb=None, clip=(0.7, 1.4)) -> float:
    """라벨 없는 배율: 관측 타격의 평균 수평 F 거리 / 사전 빈도로 본 기대 거리 Σ π_k d_k.
    회전·거울에 불변이라 정렬보다 먼저 정할 수 있다(EM 안에서 배율까지 풀면 F 쪽으로 붕괴함 — S01 확인)."""
    from .keymap3d import Keyboard3D
    kb = kb or Keyboard3D()
    r = np.concatenate([np.linalg.norm(t.P[:, :2], axis=1) for t in tracks if np.isfinite(t.P).all()])
    return float(np.clip(r.mean() / (prior * kb.d).sum(), *clip))


def fit_alignment(tracks, scorer: Scorer3D | None = None, lm=None, gamma: float = 0.5, flips=(1.0, -1.0),
                  n_theta: int = 12, iters: int = 6, sigma_yaw_em: float = np.radians(20),
                  scale: float | None = None, prior: np.ndarray | None = None, verbose: bool = False) -> Alignment:
    """라벨 없는 EM 정렬(yaw θ, 손잡이 flip). tracks: track3d.WordTrack 목록(같은 세션·같은 착용).
    scale: None이면 moment_scale(prior 필요; prior도 None이면 1.0 = '센서 이동 ≈ 손끝 이동' 물리 가정)."""
    scorer = scorer or Scorer3D()
    sc = Scorer3D(scorer.kb, "sector", scorer.w_abs, scorer.w_rel, scorer.w_dir)
    if scale is None:
        scale = moment_scale(tracks, prior, scorer.kb) if prior is not None else 1.0
    best = None
    for flip in flips:
        for th0 in np.linspace(-np.pi, np.pi, n_theta, endpoint=False):
            al = Alignment(th0, flip, scale, sigma_yaw_em, "em")
            for _ in range(iters):
                O, C, W, logz, _ = _collect(tracks, sc, al, lm, gamma)
                th, _ = _procrustes(O, C, W)
                if abs(np.angle(np.exp(1j * (th - al.theta)))) < 1e-4:
                    al = Alignment(th, flip, scale, sigma_yaw_em, "em")
                    break
                al = Alignment(th, flip, scale, sigma_yaw_em, "em")
            _, _, _, logz, per = _collect(tracks, sc, al, lm, gamma)
            if verbose:
                print(f"  flip {flip:+.0f} θ0 {np.degrees(th0):6.0f}° → θ {np.degrees(al.theta):6.1f}° s {al.scale:.2f} logZ {logz:.1f}")
            if best is None or logz > best[0]:
                best = (logz, al, per)
    logz, al, per = best
    # yaw 불확실성: 단어별 θ_w (같은 γ로) 의 원형 분산
    ths = []
    for Ow, Cw, ww in per:
        if ww.sum() > 0:
            ths.append(_procrustes(Ow, Cw, ww)[0])
    z = np.exp(1j * (np.array(ths) - al.theta)).mean() if ths else 0
    R = float(min(abs(z), 0.995))
    kappa = R * (2 - R ** 2) / (1 - R ** 2) if R > 0 else 1e-3
    al.sigma_yaw = float(np.clip(1 / np.sqrt(max(kappa, 1e-3)), np.radians(4), np.pi))
    al.theta = float(al.theta + np.angle(z)) if ths else al.theta
    al.source = f"em(logZ={logz:.1f})"
    return al


def fit_from_taps(tracks, sigma_yaw: float = np.radians(10), flips=(1.0, -1.0)) -> Alignment:
    """탭 캘리브레이션: 정답 키를 아는 몇 번의 F→키 타격(track.truth)으로 θ, flip, s 를 바로 구한다."""
    from .keymap3d import Keyboard3D
    kb = Keyboard3D()
    O, C = [], []
    for tr in tracks:
        if tr.truth is None or len(tr.truth) != tr.n:
            continue
        for k, ch in enumerate(tr.truth):
            if kb.d[KI[ch]] > 1e-6:                     # F 자신은 방향 정보 없음
                O.append(tr.P[k, :2])
                C.append(kb.C2[KI[ch]])
    O, C = np.array(O), np.array(C)
    best = None
    for flip in flips:
        Of = O * [1.0, flip]
        th, s = _procrustes(Of, C, np.ones(len(O)))
        al = Alignment(th, flip, s, sigma_yaw, f"taps(n={len(O)})")
        res = ((al.apply(np.c_[O, np.zeros(len(O))])[:, :2] - C) ** 2).sum()
        if best is None or res < best[0]:
            best = (res, al)
    return best[1]
