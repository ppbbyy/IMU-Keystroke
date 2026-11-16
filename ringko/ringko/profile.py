"""여러 사람의 라벨 있는 세션에서 '사람들이 실제로 치는 위치'를 뽑아 새 사용자에게 그대로 쓰는 공용 프로파일.

배경
  기본 모델은 키보드 물리 배열만 쓴다(학습 없음). 그런데 센서는 손끝이 아니라 손가락 마디에 있어서, 추적된 타격 위치는
  배열과 조금 체계적으로 어긋난다(먼 키일수록 덜 움직임 등). 이 어긋남은 사람이 달라도 비슷한 부분이 있으므로
  여러 사람 데이터로 평균을 내 두면 '그 사람들이 아닌' 새 사용자에게도 쓸 수 있다.

무엇을 뽑는가 (모두 키 단위 u = 키 간격으로 정규화한 키보드 좌표계에서)
  centers_u   14자음의 실제 타격 중심. 세션마다 정답 라벨로 좌표를 맞춘 뒤(회전·손잡이·배율 Procrustes) 키별 중앙값,
              세션 → 사람 → 전체 순으로 평균(사람마다 같은 가중).
  hit_sd_mm   같은 키 안에서의 산포(축당 표준편차) 중 추적 오차 σ로 설명되지 않는 몫.
  flip, gyro_sign   센서(링) 고유값: 축 손잡이와 자이로 부호 규약. 모든 세션이 일치할 때만 기록 → 단어가 적은 새 세션에서
              자동 추정(틀릴 수 있음) 대신 쓴다.
  shrink      배열 쪽으로 당기는 정도 λ∈[0,1]: centers = λ·물리 배열 + (1−λ)·실측. 사람 수가 적을 때 과적합을 막는다.

새 사용자에게는 라벨·사전 수집·재학습을 요구하지 않는다. 그 사람 세션의 좌표 정렬(yaw·배율)은 여전히 그 세션의
라벨 없는 타이핑으로 맞춘다(align3d.fit_alignment).

주의: 이것은 '학습 없음' 원칙에서 한 발 물러난 것이다. 프로파일이 실제로 도움이 되는지는 한 사람을 빼고 만든 프로파일을
그 사람에게 시험해야 알 수 있다(scripts/fit_profile.py --loso).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .align3d import fit_from_taps
from .keymap3d import KEYS, KI, Keyboard3D


def session_points(tracks, pitch_mm: float = 19.0) -> dict | None:
    """정답이 있는 WordTrack 목록(한 세션) → 정답으로 정렬한 키 단위 타격 위치.
    반환 {'xy' (N,2)[u], 'key' (N,), 'sig' (N,)[u], 'flip', 'scale'} 또는 None(쓸 타격이 너무 적음)."""
    use = [t for t in tracks if t.truth and len(t.truth) == t.n and np.isfinite(t.P).all() and t.has_return]
    if sum(t.n for t in use) < 40:
        return None
    al = fit_from_taps(use)                                  # 회전·손잡이(정답 사용)
    al.scale = 1.0
    kb = Keyboard3D(pitch_mm=pitch_mm)
    O = np.vstack([al.apply(t.P)[:, :2] for t in use])
    key = np.array([KI[c] for t in use for c in t.truth])
    # 배율 = 움직임 이득(관측 ≈ s·배열): 배열 쪽으로의 회귀라 관측 잡음에 치우치지 않는다
    s = float((O * kb.C2[key]).sum() / (kb.C2[key] ** 2).sum())
    sig = np.concatenate([t.sig for t in use])
    return {"xy": O / s / pitch_mm, "key": key, "sig": sig / s / pitch_mm, "flip": al.flip, "scale": s}


def fit_profile(sessions: list[dict], shrink: float = 0.0, min_n: int = 5) -> dict:
    """sessions: [{'subject': str, 'points': session_points(...) , 'gyro_sign': (3,) | None}, …].
    반환 프로파일 dict(JSON 저장 가능)."""
    base = Keyboard3D(pitch_mm=1.0).C2                       # 물리 배열(키 단위)
    by_subj: dict = {}
    for s in sessions:
        if s.get("points") is not None:
            by_subj.setdefault(s["subject"], []).append(s)
    if not by_subj:
        raise ValueError("프로파일을 만들 세션이 없습니다")
    cen_s, var_s, n_s = [], [], {}
    for subj, lst in by_subj.items():
        cs, vs = [], []
        for s in lst:
            pt = s["points"]
            c = np.full((14, 2), np.nan)
            res = []
            for k in range(14):
                m = pt["key"] == k
                if m.sum() >= min_n:
                    c[k] = np.median(pt["xy"][m], 0)
                    res.append(((pt["xy"][m] - c[k]) ** 2).mean() - (pt["sig"][m] ** 2).mean())
            cs.append(c)
            vs.append(np.mean(res) if res else np.nan)
        cen_s.append(np.nanmean(np.array(cs), 0))            # 사람 안에서 세션 평균
        var_s.append(np.nanmean(vs))
        n_s[subj] = int(sum(len(s["points"]["key"]) for s in lst))
    cen = np.nanmean(np.array(cen_s), 0)                     # 사람 사이 평균(같은 가중)
    cen = np.where(np.isfinite(cen), cen, base)
    cen = shrink * base + (1 - shrink) * cen
    var = float(np.nanmean(var_s))                           # 축당 초과 분산 [u²]
    flips = {float(s["points"]["flip"]) for s in sessions if s.get("points") is not None}
    signs = {tuple(float(x) for x in s["gyro_sign"]) for s in sessions if s.get("gyro_sign") is not None}
    return {"version": 1, "keys": KEYS, "centers_u": cen.round(4).tolist(),
            "hit_sd_u": float(np.sqrt(max(var, 0.05 ** 2))), "shrink": shrink,
            "flip": flips.pop() if len(flips) == 1 else None,
            "gyro_sign": list(signs.pop()) if len(signs) == 1 else None,
            "subjects": n_s, "n_subjects": len(by_subj),
            "offset_u_rms": float(np.sqrt(((cen - base) ** 2).sum(1).mean()))}


def save_profile(prof: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(prof, ensure_ascii=False, indent=1), encoding="utf-8")


def load_profile(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def keyboard_from_profile(prof: dict | None, pitch_mm: float = 19.0, use_spread: bool = True) -> Keyboard3D:
    """프로파일 → Keyboard3D. prof가 None이면 물리 배열 그대로."""
    if prof is None:
        return Keyboard3D(pitch_mm=pitch_mm)
    kw = {"centers_u": tuple(map(tuple, prof["centers_u"]))}
    if use_spread and prof.get("hit_sd_u"):
        kw["hit_sd_mm"] = prof["hit_sd_u"] * pitch_mm
    return Keyboard3D(pitch_mm=pitch_mm, **kw)


def default_profile_path() -> Path:
    return Path(__file__).resolve().parent / "profiles" / "population.json"
