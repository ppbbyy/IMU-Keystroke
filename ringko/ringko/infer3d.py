"""3차원 자음 추론 전체 파이프라인(한 세션) — 학습 없음.

  IMU 세션 ─▶ 기존 타격 검출(features.detect_presses [+ refine_presses], 수정 없음)
           ─▶ 자이로 부호(align3d.choose_gyro_sign, 라벨 없음)
           ─▶ track3d: 타격 사이 IMU 적분 → F 기준 3차원 위치 P, 이동 D, σ (닫힘·평면 보정)
           ─▶ align3d: 중력 좌표계 → 키보드 좌표계 (yaw·손잡이 EM, 배율 모멘트; 또는 탭 몇 번)
           ─▶ consonant3d: F 거리 min/max 대역 + 방향 + 이전 타격 대비 이동 → 자음 후보·점수·사후확률
           ─▶ decode.Lexicon: 자음열 사전 후보 + 순위 (기존 사전 복원 그대로)

반환 Result3D: 세션 정보 + 단어별 WordResult(후보 집합, 타격별 사후확률, 단어 순위).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .align3d import choose_gyro_sign, fit_alignment, fit_from_taps
from .consonant3d import Alignment, Scorer3D, chain_marginals
from .keymap3d import KEYS
from .track3d import TrackParams, WordTrack, strikes_from_detection, strikes_from_keylog, track_session


@dataclass
class Infer3DConfig:
    strikes: str = "imu+refine"     # imu | imu+refine | keylog(오라클, 평가용)
    gyro_sign: str = "auto"         # auto(라벨 없이 추정) | default(frames.GYRO_SIGN)
    alignment: str = "em"           # em(라벨 없음) | none(거리만, shell 모드 권장) | taps(track 정답 몇 개 필요)
    flips: tuple = (1.0, -1.0)      # 손잡이 후보. 센서 축 손잡이를 알면 하나만
    gamma: float = 0.5              # 자음 트라이그램(사전) 가중 — 사전 없이 낸 자음 후보에 쓰임
    lam: float = 0.5                # 단어 빈도 가중
    top_words: int = 10
    track: TrackParams = field(default_factory=TrackParams)
    scorer: Scorer3D = field(default_factory=Scorer3D)


@dataclass
class WordResult:
    track: WordTrack
    candidates: np.ndarray           # (n,14) bool — F 거리 min/max 대역 안
    marginals: np.ndarray            # (n,14) 자음 사후확률(사전 단어 없이, 자음 LM만)
    words: list                      # [(점수, 단어)] 상위
    consonants: str                  # 사후확률 최대 키열

    def top_consonants(self, k: int = 3):
        return [[KEYS[i] for i in np.argsort(-m)[:k]] for m in self.marginals]


@dataclass
class Result3D:
    alignment: Alignment
    gyro: dict
    info: dict
    words: list


def get_strikes(sd, how: str):
    """(strikes dict, source) — 기존 검출 함수를 그대로 호출."""
    if how == "keylog":
        return strikes_from_keylog(sd, "continuous"), "keylog"
    from .features import detect_presses, refine_presses
    det = detect_presses(sd)
    if how == "imu+refine":
        det = refine_presses(sd, det)
    return strikes_from_detection(det), "detected"


def infer_session(sd, lexicon=None, lm=None, prior=None, cfg: Infer3DConfig | None = None,
                  alignment: Alignment | None = None, taps: list | None = None, strikes=None) -> Result3D:
    """lexicon: decode.Lexicon(없으면 단어 순위 생략), lm: decode.ConsonantLM, prior: align3d.consonant_prior.
    alignment를 주면 그대로 쓴다(예: 같은 착용의 이전 세션에서 구한 것). taps: 탭 캘리브레이션용 WordTrack 목록."""
    cfg = cfg or Infer3DConfig()
    if strikes is None:
        strikes, src = get_strikes(sd, cfg.strikes)
    else:
        src = "detected"
    gyro = {"sign": None, "by": "default"}
    tp = cfg.track
    if cfg.gyro_sign == "auto":
        gyro = choose_gyro_sign(sd, strikes=strikes, source=src)
        tp = TrackParams(**{**tp.__dict__, "gyro_sign": gyro["sign"]})
    tracks, info = track_session(sd, strikes, "continuous", tp, source=src)
    sc = cfg.scorer
    if alignment is not None:
        al = alignment
    elif cfg.alignment == "em":
        al = fit_alignment(tracks, sc, lm=lm, gamma=cfg.gamma, flips=cfg.flips, prior=prior)
    elif cfg.alignment == "taps":
        al = fit_from_taps(taps or [], flips=cfg.flips)
    else:
        al = Alignment()
    out = []
    for tr in tracks:
        if not np.isfinite(tr.P).all():
            continue
        obs = sc.observe(tr, al)
        M = chain_marginals(obs, lm, cfg.gamma)
        words = lexicon.rank_words(sc, obs, cfg.lam)[:cfg.top_words] if lexicon is not None else []
        out.append(WordResult(tr, sc.candidates(tr, al), M, words, "".join(KEYS[i] for i in M.argmax(1))))
    return Result3D(al, gyro, info, out)
