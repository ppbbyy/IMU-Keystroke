"""전처리 — docs/pipeline.docx: 동기화 → 좌표 변환 → 채널별 정규화(학습 통계 기준) → 증강(학습 fold만).

동기화(clock.py, dataset.load_session)와 좌표 변환(motion.py, features.py)은 앞 단계에서 끝난다. 여기서는
  ChannelNormalizer : 원시 윈도우 (N, T, C)를 '채널마다' 학습 데이터의 평균·표준편차로 정규화. 평가 데이터 통계는 쓰지 않는다.
  WindowClassifier  : 정규화한 윈도우(기준 프레임 선형가속도 3 + 각속도 3, 48샘플) → 평탄화 → 로지스틱 회귀/SVM.
                      특징 공학 없이 신호 모양만 보는 비교 기준.
  augment_windows   : 윈도우 수준 증강(시간이동·진폭·잡음). 특징 수준 증강은 features.Augment.
"""
from __future__ import annotations

import numpy as np


class ChannelNormalizer:
    def fit(self, W: np.ndarray) -> "ChannelNormalizer":
        W = np.asarray(W, float)
        self.mu = W.mean((0, 1))
        self.sd = W.std((0, 1)) + 1e-8
        return self

    def transform(self, W: np.ndarray) -> np.ndarray:
        return (np.asarray(W, float) - self.mu) / self.sd


def augment_windows(W: np.ndarray, y: np.ndarray, n_copies: int = 2, shift: int = 2, scale: float = 0.08,
                    noise: float = 0.05, seed: int = 0):
    """(정규화 전) 윈도우 증강. 시간이동 ±shift 샘플(가장자리 반복), 채널 공통 진폭 배율, 채널별 표준편차 비례 잡음."""
    rng = np.random.default_rng(seed)
    W = np.asarray(W, float)
    sd = W.std((0, 1))
    out, ys = [W], [y]
    for _ in range(n_copies):
        A = np.empty_like(W)
        for i, w in enumerate(W):
            s = int(rng.integers(-shift, shift + 1))
            w2 = np.roll(w, s, axis=0)
            if s > 0:
                w2[:s] = w[0]
            elif s < 0:
                w2[s:] = w[-1]
            A[i] = w2 * (1 + rng.normal(0, scale)) + rng.normal(0, noise, w.shape) * sd
        out.append(A)
        ys.append(y)
    return np.concatenate(out), np.concatenate(ys)


class WindowClassifier:
    """채널 정규화 + 평탄화 + 분류기. fit에 augment=True면 학습 데이터에만 증강."""

    def __init__(self, kind: str = "logreg", augment: bool = False, seed: int = 0):
        self.kind, self.augment, self.seed = kind, augment, seed

    def _model(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        if self.kind == "svm":
            return SVC(C=3, gamma="scale", probability=True, random_state=self.seed)
        return LogisticRegression(C=0.05, max_iter=3000)

    def fit(self, W, y) -> "WindowClassifier":
        W = np.asarray(W, float)
        y = np.asarray(y)
        if self.augment:
            W, y = augment_windows(W, y, seed=self.seed)
        self.norm = ChannelNormalizer().fit(W)               # 학습(증강 포함) 통계만
        self.m = self._model().fit(self.norm.transform(W).reshape(len(W), -1), y)
        self.classes_ = self.m.classes_
        return self

    def predict_proba(self, W) -> np.ndarray:
        W = np.asarray(W, float)
        return self.m.predict_proba(self.norm.transform(W).reshape(len(W), -1))
