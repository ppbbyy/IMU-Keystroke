"""한국어 사전 기반 단어 복원 — 자음열 후보 생성 + 단어 순위.

기존 디코더에서 '사전' 부분(Lexicon, ConsonantLM, domain_vocab)만 남겼다. 2D 기하(Geometry, 착용별 A·b 보정),
학습형 행 모델(RowModel)·타이밍 모델(TimingModel)·aLeak 2D 내적은 모두 제거되었고,
타격별 기하 점수는 consonant3d.Scorer3D(F 기준 3차원 위치, 학습 없음)가 맡는다.

연결 인터페이스(기존 rank_words(geo, obs, …)와 같은 모양)
  scorer.score_strings(obs, idx) -> (m,) log 점수      obs = scorer.observe(track, alignment)  (consonant3d.Obs3D)
                                                       idx = (m, n) 후보 자음열(키 번호+1)       (Lexicon.strings)
  단어 점수 = 기하 점수(자음열) + λ · log 빈도(단어)
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .hangul import LEFT_KEYS, to_consonant_keys

KEYS = list(LEFT_KEYS)
KI = {k: i for i, k in enumerate(KEYS)}


class ConsonantLM:
    """한국어 자음열 트라이그램 log P(c_k | c_{k-2}, c_{k-1}). 상태 0 = 단어 시작, 1..14 = 키.
    사전(텍스트)의 단어 유형 수로 센다 — IMU·사용자 데이터를 쓰지 않으므로 기기·센서와 무관.
    바이그램과 보간. exclude: 평가할 단어를 빼고 만들어 누출을 막는다."""

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


def open_vocab(top_n: int = 30000, extra_words=(), max_keys: int = 12, path=None):
    """열린 어휘 [(단어, 상대빈도)]: wordfreq 한국어 목록(최대 약 3만)에서 입력 가능한 단어 전부(길이 제한 없이 자음 1~max_keys개)
    + extra_words(사용자 단어, 문서에서 뽑은 단어). 추가 단어의 빈도는 목록 중앙값(이미 있으면 그대로).
    가정: 쌍자음(ㄲㄸㅃㅆㅉ)·ㅒ·ㅖ가 든 단어는 대상에서 제외한다(Shift가 필요 — 수집 규칙과 같음). 그런 추가 단어는 건너뛴다."""
    from .hangul import is_typable
    from .plan import load_vocab
    seen, out = set(), []
    for w, f in load_vocab(path, top_n):
        if w not in seen and is_typable(w) and 1 <= len(to_consonant_keys(w)) <= max_keys:
            seen.add(w)
            out.append((w, f))
    fl = float(np.median([f for _, f in out])) if out else 1.0
    for w in extra_words:
        if w not in seen and is_typable(w) and 1 <= len(to_consonant_keys(w)) <= max_keys:
            seen.add(w)
            out.append((w, fl))
    return out


def words_from_text(text: str) -> list[str]:
    """아무 글(메모, 문서)에서 한글 단어를 뽑는다(등장 순, 중복 제거) — 사전에 없는 내 단어를 넣는 가장 쉬운 방법."""
    import re
    return list(dict.fromkeys(re.findall(r"[가-힣]+", text)))


def load_wordlist(path) -> list[str]:
    """단어 목록 파일 → 한글 단어 리스트(등장 순, 중복 제거).
      .csv : 열 '단어'(없으면 첫 열)만 읽는다. 다른 열(자음열·카테고리 등)은 무시.
      그 외 : 텍스트로 보고 '#'로 시작하는 줄을 뺀 뒤 한글 단어를 모두 추출(words_from_text)."""
    from pathlib import Path
    p = Path(path)
    if p.suffix.lower() == ".csv":
        import csv
        with open(p, encoding="utf-8-sig", newline="") as fh:
            rd = csv.reader(fh)
            header = next(rd, [])
            col = header.index("단어") if "단어" in header else 0
            if "단어" not in header and header:          # 머리글이 없으면 첫 줄도 단어
                rows = [header] + list(rd)
            else:
                rows = list(rd)
            words = [r[col].strip() for r in rows if len(r) > col and r[col].strip()]
        return list(dict.fromkeys(w for w in words if any(0xAC00 <= ord(c) <= 0xD7A3 for c in w)))
    text = "\n".join(ln for ln in p.read_text(encoding="utf-8").splitlines() if not ln.lstrip().startswith("#"))
    return words_from_text(text)


class Lexicon:
    def __init__(self, vocab):
        self.freq = dict(vocab)
        self.by_len: dict = defaultdict(lambda: defaultdict(list))
        for w, f in vocab:
            c = to_consonant_keys(w)
            self.by_len[len(c)][c].append((w, f))
        self._idx = {}

    def strings(self, n):
        """길이 n 자음열 후보 (목록, (m,n) 키 번호+1 배열)."""
        if n not in self._idx:
            cs = list(self.by_len[n])
            self._idx[n] = (cs, np.array([[KI[ch] + 1 for ch in s] for s in cs]) if cs else np.zeros((0, n), int))
        return self._idx[n]

    def rank_words(self, scorer, obs, lam: float = 0.5, unary: np.ndarray | None = None):
        """[(점수, 단어)] 내림차순. scorer: consonant3d.Scorer3D, obs: scorer.observe(...) 결과.
        unary: 추가 타격별 log 점수 (n,14)(선택)."""
        n = obs.n
        cs, idx = self.strings(n)
        if not cs:
            return []
        ll = scorer.score_strings(obs, idx)
        if unary is not None:
            ll = ll + unary[np.arange(n)[None, :], idx - 1].sum(1)
        out = [(ll[i] + lam * np.log(f), w) for i, s in enumerate(cs) for w, f in self.by_len[n][s]]
        out.sort(reverse=True)
        return out
