"""수집 플랜.

단계(phase)
  rest       : F 위에 검지를 올리고 N초 정지 -> 중력 방향/자이로 바이어스 기준(파이프라인 Stage 1의 C(0))
  calib      : F만 반복 입력 -> 착용마다 다른 센서↔키보드 회전/오프셋 추정용 기준
  isolated   : 14자음 x reps, 블록 무작위(블록마다 14자음 1회씩, 경계에서 같은 자음 연속 금지).
               매 시도 F에서 출발 -> 키 -> F 복귀. 키 간 간격이 인위적이므로 모델 선택/검증용.
  continuous : 실제 단어를 평소 속도로. 자음=왼손 검지, 모음/스페이스=오른손.
               Stage 2(연속 피크 분리)와 Stage 4(타이핑 리듬)를 실전 조건에서 검증하는 유일한 데이터.
"""
from __future__ import annotations

import csv
import zlib
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from .hangul import CONSONANT_KEY, CONSONANTS, HOME_KEY, is_typable, to_consonant_keys, to_keys

PHASES = ("axes", "rest", "calib", "isolated", "continuous")


@dataclass
class Item:
    uid: int
    phase: str      # intro | rest | calib | isolated | continuous | break | outro
    kind: str       # info | rest | timed | key | word
    target: str     # 화면에 크게 표시
    expect: str     # 기대 키열(소문자, 단어는 끝에 ' '). info/rest는 ''
    hint: str = ""
    block: int = 0


def seed_for(subject: str, wear: int, base: int = 0) -> int:
    return (zlib.crc32(f"{subject}|wear{wear}".encode()) ^ base) & 0xFFFFFFFF


# --------------------------------------------------------------------------- 단어
def load_vocab(path: str | Path | None = None, top_n: int = 30000) -> list[tuple[str, float]]:
    """(단어, 상대빈도). 파일은 '단어[ 빈도]' 줄. 없으면 wordfreq 한국어 상위 단어(빈도 ∝ 1/순위)."""
    if path:
        out = []
        for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
            parts = line.split()
            if parts and not parts[0].startswith("#"):
                out.append((parts[0], float(parts[1]) if len(parts) > 1 else 1.0 / (i + 1)))
        return out
    from wordfreq import top_n_list
    return [(w, 1.0 / (i + 1)) for i, w in enumerate(top_n_list("ko", top_n))]


def filter_vocab(vocab, min_keys: int = 2, max_keys: int = 8, max_syll: int = 4):
    seen, out = set(), []
    for w, f in vocab:
        if w in seen or len(w) > max_syll or not is_typable(w):
            continue
        if min_keys <= len(to_consonant_keys(w)) <= max_keys:
            seen.add(w)
            out.append((w, f))
    return out


def consonant_counts(words) -> dict[str, int]:
    cnt = {c: 0 for c in CONSONANTS}
    key2c = {k: c for c, k in CONSONANT_KEY.items()}
    for w in words:
        for k in to_consonant_keys(w):
            cnt[key2c[k]] += 1
    return cnt


def select_words(vocab, n: int, rng: np.random.Generator, alpha: float = 0.5,
                 min_per_consonant: int = 10, max_iter: int = 1000) -> list[str]:
    """빈도^alpha 가중 비복원 추출 후, 희귀 자음(ㅋㅌㅍㅊ 등)이 min_per_consonant회 이상 나오도록 교체."""
    if n > len(vocab):
        raise ValueError(f"어휘 {len(vocab)}개 < 요청 {n}개")
    freq = np.array([f for _, f in vocab], dtype=float) ** alpha
    idx = rng.choice(len(vocab), size=n, replace=False, p=freq / freq.sum())
    words = [vocab[i][0] for i in idx]
    keys_of = {w: set(to_consonant_keys(w)) for w, _ in vocab}
    fmap = dict(vocab)
    for _ in range(max_iter):
        cnt = consonant_counts(words)
        lacking = [c for c in CONSONANTS if cnt[c] < min_per_consonant]
        if not lacking:
            break
        c = min(lacking, key=cnt.get)
        lack_keys = {CONSONANT_KEY[x] for x in lacking}
        chosen = set(words)
        cand = [w for w, _ in vocab if w not in chosen and CONSONANT_KEY[c] in keys_of[w]]
        victims = [i for i, w in enumerate(words) if not (keys_of[w] & lack_keys)]
        if not cand or not victims:
            break
        p = np.array([fmap[w] for w in cand]) ** alpha
        words[int(rng.choice(victims))] = cand[int(rng.choice(len(cand), p=p / p.sum()))]
    rng.shuffle(words)
    return words


# --------------------------------------------------------------------------- 플랜
TEXT = {
    "intro": "입력기를 영문(A)으로 바꾸세요 (한/영 키).\n링: 왼손 검지 중위지골(둘째 마디).\n"
             "진행 키(SPACE)는 오른손으로 누릅니다.",
    "axes": "[축 확인 15초] 링 낀 왼손을 키보드에서 떼고, 천천히 여러 방향으로 돌렸다가 1초씩 멈추기를 반복하세요.\n"
            "(손목 비틀기, 손등 들기, 손가락 세우기 등. 오른손으로 SPACE를 누르면 시작)",
    "rest": "[정지] 왼손 검지를 F 위에 올리고 완전히 멈추세요.\n준비되면 오른손으로 SPACE → {s}초간 가만히.",
    "calib": "[F 보정] 'F'가 뜨면 F를 한 번 누르고 그대로 F 위에 둡니다.",
    "isolated": "[개별 키] 표시된 자음 키를 왼손 검지로 누르고 F로 돌아와 멈추세요.\n"
                "손가락이 멈추면 다음 글자가 뜹니다. 틀린 항목은 나중에 다시 나옵니다.",
    "continuous": "[연속 타이핑] 단어를 평소 속도로: 자음=왼손 검지, 모음과 SPACE=오른손.\n"
                  "단어마다 F에서 시작하고 끝나면 F로. 틀려도 BackSpace 없이 SPACE까지(나중에 다시 나옴).",
    "outro": "끝났습니다. 잠시 기다리세요.",
}


def build_plan(rng: np.random.Generator, phases=PHASES, *, reps: int = 40, words: list[str] | None = None,
               calib_n: int = 10, rest_s: float = 5.0, break_every: int = 112) -> list[Item]:
    items: list[Item] = []

    def add(phase, kind, target, expect="", hint="", block=0):
        items.append(Item(len(items) + 1, phase, kind, target, expect, hint, block))

    add("intro", "info", TEXT["intro"])
    if "axes" in phases:
        add("axes", "info", TEXT["axes"])
        add("axes", "timed", "15", hint="천천히 돌리고 멈추기")
    if "rest" in phases:
        add("rest", "info", TEXT["rest"].format(s=int(rest_s)))
        add("rest", "rest", f"{rest_s:g}", hint="F 위에서 정지")
    if "calib" in phases:
        add("calib", "info", TEXT["calib"])
        for _ in range(calib_n):
            add("calib", "key", "F", HOME_KEY, "ㄹ")
    if "isolated" in phases:
        add("isolated", "info", TEXT["isolated"])
        prev, n = None, 0
        for b in range(1, reps + 1):
            order = list(CONSONANTS)
            rng.shuffle(order)
            if order[0] == prev:
                j = int(rng.integers(1, len(order)))
                order[0], order[j] = order[j], order[0]
            prev = order[-1]
            for c in order:
                if n and break_every and n % break_every == 0:
                    add("break", "info", f"휴식 — 손을 풀어 주세요 ({n}/{reps * len(order)})")
                add("isolated", "key", c, CONSONANT_KEY[c], CONSONANT_KEY[c].upper(), b)
                n += 1
    if "continuous" in phases:
        if not words:
            raise ValueError("continuous 단계에는 단어가 필요")
        add("continuous", "info", TEXT["continuous"])
        for i, w in enumerate(words, 1):
            if i > 1 and break_every and (i - 1) % 50 == 0:
                add("break", "info", f"휴식 ({i - 1}/{len(words)} 단어)")
            add("continuous", "word", w, to_keys(w) + " ", " ".join(to_keys(w).upper()) + "  ␣", i)
    if "rest" in phases:  # 끝에도 정지 기준(드리프트 확인)
        add("rest", "info", TEXT["rest"].format(s=int(rest_s)))
        add("rest", "rest", f"{rest_s:g}", hint="F 위에서 정지")
    add("outro", "info", TEXT["outro"])
    return items


def counts(plan: list[Item]) -> dict[str, int]:
    out: dict[str, int] = {}
    for it in plan:
        if it.kind != "info":
            out[it.phase] = out.get(it.phase, 0) + 1
    return out


def estimate_minutes(plan: list[Item], key_s: float = 2.4, word_s: float = 7.0) -> int:
    c = counts(plan)
    sec = ((c.get("calib", 0) + c.get("isolated", 0)) * key_s + c.get("continuous", 0) * word_s
           + c.get("rest", 0) * 10 + c.get("axes", 0) * 20)
    return int(np.ceil(sec / 60))


_COLS = [f.name for f in fields(Item)]


def save_plan(plan: list[Item], path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_COLS)
        w.writeheader()
        for it in plan:
            w.writerow(asdict(it))


def load_plan(path) -> list[Item]:
    with open(path, newline="", encoding="utf-8") as f:
        return [Item(int(r["uid"]), r["phase"], r["kind"], r["target"], r["expect"], r["hint"], int(r["block"]))
                for r in csv.DictReader(f)]
