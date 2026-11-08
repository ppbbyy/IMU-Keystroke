"""수집 프롬프트(플랜) 생성: 동기화 / F 보정 / 개별 키(무작위 블록) / 연속 타이핑 단어.

PROTOCOL.md의 수집 설계를 그대로 코드로 옮긴 것.
 - 개별 키: 14자음 x 50회, 블록마다 14자음을 한 번씩 섞어서 제시(블록 경계에서 같은 자음 연속 금지)
 - 보정: F키 10회
 - 연속 타이핑: 어휘 목록 단어(연구자 어휘만, 실제 비밀번호/개인정보 금지)
"""
from __future__ import annotations

import csv
import math
import zlib
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from .hangul_keys import (CONSONANT_KEYS, CONSONANTS, HOME_KEY, is_typable,
                          word_to_consonant_keys, word_to_keys)

PHASES = ("sync_start", "calib", "isolated", "continuous", "sync_end")


@dataclass
class Prompt:
    prompt_id: int
    phase: str          # sync_start | calib | isolated | continuous | sync_end | info
    kind: str           # info | key | word
    block: int          # isolated: 블록 번호, continuous: 단어 순번, 그 외 0
    label: str          # 화면에 크게 표시할 글자(자음/단어/안내문)
    expect: str         # 기대 키 입력(소문자 문자열, 공백=스페이스). info는 ''
    hint: str = ""      # 보조 표시(키 이름, 입력해야 할 키열)
    consonant_keys: str = ""  # continuous: 왼손 검지로 치는 자음 키열


def seed_for(subject: str, session: str, wear: str, base: int = 0) -> int:
    """(피험자, 세션, 착용)마다 고정된 시드. 재실행해도 같은 프롬프트, 착용 간에는 다른 단어."""
    return (zlib.crc32(f"{subject}|{session}|{wear}".encode()) ^ base) & 0xFFFFFFFF


# --------------------------------------------------------------------------- 어휘
def load_vocab(path: str | Path | None = None, top_n: int = 60000) -> list[tuple[str, float]]:
    """(단어, 빈도) 목록. path가 없으면 wordfreq(ko)를 사용. 파일은 '단어[<탭/공백>빈도]' 줄 단위."""
    vocab: list[tuple[str, float]] = []
    if path:
        for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
            parts = line.strip().split()
            if not parts or parts[0].startswith("#"):
                continue
            freq = float(parts[1]) if len(parts) > 1 else 1.0 / (i + 1)  # 빈도 없으면 순위 기반 Zipf 근사
            vocab.append((parts[0], freq))
        return vocab
    try:
        from wordfreq import top_n_list
    except ImportError as e:  # pragma: no cover
        raise SystemExit("wordfreq가 없습니다: pip install wordfreq  (또는 --vocab 파일 지정)") from e
    # word_frequency()는 한국어에서 MeCab이 필요하므로, 순위 기반 Zipf 근사(빈도 ∝ 1/순위)를 쓴다.
    return [(w, 1.0 / (i + 1)) for i, w in enumerate(top_n_list("ko", top_n))]


def filter_vocab(vocab, min_cons: int = 2, max_cons: int = 8, max_syll: int = 5):
    """수집 가능한 단어만(쌍자음/ㅒ/ㅖ 제외, 자음 키 min~max개). 같은 단어 중복 제거."""
    seen, out = set(), []
    for w, f in vocab:
        if w in seen or len(w) > max_syll or not is_typable(w):
            continue
        n = len(word_to_consonant_keys(w))
        if min_cons <= n <= max_cons:
            seen.add(w)
            out.append((w, f))
    return out


def sample_words(vocab, n: int, rng: np.random.Generator, alpha: float = 0.5) -> list[str]:
    """빈도^alpha 가중으로 비복원 추출. alpha=1이면 실제 빈도, 0이면 균등."""
    if n > len(vocab):
        raise ValueError(f"어휘({len(vocab)})가 요청 단어 수({n})보다 적음")
    w = np.array([max(f, 1e-12) for _, f in vocab], dtype=float) ** alpha
    idx = rng.choice(len(vocab), size=n, replace=False, p=w / w.sum())
    return [vocab[i][0] for i in idx]


def balance_words(vocab, words: list[str], rng: np.random.Generator, min_count: int = 15,
                  alpha: float = 0.5, max_iter: int = 500) -> list[str]:
    """자음별 등장 횟수가 min_count 미만이면, 부족 자음이 든 단어로 (부족 자음이 없는) 단어를 교체.
    빈도 기반 추출만으로는 ㅋ/ㅌ/ㅍ 같은 희귀 자음이 몇 번 안 나와 분류 평가가 불가능해지기 때문."""
    words = list(words)
    freq = dict(vocab)
    key_cons = {w: set(word_to_consonant_keys(w)) for w, _ in vocab}
    for _ in range(max_iter):
        cov = consonant_coverage(words)
        lacking = [c for c, n in cov.items() if n < min_count]
        if not lacking:
            break
        c = min(lacking, key=lambda x: cov[x])
        key = CONSONANT_KEYS[c]
        chosen = set(words)
        cand = [w for w, _ in vocab if w not in chosen and key in key_cons[w]]
        if not cand:
            break
        lack_keys = {CONSONANT_KEYS[x] for x in lacking}
        victims = [i for i, w in enumerate(words) if not (key_cons[w] & lack_keys)]
        if not victims:
            break
        p = np.array([freq[w] for w in cand], dtype=float) ** alpha
        words[int(rng.choice(victims))] = cand[int(rng.choice(len(cand), p=p / p.sum()))]
    rng.shuffle(words)
    return words


def consonant_coverage(words: list[str]) -> dict[str, int]:
    cnt = {c: 0 for c in CONSONANTS}
    for w in words:
        for k in word_to_consonant_keys(w):
            cnt[next(c for c, kk in CONSONANT_KEYS.items() if kk == k)] += 1
    return cnt


# --------------------------------------------------------------------------- 플랜
class _Ids:
    def __init__(self):
        self.n = 0

    def __call__(self) -> int:
        self.n += 1
        return self.n


def _info(ids, text: str) -> Prompt:
    return Prompt(ids(), "info", "info", 0, text, "")


def _sync(ids, phase: str, n: int) -> list[Prompt]:
    return [Prompt(ids(), phase, "key", 0, "탁!", " ", "SPACE") for _ in range(n)]


def _calib(ids, n: int) -> list[Prompt]:
    return [Prompt(ids(), "calib", "key", 0, "F", HOME_KEY, "F (ㄹ)") for _ in range(n)]


def _isolated(ids, rng: np.random.Generator, n_blocks: int, rest_every: int) -> list[Prompt]:
    out: list[Prompt] = []
    prev_last = None
    count = 0
    for b in range(1, n_blocks + 1):
        order = list(CONSONANTS)
        rng.shuffle(order)
        if order[0] == prev_last:  # 블록 경계 연속 금지
            j = int(rng.integers(1, len(order)))
            order[0], order[j] = order[j], order[0]
        prev_last = order[-1]
        for c in order:
            if rest_every and count and count % rest_every == 0:
                out.append(_info(ids, f"휴식 — 손을 풀고 편히 쉬세요 ({count}/{n_blocks * len(order)} 완료)"))
            k = CONSONANT_KEYS[c]
            out.append(Prompt(ids(), "isolated", "key", b, c, k, f"{k.upper()} 키"))
            count += 1
    return out


def _continuous(ids, words: list[str], rest_every: int) -> list[Prompt]:
    out: list[Prompt] = []
    for i, w in enumerate(words, 1):
        if rest_every and i > 1 and (i - 1) % rest_every == 0:
            out.append(_info(ids, f"휴식 ({i - 1}/{len(words)} 단어 완료)"))
        keys = word_to_keys(w)
        out.append(Prompt(ids(), "continuous", "word", i, w, keys + " ",
                          f"{keys.upper()} + SPACE", word_to_consonant_keys(w)))
    return out


INTRO = {
    "start": "영문(QWERTY) 입력 상태인지 확인하세요 (한/영키로 영문).\n"
             "링을 착용하고 왼손 검지를 F 위에 올려 두세요.",
    "sync": "[동기화] 화면에 '탁!'이 뜰 때마다 왼손 검지로 스페이스바를 세게 한 번 '탁' 누르세요.",
    "calib": "[보정] F를 누르세요. 매번 F 위에서 출발해서, 누르고, F 위로 돌아옵니다.",
    "isolated": "[개별 키] 표시된 자음의 키를 왼손 검지만으로 누르고 F로 복귀하세요.\n"
                "나머지 손가락은 쓰지 않습니다. 잘못 누르면 그 항목은 나중에 다시 나옵니다.",
    "continuous": "[연속 타이핑] 단어를 평소 속도로 입력하세요: 자음=왼손 검지만, 모음=오른손, 끝에 SPACE.\n"
                  "틀려도 BackSpace를 쓰지 말고 SPACE까지 진행하세요(그 단어는 나중에 다시 나옵니다).",
    "end": "수집 완료. 센서를 그대로 두고 잠시 기다리세요.",
}


def build_plan(phases: tuple[str, ...] = PHASES, *, rng: np.random.Generator,
               words: list[str] | None = None, n_blocks: int = 50, calib_n: int = 10,
               sync_n: int = 5, rest_every_keys: int = 70, rest_every_words: int = 50) -> list[Prompt]:
    ids = _Ids()
    plan: list[Prompt] = [_info(ids, INTRO["start"])]
    if "sync_start" in phases:
        plan += [_info(ids, INTRO["sync"])] + _sync(ids, "sync_start", sync_n)
    if "calib" in phases:
        plan += [_info(ids, INTRO["calib"])] + _calib(ids, calib_n)
    if "isolated" in phases:
        plan += [_info(ids, INTRO["isolated"])] + _isolated(ids, rng, n_blocks, rest_every_keys)
    if "continuous" in phases:
        if not words:
            raise ValueError("continuous 단계에는 words가 필요")
        plan += [_info(ids, INTRO["continuous"])] + _continuous(ids, words, rest_every_words)
    if "sync_end" in phases:
        plan += [_info(ids, INTRO["sync"])] + _sync(ids, "sync_end", sync_n)
    plan.append(_info(ids, INTRO["end"]))
    return plan


def expected_counts(plan: list[Prompt]) -> dict[str, int]:
    out: dict[str, int] = {}
    for p in plan:
        if p.kind != "info":
            out[p.phase] = out.get(p.phase, 0) + 1
    return out


# --------------------------------------------------------------------------- CSV I/O
_COLS = [f.name for f in fields(Prompt)]


def save_plan(plan: list[Prompt], path: str | Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_COLS)
        w.writeheader()
        for p in plan:
            w.writerow(asdict(p))


def load_plan(path: str | Path) -> list[Prompt]:
    with open(path, newline="", encoding="utf-8") as f:
        return [Prompt(prompt_id=int(r["prompt_id"]), phase=r["phase"], kind=r["kind"],
                       block=int(r["block"]), label=r["label"], expect=r["expect"],
                       hint=r["hint"], consonant_keys=r["consonant_keys"])
                for r in csv.DictReader(f)]


def estimated_minutes(plan: list[Prompt], iti: float = 1.7, react: float = 1.2, word_s: float = 6.0) -> float:
    c = expected_counts(plan)
    keys = sum(c.get(k, 0) for k in ("sync_start", "sync_end", "calib", "isolated"))
    return math.ceil((keys * (iti + react) + c.get("continuous", 0) * (word_s + 1.2)) / 60)
