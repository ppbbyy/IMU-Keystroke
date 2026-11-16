"""한글 <-> 두벌식 키 변환, 키보드 기하.

두벌식에서 자음은 전부 왼손 영역, 모음은 전부 오른손 영역이다.
그래서 실험 규칙 "자음 = 왼손 검지(IMU), 모음/스페이스 = 오른손"이 자연스럽게 성립한다.
쌍자음과 ㅒ/ㅖ는 Shift가 필요하므로 수집 대상에서 제외한다.
"""
from __future__ import annotations

# 14자음 -> 키. 이 순서가 분류 클래스 순서.
CONSONANT_KEY: dict[str, str] = {
    "ㅂ": "q", "ㅈ": "w", "ㄷ": "e", "ㄱ": "r", "ㅅ": "t",
    "ㅁ": "a", "ㄴ": "s", "ㅇ": "d", "ㄹ": "f", "ㅎ": "g",
    "ㅋ": "z", "ㅌ": "x", "ㅊ": "c", "ㅍ": "v",
}
CONSONANTS: tuple[str, ...] = tuple(CONSONANT_KEY)
KEY_CONSONANT: dict[str, str] = {k: c for c, k in CONSONANT_KEY.items()}
LEFT_KEYS: tuple[str, ...] = tuple(CONSONANT_KEY.values())
HOME_KEY = "f"  # 검지 홈(ㄹ)

# 키보드 기하(키 단위, 1u ≈ 18~19mm). 행 0=QWERT, 1=ASDFG, 2=ZXCV.
# 표준 배열의 행 엇갈림: A행은 Q행보다 0.25u, Z행은 0.75u 오른쪽.
_ROWS = ("qwert", "asdfg", "zxcv")
_ROW_OFFSET = (0.0, 0.25, 0.75)
KEY_ROW: dict[str, int] = {k: r for r, row in enumerate(_ROWS) for k in row}
KEY_XY: dict[str, tuple[float, float]] = {
    k: (_ROW_OFFSET[r] + c, float(r)) for r, row in enumerate(_ROWS) for c, k in enumerate(row)
}

VOWEL_KEYS: dict[str, str] = {
    "ㅛ": "y", "ㅕ": "u", "ㅑ": "i", "ㅐ": "o", "ㅔ": "p",
    "ㅗ": "h", "ㅓ": "j", "ㅏ": "k", "ㅣ": "l",
    "ㅠ": "b", "ㅜ": "n", "ㅡ": "m",
    "ㅘ": "hk", "ㅙ": "ho", "ㅚ": "hl", "ㅝ": "nj", "ㅞ": "np", "ㅟ": "nl", "ㅢ": "ml",
}
CLUSTERS: dict[str, tuple[str, str]] = {
    "ㄳ": ("ㄱ", "ㅅ"), "ㄵ": ("ㄴ", "ㅈ"), "ㄶ": ("ㄴ", "ㅎ"), "ㄺ": ("ㄹ", "ㄱ"),
    "ㄻ": ("ㄹ", "ㅁ"), "ㄼ": ("ㄹ", "ㅂ"), "ㄽ": ("ㄹ", "ㅅ"), "ㄾ": ("ㄹ", "ㅌ"),
    "ㄿ": ("ㄹ", "ㅍ"), "ㅀ": ("ㄹ", "ㅎ"), "ㅄ": ("ㅂ", "ㅅ"),
}
EXCLUDED = frozenset("ㄲㄸㅃㅆㅉㅒㅖ")

_CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_JONG = ("", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ",
         "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ")


def decompose(syllable: str) -> list[str]:
    """완성형 음절 -> 입력 순서 자모(겹받침은 두 자음으로 분리)."""
    code = ord(syllable) - 0xAC00
    if not 0 <= code < 11172:
        raise ValueError(f"한글 음절이 아님: {syllable!r}")
    cho, rest = divmod(code, 21 * 28)
    jung, jong = divmod(rest, 28)
    out = [_CHO[cho], _JUNG[jung]]
    if _JONG[jong]:
        out.extend(CLUSTERS.get(_JONG[jong], (_JONG[jong],)))
    return out


def jamo_seq(word: str) -> list[str]:
    return [j for ch in word for j in decompose(ch)]


def is_typable(word: str) -> bool:
    """한글 음절로만 이루어지고 Shift 글자(쌍자음, ㅒ, ㅖ)가 없는가."""
    if not word:
        return False
    try:
        seq = jamo_seq(word)
    except ValueError:
        return False
    return all(j not in EXCLUDED and (j in CONSONANT_KEY or j in VOWEL_KEYS) for j in seq)


def _check(word: str) -> list[str]:
    if not is_typable(word):
        raise ValueError(f"수집 불가 단어: {word!r}")
    return jamo_seq(word)


def to_keys(word: str) -> str:
    """전체 키열(모음 포함). 한국 -> 'gksrnr'."""
    return "".join(CONSONANT_KEY.get(j) or VOWEL_KEYS[j] for j in _check(word))


def to_consonant_keys(word: str) -> str:
    """자음 키열만(왼손 검지가 치는 키). 한국 -> 'gsrr'."""
    return "".join(CONSONANT_KEY[j] for j in _check(word) if j in CONSONANT_KEY)


def to_consonants(word: str) -> str:
    """자음 자모열. 한국 -> 'ㅎㄴㄱㄱ'."""
    return "".join(j for j in _check(word) if j in CONSONANT_KEY)


# --------------------------------------------------------------------------- 키열 -> 한글 (두벌식 오토마타)
_KEY_VOWEL = {k: v for v, k in VOWEL_KEYS.items() if len(k) == 1}
_COMPOUND = {_KEY_VOWEL[k[0]] + _KEY_VOWEL[k[1]]: v for v, k in VOWEL_KEYS.items() if len(k) == 2}
_CLUSTER_OF = {pair: c for c, pair in CLUSTERS.items()}


def _syll(cho: str, jung: str, jong: str = "") -> str:
    return chr(0xAC00 + (_CHO.index(cho) * 21 + _JUNG.index(jung)) * 28 + _JONG.index(jong))


def compose(keys: str) -> str:
    """두벌식 키열(영문 소문자, Shift 없음) -> 한글. to_keys의 역: compose(to_keys(w)) == w.
    키 로그만 있는 자유 입력에서 '실제로 친 단어'를 복원하는 데 쓴다. 조합되지 않는 자모는 낱자로 둔다."""
    out: list[str] = []
    cho = jung = jong = ""

    def flush():
        nonlocal cho, jung, jong
        if cho and jung:
            out.append(_syll(cho, jung, jong))
        else:
            out.append(cho + jung + jong)
        cho = jung = jong = ""

    for k in keys:
        if k in KEY_CONSONANT:
            c = KEY_CONSONANT[k]
            if not cho and not jung:
                cho = c
            elif cho and jung and not jong:
                jong = c
            elif jong and (jong, c) in _CLUSTER_OF:
                jong = _CLUSTER_OF[(jong, c)]
            else:
                flush()
                cho = c
        elif k in _KEY_VOWEL:
            v = _KEY_VOWEL[k]
            if jong:                                   # 받침이 다음 음절 초성으로 넘어감
                a, b = CLUSTERS.get(jong, ("", jong))
                jong = a
                flush()
                cho, jung = b, v
            elif jung and jung + v in _COMPOUND:
                jung = _COMPOUND[jung + v]
            elif jung:
                flush()
                jung = v
            else:
                jung = v
        else:
            flush()
            out.append(k)
    flush()
    return "".join(out)

