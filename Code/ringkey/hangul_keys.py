"""한글 <-> 두벌식 QWERTY 키 변환.

수집 대상은 왼손 14자음(쌍자음 제외). 두벌식에서 자음은 전부 왼손 영역,
모음은 전부 오른손 영역이라 "자음=왼손 검지, 모음=오른손"으로 자연스럽게 나뉜다.
"""
from __future__ import annotations

# 자음 -> 소문자 QWERTY 키 (왼손 14자음). 순서가 곧 클래스 순서.
CONSONANT_KEYS: dict[str, str] = {
    "ㅂ": "q", "ㅈ": "w", "ㄷ": "e", "ㄱ": "r", "ㅅ": "t",
    "ㅁ": "a", "ㄴ": "s", "ㅇ": "d", "ㄹ": "f", "ㅎ": "g",
    "ㅋ": "z", "ㅌ": "x", "ㅊ": "c", "ㅍ": "v",
}
CONSONANTS: tuple[str, ...] = tuple(CONSONANT_KEYS)
KEY_TO_CONSONANT: dict[str, str] = {k: c for c, k in CONSONANT_KEYS.items()}
LEFT_KEYS: tuple[str, ...] = tuple(CONSONANT_KEYS.values())
# 키보드 줄(1=QWER, 2=ASDF, 3=ZXCV). 줄 분류 분석용.
ROW_OF_KEY: dict[str, int] = {
    "q": 1, "w": 1, "e": 1, "r": 1, "t": 1,
    "a": 2, "s": 2, "d": 2, "f": 2, "g": 2,
    "z": 3, "x": 3, "c": 3, "v": 3,
}
HOME_KEY = "f"  # 검지 홈 위치(ㄹ)

_VOWEL_KEYS: dict[str, str] = {
    "ㅛ": "y", "ㅕ": "u", "ㅑ": "i", "ㅐ": "o", "ㅔ": "p",
    "ㅗ": "h", "ㅓ": "j", "ㅏ": "k", "ㅣ": "l",
    "ㅠ": "b", "ㅜ": "n", "ㅡ": "m",
    "ㅘ": "hk", "ㅙ": "ho", "ㅚ": "hl",
    "ㅝ": "nj", "ㅞ": "np", "ㅟ": "nl", "ㅢ": "ml",
}
_CLUSTERS: dict[str, tuple[str, str]] = {
    "ㄳ": ("ㄱ", "ㅅ"), "ㄵ": ("ㄴ", "ㅈ"), "ㄶ": ("ㄴ", "ㅎ"),
    "ㄺ": ("ㄹ", "ㄱ"), "ㄻ": ("ㄹ", "ㅁ"), "ㄼ": ("ㄹ", "ㅂ"),
    "ㄽ": ("ㄹ", "ㅅ"), "ㄾ": ("ㄹ", "ㅌ"), "ㄿ": ("ㄹ", "ㅍ"),
    "ㅀ": ("ㄹ", "ㅎ"), "ㅄ": ("ㅂ", "ㅅ"),
}
# Shift가 필요한 글자(쌍자음, ㅒ, ㅖ)는 수집 대상에서 제외
_EXCLUDED = set("ㄲㄸㅃㅆㅉㅒㅖ")

_CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_JONG = [
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ",
    "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]


def syllable_jamo(ch: str) -> list[str]:
    """완성형 음절 한 글자를 입력 순서의 자모 리스트로 분해(겹받침은 두 자음으로)."""
    code = ord(ch) - 0xAC00
    if not 0 <= code < 11172:
        raise ValueError(f"한글 음절이 아님: {ch!r}")
    cho, rest = divmod(code, 21 * 28)
    jung, jong = divmod(rest, 28)
    out = [_CHO[cho], _JUNG[jung]]
    j = _JONG[jong]
    if j:
        out.extend(_CLUSTERS.get(j, (j,)))
    return out


def word_jamo(word: str) -> list[str]:
    out: list[str] = []
    for ch in word:
        out.extend(syllable_jamo(ch))
    return out


def is_typable(word: str) -> bool:
    """수집 가능한 단어인가: 한글 음절만, 쌍자음/ㅒ/ㅖ 없음."""
    if not word:
        return False
    try:
        jamo = word_jamo(word)
    except ValueError:
        return False
    return all(j not in _EXCLUDED and (j in CONSONANT_KEYS or j in _VOWEL_KEYS) for j in jamo)


def word_to_keys(word: str) -> str:
    """단어 전체를 두벌식 키 문자열로(모음 포함). 예: 한국 -> 'gksrnr'."""
    if not is_typable(word):
        raise ValueError(f"수집 불가 단어: {word!r}")
    return "".join(CONSONANT_KEYS.get(j) or _VOWEL_KEYS[j] for j in word_jamo(word))


def word_to_consonant_keys(word: str) -> str:
    """자음 키만 추출(모음 제외, 겹받침은 두 키). 예: 한국 -> 'gsrr'."""
    if not is_typable(word):
        raise ValueError(f"수집 불가 단어: {word!r}")
    return "".join(CONSONANT_KEYS[j] for j in word_jamo(word) if j in CONSONANT_KEYS)
