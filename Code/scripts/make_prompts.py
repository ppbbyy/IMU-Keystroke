#!/usr/bin/env python3
"""프롬프트 CSV 생성(수집 전에 확인/배포용). collect.py는 같은 규칙으로 자동 생성도 하므로 필수는 아님.

예) python scripts/make_prompts.py --subject S01 --session s1 --wear wear0 --out prompts_S01_s1_wear0.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ringkey.prompts import (PHASES, balance_words, build_plan, consonant_coverage,  # noqa: E402
                             estimated_minutes, expected_counts, filter_vocab, load_vocab, sample_words,
                             save_plan, seed_for)


def make(subject, session, wear, *, phases=PHASES, n_blocks=50, n_words=150, vocab_path=None,
         top_n=60000, alpha=0.5, base_seed=0, min_per_consonant=15):
    seed = seed_for(subject, session, wear, base_seed)
    rng = np.random.default_rng(seed)
    words = None
    if "continuous" in phases:
        vocab = filter_vocab(load_vocab(vocab_path, top_n))
        words = sample_words(vocab, n_words, rng, alpha)
        words = balance_words(vocab, words, rng, min_per_consonant, alpha)
    return build_plan(tuple(phases), rng=rng, words=words, n_blocks=n_blocks), words, seed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--session", required=True, choices=["s1", "s2", "s3"])
    ap.add_argument("--wear", required=True, choices=["wear0", "wear1"])
    ap.add_argument("--blocks", type=int, default=50, help="개별 키 블록 수(14자음 x 블록)")
    ap.add_argument("--words", type=int, default=150, help="연속 타이핑 단어 수(최소 150, 권장 300)")
    ap.add_argument("--vocab", help="단어[ 빈도] 목록 파일(없으면 wordfreq ko)")
    ap.add_argument("--top-n", type=int, default=60000)
    ap.add_argument("--alpha", type=float, default=0.5, help="단어 추출 가중: 빈도^alpha")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-per-consonant", type=int, default=15, help="연속 타이핑에서 자음별 최소 등장 횟수 보장")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    plan, words, seed = make(a.subject, a.session, a.wear, n_blocks=a.blocks, n_words=a.words,
                             vocab_path=a.vocab, top_n=a.top_n, alpha=a.alpha, base_seed=a.seed,
                             min_per_consonant=a.min_per_consonant)
    save_plan(plan, a.out)
    print(f"저장: {a.out}  (seed={seed})")
    print("항목 수:", expected_counts(plan), f"· 예상 소요 약 {estimated_minutes(plan)}분")
    if words:
        cov = consonant_coverage(words)
        print("연속 타이핑 단어 예:", " ".join(words[:10]))
        print("단어 내 자음 등장 횟수:", cov)
        low = [c for c, n in cov.items() if n < 10]
        if low:
            print(f"경고: 등장 10회 미만 자음 {low} — --words를 늘리거나 --alpha를 낮추세요")


if __name__ == "__main__":
    main()
