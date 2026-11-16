#!/usr/bin/env python3
"""입력한 단어가 추론 후보 Top-K 안에 있는지 단어별로 확인한다 — 학습 없음.

  python scripts/word_top3.py <세션폴더> [<세션폴더> ...] [옵션]

세션 = scripts/collect.py 로 모은 폴더(연속 타이핑 단계 포함). 자음은 왼손 검지(IMU), 모음·SPACE는 오른손.
제시 단어(trials.csv의 target)가 정답이고, 추론은 정답을 쓰지 않는다(정렬도 라벨 없이).

옵션
  --k 3                    Top-K (기본 3)
  --strikes imu+refine     타격 시각: imu+refine(기본, 실사용) | imu | keylog(오라클)
  --causal N               앞 N단어로만 정렬하고 그 뒤 단어만 평가(실사용 흉내). 0 = 세션 전체로 정렬(기본)
  --vocab 30000            사전: wordfreq 한국어 상위 N개에서 입력 가능한 단어
  --pitch 19.0             키보드 키 간격[mm]
  --flip auto              센서 축 손잡이: auto(라벨 없이 추정) | -1 | +1. 손잡이는 센서 고유값이라 한 번 알면 고정해도 된다
                           (이 링은 S01·S02 모두 -1). 정렬 단어가 적을 때(--causal 20~40) auto는 틀릴 수 있다.
  --csv FILE               단어별 결과 저장(utf-8-sig, 엑셀에서 바로 열림)
  --quiet                  단어별 줄을 생략하고 요약만

단어별 출력: O/X, 정답, 순위, 상위 K 후보, 그리고 X일 때 이유
  사전 밖       정답 단어가 사전에 없음 → 어떤 순위에도 나올 수 없음
  타격 수 불일치 검출한 자음 타격 수 ≠ 정답 자음 수 → 길이가 다른 자음열만 후보가 됨
  순위 밖       후보에는 있으나 K위 밖
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ringko.align3d import choose_gyro_sign, consonant_prior, fit_alignment  # noqa: E402
from ringko.consonant3d import Scorer3D  # noqa: E402
from ringko.dataset import load_session  # noqa: E402
from ringko.decode import ConsonantLM, Lexicon  # noqa: E402
from ringko.hangul import KEY_CONSONANT, to_consonant_keys, to_consonants  # noqa: E402
from ringko.infer3d import Infer3DConfig, get_strikes, infer_session  # noqa: E402
from ringko.keymap3d import Keyboard3D  # noqa: E402
from ringko.plan import filter_vocab, load_vocab  # noqa: E402
from ringko.track3d import TrackParams, track_session  # noqa: E402

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--strikes", default="imu+refine", choices=["keylog", "imu", "imu+refine"])
    ap.add_argument("--causal", type=int, default=0)
    ap.add_argument("--vocab", type=int, default=30000)
    ap.add_argument("--pitch", type=float, default=19.0)
    ap.add_argument("--flip", default="auto")
    ap.add_argument("--csv")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    vocab = filter_vocab(load_vocab(None, a.vocab), min_keys=1)
    lex, lm, prior = Lexicon(vocab), ConsonantLM(vocab), consonant_prior(vocab)
    sc = Scorer3D(Keyboard3D(pitch_mm=a.pitch))
    flips = (1.0, -1.0) if a.flip == "auto" else (float(a.flip),)
    cfg = Infer3DConfig(strikes=a.strikes, top_words=10 ** 9, scorer=sc, flips=flips)
    print(f"사전 {len(vocab)}단어 · Top-{a.k} · 타격 {a.strikes} · "
          f"정렬 {'앞 %d단어' % a.causal if a.causal else '세션 전체'}(라벨 없음) · 학습 없음")
    rows = []
    for d in a.sessions:
        sd = load_session(d)
        strikes, src = get_strikes(sd, a.strikes)
        al, skip = None, set()
        if a.causal:
            g = choose_gyro_sign(sd, strikes=strikes, source=src)["sign"]
            tracks, _ = track_session(sd, strikes, "continuous", TrackParams(gyro_sign=g), source=src)
            head = tracks[:a.causal]
            skip = {w.trial_id for w in head}
            al = fit_alignment(head, sc, lm=lm, flips=cfg.flips, prior=prior)
        res = infer_session(sd, lex, lm, prior, cfg, alignment=al, strikes=strikes if src == "detected" else None)
        n_ok = int(((sd.trials.phase == "continuous") & (sd.trials.status == "ok")).sum())
        print(f"\n=== {Path(d).name} — 정상 단어 {n_ok}개 중 추적 {len(res.words)}개"
              + (f", 정렬용 앞 {len(skip)}개 제외" if skip else ""))
        sess = Path(d).name
        print(f"    정렬: yaw {np.degrees(res.alignment.theta):.0f}° · 손잡이 {res.alignment.flip:+.0f} · 배율 {res.alignment.scale:.2f}")
        s_rows = []
        for wr in res.words:
            tr = wr.track
            if tr.trial_id in skip:
                continue
            word = tr.target
            n_true = len(to_consonant_keys(word))
            ranked = [w for _, w in wr.words]
            rank = ranked.index(word) + 1 if word in ranked else None
            hit = rank is not None and rank <= a.k
            why = "" if hit else ("사전 밖" if word not in lex.freq else
                                  "타격 수 불일치" if tr.n != n_true else "순위 밖")
            s_rows.append({"session": sess, "trial_id": tr.trial_id, "word": word, "hit": int(hit),
                           "rank": rank if rank else "", "reason": why, "consonants": to_consonants(word),
                           "pred_consonants": "".join(KEY_CONSONANT[k] for k in wr.consonants),
                           "n_true": n_true, "n_detected": tr.n, "top": " ".join(ranked[:a.k])})
            if not a.quiet:
                r = s_rows[-1]
                print(f"  {'O' if hit else 'X'} {word:　<5s} 순위 {str(r['rank'] or '-'):>4s} | {r['top']:　<14s}"
                      + (f" ← {why} (검출 {tr.n}타 / 정답 {n_true}타)" if why == "타격 수 불일치" else f" ← {why}" if why else ""))
        df = pd.DataFrame(s_rows)
        rows += s_rows
        if df.empty:
            continue
        in_v = df[df.reason != "사전 밖"]
        print(f"  ▶ Top-{a.k} 포함: {df.hit.sum()}/{len(df)} = {df.hit.mean():.1%}"
              f"  (사전에 있는 단어만: {in_v.hit.sum()}/{len(in_v)} = {in_v.hit.mean() if len(in_v) else float('nan'):.1%})"
              f" · Top-1 {np.mean(df['rank'] == 1):.1%}")
        fails = df[df.hit == 0].reason.value_counts()
        if len(fails):
            print("    X 이유: " + ", ".join(f"{k} {v}" for k, v in fails.items()))
    all_df = pd.DataFrame(rows)
    if len(a.sessions) > 1 and len(all_df):
        print(f"\n전체 {len(all_df)}단어: Top-{a.k} 포함 {all_df.hit.mean():.1%} · Top-1 {np.mean(all_df['rank'] == 1):.1%}")
    if a.csv and len(all_df):
        all_df.to_csv(a.csv, index=False, encoding="utf-8-sig")
        print(f"단어별 결과 저장: {a.csv}")


if __name__ == "__main__":
    main()
