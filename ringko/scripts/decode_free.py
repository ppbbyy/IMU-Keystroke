#!/usr/bin/env python3
"""자유 입력 추론: IMU만으로 단어 경계 → 타격 → 자음 → 단어 후보. 제시 단어·키 로그를 쓰지 않는다. 학습 없음.

  python scripts/decode_free.py <세션폴더> [옵션]

세션폴더
  · scripts/record_free.py 로 녹화한 자유 입력 세션 → 녹화 전체를 추론
  · scripts/collect.py 로 모은 세션 → 연속 타이핑 구간의 IMU만 떼어 '자유 입력인 것처럼' 추론(검증용)
어느 쪽이든 추론에는 IMU만 쓴다. 키 로그/제시 단어가 있으면 '실제로 친 단어'를 복원해 채점에만 쓴다(--no-truth 로 끔).

출력(단어 후보마다)
  자음열 후보  사전 없이 가장 그럴듯한 자음열 K개(예: ㅎㄴㄱㄱ) — 사전에 없는 단어도 여기에는 나올 수 있다
  단어 후보    열린 사전(wordfreq에서 입력 가능한 단어 전부 + --add-words 로 넣은 내 단어)에서 K개
가정: 쌍자음(ㄲㄸㅃㅆㅉ)·ㅒ·ㅖ가 든 단어는 대상에서 제외(Shift 필요). --add-words 에 있어도 건너뛴다.

옵션
  --k 3                Top-K
  --add-words FILE     내 단어 추가. 한 줄에 한 단어든 아무 글(메모·문서)이든 된다 — 한글 단어를 모두 뽑아 사전에 넣는다
  --flip auto|-1|+1    센서 축 손잡이. 단어가 적은 녹화(< 60단어)는 auto가 틀릴 수 있으니 아는 값을 준다(이 링은 -1)
  --still-dps 15 --min-still 0.4    단어 경계: 자이로가 이 값 아래로 이 시간[s] 이상 = F 위 정지
  --pitch 19.0         키 간격[mm]
  --gyro auto|default  자이로 부호: 라벨 없이 추정(기본) | frames.GYRO_SIGN
  --csv FILE           결과 저장(utf-8-sig)
  --no-truth           키 로그가 있어도 채점하지 않음
  --drop-truth         (검증용) 실제로 친 단어를 사전에서 빼고 추론 = '사전에 없는 단어를 쳤을 때' 재현.
                       --add-words 와 함께 쓰면 '사전에 없던 단어를 내 단어로 넣었을 때'가 된다
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ringko.align3d import consonant_prior  # noqa: E402
from ringko.consonant3d import Scorer3D, kbest_strings  # noqa: E402
from ringko.dataset import load_session  # noqa: E402
from ringko.decode import ConsonantLM, Lexicon, open_vocab, words_from_text  # noqa: E402
from ringko.free import match_truth, plausible_word, typed_words, virtual_session  # noqa: E402
from ringko.hangul import KEY_CONSONANT, LEFT_KEYS, is_typable  # noqa: E402
from ringko.hangul import to_consonant_keys as skeleton_keys  # noqa: E402
from ringko.infer3d import Infer3DConfig, infer_session  # noqa: E402
from ringko.keymap3d import Keyboard3D  # noqa: E402

warnings.filterwarnings("ignore")
jamo = lambda ks: "".join(KEY_CONSONANT[k] for k in ks)  # noqa: E731


def collected_truth(sd):
    """수집 세션: 정상 연속 타이핑 단어와 왼손 키 시각, 그리고 연속 타이핑 시간 범위."""
    tr = sd.trials[sd.trials.phase == "continuous"]
    kd = sd.keys[sd.keys.event == "down"]
    rows = []
    for x in tr[tr.status == "ok"].itertuples():
        k = kd[(kd.trial_id == x.trial_id) & kd.key.isin(LEFT_KEYS)]
        if len(k):
            rows.append({"word": x.target, "t_first": k.host_ns.min() / 1e9, "t_last": k.host_ns.max() / 1e9,
                         "n_left": len(k)})
    return pd.DataFrame(rows), (tr.cue_ns.min() / 1e9 - 3, tr.end_ns.max() / 1e9 + 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--add-words")
    ap.add_argument("--flip", default="auto")
    ap.add_argument("--still-dps", type=float, default=15.0)
    ap.add_argument("--min-still", type=float, default=0.4)
    ap.add_argument("--pitch", type=float, default=19.0)
    ap.add_argument("--gyro", default="auto", choices=["auto", "default"])
    ap.add_argument("--csv")
    ap.add_argument("--no-truth", action="store_true")
    ap.add_argument("--drop-truth", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    sd = load_session(a.session)
    collected = (sd.trials.phase == "continuous").any()
    truth, t_range = (collected_truth(sd) if collected else (typed_words(sd.keys), None))
    if a.no_truth:
        truth = truth.iloc[:0]
    extra = words_from_text(Path(a.add_words).read_text(encoding="utf-8")) if a.add_words else []
    drop = set(truth.word) if a.drop_truth else set()
    vocab = [(w, f) for w, f in open_vocab() if w not in drop]            # --drop-truth: 친 단어를 사전에서 뺌(사전 밖 재현)
    have = {w for w, _ in vocab}
    fl = float(np.median([f for _, f in vocab]))
    skipped = [w for w in extra if not is_typable(w)]
    added = [w for w in dict.fromkeys(extra) if is_typable(w) and w not in have]
    vocab += [(w, fl) for w in added]                                     # 내 단어: 빈도는 사전 중앙값
    lex, lm, prior = Lexicon(vocab), ConsonantLM(vocab), consonant_prior(vocab)
    v, seg = virtual_session(sd, t_range, still_dps=a.still_dps, min_still_s=a.min_still)
    flips = (1.0, -1.0) if a.flip == "auto" else (float(a.flip),)
    sc = Scorer3D(Keyboard3D(pitch_mm=a.pitch))
    cfg = Infer3DConfig(strikes="imu+refine", top_words=a.k, scorer=sc, flips=flips, gyro_sign=a.gyro)
    ok = lambda t: plausible_word(t, a.pitch)  # noqa: E731
    res = infer_session(v, lex, lm, prior, cfg, align_filter=ok)
    n_all = len(res.words)
    res.words = [w for w in res.words if ok(w.track)]
    al = res.alignment
    seg = match_truth(seg, truth)
    T = seg.attrs["truth"]
    seg_word = dict(zip(seg.trial_id, seg.word))
    print(f"{Path(a.session).name} · {'수집 세션의 연속 타이핑 구간' if collected else '자유 입력 녹화'} · IMU만 사용")
    if skipped:
        print(f"추가 단어 중 쌍자음·ㅒ·ㅖ가 있어 건너뜀 {len(skipped)}개: {' '.join(skipped[:8])}{' …' if len(skipped) > 8 else ''}")
    print(f"사전 {len(vocab)}단어(추가 {len(added)}{', 친 단어 %d개 제외' % len(drop) if drop else ''}) · 단어 후보(움직임 구간) {len(seg)}개 → 타격이 있는 것 {n_all}개"
          f" → 단어로 볼 만한 것 {len(res.words)}개"
          f" · 정렬 yaw {np.degrees(al.theta):.0f}° 손잡이 {al.flip:+.0f} 배율 {al.scale:.2f}"
          + (f" · 실제 단어 {len(T)}개(채점용)" if len(T) else " · 정답 없음(채점 생략)"))
    if len(res.words) < 60 and a.flip == "auto":
        print("주의: 단어가 60개 미만이면 손잡이 자동 추정이 틀릴 수 있습니다. 같은 링이면 --flip -1 처럼 고정하세요.")
    t0 = v.imu.host_ns.iloc[0] / 1e9
    rows = []
    for wr in res.words:
        tr = wr.track
        obs = sc.observe(tr, al)
        sk = [s for _, s in kbest_strings(obs, lm, cfg.gamma, a.k)]
        wd = [w for _, w in wr.words]
        true = seg_word.get(tr.trial_id, "")
        r = {"id": tr.trial_id, "t_s": round(float(tr.t_strike[0] - t0), 2), "n_strikes": tr.n,
             "skeletons": " ".join(jamo(s) for s in sk), "words": " ".join(wd), "truth": true,
             "truth_skeleton": jamo(skeleton_keys(true)) if true else "",
             "word_hit": int(true in wd) if true else "", "skeleton_hit": int(skeleton_keys(true) in sk) if true else "",
             "in_vocab": int(true in lex.freq) if true else ""}
        rows.append(r)
        if not a.quiet:
            mark = ("O" if r["word_hit"] else "X") if true else " "
            print(f"  {mark} {r['t_s']:7.1f}s {tr.n}타 | 자음열 {r['skeletons']:　<16s} | 단어 {r['words']:　<14s}"
                  + (f" | 정답 {true} ({r['truth_skeleton']})" if true else ""))
    df = pd.DataFrame(rows)
    if len(T) and len(df):
        by_truth = {r["truth"]: r for r in rows if r["truth"]}
        why = []
        for x in T.itertuples():
            r = by_truth.get(x.word) if x.seg_id >= 0 else None
            if x.seg_id < 0:
                why.append("경계: 쪼개짐" if x.n_seg > 1 else "경계: 합쳐짐·누락")
            elif r is None:
                why.append("타격 없음")
            elif r["word_hit"]:
                why.append("hit")
            elif r["n_strikes"] != len(skeleton_keys(x.word)):
                why.append("타격 수 불일치")
            elif not r["in_vocab"]:
                why.append("사전 밖")
            else:
                why.append("순위 밖")
        n = len(T)
        w_hit = why.count("hit")
        s_hit = sum(int(r["skeleton_hit"]) for r in by_truth.values())
        print(f"\n▶ 실제 단어 {n}개 기준  단어 Top-{a.k} 포함 {w_hit}/{n} = {w_hit / n:.1%} · "
              f"자음열 Top-{a.k} 포함 {s_hit}/{n} = {s_hit / n:.1%}")
        miss = pd.Series([w for w in why if w != "hit"]).value_counts()
        print("  놓친 이유: " + (", ".join(f"{k} {v}" for k, v in miss.items()) if len(miss) else "없음"))
        print(f"  가짜 단어(실제 단어가 없는데 나온 후보): {int((df.truth == '').sum())}개")
    if a.csv and len(df):
        df.to_csv(a.csv, index=False, encoding="utf-8-sig")
        print(f"저장: {a.csv}")


if __name__ == "__main__":
    main()
