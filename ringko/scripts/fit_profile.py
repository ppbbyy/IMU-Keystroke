#!/usr/bin/env python3
"""여러 사람의 수집 세션으로 공용 프로파일을 만든다 → 새 사용자는 라벨 없이 그대로 쓴다(scripts/decode_free.py가 자동으로 읽음).

  python scripts/fit_profile.py data/raw/*/*                 # 프로파일 생성 → ringko/profiles/population.json
  python scripts/fit_profile.py data/raw/*/* --loso          # + 한 사람을 빼고 만든 프로파일을 그 사람에게 시험

프로파일 내용(ringko/profile.py)
  flip, gyro_sign   센서(링) 고유값. 모든 세션에서 같을 때만 기록. 새 세션에서 자동 추정 대신 쓰면 단어 20개쯤으로도 정렬된다
  centers_u         사람들이 실제로 치는 키 중심(키 단위). decode_free.py --use-centers 일 때만 쓴다(기본 끔 — 아래 --loso 로 확인)
  hit_sd_u          키 안 산포

--loso 는 '그 사람들 말고 새 사람에게 통하는가'를 재는 유일한 방법이다: 사람 A를 빼고 나머지로 프로파일을 만들어
A의 세션(IMU 타격 검출, 라벨 없는 정렬)에 적용하고, 물리 배열만 쓴 결과와 나란히 낸다.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ringko.align3d import choose_gyro_sign, consonant_prior, fit_alignment  # noqa: E402
from ringko.consonant3d import Scorer3D, chain_marginals  # noqa: E402
from ringko.dataset import load_session  # noqa: E402
from ringko.decode import ConsonantLM, Lexicon, open_vocab  # noqa: E402
from ringko.hangul import to_consonant_keys  # noqa: E402
from ringko.infer3d import get_strikes  # noqa: E402
from ringko.keymap3d import KEYS, KI, Keyboard3D  # noqa: E402
from ringko.profile import default_profile_path, fit_profile, keyboard_from_profile, save_profile, session_points  # noqa: E402
from ringko.track3d import TrackParams, strikes_from_keylog, track_session  # noqa: E402

warnings.filterwarnings("ignore")


def load(d, need_det):
    sd = load_session(d)
    st, src = get_strikes(sd, "imu+refine")
    g = choose_gyro_sign(sd, strikes=st, source=src)["sign"]
    tp = TrackParams(gyro_sign=g)
    key, _ = track_session(sd, strikes_from_keylog(sd, "continuous"), "continuous", tp, source="keylog")
    det = track_session(sd, st, "continuous", tp, source=src)[0] if need_det else []
    return {"session": str(d), "subject": sd.meta.get("subject", "?"), "gyro_sign": g, "points": session_points(key),
            "det": det}


def evaluate(s, kb, lex, lm, prior, flip):
    sc = Scorer3D(kb)
    tr = [t for t in s["det"] if np.isfinite(t.P).all()]
    al = fit_alignment(tr, sc, lm=lm, prior=prior, flips=(flip,) if flip else (1.0, -1.0))
    h1 = h3 = n = w3 = 0
    for t in tr:
        obs = sc.observe(t, al)
        truth = to_consonant_keys(t.target)
        if len(truth) == t.n:
            o = np.argsort(-chain_marginals(obs, lm, 0.5), 1)
            y = [KI[c] for c in truth]
            h1 += sum(o[i, 0] == y[i] for i in range(t.n))
            h3 += sum(y[i] in o[i, :3] for i in range(t.n))
            n += t.n
        w3 += t.target in [w for _, w in lex.rank_words(sc, obs, 0.5)[:3]]
    return h1 / max(n, 1), h3 / max(n, 1), w3 / max(len(tr), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--out", default=str(default_profile_path()))
    ap.add_argument("--shrink", type=float, default=0.0, help="실측 중심을 물리 배열 쪽으로 당기는 정도 0~1")
    ap.add_argument("--loso", action="store_true")
    a = ap.parse_args()
    S = []
    for d in a.sessions:
        if not (Path(d) / "meta.json").exists():
            continue
        s = load(d, a.loso)
        S.append(s)
        n = 0 if s["points"] is None else len(s["points"]["key"])
        print(f"  {s['subject']} {Path(d).name[:18]}: 라벨 타격 {n} · 자이로 부호 {s['gyro_sign']}"
              + ("" if s["points"] is None else f" · 손잡이 {s['points']['flip']:+.0f} · 움직임 이득 {s['points']['scale']:.2f}"))
    subs = sorted({s["subject"] for s in S})
    loso = {}
    if a.loso and len(subs) >= 2:
        vocab = open_vocab()
        lex, lm, prior = Lexicon(vocab), ConsonantLM(vocab), consonant_prior(vocab)
        print("\n한 사람을 빼고 만든 프로파일을 그 사람에게 시험 (자음 Top-1 / Top-3 · 단어 Top-3)")
        print(f"  {'세션':28s} | 물리 배열              | 다른 사람들의 실측 중심")
        for s in S:
            prof = fit_profile([x for x in S if x["subject"] != s["subject"]], a.shrink)
            b = evaluate(s, Keyboard3D(), lex, lm, prior, prof["flip"])
            p = evaluate(s, keyboard_from_profile(prof), lex, lm, prior, prof["flip"])
            name = f"{s['subject']} {Path(s['session']).name[:18]}"
            loso[name] = {"physical": b, "centers": p}
            print(f"  {name:28s} | {b[0]:5.1%} / {b[1]:5.1%} · {b[2]:5.1%} | {p[0]:5.1%} / {p[1]:5.1%} · {p[2]:5.1%}")
        mb = np.mean([v["physical"][2] for v in loso.values()])
        mp = np.mean([v["centers"][2] for v in loso.values()])
        print(f"  단어 Top-3 평균: 물리 배열 {mb:.1%} → 실측 중심 {mp:.1%} ({mp - mb:+.1%}p)"
              f" — 차이가 세션 간 흔들림(±5~10%p)보다 작으면 실측 중심은 쓰지 않는 편이 안전")
    prof = fit_profile(S, a.shrink)
    if loso:
        prof["loso"] = {k: {m: [round(float(x), 4) for x in v[m]] for m in v} for k, v in loso.items()}
    save_profile(prof, a.out)
    base = Keyboard3D(pitch_mm=1.0).C2
    off = np.array(prof["centers_u"]) - base
    print(f"\n프로파일 저장: {a.out}")
    print(f"  사람 {prof['n_subjects']}명 {prof['subjects']} · 손잡이 {prof['flip']} · 자이로 부호 {prof['gyro_sign']}")
    print(f"  실측 중심 − 물리 배열 (키 단위, RMS {prof['offset_u_rms']:.2f}u): "
          + " ".join(f"{k}({x:+.2f},{y:+.2f})" for k, (x, y) in zip(KEYS, off)))
    if prof["flip"] is None or prof["gyro_sign"] is None:
        print("  주의: 세션마다 손잡이/자이로 부호가 달라 센서 고유값을 기록하지 못했습니다(센서가 섞였거나 추정 실패).")


if __name__ == "__main__":
    main()
