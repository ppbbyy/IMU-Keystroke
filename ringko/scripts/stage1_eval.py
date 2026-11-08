#!/usr/bin/env python3
"""Stage 1 평가: 개별 키 시도의 F→키 수평 변위로 키/행을 얼마나 구분하는가 (설정별 비교).

  python scripts/stage1_eval.py <세션폴더> [--sweep]

지표
  LOO 중심   : 레이아웃 정보 없이, 키별 중앙값 중심 + leave-one-out 최근접 (데이터가 키를 구분하는 상한에 가까운 값)
  레이아웃 맞춤: 측정 → 실제 키보드 좌표 2D 유사변환(Huber 강건 맞춤) 후 최근접 키 (학습 없이 레이아웃만 쓰는 경우)
우연 수준: 키 1/14 = 7%, 행 1/3 = 33%
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ringko.dataset import load_session  # noqa: E402
from ringko.motion import MotionParams, isolated_displacements, loo_centroid_accuracy, robust_layout_fit  # noqa: E402

KEYS = list("qwertasdfgzxcv")


def evaluate(sd, p: MotionParams, verbose=False) -> dict:
    df = isolated_displacements(sd, p)
    iso = df[df.phase == "isolated"]
    good = iso[iso.ok]
    calib = df[df.phase == "calib"]
    k_loo, r_loo = loo_centroid_accuracy(good, KEYS)
    fit = robust_layout_fit(good)
    out = {"n": len(iso), "n_ok": len(good), "onset_cue": int((iso.onset_src == "cue").sum()),
           "loo_key": k_loo, "loo_row": r_loo, "fit_key": fit["key_acc"], "fit_row": fit["row_acc"],
           "fit_scale": fit["scale"], "calib_mm": float(np.median(np.hypot(calib.u, calib.w))) if len(calib) else np.nan,
           "dur_med": float(good.dur.median()), "rot_med": float(good.rot_deg.median())}
    if verbose:
        g = good.groupby("key")
        print(g[["u", "w"]].median().round(1).assign(spread=g[["u", "w"]].std().mean(axis=1).round(1),
                                                       n=g.size()).loc[KEYS].T.to_string())
        from collections import Counter
        conf = Counter((k, p_) for k, p_ in zip(good.key, fit["pred"]) if k != p_)
        print("자주 틀린 쌍(정답→예측):", conf.most_common(10))
    return out


def fmt(p, r):
    return (f"end={p.end:6s} zupt={p.zupt:10s} still={p.still_dps:4.1f} | n={r['n']} ok={r['n_ok']} cue대체={r['onset_cue']:3d} | "
            f"LOO 키 {r['loo_key']:.0%} 행 {r['loo_row']:.0%} | 레이아웃 키 {r['fit_key']:.0%} 행 {r['fit_row']:.0%} "
            f"(스케일 {r['fit_scale']:.2f}) | F→F {r['calib_mm']:.1f}mm · 이동 {r['dur_med']:.2f}s · 회전 {r['rot_med']:.1f}°")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    sd = load_session(a.session)
    if a.sweep:
        for end, zupt, still in itertools.product(["before", "after"], ["horizontal", "full"], [4.0, 8.0]):
            p = MotionParams(end=end, zupt=zupt, still_dps=still)
            print(fmt(p, evaluate(sd, p)), flush=True)
    else:
        p = MotionParams()
        print(fmt(p, evaluate(sd, p, verbose=True)))


if __name__ == "__main__":
    main()
