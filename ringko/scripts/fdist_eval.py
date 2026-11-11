#!/usr/bin/env python3
"""F 기준 3차원 거리 디코더(ringko.fdist) 평가 — 학습 없음(키 위치·회전 보정·분류기 전부 안 씀).

  python scripts/fdist_eval.py data/raw/S01/<세션> [<세션> ...] [--imu] [--tol 12] [--scale 0.8,1.25]

세션마다 연속 타이핑 단어를 그대로 디코딩한다(fold 없음 — 학습하는 것이 없으므로).
출력
  후보 재현율   정답 키가 F 거리 [min, max] 대역 후보에 든 비율, 타격당 평균 후보 수
  자음 Top-1/3  사전 없이 대역 + 이웃 거리 사슬
  단어 Top-1/3/5 사전(어휘 크기별), λ=0.5
  d/g 구분     정답이 d 또는 g인 타격에서 P(정답) > P(반대쪽)인 비율(사전 없이)
  설정을 나란히: 거리만(닫힘 보정 없음 u,w,z) / 거리만(닫힘 보정 px,py,pz) / 중력 + 세션 yaw(닫힘 보정)
  '중력 + 세션 yaw'는 시험 세션의 라벨 없는 단어로 손잡이·yaw를 추정한다(FDistModel.calibrate).
--imu  타격 시각을 키 로그 대신 detect_presses(Stage 2, 기존 코드 그대로)로. 이때 자음 지표는 타격 수가 맞는 단어만.
"""
import argparse
import sys
import warnings
from pathlib import Path

import copy

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ringko.dataset import load_session  # noqa: E402
from ringko.decode import KI, Lexicon  # noqa: E402
from ringko.fdist import FDistModel  # noqa: E402
from ringko.features import detect_presses, word_presses  # noqa: E402
from ringko.hangul import to_consonant_keys  # noqa: E402
from ringko.plan import filter_vocab, load_vocab  # noqa: E402

warnings.filterwarnings("ignore")
POS = {"거리만 · 닫힘 보정 없음": (["u", "w", "z"], False), "거리만 · 닫힘 보정": (["px", "py", "pz"], False),
       "중력 + 세션 yaw": (["px", "py", "pz"], True)}
DG = (KI["d"], KI["g"])


def evaluate(wp, fd0, lexes, lam):
    out = {}
    cal_lex = list(lexes.values())[-1]
    for name, (cols, orient) in POS.items():
        fd = copy.deepcopy(fd0)
        if orient:
            fd.calibrate([g[cols].to_numpy(float) for _, g in wp.groupby("trial_id", sort=False)], cal_lex)
        dg_ok = dg_n = 0
        rec = size = 0
        c_hit = {1: 0, 3: 0}
        n_c = 0
        w_hit = {ln: {1: 0, 3: 0, 5: 0} for ln in lexes}
        w_n = {ln: 0 for ln in lexes}
        for _, g in wp.groupby("trial_id", sort=False):
            pos = g[cols].to_numpy(float)
            if not np.isfinite(pos).all():
                continue
            truth = to_consonant_keys(g.word.iloc[0])
            if len(truth) == len(pos):
                y = np.array([KI[c] for c in truth])
                C = fd.candidates(pos)
                rec += C[np.arange(len(y)), y].sum()
                size += C.sum()
                M = fd.marginals(pos)
                order = np.argsort(-M, 1)
                for i in range(len(y)):
                    if y[i] in DG:
                        dg_n += 1
                        dg_ok += M[i, y[i]] > M[i, DG[0] if y[i] == DG[1] else DG[1]]
                for k in c_hit:
                    c_hit[k] += sum(y[i] in order[i, :k] for i in range(len(y)))
                n_c += len(y)
            for ln, lex in lexes.items():
                w = g.word.iloc[0]
                if w not in lex.freq:
                    continue
                ranked = [x for _, x in fd.rank_words(lex, pos, lam)]
                r = ranked.index(w) if w in ranked else 10 ** 9
                w_n[ln] += 1
                for k in w_hit[ln]:
                    w_hit[ln][k] += r < k
        cal = f"손잡이 {fd.flip:+.0f} · yaw {np.degrees(fd.mu):.0f}° · κ {fd.kappa:.1f}" if fd.oriented else ""
        out[name] = (rec / max(n_c, 1), size / max(n_c, 1), {k: v / max(n_c, 1) for k, v in c_hit.items()}, n_c,
                     {ln: ({k: v / max(w_n[ln], 1) for k, v in w_hit[ln].items()}, w_n[ln]) for ln in lexes},
                     (dg_ok / max(dg_n, 1), dg_n), cal)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--imu", action="store_true", help="타격 시각 = detect_presses(키 로그 안 씀)")
    ap.add_argument("--tol", type=float, default=12.0, help="대역 여유[mm]")
    ap.add_argument("--scale", default="0.8,1.25", help="배율 대역 s_lo,s_hi")
    ap.add_argument("--press", type=float, default=4.0, help="키 누름 깊이[mm]")
    ap.add_argument("--pitch", type=float, default=19.0, help="키 간격[mm]")
    ap.add_argument("--lam", type=float, default=0.5)
    a = ap.parse_args()
    s_lo, s_hi = map(float, a.scale.split(","))
    fd = FDistModel(a.pitch, a.press, (s_lo, s_hi), a.tol)
    lexes = {}
    for n in (10000, 30000):
        v = filter_vocab(load_vocab(None, n), min_keys=1)
        lexes[f"{len(v)}단어"] = Lexicon(v)
    print(f"F 거리 대역: pitch {a.pitch}mm · 누름 {a.press}mm · 배율 [{s_lo}, {s_hi}] · 여유 {a.tol}mm · "
          f"타격 {'IMU 검출' if a.imu else '키 로그'} · 학습 없음")
    for d in a.sessions:
        sd = load_session(d)
        times = None
        if a.imu:
            det = detect_presses(sd)
            times = dict(zip(det.trial_id, det.det))
        wp = word_presses(sd, times=times)
        print(f"\n{Path(d).name} · 단어 {wp.trial_id.nunique()} · 닫힘 오차 중앙값 "
              f"{wp.groupby('trial_id').closure.first().median():.1f}mm")
        for name, (rec, size, ch, n_c, wh, (dg, dgn), cal) in evaluate(wp, fd, lexes, a.lam).items():
            print(f"  [{name}] {cal}")
            print(f"      후보 재현율 {rec:5.1%} · 평균 후보 {size:4.1f}/14 · 자음 Top-1/3 "
                  f"{ch[1]:5.1%} / {ch[3]:5.1%} ({n_c}타) · d/g 구분 {dg:5.1%} ({dgn}타)")
            for ln, (r, n) in wh.items():
                print(f"      단어 [{ln}, {n}개] Top-1/3/5 {r[1]:5.1%} / {r[3]:5.1%} / {r[5]:5.1%}")


if __name__ == "__main__":
    main()
