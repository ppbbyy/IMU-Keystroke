#!/usr/bin/env python3
"""3차원 자음 추론 평가 — 학습 없음. 세션마다 그 세션의 라벨 없는 타이핑만으로 정렬하고 디코딩한다.

  python scripts/eval3d.py data/raw/S01/<세션> [<세션> ...] [옵션]

옵션
  --strikes keylog|imu|imu+refine   타격 시각: 키 로그(오라클) / 기존 Stage 2 검출 / 검출 + 닫힘 보정(기본)
  --mode sector|box|shell           허용 영역: 극좌표(거리+방향, 기본) / 직교 상자 / 거리 구간만
  --align em|none|taps              정렬: 라벨 없는 EM(기본) / 없음 / 탭 캘리브레이션(--taps)
  --taps N                          같은 세션 개별 키 단계의 F→키 시도를 키별 N개(g,d,r,c,v,t,e,s) 써서 정렬
  --causal K                        앞 K단어만으로 정렬(라벨 없음)하고 나머지 단어로 평가(실시간 사용 흉내)
  --closure variance|linear|none, --no-plane    추적 설정(비교용)
  --gyro auto|default               자이로 부호: 라벨 없이 추정(기본) / frames.GYRO_SIGN
  --perturb none|mount|sensor       모의 조건: 부착 방향 회전 / 다른 IMU(축 규약·배율·바이어스·잡음·119Hz)
  --pitch MM                        키보드 키 간격 사양(키보드가 다르면 이것만 바꾼다)
  --flip +1|-1|auto                 센서 축 손잡이(기본 auto: EM이 고름)
  --json FILE                       결과 저장

지표: 후보 재현율·평균 후보 수(① F 거리 min/max 대역, ② 3D 점수 사후확률 누적 90% 집합), 자음 Top-1/Top-3, Macro-F1(14클래스),
      단어 Top-1/3/5(어휘 크기별), 타격 수 일치율(IMU), 정렬 결과와 걸린 시간·쓴 데이터.
--strikes imu*에서 자음 지표는 타격 수가 정답과 같은 단어만, 단어 지표는 전체 단어.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ringko.align3d import consonant_prior, fit_from_taps  # noqa: E402
from ringko.consonant3d import Scorer3D  # noqa: E402
from ringko.dataset import load_session  # noqa: E402
from ringko.decode import ConsonantLM, Lexicon  # noqa: E402
from ringko.hangul import to_consonant_keys  # noqa: E402
from ringko.infer3d import Infer3DConfig, get_strikes, infer_session  # noqa: E402
from ringko.keymap3d import KI, Keyboard3D  # noqa: E402
from ringko.plan import filter_vocab, load_vocab  # noqa: E402
from ringko.track3d import TrackParams, strikes_from_keylog, track_session  # noqa: E402

warnings.filterwarnings("ignore")
TAP_KEYS = "gdrcvtes"


def macro_f1(y, pred, k=14):
    f = []
    for c in range(k):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        if tp + fn == 0:
            continue
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f)) if f else float("nan")


def cumulative_set(M, mass=0.9):
    """(n,14) bool: 사후확률 큰 순으로 누적 mass에 처음 닿을 때까지의 자음 = 최종 후보군."""
    o = np.argsort(-M, 1)
    cs = np.cumsum(np.take_along_axis(M, o, 1), 1)
    keep = np.concatenate([np.ones((len(M), 1), bool), cs[:, :-1] < mass], 1)
    out = np.zeros_like(M, bool)
    np.put_along_axis(out, o, keep, 1)
    return out


def metrics(words, lexes, skip=()):
    y, pred, top3, rec, size, rec90, size90 = [], [], [], 0, 0, 0, 0
    w_hit = {ln: {1: 0, 3: 0, 5: 0} for ln in lexes}
    w_n = {ln: 0 for ln in lexes}
    n_match = n_all = 0
    for wr in words:
        tr = wr.track
        if tr.trial_id in skip:
            continue
        truth = to_consonant_keys(tr.target)
        n_all += 1
        if len(truth) == tr.n:
            n_match += 1
            yy = np.array([KI[c] for c in truth])
            o = np.argsort(-wr.marginals, 1)
            y.extend(yy)
            pred.extend(o[:, 0])
            top3.extend([yy[i] in o[i, :3] for i in range(len(yy))])
            rec += wr.candidates[np.arange(len(yy)), yy].sum()
            size += wr.candidates.sum()
            C90 = cumulative_set(wr.marginals, 0.9)
            rec90 += C90[np.arange(len(yy)), yy].sum()
            size90 += C90.sum()
        for ln, (lex, ranked) in lexes.items():
            if tr.target not in lex.freq:
                continue
            r = ranked[tr.trial_id]
            w_n[ln] += 1
            for k in w_hit[ln]:
                w_hit[ln][k] += r < k
    y, pred = np.array(y), np.array(pred)
    n = max(len(y), 1)
    return {"n_words": n_all, "count_match": n_match / max(n_all, 1), "n_strikes": int(len(y)),
            "cand_recall": rec / n, "cand_size": size / n, "cand90_recall": rec90 / n, "cand90_size": size90 / n, "top1": float(np.mean(pred == y)) if len(y) else np.nan,
            "top3": float(np.mean(top3)) if len(y) else np.nan, "macro_f1": macro_f1(y, pred),
            "word": {ln: {k: v / max(w_n[ln], 1) for k, v in w_hit[ln].items()} | {"n": w_n[ln]} for ln in lexes}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--strikes", default="imu+refine", choices=["keylog", "imu", "imu+refine"])
    ap.add_argument("--mode", default="sector", choices=["sector", "box", "shell"])
    ap.add_argument("--align", default="em", choices=["em", "none", "taps"])
    ap.add_argument("--taps", type=int, default=1)
    ap.add_argument("--causal", type=int, default=0)
    ap.add_argument("--closure", default="variance", choices=["variance", "linear", "none"])
    ap.add_argument("--no-plane", action="store_true")
    ap.add_argument("--gyro", default="auto", choices=["auto", "default"])
    ap.add_argument("--perturb", default="none", choices=["none", "mount", "sensor"])
    ap.add_argument("--pitch", type=float, default=19.0)
    ap.add_argument("--flip", default="auto")
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--json")
    a = ap.parse_args()

    vocab = filter_vocab(load_vocab(None, 30000), min_keys=1)
    lexes0 = {}
    for n in (10000, 30000):
        v = filter_vocab(load_vocab(None, n), min_keys=1)
        lexes0[f"{len(v)}단어"] = Lexicon(v)
    big = list(lexes0.values())[-1]
    lm = ConsonantLM(vocab)
    prior = consonant_prior(vocab)
    flips = (1.0, -1.0) if a.flip == "auto" else (float(a.flip),)
    cfg = Infer3DConfig(strikes=a.strikes, gyro_sign=a.gyro, alignment=a.align if a.align != "taps" else "none",
                        flips=flips, gamma=a.gamma, lam=a.lam, top_words=10 ** 9,
                        track=TrackParams(closure=a.closure, plane=not a.no_plane),
                        scorer=Scorer3D(Keyboard3D(pitch_mm=a.pitch), mode=a.mode))
    print(f"설정: 타격 {a.strikes} · 영역 {a.mode} · 정렬 {a.align}{'(앞 %d단어)' % a.causal if a.causal else ''} · "
          f"닫힘 {a.closure} · 평면 {not a.no_plane} · 자이로 {a.gyro} · 모의 {a.perturb} · pitch {a.pitch}mm · 학습 없음")
    allres = {}
    for d in a.sessions:
        sd = load_session(d)
        if a.perturb == "mount":
            from ringko.simulate import mount_rotation
            sd = mount_rotation(sd)
        elif a.perturb == "sensor":
            from ringko.simulate import other_sensor
            sd = other_sensor(sd)
        t0 = time.time()
        strikes, src = get_strikes(sd, a.strikes)
        t_det = time.time() - t0
        al, cal_note, skip = None, "", set()
        if a.align == "taps" or a.causal:
            from ringko.align3d import choose_gyro_sign, fit_alignment
            g = choose_gyro_sign(sd, strikes=strikes, source=src)["sign"] if a.gyro == "auto" else None
            tp = TrackParams(closure=a.closure, plane=not a.no_plane, gyro_sign=g)
            t1 = time.time()
            if a.align == "taps":
                if not (sd.trials.phase == "isolated").any():
                    print(f"\n{Path(d).name}: 개별 키 단계가 없어 탭 캘리브레이션 불가 — 건너뜀")
                    continue
                iso, _ = track_session(sd, strikes_from_keylog(sd, "isolated"), "isolated", tp)
                taps = []
                for k in TAP_KEYS:
                    taps += [w for w in iso if w.truth == k][:a.taps]
                al = fit_from_taps(taps, flips=flips)
                cal_note = f"탭 {len(taps)}회(키 {TAP_KEYS}, 각 {a.taps}회 ≈ {len(taps) * 1.5:.0f}초)"
            else:
                tracks, _ = track_session(sd, strikes, "continuous", tp, source=src)
                head = tracks[:a.causal]
                skip = {w.trial_id for w in head}
                al = fit_alignment(head, cfg.scorer, lm=lm, gamma=a.gamma, flips=flips, prior=prior)
                cal_note = f"앞 {len(head)}단어(라벨 없음)로 정렬, 그 단어는 평가 제외"
            cal_note += f" · 정렬 계산 {time.time() - t1:.1f}s"
        t1 = time.time()
        res = infer_session(sd, big, lm, prior, cfg, alignment=al, strikes=strikes if src == "detected" else None)
        t_inf = time.time() - t1
        if not cal_note:
            cal_note = f"세션 전체 {len(res.words)}단어(라벨 없음)로 정렬" if a.align == "em" else "정렬 없음"
        ranked = {}
        for ln, lex in lexes0.items():
            rk = {}
            for wr in res.words:
                if lex is big:
                    lst = [w for _, w in wr.words]
                else:
                    lst = [w for _, w in lex.rank_words(cfg.scorer, cfg.scorer.observe(wr.track, res.alignment), a.lam)]
                rk[wr.track.trial_id] = lst.index(wr.track.target) if wr.track.target in lst else 10 ** 9
            ranked[ln] = (lex, rk)
        m = metrics(res.words, ranked, skip)
        al = res.alignment
        gs = res.gyro.get("sign")
        m.update({"alignment": {"theta_deg": float(np.degrees(al.theta)), "flip": al.flip, "scale": al.scale,
                                "sigma_yaw_deg": float(np.degrees(al.sigma_yaw)), "source": al.source},
                  "gyro_sign": None if gs is None else list(map(float, gs)), "gyro_by": res.gyro.get("by"),
                  "closure_med_mm": res.info["closure_med_mm"], "q": res.info["q"], "calibration": cal_note,
                  "time_detect_s": t_det, "time_infer_s": t_inf})
        allres[Path(d).name] = m
        print(f"\n{Path(d).name} · 단어 {m['n_words']} · 타격 수 일치 {m['count_match']:.0%} · 닫힘 중앙 {m['closure_med_mm']:.1f}mm")
        print(f"  자이로 부호 {m['gyro_sign']}({m['gyro_by']}) · 정렬 θ {m['alignment']['theta_deg']:.0f}° "
              f"손잡이 {al.flip:+.0f} 배율 {al.scale:.2f} σψ {m['alignment']['sigma_yaw_deg']:.0f}° · {cal_note}")
        print(f"  후보 ① 거리 대역: 재현율 {m['cand_recall']:5.1%} · 평균 {m['cand_size']:4.1f}/14 | "
              f"② 3D 점수 누적 90%: 재현율 {m['cand90_recall']:5.1%} · 평균 {m['cand90_size']:4.1f}/14")
        print(f"  자음 Top-1 {m['top1']:5.1%} · "
              f"Top-3 {m['top3']:5.1%} · Macro-F1 {m['macro_f1']:.3f} ({m['n_strikes']}타)")
        for ln, r in m["word"].items():
            print(f"  단어 [{ln}, {r['n']}개] Top-1/3/5 {r[1]:5.1%} / {r[3]:5.1%} / {r[5]:5.1%}")
        print(f"  시간: 검출 {t_det:.1f}s · 추적+정렬+디코딩 {t_inf:.1f}s")
    if a.json:
        Path(a.json).write_text(json.dumps({"args": vars(a), "results": allres}, ensure_ascii=False, indent=1,
                                           default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
