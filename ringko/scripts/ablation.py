#!/usr/bin/env python3
"""파이프라인 구성요소별 기여 측정(단일 사용자, 같은 세션 안 CV). docs/pipeline.docx의 각 항목을 켜고/끄며 비교한다.

  python scripts/ablation.py <세션폴더> [--only kalman,mag,stage2,row,timing,inner,wear,aug]

항목
  kalman : Stage 1-(5) 칼만 스무딩 q/r 비(rho) — 개별 키 LOO 중심 키/행, 연속 기하 사슬 Top-1
  mag    : 자력계 조건부 요 보정(ΔB 임계) — 같은 지표
  stage2 : Stage 2 Algorithm 1의 타이핑 리듬 = 중앙 간격 vs FFT f_type
  row    : (A) QwertyRing 행 모델(윈도우 평균 tilt 정규분포) — 행 정확도 단독, 개별 키 분류 특징 추가, 사슬 unary
  timing : Stage 4 keystroke interval — 사슬 쌍 퍼텐셜(자음), 단어 점수(사이 오른손 키 수 포함)
  inner  : (B) aLeak 내적 매칭 — 단어 점수 gauss/inner/both, 자음 k-best 재순위
  wear   : 착용별 회전·오프셋 보정 — 시험 단어를 회전·축소해 '새 착용'을 흉내 내고, 보정 없음 / 보정 단어 10개 / 정답 없는 EM
  aug    : 증강(학습 fold만) — 개별 키 ExtraTrees(특징 수준), 윈도우 분류기(채널 정규화 ± 증강)
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ringko.dataset import load_session  # noqa: E402
from ringko.decode import (KEYS, ConsonantLM, Geometry, Lexicon, RowModel, TimingModel, chain_marginals,  # noqa: E402
                           kbest_sequences, rerank_inner, timing_pairs)
from ringko.features import Augment, detect_presses, isolated_features, word_presses  # noqa: E402
from ringko.hangul import KEY_ROW  # noqa: E402
from ringko.motion import MotionParams, loo_centroid_accuracy  # noqa: E402
from ringko.plan import filter_vocab, load_vocab  # noqa: E402
from ringko.preprocess import WindowClassifier  # noqa: E402

warnings.filterwarnings("ignore")
from sklearn.ensemble import ExtraTreesClassifier  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

KI = {k: i for i, k in enumerate(KEYS)}
ROW_OF = np.array([KEY_ROW[k] for k in KEYS])


def iso_feats(df, extra=()):
    base = ["dur", "u", "w", "vert"] + [f"p{a}{k}" for k in range(8) for a in "uwz"] + \
        ["rot_u", "rot_w", "rot_z", "vmax_h", "gyr_peak"] + [c for c in df.columns if c.startswith("raw")]
    return base + list(extra)


def topk(P, y, ks=(1, 3, 5)):
    o = np.argsort(-P, 1)
    return {k: float(np.mean([y[i] in o[i, :k] for i in range(len(y))])) for k in ks}


def rowacc(P, y):
    return float(np.mean(ROW_OF[P.argmax(1)] == ROW_OF[y]))


def f3(r):
    return f"{r[1]:5.1%} / {r[3]:5.1%} / {r[5]:5.1%}"


# --------------------------------------------------------------------------- 공통 평가
class Words:
    """연속 타이핑 단어 단위 5-fold."""

    def __init__(self, wp):
        self.wp = wp.reset_index(drop=True)
        self.tids = list(dict.fromkeys(self.wp.trial_id))
        self.fold = {t: i % 5 for i, t in enumerate(self.tids)}
        self.rows = {t: np.where(self.wp.trial_id.to_numpy() == t)[0] for t in self.tids}
        self.obs = {t: self.wp[["du", "dw"]].to_numpy(float)[r] for t, r in self.rows.items()}
        self.cstr = {t: "".join(self.wp.key.to_numpy()[r]) for t, r in self.rows.items()}
        self.ikt = {t: self.wp.ikt.to_numpy(float)[r] for t, r in self.rows.items()}
        self.tail = ({t: float(self.wp["tail"].to_numpy(float)[r[-1]]) for t, r in self.rows.items()}
                     if "tail" in self.wp.columns else {})
        self.word = {t: self.wp.word.to_numpy()[r[0]] for t, r in self.rows.items()}
        self.y = self.wp.key.map(KI).to_numpy()

    def train(self, f):
        return [t for t in self.tids if self.fold[t] != f]

    def test(self, f):
        return [t for t in self.tids if self.fold[t] == f]

    def geo(self, iso, f, offset=True, base=None):
        g = (base.copy() if base is not None else Geometry.from_isolated(iso))
        tr = self.train(f)
        return g.fit([self.obs[t] for t in tr], [self.cstr[t] for t in tr], offset=offset)


def chain_eval(W, iso, lm=None, gamma=0.0, offset=True, row_eta=0.0, row_cols=("tilt_a", "tilt_b"),
               timing_w=0.0, inner_kappa=0.0):
    P = np.zeros((len(W.y), 14))
    for f in range(5):
        geo = W.geo(iso, f, offset)
        trm = np.isin(W.wp.trial_id, W.train(f))
        rm = RowModel(row_cols).fit(W.wp[trm]) if row_eta else None
        tm = TimingModel(with_nv=False).fit(geo, W.wp[trm]) if timing_w else None
        for t in W.test(f):
            r = W.rows[t]
            un = row_eta * rm.unary(W.wp.iloc[r]) if rm is not None else None
            pe = timing_pairs(tm, geo, W.ikt[t], timing_w)
            if inner_kappa:
                c = kbest_sequences(geo, W.obs[t], un, lm, gamma, k=300, pair_extra=pe)
                _, P[r] = rerank_inner(geo, W.obs[t], c, inner_kappa)
            else:
                P[r] = chain_marginals(geo, W.obs[t], un, lm, gamma, pe)
    return P


def word_eval(W, iso, lex, lam=0.5, score="gauss", kappa=20.0, timing_tau=0.0, geos=None, obs=None, unary=None,
              tail=False):
    hits = {1: 0, 3: 0, 5: 0}
    n = 0
    for f in range(5):
        geo = geos[f] if geos is not None else W.geo(iso, f)
        tm = None
        if timing_tau:
            trm = np.isin(W.wp.trial_id, W.train(f))
            tm = TimingModel(with_nv=True).fit(geo, W.wp[trm])
        for t in W.test(f):
            if W.word[t] not in lex.freq:
                continue
            o = obs[t] if obs is not None else W.obs[t]
            ranked = [w for _, w in lex.rank_words(geo, o, unary.get(t) if unary else None, lam, score, kappa,
                                                   tm, W.ikt[t], timing_tau, W.tail.get(t) if tail else None)]
            r = ranked.index(W.word[t]) if W.word[t] in ranked else 10 ** 9
            n += 1
            for k in hits:
                hits[k] += r < k
    return {k: v / max(n, 1) for k, v in hits.items()}, n


def iso_cv(iso, feats, aug_df=None, model=None):
    X = iso[feats].to_numpy(float)
    y = iso.key.map(KI).to_numpy()
    P = np.zeros((len(y), 14))
    for tr, te in GroupKFold(5).split(X, y, iso.block):
        Xtr, ytr = X[tr], y[tr]
        if aug_df is not None:
            a = aug_df[aug_df.trial_id.isin(iso.trial_id.iloc[tr])]
            Xtr = np.vstack([Xtr, a[feats].to_numpy(float)])
            ytr = np.concatenate([ytr, a.key.map(KI).to_numpy()])
        m = (model or (lambda: ExtraTreesClassifier(n_estimators=500, random_state=0, n_jobs=-1)))()
        m.fit(Xtr, ytr)
        Pt = np.zeros((len(te), 14))
        Pt[:, m.classes_] = m.predict_proba(X[te])
        P[te] = Pt
    return P, y


# --------------------------------------------------------------------------- 항목
def stage1_line(sd, p, tag, lm):
    iso = isolated_features(sd, p)
    iso = iso[iso.ok_dur].reset_index(drop=True)
    k, r = loo_centroid_accuracy(iso, KEYS)
    W = Words(word_presses(sd, p))
    P = chain_eval(W, iso)
    Pl = chain_eval(W, iso, lm, 0.5)
    extra = ""
    if "mag_used" in iso and p.mag_db_max > 0:
        extra = f" · 자력계 사용 개별 {iso.mag_used.mean():.0%} / 단어 {W.wp.groupby('trial_id').mag_used.first().mean():.0%}"
    print(f"  {tag:22s} 개별 LOO 키 {k:5.1%} 행 {r:5.1%} | 연속 기하 Top-1/3/5 {f3(topk(P, W.y))} | +LM {f3(topk(Pl, W.y))}{extra}")


def run_kalman(sd, lm):
    print("\n[kalman] Stage 1-(5) 칼만 스무딩 (rho = q/r, 0 = 끔·기존)")
    for rho in (0.0, 1e-6, 1e-5, 1e-4, 1e-3):
        stage1_line(sd, MotionParams(kalman_rho=rho), f"rho={rho:g}", lm)


def run_mag(sd, lm):
    from ringko.magnet import session_hard_iron
    print(f"\n[mag] 자력계 조건부 요 보정 (hard-iron {np.round(session_hard_iron(sd), 1)} uT)")
    for thr in (0.0, 3.0, 5.0, 8.0, 1e9):
        stage1_line(sd, MotionParams(mag_db_max=thr), "6축만(끔)" if thr == 0 else f"ΔB ≤ {thr:g} uT" if thr < 1e8 else "항상 사용", lm)


def eval_det(det, tol=(-0.06, 0.03)):
    tp = fp = fn = 0
    for _, r in det.iterrows():
        used = set()
        for kt in r.left:
            m = [i for i, d in enumerate(r.det) if i not in used and tol[0] <= d - kt <= tol[1]]
            if m:
                used.add(m[0])
                tp += 1
            else:
                fn += 1
        fp += len(r.det) - len(used)
    p, rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return p, rc, float(np.mean([len(r.det) == len(r.left) for _, r in det.iterrows()]))


def run_stage2(sd):
    print("\n[stage2] Algorithm 1 타이핑 리듬 추정: 중앙 간격 vs FFT f_type")
    for pace in ("median", "fft"):
        for alpha in (0.5, 0.75):
            det = detect_presses(sd, alpha=alpha, pace=pace)
            p, r, e = eval_det(det)
            extra = ""
            if pace == "fft":
                true_pace = [np.median(np.diff(x.left)) for _, x in det.iterrows() if len(x.left) >= 3]
                est = [x.tpace for _, x in det.iterrows() if len(x.left) >= 3]
                extra = f" · FFT 간격 vs 실제 중앙 간격 상관 {np.corrcoef(true_pace, est)[0, 1]:.2f}"
            print(f"  pace={pace:6s} alpha={alpha:.2f}  정밀도 {p:5.1%} 재현율 {r:5.1%} 개수 일치 {e:4.0%}{extra}")


def run_row(sd, lm):
    print("\n[row] (A) QwertyRing 행 모델 — 윈도우 전체 평균 자세(tilt)")
    iso = isolated_features(sd)
    iso = iso[iso.ok_dur].reset_index(drop=True)
    y = iso.key.map(KI).to_numpy()
    for cols in (("tilt_a", "tilt_b"), ("dtilt_a", "dtilt_b"), ("tilt_a", "tilt_b", "dtilt_a", "dtilt_b")):
        L = np.zeros((len(iso), 3))
        for tr, te in GroupKFold(5).split(iso, y, iso.block):
            L[te] = RowModel(cols).fit(iso.iloc[tr]).row_logpost(iso.iloc[te])
        print(f"  개별 키 행 정확도(행 모델 단독, {'+'.join(cols)}): {np.mean(L.argmax(1) == ROW_OF[y]):5.1%} (우연 33%)")
    P0, _ = iso_cv(iso, iso_feats(iso))
    P1, _ = iso_cv(iso, iso_feats(iso, ("tilt_a", "tilt_b", "dtilt_a", "dtilt_b")))
    print(f"  개별 키 ExtraTrees Top-1/3/5  기존 {f3(topk(P0, y))} 행 {rowacc(P0, y):5.1%} | +tilt {f3(topk(P1, y))} 행 {rowacc(P1, y):5.1%}")
    W = Words(word_presses(sd))
    L = np.zeros((len(W.y), 3))
    for f in range(5):
        trm = np.isin(W.wp.trial_id, W.train(f))
        L[~trm] = RowModel().fit(W.wp[trm]).row_logpost(W.wp[~trm])
    print(f"  연속 타격 행 정확도(행 모델 단독, tilt_a+tilt_b): {np.mean(L.argmax(1) == ROW_OF[W.y]):5.1%}")
    for eta in (0.0, 0.5, 1.0, 2.0):
        P = chain_eval(W, iso, lm, 0.5, row_eta=eta)
        print(f"  사슬+LM + η={eta:3.1f}·행 unary  Top-1/3/5 {f3(topk(P, W.y))} · 행 {rowacc(P, W.y):5.1%}")


def run_timing(sd, lm, lexes):
    print("\n[timing] Stage 4 keystroke interval")
    iso = isolated_features(sd)
    iso = iso[iso.ok_dur].reset_index(drop=True)
    W = Words(word_presses(sd))
    tm = TimingModel(True).fit(Geometry.from_isolated(iso).fit(list(W.obs.values()), list(W.cstr.values())), W.wp)
    print(f"  간격 = {tm.coef[0]:.3f} + {tm.coef[1] * 1000:.2f}ms/mm·거리 + {tm.coef[2]:.3f}·첫타격 + {tm.coef[3]:.3f}·오른손키수, s={tm.s:.3f}s")
    if tm.tail_coef is not None:
        c = tm.tail_coef
        print(f"  꼬리(F 복귀 정지) = {c[0]:.3f} + {c[1] * 1000:.2f}ms/mm·끝키거리 + {c[2]:.3f}·[끝 자음 뒤 모음], s={tm.s_tail:.3f}s")
    for w_ in (0.0, 0.5, 1.0):
        P = chain_eval(W, iso, lm, 0.5, timing_w=w_)
        print(f"  사슬+LM + 타이밍 가중 {w_:3.1f}   Top-1/3/5 {f3(topk(P, W.y))}")
    for name, lex in lexes.items():
        for tau in (0.0, 0.5, 1.0, 2.0):
            for tl in ((False, True) if tau else (False,)):
                r, n = word_eval(W, iso, lex, timing_tau=tau, tail=tl)
                print(f"  단어 [{name}, {n}개] τ={tau:3.1f}{' +꼬리' if tl else '      '}  Top-1/3/5 {f3(r)}")


def run_inner(sd, lm, lexes):
    print("\n[inner] (B) aLeak 내적(코사인) 기하 매칭")
    iso = isolated_features(sd)
    iso = iso[iso.ok_dur].reset_index(drop=True)
    W = Words(word_presses(sd))
    for kap in (0.0, 5.0, 20.0):
        P = chain_eval(W, iso, lm, 0.5, inner_kappa=kap) if kap else chain_eval(W, iso, lm, 0.5)
        print(f"  자음: 사슬+LM k-best 300 재순위 κ={kap:4.1f}  Top-1/3/5 {f3(topk(P, W.y))}")
    for name, lex in lexes.items():
        for score, kap in (("gauss", 0), ("inner", 20.0), ("both", 5.0), ("both", 20.0)):
            r, n = word_eval(W, iso, lex, score=score, kappa=kap)
            print(f"  단어 [{name}, {n}개] {score:5s} κ={kap:4.1f}  Top-1/3/5 {f3(r)}")


def rot(th):
    c, s = np.cos(np.radians(th)), np.sin(np.radians(th))
    return np.array([[c, s], [-s, c]])


def run_wear(sd, lm, lexes):
    print("\n[wear] 착용별 회전·오프셋 보정 — 시험 단어 관측을 회전 θ·배율 s로 바꿔 '다시 낀 링'을 흉내")
    iso = isolated_features(sd)
    iso = iso[iso.ok_dur].reset_index(drop=True)
    W = Words(word_presses(sd))
    lex = list(lexes.values())[-1]
    for off in (False, True):
        P = chain_eval(W, iso, lm, 0.5, offset=off)
        r, n = word_eval(W, iso, lex, geos={f: W.geo(iso, f, off) for f in range(5)})
        print(f"  (변형 없음) 시작 오프셋 b {'켬' if off else '끔'}: 자음 Top-1/3/5 {f3(topk(P, W.y))} | 단어 {f3(r)}")
    # 레이아웃만(개별 키 없이) — 회전·배율 A만 연속 단어로 맞춤
    geos = {f: W.geo(iso, f, True, Geometry.from_layout()) for f in range(5)}
    r, _ = word_eval(W, iso, lex, geos=geos)
    print(f"  레이아웃만(개별 키 데이터 없이, A·b를 학습 단어로): 단어 {f3(r)}")
    rng = np.random.default_rng(0)
    for th, sc in ((15, 0.85), (30, 0.7)):
        T = rot(th) * sc
        obs2 = {t: o @ T for t, o in W.obs.items()}
        res = {}
        for mode in ("보정 없음", "보정 단어 10개", "정답 없는 EM"):
            geos = {}
            for f in range(5):
                g = W.geo(iso, f)
                te = W.test(f)
                if mode == "보정 단어 10개":
                    cal = list(rng.choice(W.train(f), 10, replace=False))
                    g = g.fit([obs2[t] for t in cal], [W.cstr[t] for t in cal], ridge=20.0)
                elif mode == "정답 없는 EM":
                    g = g.copy().adapt([obs2[t] for t in te], lex, iters=6, ridge=20.0)
                geos[f] = g
            r, _ = word_eval(W, iso, lex, geos=geos, obs=obs2)
            res[mode] = r
        print(f"  θ={th:2d}° s={sc:.2f}: " + " | ".join(f"{m} {f3(r)}" for m, r in res.items()))


def run_aug(sd):
    print("\n[aug] 증강(학습 fold만) · 채널별 정규화(학습 통계)")
    iso = isolated_features(sd, windows=True)
    keep = iso.ok_dur.to_numpy()
    iso = iso[keep].reset_index(drop=True)
    y = iso.key.map(KI).to_numpy()
    feats = iso_feats(iso, ("tilt_a", "tilt_b", "dtilt_a", "dtilt_b"))
    P0, _ = iso_cv(iso, feats)
    augs = pd.concat([isolated_features(sd, aug=Augment(seed=s)) for s in (1, 2, 3)], ignore_index=True)
    P1, _ = iso_cv(iso, feats, augs)
    print(f"  개별 키 ExtraTrees  증강 없음 {f3(topk(P0, y))} | 특징 수준 증강 ×3(학습 fold만) {f3(topk(P1, y))}")
    Wn = np.stack(iso.win.to_numpy())
    for kind in ("logreg", "svm"):
        for aug in (False, True):
            P = np.zeros((len(y), 14))
            for tr, te in GroupKFold(5).split(Wn, y, iso.block):
                m = WindowClassifier(kind, aug).fit(Wn[tr], y[tr])
                P[np.ix_(te, m.classes_)] = m.predict_proba(Wn[te])
            print(f"  윈도우 분류기(채널 정규화) {kind:6s} 증강 {'O' if aug else 'X'}  {f3(topk(P, y))} · 행 {rowacc(P, y):5.1%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--only", default="kalman,mag,stage2,row,timing,inner,wear,aug")
    a = ap.parse_args()
    sd = load_session(a.session)
    only = a.only.split(",")
    probe = word_presses(sd)
    lm = ConsonantLM(filter_vocab(load_vocab(None, 30000), min_keys=1), exclude=set(probe.word))
    lexes = {}
    if {"timing", "inner", "wear"} & set(only):
        for n in (10000, 30000):
            v = filter_vocab(load_vocab(None, n), min_keys=1)
            lexes[f"{len(v)}단어"] = Lexicon(v)
    print(f"세션 {Path(a.session).name} · 자음 Top-1/3/5 = 단어 단위 5-fold · 단어 λ=0.5")
    for k in only:
        {"kalman": lambda: run_kalman(sd, lm), "mag": lambda: run_mag(sd, lm), "stage2": lambda: run_stage2(sd),
         "row": lambda: run_row(sd, lm), "timing": lambda: run_timing(sd, lm, lexes),
         "inner": lambda: run_inner(sd, lm, lexes), "wear": lambda: run_wear(sd, lm, lexes),
         "aug": lambda: run_aug(sd)}[k]()


if __name__ == "__main__":
    main()
