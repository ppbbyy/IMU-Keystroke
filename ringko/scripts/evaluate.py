#!/usr/bin/env python3
"""성능 평가(단일 사용자). docx 기준: 자음 분류 Top-1/3/5 와 단어 복원 Top-1/3/5 를 분리 보고.

  python scripts/evaluate.py <세션폴더> [<세션폴더> ...]

1) 개별 키 자음 분류: 블록 단위 GroupKFold(5) — 같은 블록(시간대)의 시도가 학습/평가에 섞이지 않게.
2) 연속 타이핑 자음 분류: 단어 내 타격 위치(F 기준 누적 변위) 등. 분할은 키 로그(오라클). 단어 단위 GroupKFold(5).
3) Stage 2: IMU만으로 왼손 타격 검출 정밀도/재현율.
4) 단어 복원: 사전(wordfreq) × 타격별 자음 확률. 어휘 크기별. '자음 100% 정확' 상한도 함께.

파이프라인 구성요소(docs/pipeline.docx ⑧)별 결과도 함께 낸다. 켜고/끄는 세부 비교는 scripts/ablation.py.
  [1] 특징 + 행 자세(tilt), 윈도우 분류기(채널 정규화 ± 증강, 학습 fold만), 특징 수준 증강
  [2] (f) 행 모델(QwertyRing) unary, (g) 타이밍(Stage 4) 쌍 퍼텐셜, (h) aLeak 내적 k-best 재순위
  [3] Algorithm 1 리듬: 중앙 간격 / FFT(단어) / FFT(세션)
  [4] 기하 가우스 + aLeak 내적(κ) + 타이밍(τ) — 채택 설정 'both κ=20 τ=1'
"""
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ringko.dataset import load_session  # noqa: E402
from ringko.decode import (ConsonantLM, Geometry, Lexicon, RowModel, TimingModel, chain_marginals,  # noqa: E402
                           kbest_sequences, rerank_inner, timing_pairs)
from ringko.features import Augment, detect_presses, isolated_features, word_presses  # noqa: E402
from ringko.preprocess import WindowClassifier  # noqa: E402
from ringko.hangul import KEY_ROW, LEFT_KEYS  # noqa: E402
from ringko.plan import filter_vocab, load_vocab  # noqa: E402

warnings.filterwarnings("ignore")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis  # noqa: E402
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

CLASSES = list(LEFT_KEYS)
CI = {k: i for i, k in enumerate(CLASSES)}


def models():
    return {
        "QDA(u,w)": None,  # 특수: 변위 2D만
        "LDA": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)),
        "SVM-RBF": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", probability=True, random_state=0)),
        "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=9, weights="distance")),
        "RandomForest": RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=800, random_state=0, n_jobs=-1),
    }


def full_proba(model, X, n_classes=14):
    """학습에 없는 클래스가 있어도 14열 확률."""
    P = np.zeros((len(X), n_classes))
    P[:, model.classes_] = model.predict_proba(X)
    return P


def topk(P, y, ks=(1, 3, 5)):
    order = np.argsort(-P, axis=1)
    return {k: float(np.mean([y[i] in order[i, :k] for i in range(len(y))])) for k in ks}


def row_acc(P, y):
    pred = P.argmax(1)
    return float(np.mean([KEY_ROW[CLASSES[a]] == KEY_ROW[CLASSES[b]] for a, b in zip(pred, y)]))


def cv_proba(make, X, y, groups, n_splits=5):
    P = np.zeros((len(y), 14))
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        m = make()
        m.fit(X[tr], y[tr])
        P[te] = full_proba(m, X[te])
    return P


FOLD, OBS, CSTR, ROWS = {}, {}, {}, {}


def setup_folds(wp, n_splits=5):
    """단어(trial) 단위 fold를 한 번 정해 분류기와 기하 적합에 똑같이 쓴다."""
    tids = list(dict.fromkeys(wp.trial_id))
    for i, t_ in enumerate(tids):
        FOLD[t_] = i % n_splits
        rows = np.where(wp.trial_id.to_numpy() == t_)[0]
        ROWS[t_] = rows
        OBS[t_] = wp[["du", "dw"]].to_numpy()[rows]
        CSTR[t_] = "".join(wp.key.to_numpy()[rows])


def folds_rows(wp):
    f = np.array([FOLD[t_] for t_ in wp.trial_id])
    return [(np.where(f != k)[0], np.where(f == k)[0]) for k in sorted(set(f))]


def fmt_top(r):
    return f"Top-1 {r[1]:.1%}  Top-3 {r[3]:.1%}  Top-5 {r[5]:.1%}"


# --------------------------------------------------------------------------- 1) 개별 키
BASE_FEATS = ["dur", "u", "w", "vert"] + [f"p{a}{k}" for k in range(8) for a in "uwz"] + \
    ["rot_u", "rot_w", "rot_z", "vmax_h", "gyr_peak"]
TILT_FEATS = ["tilt_a", "tilt_b", "dtilt_a", "dtilt_b"]


def iso_feature_cols(df):
    return BASE_FEATS + [c for c in df.columns if c.startswith("raw")] + TILT_FEATS


def eval_isolated(df, aug_df=None):
    feat = iso_feature_cols(df)
    X = df[feat].to_numpy(float)
    y = df.key.map(CI).to_numpy()
    g = df.block.to_numpy()
    print(f"\n[1] 개별 키 자음 분류 — n={len(df)} (14클래스, 블록 GroupKFold 5) · 특징 {len(feat)}개 · 우연 Top-1 7.1%")
    best = (None, -1, None)
    for name, mdl in models().items():
        if mdl is None:
            Xs = df[["u", "w"]].to_numpy(float)
            P = cv_proba(lambda: QuadraticDiscriminantAnalysis(reg_param=0.1), Xs, y, g)
        else:
            from sklearn.base import clone
            P = cv_proba(lambda: clone(mdl), X, y, g)
        r = topk(P, y)
        print(f"  {name:13s} {fmt_top(r)}  · 행 {row_acc(P, y):.1%}")
        if r[1] > best[1]:
            best = (name, r[1], P)
    # 윈도우 분류기: 기준 프레임 원시 윈도우(48샘플 × 6채널) → 채널별 정규화(학습 fold 통계) ± 증강(학습 fold만)
    if "win" in df:
        Wn = np.stack(df.win.to_numpy())
        for aug in (False, True):
            P = np.zeros((len(y), 14))
            for tr, te in GroupKFold(5).split(Wn, y, g):
                m = WindowClassifier("svm", aug).fit(Wn[tr], y[tr])
                P[np.ix_(te, m.classes_)] = m.predict_proba(Wn[te])
            print(f"  {'Win-SVM' + ('+증강' if aug else ''):13s} {fmt_top(r := topk(P, y))}  · 행 {row_acc(P, y):.1%}"
                  "  (원시 윈도우, 채널 정규화)")
    # 특징 수준 증강(진폭·잡음·시간이동, 학습 fold의 시도에만 붙임)
    if aug_df is not None and len(aug_df):
        P = np.zeros((len(y), 14))
        for tr, te in GroupKFold(5).split(X, y, g):
            a = aug_df[aug_df.trial_id.isin(df.trial_id.iloc[tr])]
            m = ExtraTreesClassifier(n_estimators=800, random_state=0, n_jobs=-1)
            m.fit(np.vstack([X[tr], a[feat].to_numpy(float)]), np.concatenate([y[tr], a.key.map(CI).to_numpy()]))
            P[te] = full_proba(m, X[te])
        r = topk(P, y)
        print(f"  {'ET+증강×3':13s} {fmt_top(r)}  · 행 {row_acc(P, y):.1%}  (학습 fold에만 증강본)")
        if r[1] > best[1]:
            best = ("ExtraTrees+증강", r[1], P)
    name, _, P = best
    pred = P.argmax(1)
    conf = defaultdict(int)
    for a, b in zip(y, pred):
        if a != b:
            conf[(CLASSES[a], CLASSES[b])] += 1
    per = {CLASSES[c]: float(np.mean(pred[y == c] == c)) for c in range(14)}
    print(f"  최고 {name}: 키별 Top-1 " + " ".join(f"{k}{v:.0%}" for k, v in per.items()))
    print("  자주 틀린 쌍:", sorted(conf.items(), key=lambda kv: -kv[1])[:8])
    return name


# --------------------------------------------------------------------------- 2) 연속 타이핑
def eval_continuous(iso, wp):
    y = wp.key.map(CI).to_numpy()
    g = wp.trial_id.to_numpy()
    raw = [c for c in wp.columns if c.startswith("raw")]
    print(f"\n[2] 연속 타이핑 자음 분류 — 단어 {wp.trial_id.nunique()}개, 자음 타격 {len(wp)}개 (분할=키 로그)")
    res = {}
    # a) 개별 키로만 학습 -> 연속 평가(전이)
    Xi = iso[["u", "w"] + raw].to_numpy(float)
    yi = iso.key.map(CI).to_numpy()
    Xc = wp[["u", "w"] + raw].to_numpy(float)
    for name, mk in (("QDA(u,w)", lambda: QuadraticDiscriminantAnalysis(reg_param=0.1)),
                     ("ExtraTrees(u,w,raw)", lambda: ExtraTreesClassifier(n_estimators=800, random_state=0, n_jobs=-1))):
        cols = slice(0, 2) if "QDA" in name else slice(None)
        m = mk().fit(Xi[:, cols], yi)
        P = full_proba(m, Xc[:, cols])
        r = topk(P, y)
        res[f"iso->cont {name}"] = P
        print(f"  (a) 개별 키로 학습 → 연속 평가  {name:20s} {fmt_top(r)} · 행 {row_acc(P, y):.1%}")
    # b) 연속 안에서 단어 단위 CV,  c) 개별 키 + 연속 학습
    feats = ["u", "w", "du", "dw", "idx", "n", "word_t"] + raw
    Xw = wp[feats].to_numpy(float)
    for name, mk in (("ExtraTrees", lambda: ExtraTreesClassifier(n_estimators=800, random_state=0, n_jobs=-1)),
                     ("LogReg", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)))):
        P = cv_proba(mk, Xw, y, g)
        res[f"cont-cv {name}"] = P
        print(f"  (b) 연속 안에서 단어 CV          {name:20s} {fmt_top(r := topk(P, y))} · 행 {row_acc(P, y):.1%}")
    # c) 개별 키 데이터를 학습에 추가(공통 특징: u,w,raw; du,dw는 개별 키에선 u,w와 같음)
    iso_c = iso.assign(du=iso.u, dw=iso.w, idx=0, n=1, word_t=iso.dur)
    Xi2 = iso_c[feats].to_numpy(float)
    P = np.zeros((len(y), 14))
    for tr, te in folds_rows(wp):
        m = ExtraTreesClassifier(n_estimators=800, random_state=0, n_jobs=-1)
        m.fit(np.vstack([Xw[tr], Xi2]), np.concatenate([y[tr], yi]))
        P[te] = full_proba(m, Xw[te])
    res["cont-cv+iso ExtraTrees"] = P
    print(f"  (c) 개별 키 + 연속(단어 CV) 학습  {'ExtraTrees':20s} {fmt_top(topk(P, y))} · 행 {row_acc(P, y):.1%}")
    by_idx = wp.assign(ok=(P.argmax(1) == y)).groupby("idx").ok.mean()
    print("      단어 내 타격 순서별 Top-1(위치 누적 오차 확인): " + " ".join(f"{i + 1}번째 {v:.0%}" for i, v in by_idx.items() if i < 6))
    # d) 기하 사슬 모델(상대 벡터) ± 분류기 — 사전 없이 forward-backward 주변확률
    for beta in (0.0, 0.5, 1.0):
        Pg = np.zeros_like(P)
        for fold in sorted(set(FOLD.values())):
            trw = [t_ for t_ in OBS if FOLD[t_] != fold]
            geo = Geometry.from_isolated(iso).fit([OBS[t_] for t_ in trw], [CSTR[t_] for t_ in trw])
            for t_ in OBS:
                if FOLD[t_] == fold:
                    rows = ROWS[t_]
                    Pg[rows] = chain_marginals(geo, OBS[t_], beta * np.log(np.clip(P[rows], 1e-4, 1)))
        tag = "기하만" if beta == 0 else f"기하+분류기 β={beta}"
        print(f"  (d) 기하 사슬 모델(단어 CV)       {tag:20s} {fmt_top(topk(Pg, y))} · 행 {row_acc(Pg, y):.1%}")
    # e) + 자음 트라이그램 언어모델(평가 단어는 언어모델에서 제외)
    lm = ConsonantLM(filter_vocab(load_vocab(None, 30000), min_keys=1), exclude=set(wp.word))
    for gamma in (0.5,):
        Pg = np.zeros_like(P)
        for fold in sorted(set(FOLD.values())):
            trw = [t_ for t_ in OBS if FOLD[t_] != fold]
            geo = Geometry.from_isolated(iso).fit([OBS[t_] for t_ in trw], [CSTR[t_] for t_ in trw])
            for t_ in OBS:
                if FOLD[t_] == fold:
                    Pg[ROWS[t_]] = chain_marginals(geo, OBS[t_], None, lm, gamma)
        print(f"  (e) 기하 + 자음 언어모델 γ={gamma}    {'':14s} {fmt_top(topk(Pg, y))} · 행 {row_acc(Pg, y):.1%}")
    # f) 행 모델(QwertyRing), g) 타이밍(Stage 4), h) aLeak 내적 재순위 — 모두 (e) 위에 추가
    ikt = {t_: wp.ikt.to_numpy(float)[ROWS[t_]] for t_ in OBS}
    for tag, eta, tw, kap in (("(f) + 행 모델 η=0.5", 0.5, 0, 0), ("(g) + 타이밍 쌍 w=0.5", 0, 0.5, 0),
                              ("(h) + aLeak 내적 κ=5", 0, 0, 5.0)):
        Pg = np.zeros_like(P)
        for fold in sorted(set(FOLD.values())):
            trw = [t_ for t_ in OBS if FOLD[t_] != fold]
            trm = wp.trial_id.isin(trw).to_numpy()
            geo = Geometry.from_isolated(iso).fit([OBS[t_] for t_ in trw], [CSTR[t_] for t_ in trw])
            rm = RowModel(("dtilt_a", "dtilt_b")).fit(wp[trm]) if eta else None
            tm = TimingModel(False).fit(geo, wp[trm]) if tw else None
            for t_ in OBS:
                if FOLD[t_] != fold:
                    continue
                rows = ROWS[t_]
                un = eta * rm.unary(wp.iloc[rows]) if rm is not None else None
                pe = timing_pairs(tm, geo, ikt[t_], tw)
                if kap:
                    c = kbest_sequences(geo, OBS[t_], un, lm, 0.5, k=300, pair_extra=pe)
                    _, Pg[rows] = rerank_inner(geo, OBS[t_], c, kap)
                else:
                    Pg[rows] = chain_marginals(geo, OBS[t_], un, lm, 0.5, pe)
        print(f"  {tag:34s}{'':6s} {fmt_top(topk(Pg, y))} · 행 {row_acc(Pg, y):.1%}")
    return P, "cont-cv+iso ExtraTrees"


# --------------------------------------------------------------------------- 3) Stage 2 검출
def eval_detection(det, tol=(-0.06, 0.03)):
    tp = fp = fn = fp_right = 0
    for _, r in det.iterrows():
        used = set()
        for kt in r.left:
            m = [i for i, d in enumerate(r.det) if i not in used and tol[0] <= d - kt <= tol[1]]
            if m:
                used.add(m[0])
                tp += 1
            else:
                fn += 1
        for i, d in enumerate(r.det):
            if i not in used:
                fp += 1
                if any(tol[0] <= d - k <= tol[1] for k in r.right):
                    fp_right += 1
    p, rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
    print(f"\n[3] Stage 2 IMU만으로 왼손 타격 검출 — 정밀도 {p:.1%} · 재현율 {rc:.1%} · F1 {2 * p * rc / max(1e-9, p + rc):.1%}"
          f" (오검출 {fp}개 중 오른손 모음 타건 시점 {fp_right}개)")
    exact = np.mean([len(r.det) == len(r.left) for _, r in det.iterrows()])
    print(f"    단어별 타격 개수 정확히 맞춤: {exact:.0%}")
    return p, rc, exact


# --------------------------------------------------------------------------- 4) 단어 복원
def eval_words(wp, iso, P):
    print("\n[4] 단어 복원 — 단어 CV(5). 점수 = 기하 Σlogψ + β·Σlog P_cls + λ·log 빈도")
    print("    (β/λ 격자는 같은 100단어에서 고른 값이라 약간 낙관적. 격자 전체를 함께 표시)")
    logP = np.log(np.clip(P, 1e-4, 1))
    words = {t_: wp.word[ROWS[t_][0]] for t_ in OBS}
    geos = {}
    for fold in sorted(set(FOLD.values())):
        trw = [t_ for t_ in OBS if FOLD[t_] != fold]
        geos[fold] = Geometry.from_isolated(iso).fit([OBS[t_] for t_ in trw], [CSTR[t_] for t_ in trw])
    for top_n in (5000, 10000, 30000):
        vocab = filter_vocab(load_vocab(None, top_n), min_keys=1)
        lex = Lexicon(vocab)
        test = [t_ for t_ in OBS if words[t_] in lex.freq]
        grid = {}
        for beta in (0.0, 0.5, 1.0):
            for lam in (0.0, 0.5, 1.0):
                hits = {1: 0, 3: 0, 5: 0}
                for t_ in test:
                    un = beta * logP[ROWS[t_]] if beta else None
                    ranked = [w for _, w in lex.rank_words(geos[FOLD[t_]], OBS[t_], un, lam)]
                    r = ranked.index(words[t_])
                    for k in hits:
                        hits[k] += r < k
                grid[(beta, lam)] = {k: v / len(test) for k, v in hits.items()}
        # 채택 설정: 기하 가우스 + aLeak 내적 κ=20 + Stage 4 타이밍 τ=1 (β=0, λ=0.5)
        tms = {f: TimingModel(True).fit(geos[f], wp[wp.trial_id.isin([t_ for t_ in OBS if FOLD[t_] != f]).to_numpy()])
               for f in geos}
        ikt = {t_: wp.ikt.to_numpy(float)[ROWS[t_]] for t_ in OBS}
        hits = {1: 0, 3: 0, 5: 0}
        for t_ in test:
            ranked = [w for _, w in lex.rank_words(geos[FOLD[t_]], OBS[t_], None, 0.5, "both", 20.0,
                                                   tms[FOLD[t_]], ikt[t_], 1.0)]
            r = ranked.index(words[t_])
            for k in hits:
                hits[k] += r < k
        adopted = {k: v / len(test) for k, v in hits.items()}
        ceil = {1: 0, 3: 0, 5: 0}
        for t_ in test:
            same = sorted(lex.by_len[len(CSTR[t_])][CSTR[t_]], key=lambda wf: -wf[1])
            r = [w for w, _ in same].index(words[t_])
            for k in ceil:
                ceil[k] += r < k
        ceil = {k: v / len(test) for k, v in ceil.items()}
        g0 = grid[(0.0, 0.0)]
        bk = max(grid, key=lambda k: (grid[k][1], grid[k][5]))
        print(f"  어휘 {len(vocab):6d}단어 (시험 {len(test)}단어):  기하만 {fmt_top(g0)}  |  최고 β={bk[0]} λ={bk[1]}: "
              f"{fmt_top(grid[bk])}  |  자음 100% 상한 {fmt_top(ceil)}")
        print(f"     채택(기하+내적 κ=20+타이밍 τ=1): {fmt_top(adopted)}")
        print("     격자 Top-1: " + " ".join(f"(β{b},λ{l_}) {v[1]:.0%}" for (b, l_), v in grid.items()))


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    isos, wps, augs, dets = [], [], [], {"median": [], "fft": [], "fft_session": []}
    for si, s in enumerate(sys.argv[1:]):
        sd = load_session(s)
        # 여러 세션이면 trial_id가 겹치므로 세션 번호로 구분
        off = si * 100000
        isos.append(isolated_features(sd, windows=True).assign(trial_id=lambda d: d.trial_id + off))
        augs.append(pd.concat([isolated_features(sd, aug=Augment(seed=k)) for k in (1, 2, 3)],
                              ignore_index=True).assign(trial_id=lambda d: d.trial_id + off))
        wps.append(word_presses(sd).assign(trial_id=lambda d: d.trial_id + off))
        for pace in dets:
            dets[pace].append(detect_presses(sd, pace=pace))
    iso = pd.concat(isos, ignore_index=True)
    wp = pd.concat(wps, ignore_index=True)
    aug = pd.concat(augs, ignore_index=True)
    iso = iso[iso.ok_dur].reset_index(drop=True)
    aug = aug[aug.ok_dur & aug.trial_id.isin(iso.trial_id)]
    setup_folds(wp)
    eval_isolated(iso, aug)
    P, _ = eval_continuous(iso, wp)
    for pace, d in dets.items():
        print(f"\n  (리듬 추정 = {pace})", end="")
        eval_detection(pd.concat(d, ignore_index=True))
    eval_words(wp, iso, P)


if __name__ == "__main__":
    main()
