#!/usr/bin/env python3
"""실제 타이핑 점검: 학습 세션으로 모델을 만들고, 새로 친 단어를 전체 파이프라인에 통과시켜 정확도를 잰다.

  # 1) 시험용 단어를 새로 친다(학습과 다른 단어가 나오도록 seed를 바꾸거나 착용 번호를 올림)
  python scripts/collect.py --subject S01 --wear 2 --phases rest,calib,continuous --words 50
  # 2) 점검
  python scripts/pipeline_test.py --train data/raw/S01/20261103_171029_w1_* --test data/raw/S01/<새 세션>
  # 시험 세션이 아직 없으면: 학습 세션 안에서 단어 단위 5-fold로 점검(같은 착용 → 낙관적)
  python scripts/pipeline_test.py --train data/raw/S01/20261103_171029_w1_*

두 가지 분할로 같은 단어를 끝까지 돌린다.
  [오라클] 타격 시각 = 키 로그(키다운).        → Stage 1·디코더·사전만의 성능
  [IMU만]  타격 시각 = Stage 2 검출(detect_presses). → 실제 사용 상황의 성능(키 로그를 전혀 쓰지 않음)
  단어 경계(시작 F 정지 ~ SPACE)는 두 경우 모두 시도 구간을 쓴다.

출력: 자음 Top-1/3/5(오라클), 자음 오류율 CER(편집거리), 타격 수 일치율, 단어 Top-1/3/5(어휘 크기별),
      단어별 결과 CSV(--csv).

파이프라인 선택(기본값 = S01 ablation(scripts/ablation.py)에서 고른 '인식 알고리즘')
  Stage 1   --kalman-rho(칼만 스무딩, 기본 끔) · --mag-db(자력계 조건부 결합 ΔB 임계, 기본 끔 = 6축)
  Stage 2   --pace median|fft|fft_session (aLeak Algorithm 1 리듬 추정)
  디코더     --geometry isolated|layout · --score gauss|inner|both, --kappa (aLeak 내적) · --tau (Stage 4 타이밍, 단어)
            --timing-w (타이밍, 자음 사슬) · --row-eta (QwertyRing 행 모델) · --beta (분류기) · --gamma (자음 LM) · --lam (빈도)
  착용 보정  --calib-words N (시험 세션 앞 N단어를 정답과 함께 써서 A·b 재적합, 그 단어는 평가 제외)
            --adapt (정답 없이 시험 단어로 EM 재보정)
  사전      --domain FILE (맥락 사전: 분야 단어 목록으로 범위를 좁힌 어휘를 따로 보고)
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
                           domain_vocab, timing_pairs)
from ringko.features import detect_presses, isolated_features, word_presses  # noqa: E402
from ringko.hangul import KEY_ROW, to_consonant_keys  # noqa: E402
from ringko.motion import MotionParams  # noqa: E402
from ringko.plan import filter_vocab, load_vocab  # noqa: E402

warnings.filterwarnings("ignore")
from sklearn.ensemble import ExtraTreesClassifier  # noqa: E402

KI = {k: i for i, k in enumerate(KEYS)}
KEY2JAMO = dict(zip("qwertasdfgzxcv", "ㅂㅈㄷㄱㅅㅁㄴㅇㄹㅎㅋㅌㅊㅍ"))


# --------------------------------------------------------------------------- 데이터
def load_all(dirs, p=None, pace="median", base=0):
    """세션들 → 개별 키 특징, 오라클 단어 타격, IMU 검출 단어 타격. 시도 키 = (세션번호, trial_id)."""
    isos, ors, ims, dets = [], [], [], []
    for si, d in enumerate(dirs, base):
        sd = load_session(d)
        iso = isolated_features(sd, p) if (sd.trials.phase == "isolated").any() else pd.DataFrame()
        if len(iso):
            isos.append(iso[iso.ok_dur].assign(sess=si))
        wo = word_presses(sd, p)
        det = detect_presses(sd, pace=pace)
        times = dict(zip(det.trial_id, det.det))
        wi = word_presses(sd, p, times=times)
        for df, out in ((wo, ors), (wi, ims)):
            if len(df):
                out.append(df.assign(sess=si, tkey=[(si, t_) for t_ in df.trial_id]))
        dets.append(det.assign(sess=si))
        print(f"  {Path(d).name}: 개별 키 {len(iso)} · 단어 {sd.trials[(sd.trials.phase == 'continuous') & (sd.trials.status == 'ok')].shape[0]}"
              f" · 오라클 타격 {len(wo)} · IMU 검출 타격 {len(wi)}")
    cat = lambda L: pd.concat(L, ignore_index=True) if L else pd.DataFrame()  # noqa: E731
    return cat(isos), cat(ors), cat(ims), cat(dets)


def per_word(wp):
    """{tkey: (word, obs(n,2), rows, ikt(n,))}"""
    out = {}
    for tk, g in wp.groupby("tkey", sort=False):
        out[tk] = (g.word.iloc[0], g[["du", "dw"]].to_numpy(float), g.index.to_numpy(), g.ikt.to_numpy(float))
    return out


# --------------------------------------------------------------------------- 모델
CLS_FEATS = ["u", "w", "du", "dw"]


def raw_cols(df):
    return [c for c in df.columns if c.startswith("raw")]


def fit_models(iso, wp_tr, use_cls, a=None):
    """학습 단어로 기하(A,b), 분류기, 행 모델, 타이밍 모델. 반환 dict."""
    words = per_word(wp_tr)
    layout = iso.empty or (a is not None and a.geometry == "layout")
    geo = Geometry.from_layout() if layout else Geometry.from_isolated(iso)
    geo.fit([o for _, o, _, _ in words.values()], ["".join(wp_tr.key[r]) for _, _, r, _ in words.values()])
    m = {"geo": geo, "clf": None, "row": None, "tm_word": None, "tm_chain": None}
    if a is not None and a.row_eta > 0:
        m["row"] = RowModel(("dtilt_a", "dtilt_b")).fit(wp_tr)
    if a is not None and (a.tau > 0 or a.timing_w > 0):
        m["tm_word"] = TimingModel(True).fit(geo, wp_tr)
        m["tm_chain"] = TimingModel(False).fit(geo, wp_tr)
    if use_cls:
        feats = CLS_FEATS + raw_cols(wp_tr)
        X = wp_tr[feats].to_numpy(float)
        y = wp_tr.key.map(KI).to_numpy()
        if not iso.empty:
            X = np.vstack([iso.assign(du=iso.u, dw=iso.w)[feats].to_numpy(float), X])
            y = np.concatenate([iso.key.map(KI), y])
        m["clf"] = ExtraTreesClassifier(n_estimators=600, random_state=0, n_jobs=-1).fit(X, y)
    return m


def unary(clf, df, rows, beta):
    if clf is None or beta == 0:
        return None
    P = np.zeros((len(rows), 14))
    P[:, clf.classes_] = clf.predict_proba(df.loc[rows, CLS_FEATS + raw_cols(df)].to_numpy(float))
    return beta * np.log(np.clip(P, 1e-4, 1))


# --------------------------------------------------------------------------- 지표
def edit_distance(a, b):
    d = np.arange(len(b) + 1)
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return int(d[-1])


def jamo(s):
    return "".join(KEY2JAMO.get(c, "?") for c in s)


def decode_word(geo, obs, un, lexes, a, lm=None, ikt=None, models=None, un_row=None, tail=None):
    """자음 주변확률(자음 언어모델·타이밍·행 모델 포함) → 최선 자음열,
    어휘별 단어 순위 목록(사전이 있으므로 언어모델 없이; 기하 + aLeak 내적 + Stage 4 타이밍(+ F 복귀 꼬리) + 빈도)."""
    models = models or {}
    pe = timing_pairs(models.get("tm_chain"), geo, ikt, a.timing_w)
    u = un
    if un_row is not None:
        u = un_row if u is None else u + un_row
    P = chain_marginals(geo, obs, u, lm, a.gamma, pe)
    best = "".join(KEYS[i] for i in P.argmax(1))
    ranked = {name: [w for _, w in lex.rank_words(geo, obs, un, a.lam, a.score, a.kappa, models.get("tm_word"), ikt, a.tau,
                                                 tail if a.tail else None)]
              for name, lex in lexes.items()}
    return P, best, ranked


def summarize(rows, lex_names, title):
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"\n{title}: 결과 없음")
        return df
    print(f"\n{title} — 단어 {len(df)}개")
    n_true = df.n_true.sum()
    print(f"  타격 수: 정답 {n_true} · 검출 {df.n_det.sum()} · 단어별 개수 일치 {np.mean(df.n_true == df.n_det):.0%}")
    print(f"  자음열 오류율 CER(편집거리/정답 길이) {df.edits.sum() / n_true:.1%} · 자음열 완전 일치 {np.mean(df.edits == 0):.0%}")
    if "c_top1" in df:
        m = df.n_true == df.n_det
        tot = df[m].n_true.sum()
        if tot:
            print(f"  자음(위치별, 개수 일치 단어) Top-1 {df[m].c_top1.sum() / tot:.1%}  Top-3 {df[m].c_top3.sum() / tot:.1%}"
                  f"  Top-5 {df[m].c_top5.sum() / tot:.1%} · 행 {df[m].c_row.sum() / tot:.1%}")
    for name in lex_names:
        r = df[f"rank_{name}"]
        inv = r == -2
        ok = ~inv
        hits = {k: np.mean((r[ok] >= 0) & (r[ok] < k)) for k in (1, 3, 5)}
        print(f"  단어 [{name}] (사전 밖 {inv.sum()}개 제외 {ok.sum()}개) Top-1 {hits[1]:.1%}  Top-3 {hits[3]:.1%}  Top-5 {hits[5]:.1%}")
    return df


# --------------------------------------------------------------------------- 실행
def wear_calibrate(models, test_wp, a, lexes, skip):
    """착용별 보정: --calib-words(정답 있는 보정 단어) 또는 --adapt(정답 없는 EM)로 기하 A·b를 시험 착용에 다시 맞춘다."""
    geo = models["geo"].copy()
    words = per_word(test_wp)
    if a.calib_words:
        cal = [tk for tk in words if tk in skip]
        if cal:
            geo.fit([words[tk][1] for tk in cal], [to_consonant_keys(words[tk][0]) for tk in cal], ridge=30.0)
    if a.adapt:
        obs = [o for tk, (_, o, _, _) in words.items() if tk not in skip]
        geo.adapt(obs, list(lexes.values())[-1], iters=6, ridge=30.0)
    return dict(models, geo=geo)


def run(iso, test_wp, train_wp_for, lexes, a, oracle, lm=None, skip=frozenset(), calib=False):
    rows = []
    words = per_word(test_wp)
    cache = {}
    for tk, (word, obs, idx, ikt) in words.items():
        if tk in skip:
            continue
        fid, wp_tr = train_wp_for(tk)
        if fid not in cache:
            cache[fid] = fit_models(iso, wp_tr, a.beta > 0, a)
            if calib:
                cache[fid] = wear_calibrate(cache[fid], test_wp, a, lexes, skip)
        models = cache[fid]
        geo, clf = models["geo"], models["clf"]
        truth = to_consonant_keys(word)
        un = unary(clf, test_wp, idx, a.beta)
        un_row = a.row_eta * models["row"].unary(test_wp.loc[idx]) if models["row"] is not None else None
        tail = float(test_wp["tail"][idx[-1]]) if "tail" in test_wp.columns else None
        P, best, ranked = decode_word(geo, obs, un, lexes, a, lm, ikt, models, un_row, tail)
        r = {"session": tk[0], "trial_id": tk[1], "word": word, "truth": jamo(truth), "pred": jamo(best),
             "n_true": len(truth), "n_det": len(obs), "edits": edit_distance(best, truth)}
        if oracle and len(obs) == len(truth):
            order = np.argsort(-P, 1)
            y = [KI[c] for c in truth]
            for k in (1, 3, 5):
                r[f"c_top{k}"] = sum(y[i] in order[i, :k] for i in range(len(y)))
            r["c_row"] = sum(KEY_ROW[best[i]] == KEY_ROW[truth[i]] for i in range(len(y)))
        for name, lex in lexes.items():
            if word not in lex.freq:
                r[f"rank_{name}"] = -2
            else:
                r[f"rank_{name}"] = ranked[name].index(word) if word in ranked[name] else -1
            r[f"top5_{name}"] = " ".join(ranked[name][:5])
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", required=True, help="학습 세션(개별 키 + 연속 타이핑 포함)")
    ap.add_argument("--test", nargs="*", default=[], help="시험 세션. 생략하면 학습 세션 안에서 단어 5-fold")
    ap.add_argument("--vocab", help="단어[ 빈도] 파일(없으면 wordfreq)")
    ap.add_argument("--vocab-sizes", default="5000,30000", help="wordfreq 상위 N (쉼표)")
    ap.add_argument("--beta", type=float, default=0.0, help="분류기 가중(0=기하만; S01에서 단어는 0, 자음은 1.0이 최고)")
    ap.add_argument("--lam", type=float, default=0.5, help="단어 빈도 가중")
    ap.add_argument("--gamma", type=float, default=0.5, help="자음 트라이그램 언어모델 가중(0=끔). 학습·시험 단어는 언어모델에서 제외")
    ap.add_argument("--score", default="both", choices=("gauss", "inner", "both"),
                    help="단어 기하 점수: 가우스 | aLeak 내적(코사인) | 둘 다 (S01 CV: both κ=20)")
    ap.add_argument("--kappa", type=float, default=20.0, help="aLeak 내적 점수 가중")
    ap.add_argument("--tau", type=float, default=1.0, help="Stage 4 타이밍(타격 간격) 단어 점수 가중(0=끔)")
    ap.add_argument("--tail", action="store_true",
                    help="단어 끝 F 복귀 정지 시간(꼬리) 항 켜기(τ로 가중). S01에서 효과 중립(±1단어)이라 기본 끔")
    ap.add_argument("--timing-w", type=float, default=0.0, help="타이밍 쌍 퍼텐셜 가중(자음 사슬; S01에서 효과 없음)")
    ap.add_argument("--row-eta", type=float, default=0.0, help="QwertyRing 행 모델 unary 가중(자음 사슬; S01에서 효과 없음)")
    ap.add_argument("--geometry", default="isolated", choices=("isolated", "layout"),
                    help="키 위치 M: 개별 키 중앙값 | 두벌식 물리 배열만(개별 키 데이터 불필요)")
    ap.add_argument("--kalman-rho", type=float, default=0.0, help="Stage 1 칼만 스무딩 q/r(0=끔; S01에서 ≤1e-5 무변화, 1e-3 악화)")
    ap.add_argument("--mag-db", type=float, default=0.0, help="자력계 조건부 요 보정 ΔB 임계[uT](0=6축만; S01에서 효과 없음)")
    ap.add_argument("--pace", default="median", choices=("median", "fft", "fft_session"), help="Stage 2 리듬 추정")
    ap.add_argument("--calib-words", type=int, default=0, help="시험 세션 앞 N단어로 착용 보정(정답 사용, 평가 제외)")
    ap.add_argument("--adapt", action="store_true", help="정답 없이 시험 단어로 착용 보정(EM)")
    ap.add_argument("--domain", help="맥락 사전 단어 목록 파일(한 줄에 한 단어)")
    ap.add_argument("--csv", help="단어별 결과 저장 경로")
    a = ap.parse_args()
    mp = MotionParams(kalman_rho=a.kalman_rho, mag_db_max=a.mag_db)

    train_dirs = [str(Path(d).resolve()) for d in a.train]
    test_dirs = [str(Path(d).resolve()) for d in a.test]
    overlap = set(train_dirs) & set(test_dirs)
    if overlap:
        sys.exit(f"학습과 시험에 같은 세션이 있습니다: {overlap}. 같은 세션 점검은 --test를 빼면 단어 5-fold로 합니다.")

    print("학습 세션 로드")
    iso, tr_or, tr_im, tr_det = load_all(train_dirs, mp, a.pace)
    if tr_or.empty or (iso.empty and a.geometry == "isolated"):
        sys.exit("학습 세션에 연속 타이핑(그리고 --geometry isolated면 개별 키)이 필요합니다.")

    lexes = {}
    for n in (int(s) for s in a.vocab_sizes.split(",")):
        v = filter_vocab(load_vocab(a.vocab, n), min_keys=1)
        lexes[f"{len(v)}단어"] = Lexicon(v)
        if a.vocab:
            break
    if a.domain:
        dw = [ln.split()[0] for ln in Path(a.domain).read_text(encoding="utf-8").splitlines() if ln.strip()]
        dv = filter_vocab(domain_vocab(load_vocab(a.vocab, 30000), dw), min_keys=1)
        lexes = {f"맥락 {len(dv)}단어": Lexicon(dv), **lexes}

    if test_dirs:
        print("시험 세션 로드")
        _, te_or, te_im, _ = load_all(test_dirs, mp, a.pace, base=len(train_dirs))
        mode = "시험 세션(학습과 분리)"
        whole = ("all", tr_or)
        train_for = lambda tk: whole  # noqa: E731
    else:
        te_or, te_im = tr_or, tr_im
        mode = "학습 세션 단어 5-fold(같은 착용 → 낙관적)"
        tkeys = list(dict.fromkeys(tr_or.tkey))
        fold_of = {tk: i % 5 for i, tk in enumerate(tkeys)}
        folds = {f: (f, tr_or[[fold_of[tk] != f for tk in tr_or.tkey]].reset_index(drop=True)) for f in range(5)}
        train_for = lambda tk: folds[fold_of.get(tk, 0)]  # noqa: E731

    lm = None
    if a.gamma > 0:
        # 평가 단어(학습·시험 모두)를 뺀 어휘로 자음 언어모델 → 누출 없음
        lm = ConsonantLM(filter_vocab(load_vocab(a.vocab, 30000), min_keys=1),
                         exclude=set(tr_or.word) | set(te_or.word) | set(te_im.word))

    skip = set()
    if a.calib_words and test_dirs:
        skip = set(list(dict.fromkeys(te_or.tkey))[:a.calib_words])
    calib = bool(test_dirs) and (a.calib_words > 0 or a.adapt)
    cal_txt = ""
    if calib:
        cal_txt = " · 착용 보정" + (f" 보정 단어 {a.calib_words}" if a.calib_words else "") + (" EM" if a.adapt else "")
    print(f"\n=== 파이프라인 점검: {mode} · β={a.beta} λ={a.lam} γ(자음 LM)={a.gamma} · 점수 {a.score} κ={a.kappa}"
          f" · τ(타이밍)={a.tau} · 기하 {a.geometry}{cal_txt} ===")
    rows_or = run(iso, te_or, train_for, lexes, a, oracle=True, lm=lm, skip=skip, calib=calib)
    df_or = summarize(rows_or, lexes, "[오라클 분할] 타격 시각 = 키 로그")
    rows_im = run(iso, te_im, train_for, lexes, a, oracle=False, lm=lm, skip=skip, calib=calib)
    # IMU 쪽에서 검출이 0개라 빠진 단어는 실패로 추가
    seen = {(r["session"], r["trial_id"]) for r in rows_im}
    for r in rows_or:
        if (r["session"], r["trial_id"]) not in seen:
            miss = {k: r[k] for k in ("session", "trial_id", "word", "truth", "n_true")}
            miss.update(pred="", n_det=0, edits=r["n_true"],
                        **{f"rank_{n}": (-2 if r[f"rank_{n}"] == -2 else -1) for n in lexes},
                        **{f"top5_{n}": "" for n in lexes})
            rows_im.append(miss)
    df_im = summarize(rows_im, lexes, "[IMU만] 타격 시각 = Stage 2 검출 (실사용)")

    if len(df_im):
        m = df_im.n_true == df_im.n_det
        print(f"\n  참고: IMU만 — 타격 수가 맞은 단어 {m.sum()}개에서의 단어 Top-1: " + " · ".join(
            f"[{n}] {np.mean(df_im[m][f'rank_{n}'] == 0):.0%}" for n in lexes))
        print("\n  예시(IMU만, 앞 10단어): 정답 → 예측자음열 | 상위 후보")
        name = list(lexes)[-1]
        for _, r in df_im.head(10).iterrows():
            print(f"    {r.word:6s} {r.truth:8s} → {r.pred:10s} | {r[f'top5_{name}']}")

    if a.csv:
        out = pd.concat([df_or.assign(split="oracle"), df_im.assign(split="imu")], ignore_index=True)
        out.to_csv(a.csv, index=False, encoding="utf-8-sig")
        print(f"\n단어별 결과 저장: {a.csv}")


if __name__ == "__main__":
    main()
