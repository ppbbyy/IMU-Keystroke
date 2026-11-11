"""검증: python tests/test_ringko.py  (pytest로도 실행 가능)"""
import json
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ringko.clock import dev_to_host_ns, fit_clock  # noqa: E402
from ringko.dataset import gaps, impact_lags_ms, load_session, quality_report, trial_windows  # noqa: E402
from ringko.device import Device, Unwrapper  # noqa: E402
from ringko.hangul import (CONSONANTS, KEY_XY, is_typable, to_consonant_keys, to_consonants,  # noqa: E402
                           to_keys)
from ringko.plan import (build_plan, consonant_counts, counts, filter_vocab, load_plan, save_plan,  # noqa: E402
                         select_words)
from ringko.session import Params, Session  # noqa: E402


# --------------------------------------------------------------------------- 한글
def test_hangul():
    assert to_keys("한국") == "gksrnr" and to_consonant_keys("한국") == "gsrr" and to_consonants("한국") == "ㅎㄴㄱㄱ"
    assert to_keys("값") == "rkqt" and to_consonant_keys("값") == "rqt"
    assert to_keys("의사") == "dmltk" and to_keys("과일") == "rhkdlf"
    for w in ("꽃", "있다", "얘기", "abc", "", "한1"):
        assert not is_typable(w), w
    assert KEY_XY["f"] == (3.25, 1.0) and KEY_XY["z"] == (0.75, 2.0)


# --------------------------------------------------------------------------- 플랜
def test_plan():
    rng = np.random.default_rng(0)
    vocab = filter_vocab([(w, 1 / (i + 1)) for i, w in enumerate(
        "한국 사람 학교 하늘 바다 친구 시간 마음 생각 음악 도서관 나무 사랑 가족 여행 컴퓨터 사진 노래 택시 포도".split())])
    words = select_words(vocab, 8, rng, min_per_consonant=1)
    plan = build_plan(rng, reps=40, words=words)
    c = counts(plan)
    assert c == {"axes": 1, "rest": 2, "calib": 10, "isolated": 560, "continuous": 8}, c
    iso = [it for it in plan if it.phase == "isolated" and it.kind == "key"]
    assert Counter(it.target for it in iso) == {k: 40 for k in CONSONANTS}
    assert all(x.target != y.target for x, y in zip(iso, iso[1:])), "같은 자음 연속"
    assert len({it.uid for it in plan}) == len(plan)
    with tempfile.TemporaryDirectory() as d:
        save_plan(plan, Path(d) / "p.csv")
        assert load_plan(Path(d) / "p.csv") == plan


def test_wordfreq_words():
    from ringko.plan import load_vocab
    v = filter_vocab(load_vocab(None, 20000))
    w = select_words(v, 100, np.random.default_rng(1), min_per_consonant=10)
    assert len(set(w)) == 100 and min(consonant_counts(w).values()) >= 10, consonant_counts(w)


# --------------------------------------------------------------------------- 시계
def test_clock_fit():
    rng = np.random.default_rng(0)
    t_us = np.sort(rng.uniform(0, 600e6, 600)) + 5e9
    true = lambda t: 7e12 + 1000.05 * (t - 5e9)  # noqa: E731  (50ppm 드리프트)
    lat_out = rng.exponential(0.6e6, 600) + 0.2e6   # 호스트->장치
    lat_in = rng.exponential(0.6e6, 600) + 0.2e6    # 장치->호스트
    lat_in[::7] += 8e6                               # 가끔 큰 지연
    send = true(t_us) - lat_out
    recv = true(t_us) + lat_in
    m = fit_clock(send, recv, t_us)
    err = dev_to_host_ns(t_us, m) - true(t_us)
    assert np.abs(err).max() < 1.0e6, np.abs(err).max()
    assert abs(m["drift_ppm"] - 50) < 5, m["drift_ppm"]


def test_clock_wander():
    """내부 RC 발진기처럼 -1.6% 드리프트 + 수십 ms 흔들림이 있어도 구간 보정으로 1ms 이내."""
    rng = np.random.default_rng(1)
    t_us = np.arange(0, 540e6, 1e6) + 1.7e8
    wander = 25e6 * np.sin(2 * np.pi * (t_us - t_us[0]) / 300e6) + 8e6 * np.sin(2 * np.pi * (t_us - t_us[0]) / 97e6)
    true = 1.5e14 + 983.9 * (t_us - t_us[0]) + wander
    send = true - (rng.exponential(0.3e6, len(t_us)) + 0.2e6)
    recv = true + (rng.exponential(0.3e6, len(t_us)) + 0.2e6)
    m = fit_clock(send, recv, t_us)
    tt = np.linspace(t_us[0], t_us[-1], 5000)
    w = 25e6 * np.sin(2 * np.pi * (tt - t_us[0]) / 300e6) + 8e6 * np.sin(2 * np.pi * (tt - t_us[0]) / 97e6)
    err = dev_to_host_ns(tt, m) - (1.5e14 + 983.9 * (tt - t_us[0]) + w)
    assert m["linear_resid_ms_max"] > 20
    assert np.abs(err).max() < 1.0e6, np.abs(err).max() / 1e6


def test_winkeys_hook():
    """Windows: 저수준 훅이 주입 키(F down/up)를 VK/스캔코드로 받는지. 주입 키는 다른 창으로 새지 않게 차단."""
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes
    from ringko.winkeys import KeyHook
    h = KeyHook(swallow_injected=True)
    h.start()

    class KI(ctypes.Structure):
        _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]

    class INPUT(ctypes.Structure):
        class _U(ctypes.Union):
            _fields_ = [("ki", KI), ("pad", ctypes.c_byte * 32)]
        _anonymous_ = ("u",)
        _fields_ = [("type", wintypes.DWORD), ("u", _U)]

    def send(vk, up):
        inp = INPUT(type=1)
        inp.ki = KI(vk, 0x21 if vk == 0x46 else 0, 0x2 if up else 0, 0, 0)
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

    send(0x46, False)
    send(0x46, True)
    time.sleep(0.3)
    ev = h.drain()
    h.stop()
    got = [(d, k) for _, d, k, _, _, inj in ev if inj]
    assert got == [(True, "f"), (False, "f")], ev


def _synthetic_move(D=(0.040, -0.020), T=0.5, fs=233.0, tilt_deg=3.0, yaw_deg=8.0, lift=0.02):
    """세계 좌표 궤적(최소저크 수평 이동 + 들어올림 호 + 회전) -> body 비력/각속도."""
    from ringko.motion import rotvec_to_mat
    t = np.arange(0, 0.3 + T, 1 / fs)
    s = np.clip((t - 0.3) / T, 0, 1)
    mj = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    pos = np.stack([D[0] * mj, D[1] * mj, lift * np.sin(np.pi * s)], 1)
    dt = 1 / fs
    vel = np.gradient(pos, dt, axis=0)
    accw = np.gradient(vel, dt, axis=0)
    base = rotvec_to_mat(np.radians([20.0, -70.0, 10.0]))  # 착용 자세(임의)
    Rs = [base @ rotvec_to_mat(np.radians([tilt_deg * m, 0.5 * tilt_deg * m, yaw_deg * m])) for m in mj]  # R_wb
    f_b = np.array([R.T @ (a + [0, 0, 9.80665]) for R, a in zip(Rs, accw)])
    w_b = []
    for i in range(len(t)):
        j = min(i, len(t) - 2)
        dR = Rs[j].T @ Rs[j + 1]
        th = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
        vee = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
        w_b.append(vee * (th / (2 * np.sin(th)) if th > 1e-9 else 0.5) / dt)
    return t, f_b, np.array(w_b), Rs[0], np.array([D[0], D[1], 0.0])


def test_motion_integrate():
    from ringko.motion import integrate
    t, f, w, R0, D = _synthetic_move()
    i0 = int(0.25 * 233)
    g0 = f[:i0].mean(0)
    res = integrate(t[i0:], f[i0:], w[i0:], g0, np.zeros(3), "horizontal")
    d_true = R0.T @ D                       # 기준(body t0) 좌표의 실제 변위
    gz = g0 / np.linalg.norm(g0)
    h = lambda v: v - (v @ gz) * gz         # noqa: E731
    err = np.linalg.norm(h(res["disp"]) - h(d_true)) * 1000
    assert err < 1.5, f"수평 변위 오차 {err:.2f}mm"
    # 자이로 부호를 틀리면(원시 그대로) 크게 틀려야 한다 -> 부호 보정이 실제로 중요함을 확인
    bad = integrate(t[i0:], f[i0:], -w[i0:], g0, np.zeros(3), "horizontal")
    assert np.linalg.norm(h(bad["disp"]) - h(d_true)) * 1000 > 5 * err


def test_decoder():
    from ringko.decode import KEYS, Geometry, Lexicon, chain_marginals
    pos = np.array([[(KEY_XY[k][0] - KEY_XY["f"][0]) * 19, (KEY_XY[k][1] - KEY_XY["f"][1]) * 19] for k in KEYS])
    geo = Geometry(pos)
    geo.sigma = 4.0
    rng = np.random.default_rng(0)
    lex = Lexicon([(w, 1.0) for w in ["한국", "학교", "사람", "바다", "하늘", "시간", "친구", "택시"]])
    for w in ["한국", "택시", "바다"]:
        obs = geo.expected(to_consonant_keys(w)) + rng.normal(0, 2.0, (len(to_consonant_keys(w)), 2))
        assert lex.rank_words(geo, obs)[0][1] == w
        P = chain_marginals(geo, obs)
        assert "".join(KEYS[i] for i in P.argmax(1)) == to_consonant_keys(w)
    # fit은 스케일/기울기를 복원
    obs_l = [geo.expected(s) @ np.array([[0.8, 0.1], [0.0, 0.7]]) for s in ["gsrr", "xt", "qe", "gfs"]]
    g2 = Geometry(pos).fit(obs_l, ["gsrr", "xt", "qe", "gfs"])
    assert np.allclose(g2.A, [[0.8, 0.1], [0.0, 0.7]], atol=1e-6) and g2.sigma < 1e-6


def test_consonant_lm():
    from ringko.decode import ConsonantLM, Geometry, chain_marginals
    M = np.array([[(KEY_XY[k][0] - KEY_XY["f"][0]) * 19, (KEY_XY[k][1] - KEY_XY["f"][1]) * 19] for k in "qwertasdfgzxcv"])
    geo = Geometry(M)
    lm = ConsonantLM([("가나", 1), ("가다", 1), ("나라", 1)])
    assert np.allclose(np.exp(lm.logp).sum(2), 1)
    obs = np.diff(np.vstack([np.zeros(2), M[[3, 6]]]), axis=0) + 3.0
    P0 = chain_marginals(geo, obs)
    P1 = chain_marginals(geo, obs, None, lm, 1e-9)  # γ≈0 → 언어모델 없는 경우와 같아야
    assert np.allclose(P0, P1, atol=1e-6)
    P2 = chain_marginals(geo, obs, None, lm, 0.5)
    assert np.allclose(P2.sum(1), 1) and P2[0].argmax() == 3


def test_kalman_trend():
    from ringko.motion import kalman_trend
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 233)
    drift = np.stack([0.3 * t, -0.2 * t ** 2, 0 * t], 1)                     # 중력 추정 오차(저주파)
    motion = np.stack([np.sin(2 * np.pi * 4 * t), np.cos(2 * np.pi * 4 * t), 0 * t], 1)
    tr = kalman_trend(drift + motion + rng.normal(0, 0.05, drift.shape), 1e-4)
    assert np.abs(tr - drift)[20:-20].max() < 0.25, "추세만 남아야 함"
    assert np.allclose(kalman_trend(drift, 0.0), 0)                          # rho=0 → 끔
    flat = kalman_trend(drift + motion, 1e-12)                                # rho→0 → 거의 상수(구간 평균)
    assert np.ptp(flat[:, 0]) < 0.05


def test_hard_iron_and_yaw():
    from ringko.magnet import fit_hard_iron, yaw_correction
    from ringko.motion import rotvec_to_mat
    rng = np.random.default_rng(1)
    h = np.array([12.0, -30.0, 5.0])
    Bw = np.array([20.0, 0.0, -40.0])
    B = np.array([rotvec_to_mat(rng.normal(0, 1.5, 3)).T @ Bw + h for _ in range(300)])
    h2, r = fit_hard_iron(B + rng.normal(0, 0.3, B.shape))
    assert np.linalg.norm(h2 - h) < 0.5 and abs(r - np.linalg.norm(Bw)) < 0.5
    gz = np.array([0, 0, 1.0])
    Rs = [rotvec_to_mat(gz * np.radians(10) * s) for s in np.linspace(0, 1, 11)]
    Rc = yaw_correction(Rs, gz, 10.0, np.linspace(0, 1, 11))
    assert np.allclose(Rc[-1], np.eye(3), atol=1e-9)


def test_typing_pace_fft():
    from ringko.features import typing_pace_fft
    fs = 233.0
    t = np.arange(0, 2.5, 1 / fs)
    j = np.zeros_like(t)
    for c in np.arange(0.2, 2.4, 0.4):                                        # 0.4초 간격 펄스
        j[int(c * fs)] = 1.0
    assert abs(typing_pace_fft(t, j) - 0.4) < 0.03


def test_row_timing_inner():
    from ringko.decode import (KEYS, Geometry, RowModel, TimingModel, inner_score, kbest_sequences, rerank_inner,
                               right_counts)
    rng = np.random.default_rng(2)
    rows = {k: KEY_XY[k][1] for k in KEYS}
    keys = rng.choice(KEYS, 600)
    df = pd.DataFrame({"key": keys, "tilt_a": [rows[k] * 2.0 + rng.normal(0, 0.5) for k in keys],
                       "tilt_b": rng.normal(0, 1, 600)})
    rm = RowModel().fit(df.iloc[:400])
    pred = rm.row_logpost(df.iloc[400:]).argmax(1)
    assert np.mean(pred == df.iloc[400:].key.map(rows).to_numpy()) > 0.9
    assert right_counts("한국") == [0, 1, 0, 1] and right_counts("과일") == [0, 2, 1]
    M = np.array([[(KEY_XY[k][0] - KEY_XY["f"][0]) * 19, (KEY_XY[k][1] - KEY_XY["f"][1]) * 19] for k in KEYS])
    geo = Geometry(M)
    geo.sigma = 4.0
    obs = geo.expected("gsrr") * 0.85                                         # 크기만 작아진 동작
    assert abs(inner_score(geo, obs, "gsrr") - 1.0) < 1e-9 and inner_score(geo, obs, "qqqq") < 0.9
    geo.sigma = 20.0                                                          # 동작이 작아도 후보에 들도록 넓게
    cands = kbest_sequences(geo, obs, k=200)
    assert len(cands) == 200 and len(cands[0][1]) == 4 and "gsrr" in [c for _, c in cands]
    ranked, P = rerank_inner(geo, obs, cands, kappa=200.0, mix=0.0)
    assert ranked[0][1] == "gsrr" and np.allclose(P.sum(1), 1)
    # 타이밍: 거리·오른손 키 수에 선형인 간격을 복원
    recs = []
    for i, w in enumerate(["한국", "사람", "바다", "택시", "학교", "하늘"] * 5):
        cs, nv = to_consonant_keys(w), right_counts(w)
        d = np.linalg.norm(geo.expected(cs), axis=1)
        for k in range(len(cs)):
            recs.append({"trial_id": i, "key": cs[k], "idx": k, "nv": nv[k],
                         "ikt": 0.2 + 0.004 * d[k] + 0.1 * (k == 0) + 0.15 * nv[k] + rng.normal(0, 0.01)})
    tm = TimingModel(True).fit(geo, pd.DataFrame(recs))
    assert np.allclose(tm.coef, [0.2, 0.004, 0.1, 0.15], atol=0.02), tm.coef


def test_tail_return():
    """단어 끝 F 복귀 정지 시간(꼬리)으로 자음열이 같은 '눈'/'누나'를 가른다."""
    from ringko.decode import KEYS, Geometry, Lexicon, TimingModel, tail_right
    from ringko.features import tail_settle
    assert tail_right("눈") == 0 and tail_right("누나") == 1 and tail_right("과일") == 0
    assert to_consonant_keys("눈") == to_consonant_keys("누나")
    # tail_settle: 타격 뒤 0.4s 움직이다 멈춘 자이로 → 약 0.4s
    t = np.arange(0, 3, 1 / 238)
    gyr = np.zeros((len(t), 3))
    ip = 100
    gyr[ip:ip + int(0.4 * 238), 0] = np.radians(60)
    assert abs(tail_settle(t, gyr, ip) - 0.4) < 0.02
    # 꼬리 모델: 끝 자음 뒤 모음이 있으면 0.12s 늦게 멈추는 데이터로 학습
    rng = np.random.default_rng(5)
    M = np.array([[(KEY_XY[k][0] - KEY_XY["f"][0]) * 19, (KEY_XY[k][1] - KEY_XY["f"][1]) * 19] for k in KEYS])
    geo = Geometry(M)
    recs = []
    for i, w in enumerate(["눈", "누나", "사람", "바다", "한국", "하늘", "가게", "강"] * 6):
        cs = to_consonant_keys(w)
        tail = 0.55 + 0.12 * (tail_right(w) > 0) + rng.normal(0, 0.03)
        for k in range(len(cs)):
            recs.append({"trial_id": i, "key": cs[k], "idx": k, "nv": 0, "ikt": 0.3, "tail": tail,
                         "nvt": tail_right(w)})
    tm = TimingModel(True).fit(geo, pd.DataFrame(recs))
    assert tm.tail_coef is not None and abs(tm.tail_coef[2] - 0.12) < 0.03, tm.tail_coef
    lex = Lexicon([("눈", 100.0), ("누나", 100.0)])
    obs = geo.expected(to_consonant_keys("눈"))
    ikt = np.array([0.3, 0.3])
    top = lambda tail: lex.rank_words(geo, obs, timing=tm, ikt=ikt, tail=tail)[0][1]  # noqa: E731
    assert top(0.55) == "눈" and top(0.68) == "누나"
    # 꼬리 관측이 없으면 두 단어 점수가 같다
    sc = lex.rank_words(geo, obs, timing=tm, ikt=ikt)
    assert abs(sc[0][0] - sc[1][0]) < 1e-9


def test_wear_calibration():
    from ringko.decode import KEYS, Geometry, Lexicon
    words = ["한국", "학교", "사람", "바다", "하늘", "시간", "친구", "택시", "나라", "가족", "도시", "마음"]
    lex = Lexicon([(w, 1.0) for w in words])
    geo = Geometry.from_layout()
    geo.sigma = 4.0
    th = np.radians(25)
    T = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]]) * 0.8
    rng = np.random.default_rng(3)
    obs = [geo.expected(to_consonant_keys(w)) @ T + np.array([3.0, -2.0]) * (np.arange(len(to_consonant_keys(w)))[:, None] == 0)
           + rng.normal(0, 1.0, (len(to_consonant_keys(w)), 2)) for w in words]
    g2 = geo.copy().fit(obs, [to_consonant_keys(w) for w in words])          # 보정 단어(정답 있음)
    assert np.allclose(g2.A, T, atol=0.05) and np.allclose(g2.b, [3, -2], atol=1.5)
    g3 = geo.copy()
    g3.sigma = 15.0
    g3.adapt(obs, lex, iters=6, ridge=1.0)                                     # 정답 없이
    assert sum(lex.rank_words(g3, o)[0][1] == w for o, w in zip(obs, words)) >= 10
    assert len(KEYS) == 14


def test_preprocess():
    from ringko.decode import domain_vocab
    from ringko.preprocess import ChannelNormalizer, WindowClassifier, augment_windows
    rng = np.random.default_rng(4)
    W = rng.normal([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], (40, 48, 6))
    Z = ChannelNormalizer().fit(W).transform(W)
    assert np.allclose(Z.mean((0, 1)), 0, atol=1e-9) and np.allclose(Z.std((0, 1)), 1, atol=1e-6)
    y = np.repeat([0, 1], 20)
    W[y == 1, 20:30, 0] += 5
    Wa, ya = augment_windows(W, y, n_copies=2)
    assert len(Wa) == 120 and (ya[:40] == y).all()
    m = WindowClassifier("logreg", augment=True).fit(W, y)
    assert (m.predict_proba(W).argmax(1) == y).mean() > 0.9
    assert domain_vocab([("한국", 0.5), ("학교", 0.1)], ["학교", "대학", "학교"]) == [("학교", 0.1), ("대학", 0.1)]


def test_unwrap_and_gaps():
    u = Unwrapper()
    out = [u(t) for t in [(1 << 32) - 10, (1 << 32) - 2, 5, 100]]
    assert out == [(1 << 32) - 10, (1 << 32) - 2, (1 << 32) + 5, (1 << 32) + 100]
    t = np.arange(0, 1000) * 4202.0
    t = np.delete(t, [100, 101, 102, 500])
    assert gaps(t) == (2, 4)


# --------------------------------------------------------------------------- 상태기계(가상 시계)
class StubDev:
    def __init__(self):
        self.still = True
        self.peak = 2.0

    def is_still(self, w, thr):
        return self.still

    def recent_gyro_peak(self, w):
        return self.peak

    def silence_s(self):
        return 0.0


def test_session_state_machine():
    rng = np.random.default_rng(3)
    plan = build_plan(rng, reps=1, words=["한국", "바다"], calib_n=2)
    dev = StubDev()
    with tempfile.TemporaryDirectory() as d:
        s = Session(plan, Params(), d, device=dev)
        t = [1_000_000_000]

        def adv(sec):
            t[0] += int(sec * 1e9)
            s.tick(t[0])

        def key(k):
            s.on_key(True, k, t[0])
            adv(0.05)
            s.on_key(False, k, t[0])

        s.start(t[0])
        wrong_once = moved_once = unsettled_once = False
        guard = 0
        while not s.done and guard < 5000:
            guard += 1
            if s.state == "info":
                key(" ")
            elif s.state == "rest_ready":
                if s.cur.kind == "rest":
                    dev.peak = 50.0 if not moved_once else 2.0  # 첫 정지 측정은 움직임 -> 재측정
                    moved_once = True
                else:
                    dev.peak = 300.0  # 축 확인은 움직여도 ok
                key(" ")
            elif s.state == "rest":
                adv(0.5)
            elif s.state == "settle":
                if not unsettled_once:
                    dev.still = False  # 한 번은 정지 안 됨 -> settle_max 후 cue
                adv(0.1)
                if s.state == "cue":
                    unsettled_once, dev.still = True, True
            elif s.state == "cue":
                it = s.cur
                if it.phase == "isolated" and not wrong_once:
                    wrong_once = True
                    key("p")
                elif it.kind == "key":
                    key(it.expect)
                else:
                    for ch in it.expect:
                        key(ch)
                adv(0.3)
        s.close()
        assert s.done
        tr = pd.read_csv(Path(d) / "trials.csv", keep_default_na=False)
        ks = pd.read_csv(Path(d) / "keys.csv", keep_default_na=False)
    ok = tr[tr.status == "ok"]
    assert ok.phase.value_counts().to_dict() == {"isolated": 14, "calib": 2, "rest": 2, "continuous": 2, "axes": 1}
    assert (tr.status == "wrong").sum() == 1 and (tr.status == "moved").sum() == 1
    wrong = tr[tr.status == "wrong"].iloc[0]
    assert ((tr.uid == wrong.uid) & (tr.status == "ok")).any(), "틀린 항목 재시도"
    assert s.unsettled == 1 and (tr.still_at_cue == 0).sum() == 1
    assert tr[tr.phase == "rest"].iloc[0].status == "moved" and tr[tr.phase == "rest"].iloc[1].status == "ok"
    assert tr.iloc[-1].phase == "rest", "마지막 정지는 맨 끝"
    w = ok[ok.phase == "continuous"]
    assert set(w.typed) == {"gksrnr ", "qkek "}
    assert ks.host_ns.is_monotonic_increasing and (ks[ks.trial_id > 0].event == "down").sum() >= len(ok) - 2


# --------------------------------------------------------------------------- 가짜 장치 + 종단 테스트
class FakeArduino:
    """ring_imu 펌웨어 흉내: 238Hz D줄, 80Hz M줄, P 핑 응답. micros 래핑 직전에서 시작, 50ppm 드리프트."""
    ODR = 238.0

    def __init__(self, port, baud, timeout=0.02):
        self.timeout = timeout
        self._buf = bytearray()
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._g_ns = None
        self._t0 = (1 << 32) - 3_000_000
        self._impacts: list[float] = []
        self._th = None
        FakeArduino.last = self

    def dev_us(self, now_ns=None):
        now_ns = time.perf_counter_ns() if now_ns is None else now_ns
        return self._t0 + (now_ns - self._g_ns) / 1000.0 * (1 + 50e-6)

    def impact(self):
        self._impacts.append(self.dev_us())

    @property
    def in_waiting(self):
        with self._cv:
            return len(self._buf)

    def reset_input_buffer(self):
        with self._cv:
            self._buf.clear()

    def flush(self):
        pass

    def close(self):
        self._stop.set()

    def write(self, data):
        for c in bytes(data):
            if c == ord("I"):
                self._push(b"H,ringko-fw-1,238.0,0.244,70.000,0.140\n")
            elif c == ord("G") and self._th is None:
                self._g_ns = time.perf_counter_ns()
                self._th = threading.Thread(target=self._gen, daemon=True)
                self._th.start()
            elif c == ord("P") and self._g_ns is not None:
                self._push(f"P,{int(self.dev_us()) % (1 << 32)}\n".encode())
            elif c == ord("X") and self._th is not None:
                self._stop.set()
        return len(data)

    def read(self, n=1):
        with self._cv:
            if not self._buf:
                self._cv.wait(self.timeout)
            out = bytes(self._buf[:n])
            del self._buf[:n]
            return out

    def _push(self, b):
        with self._cv:
            self._buf.extend(b)
            self._cv.notify_all()

    def _gen(self):  # noqa: C901
        rng = np.random.default_rng(0)
        period = 1e6 / self.ODR
        i = m = 0
        while not self._stop.is_set():
            now_dev = self.dev_us()
            lines = []
            while self._t0 + i * period <= now_dev:
                t = self._t0 + i * period
                a = np.array([0.02, -0.05, 1.0]) + rng.normal(0, 0.003, 3)
                g = rng.normal(0, 0.4, 3)
                for ti in self._impacts:
                    dt = (t - ti) / 1e6
                    if 0 <= dt < period / 1e6:
                        a += [0.8, -0.6, 2.2]           # 타격 스파이크(1샘플)
                    if -0.08 < dt < 0.05:
                        g += [120.0, 40.0, -30.0]       # 손가락 움직임
                raw_a = np.round(a / 0.000244).astype(int)
                raw_g = np.round(g / 0.070).astype(int)
                lines.append(f"D,{i},{int(t) % (1 << 32)},{raw_a[0]},{raw_a[1]},{raw_a[2]},"
                             f"{raw_g[0]},{raw_g[1]},{raw_g[2]}\n")
                if self._t0 + m * 12500 <= t:
                    lines.append(f"M,{int(t) % (1 << 32)},{1500},{-800},{3000}\n")
                    m += 1
                i += 1
            if lines:
                self._push("".join(lines).encode())
            time.sleep(0.002)


def test_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        rng = np.random.default_rng(5)
        plan = build_plan(rng, ("rest", "calib", "isolated"), reps=1, calib_n=3)
        dev = Device("FAKE", d, serial_factory=FakeArduino, ping_interval_s=0.2)
        dev.start()
        fake = FakeArduino.last
        p = Params(settle_still_s=0.1, min_gap_s=0.05, rest_s=0.6)
        s = Session(plan, p, d, device=dev)
        (d / "meta.json").write_text(json.dumps({"subject": "T", "wear": 1, "site": "middle_phalanx",
                                                 "device": dev.header, "complete": True}), encoding="utf-8")
        s.start(time.perf_counter_ns())
        t_end = time.time() + 60
        while not s.done and time.time() < t_end:
            now = time.perf_counter_ns()
            s.tick(now)
            if s.state in ("info", "rest_ready"):
                s.on_key(True, " ", now)
                s.on_key(False, " ", now)
            elif s.state == "cue":
                fake.impact()
                time.sleep(0.010)                 # 키 이벤트는 타격 10ms 뒤
                now = time.perf_counter_ns()
                s.on_key(True, s.cur.expect, now)
                s.on_key(False, s.cur.expect, now + 50_000_000)
            time.sleep(0.003)
        time.sleep(0.5)
        dev.stop()
        s.close()
        assert s.done, s.state
        st = dev.stats()
        assert st["seq_gaps"] == 0 and st["bad_lines"] == 0, st
        assert abs(st["rate_hz"] - 238 * (1 + 50e-6)) < 3, st["rate_hz"]
        sd = load_session(d)
        assert sd.imu.t_us.is_monotonic_increasing and sd.imu.t_us.iloc[-1] > (1 << 32), "래핑 해제"
        assert abs(sd.imu.az.median() - 1.0) < 0.01 and abs(sd.mag.mx.iloc[0] - 21.0) < 1e-6
        assert sd.clock["resid_ms_max"] < 3.0, sd.clock
        lag = impact_lags_ms(sd)
        assert len(lag) == 17 and -25 < np.median(lag) < 0, lag
        wins = trial_windows(sd, "isolated")
        assert len(wins) == 14 and all(len(w["imu"]) > 300 for w in wins)
        lines, warns = quality_report(sd)
        assert any("자음별" in line for line in lines), lines


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            t0 = time.time()
            try:
                fn()
                print(f"PASS {name} ({time.time() - t0:.1f}s)")
            except Exception as e:  # noqa: BLE001
                fails += 1
                import traceback
                print("FAIL", name, "->", repr(e))
                traceback.print_exc()
    sys.exit(1 if fails else 0)
