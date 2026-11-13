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


def test_closure_zupt_principle():
    """refine_presses의 근거: 진짜 정지(타격) 시각에 속도 0을 넣으면 F 복귀 닫힘 오차가 줄고, 움직이는 중간에 넣으면 는다."""
    from ringko.features import _segment_disp
    fs = 238
    t = np.arange(0, 1.5, 1 / fs)
    # F(0) → 키(+38mm, t=0.5에서 정지) → F(0, t=1.5): 각 0.5/1.0초 구간 사인형 가속, 가속도계 바이어스 0.05 m/s²
    def leg(t0, t1, dx):
        T = t1 - t0
        tau = np.clip((t - t0) / T, 0, 1)
        inside = (t >= t0) & (t <= t1)
        return np.where(inside, dx * 2 * np.pi / T ** 2 * np.sin(2 * np.pi * tau), 0.0)
    ax = leg(0.0, 0.5, 0.038) + leg(0.5, 1.5, -0.038) + 0.05
    a = np.stack([ax, np.zeros_like(ax), np.zeros_like(ax)], 1)
    gz = np.array([0.0, 0.0, 1.0])

    def closure(split_s):
        pos, start = np.zeros(3), 0
        for ie in [int(split_s * fs), len(t) - 1]:
            d, _, _ = _segment_disp(t[start:ie + 1], a[start:ie + 1], gz, full_zupt=True)
            pos, start = pos + d, ie
        return np.linalg.norm(pos) * 1000

    whole, _, _ = _segment_disp(t, a, gz, full_zupt=True)
    c_none = np.linalg.norm(whole) * 1000
    assert closure(0.5) < 0.6 * c_none                 # 진짜 정지에 ZUPT → 닫힘 개선
    assert closure(0.25) > closure(0.5) * 3            # 움직이는 중간(속도 최대)에 ZUPT → 크게 나빠짐


# --------------------------------------------------------------------------- 3차원 자음 추론
def _synthetic_word(keys="ds", mount=(20.0, -70.0, 10.0), fs=238.0, leg=0.35, dwell=0.08, tilt_deg=6.0, seed=0):
    """F 정지 → 키들 → F 정지. 세계 좌표(x 오른쪽, y 위 행, z 위) 최소저크 이동 + 손가락 회전, 센서는 mount 회전으로 부착.
    반환 t, 비력 f_b, 각속도 w_b(오른손), 타격 인덱스, 세계 타격 위치[m]."""
    from ringko.keymap3d import KI, Keyboard3D
    from ringko.motion import rotvec_to_mat
    kb = Keyboard3D()
    pts = [np.zeros(3)] + [np.r_[kb.C2[KI[k]] / 1000, -0.003] for k in keys] + [np.zeros(3)]
    p_parts, m_parts = [], []
    dt = 1 / fs
    ips = []
    for a, b in zip(pts[:-1], pts[1:]):
        n = int(leg * fs)
        s = np.arange(n) / n
        mj = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
        p_parts.append(a + np.outer(mj, b - a) + np.outer(np.sin(np.pi * s) ** 2, [0, 0, 0.012]))
        m_parts.append(np.sin(np.pi * s))
        p_parts.append(np.repeat(b[None], int(dwell * fs), 0))
        m_parts.append(np.zeros(int(dwell * fs)))
    pre = np.zeros((int(0.6 * fs), 3))
    post = np.zeros((int(1.0 * fs), 3))
    pos = np.vstack([pre] + p_parts + [post])
    mot = np.concatenate([np.zeros(len(pre))] + m_parts + [np.zeros(len(post))])
    t = np.arange(len(pos)) * dt
    k0 = len(pre)
    for k in range(len(keys)):
        k0 += int(leg * fs)
        ips.append(k0 + 2)
        k0 += int(dwell * fs)
    vel = np.gradient(pos, dt, axis=0)
    accw = np.gradient(vel, dt, axis=0)
    base = rotvec_to_mat(np.radians(mount))
    Rs = [base @ rotvec_to_mat(np.radians([tilt_deg * m, 0.5 * tilt_deg * m, 0.3 * tilt_deg * m])) for m in mot]
    f_b = np.array([R.T @ (a + [0, 0, 9.80665]) for R, a in zip(Rs, accw)])
    w_b = []
    for i in range(len(t)):
        j = min(i, len(t) - 2)
        dR = Rs[j].T @ Rs[j + 1]
        th = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
        vee = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
        w_b.append(vee * (th / (2 * np.sin(th)) if th > 1e-9 else 0.5) / dt)
    rng = np.random.default_rng(seed)
    f_b = f_b + rng.normal(0, 0.002, f_b.shape)
    return t, f_b, np.array(w_b), ips, np.array(pts[1:-1])


def test_track3d_mount_invariant():
    """타격 사이 적분 + 타격 ZUPT + F 복귀 닫힘: 센서를 어떻게 붙여도 F 거리·타격 간 거리가 실제와 같다."""
    from ringko.track3d import TrackParams, _finish, track_trial
    for mount in [(20.0, -70.0, 10.0), (-110.0, 30.0, 60.0)]:
        t, f, w, ips, true = _synthetic_word("dsr", mount)
        ref = np.eye(3)[int(np.argmin(np.abs(f[:50].mean(0))))]
        raw = track_trial(t, f, w, ips, 0.3, ref, TrackParams())
        assert raw is not None and raw["has_ret"]
        x = pd.Series({"trial_id": 1, "phase": "continuous", "target": "x"})
        tr = _finish(raw, x, "dsr", TrackParams(), q=100.0)
        r_true = np.linalg.norm(true[:, :2], axis=1) * 1000
        r_est = np.linalg.norm(tr.P[:, :2], axis=1)
        assert np.abs(r_est - r_true).max() < 2.0, (mount, r_est, r_true)
        d_true = np.linalg.norm(np.diff(true[:, :2], axis=0), axis=1) * 1000
        d_est = np.linalg.norm(np.diff(tr.P[:, :2], axis=0), axis=1)
        assert np.abs(d_est - d_true).max() < 2.0
        assert np.linalg.norm(tr.closure) < 5.0                         # 깨끗한 합성 데이터면 닫힘 오차도 작다


def test_track3d_closure_weights():
    """닫힘 오차는 구간 분산(∝T³)에 비례해 나뉘고, 평면 제약은 타격 높이를 맞추며, 닫힘은 정확히 0이 된다."""
    from ringko.track3d import TrackParams, _correct
    d = np.array([[10.0, 0, 3], [5.0, 2, -1], [-12.0, -1, 4]])           # 타격 2개 + 복귀
    T = np.array([0.2, 0.4, 0.2])
    dc, _ = _correct(d, T, True, TrackParams(plane=False), q=1.0)
    assert np.allclose(dc.sum(0), 0, atol=1e-9)
    c = dc - d
    assert np.allclose(c[:, 0] / c[:, 0].sum(), (T ** 3 + 1e-4) / (T ** 3 + 1e-4).sum())
    dl, _ = _correct(d, T, True, TrackParams(closure="linear", plane=False), q=1.0)
    assert np.allclose((dl - d)[:, 0] / (dl - d)[:, 0].sum(), (T + 1e-4) / (T + 1e-4).sum())
    dp, zr = _correct(d, T, True, TrackParams(plane=True, sigma_plane_mm=0.5), q=1000.0)
    Pz = np.cumsum(dp[:, 2])[:2]
    assert abs(Pz[0] - Pz[1]) < 1.0 and abs(dp[:, 2].sum()) < 1e-6


def test_keymap_bands():
    """F 기준 기준 거리·min/max: 물리 배열에서 나오고, 키 단위로 정규화하면 키보드 크기와 무관."""
    from ringko.keymap3d import KI, Keyboard3D
    kb = Keyboard3D(pitch_mm=19.0)
    assert np.allclose(kb.C2[KI["f"]], 0) and abs(kb.d[KI["s"]] - 38.0) < 1e-9 and abs(kb.d[KI["d"]] - 19.0) < 1e-9
    assert kb.C2[KI["r"]][1] > 0 and kb.C2[KI["c"]][1] < 0 and kb.C2[KI["g"]][0] > 0   # y = 위 행, x = 오른쪽
    dk, lo, hi = kb.band3d([0.0], [5.0])
    assert (lo <= dk).all() and (dk <= hi).all() and lo[0, KI["f"]] == 0
    t1, t2 = Keyboard3D(pitch_mm=18.0).table(), Keyboard3D(pitch_mm=19.05).table()
    assert len(t1) == 14 and np.allclose(t1["d_k[u]"], t2["d_k[u]"], atol=0.02)
    lo2, hi2 = kb.pair_band(5.0)
    assert np.allclose(lo2, lo2.T) and (lo2[np.arange(14), np.arange(14)] == 0).all()


def _tracks_from_layout(words, R2, scale=1.0, noise=3.0, seed=0):
    """정답 배열 위치를 중력 좌표계로 돌려(R2: 키보드→센서 수평 변환) 만든 합성 WordTrack."""
    from ringko.keymap3d import KI, Keyboard3D
    from ringko.track3d import WordTrack
    kb = Keyboard3D()
    rng = np.random.default_rng(seed)
    out = []
    for i, w in enumerate(words):
        cs = to_consonant_keys(w)
        K = np.c_[kb.C2[[KI[c] for c in cs]], np.full(len(cs), -4.0)]
        P = np.c_[K[:, :2] @ R2.T, K[:, 2]] * scale + rng.normal(0, noise, K.shape)
        D = np.diff(np.vstack([np.zeros(3), P]), axis=0)
        n = len(cs)
        out.append(WordTrack(i, "continuous", w, cs, np.arange(n), P, D, P, np.full(n, 6.0), np.full(n, 6.0),
                             np.zeros(3), np.full(n + 1, 0.3), -4.0))
    return out


def test_consonant3d_and_alignment():
    """라벨 없는 EM이 yaw·손잡이를 복원하고, 방향 항이 거리만으로는 못 가르는 ㅇ(d)/ㅎ(g)를 가른다."""
    from ringko.align3d import consonant_prior, fit_alignment, fit_from_taps
    from ringko.consonant3d import Alignment, Scorer3D, chain_marginals
    from ringko.decode import ConsonantLM, Lexicon
    from ringko.keymap3d import KEYS, KI
    words = ["사람", "한국", "학교", "나라", "가게", "하늘", "바다", "어디", "다리", "기도", "도시", "하다", "택시",
             "친구", "마음", "시간", "편지", "자동차", "커피", "방법"]
    yaw = np.radians(95.0)
    R2 = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]) @ np.diag([1.0, -1.0])   # 회전 + 거울
    tracks = _tracks_from_layout(words * 2, R2, 0.95)
    vocab = [(w, 10.0) for w in words]
    sc = Scorer3D()
    al = fit_alignment(tracks, sc, prior=consonant_prior(vocab))
    assert al.flip == -1.0 and abs(np.degrees(np.angle(np.exp(1j * (al.theta - yaw))))) < 8, (al.flip, np.degrees(al.theta))
    assert abs(al.scale - 0.95) < 0.15 and al.oriented
    hit = sum(int((chain_marginals(sc.observe(tr, al)).argmax(1) == [KI[c] for c in tr.truth]).sum()) for tr in tracks)
    assert hit / sum(tr.n for tr in tracks) > 0.85
    lex = Lexicon(vocab)
    assert sum(lex.rank_words(sc, sc.observe(tr, al))[0][1] == tr.target for tr in tracks) >= 0.9 * len(tracks)
    # 거리만(shell, 정렬 없음): 회전에 완전 불변, d와 g는 F에서 같은 거리라 점수가 같다
    shell = Scorer3D(mode="shell")
    one = _tracks_from_layout(["오"], np.eye(2), noise=0.0)[0]      # ㅇ = d 한 타격
    rot = _tracks_from_layout(["오"], R2, noise=0.0)[0]
    Ma, Mb = chain_marginals(shell.observe(one, Alignment())), chain_marginals(shell.observe(rot, Alignment()))
    assert np.allclose(Ma, Mb) and abs(Ma[0, KI["d"]] - Ma[0, KI["g"]]) < 1e-9
    Mo = chain_marginals(sc.observe(rot, al))
    assert Mo[0, KI["d"]] > 5 * Mo[0, KI["g"]]
    # 탭 캘리브레이션(정답 키 몇 개)도 같은 정렬
    taps = _tracks_from_layout(["호", "오", "고", "초"], R2, 0.95)
    at = fit_from_taps(taps)
    assert at.flip == -1.0 and abs(np.degrees(np.angle(np.exp(1j * (at.theta - yaw))))) < 8
    # 자음 LM: 확률 정규화, γ→0 이면 LM 없는 경우와 같다
    lm = ConsonantLM(vocab)
    assert np.allclose(np.exp(lm.logp).sum(2), 1)
    obs = sc.observe(tracks[0], al)
    assert np.allclose(chain_marginals(obs), chain_marginals(obs, lm, 1e-9), atol=1e-6)
    assert len(KEYS) == 14


def test_gyro_sign_estimate():
    """정지 자세 사이 중력 방향 변화로 자이로 부호 규약을 라벨 없이 찾는다."""
    from ringko.align3d import estimate_gyro_sign
    from ringko.motion import rotvec_to_mat
    fs = 200.0
    rng = np.random.default_rng(3)
    R = rotvec_to_mat(np.radians([10, 20, 30]))
    T, A, W = [], [], []
    tt = 0.0
    for k in range(12):
        n = int(0.5 * fs)
        for _ in range(n):                                            # 정지
            T.append(tt); A.append(R.T @ [0, 0, 9.80665]); W.append(np.zeros(3)); tt += 1 / fs
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        wv = axis * np.radians(60)                                    # body 각속도
        for _ in range(int(0.4 * fs)):
            T.append(tt); A.append(R.T @ [0, 0, 9.80665]); W.append(wv.copy()); tt += 1 / fs
            R = R @ rotvec_to_mat(wv / fs)
    T, A, W = np.array(T), np.array(A), np.array(W)
    true = np.array([-1.0, 1.0, -1.0])
    r = estimate_gyro_sign(T, A, W * true, max_pairs=40)              # 원시 = 물리 × 부호
    assert np.allclose(r["sign"], true), r["errors"]


def test_gyro_sign_few_large_rotations_among_many_small():
    """axes 단계의 큰 회전 몇 번 + 타이핑의 작은 회전 다수(가속도 잡음에 묻힘).
    작은 쌍이 다수여도 큰 회전 쌍이 부호를 정해야 한다(중앙값 점수는 여기서 애매해져 오판했다, S01 w3)."""
    from ringko.align3d import compose_rotvecs, estimate_gyro_sign
    from ringko.motion import rotvec_to_mat
    fs = 200.0
    rng = np.random.default_rng(7)
    R = rotvec_to_mat(np.radians([5, -10, 15]))
    T, A, W = [], [], []
    tt = 0.0

    def still(n):
        nonlocal tt
        g = R.T @ [0, 0, 9.80665] + rng.normal(0, 0.15, 3)            # 자세마다 다른 가속도 치우침(≈1°)
        for _ in range(n):
            T.append(tt); A.append(g + rng.normal(0, 0.01, 3)); W.append(np.zeros(3)); tt += 1 / fs

    def move(wv, n):
        nonlocal R, tt
        for _ in range(n):
            T.append(tt); A.append(R.T @ [0, 0, 9.80665]); W.append(wv.copy()); tt += 1 / fs
            R = R @ rotvec_to_mat(wv / fs)

    for k in range(4):                                                # axes: 큰 회전 4번(축 고르게)
        still(60)
        move(np.eye(3)[k % 3] * np.radians(150), 80)
    for _ in range(60):                                               # 타이핑: 작은 회전 60번
        still(60)
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        move(axis * np.radians(15), 16)
    still(60)
    T, A, W = np.array(T), np.array(A), np.array(W)
    true = np.array([-1.0, -1.0, -1.0])
    r = estimate_gyro_sign(T, A, W * true)
    assert np.allclose(r["sign"], true), r["errors"]
    assert r["ambiguous"] == [(-1, -1, -1)], r["errors"]
    v = rng.normal(size=(5, 3)) * 0.1                                  # 벡터화 합성 = 순차 곱
    Rl = np.eye(3)
    for x in v:
        Rl = Rl @ rotvec_to_mat(x)
    assert np.allclose(compose_rotvecs(v), Rl)


def test_lexicon_domain_vocab():
    from ringko.decode import domain_vocab
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
