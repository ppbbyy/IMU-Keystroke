"""수집 파이프라인 검증. 실행: python tests/test_pipeline.py"""
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ringkey.hangul_keys import (CONSONANTS, is_typable, word_to_consonant_keys, word_to_keys)  # noqa: E402
from ringkey.imu_stream import IMUReader  # noqa: E402
from ringkey.keylog import KeyLog  # noqa: E402
from ringkey.prompts import (build_plan, consonant_coverage, expected_counts, filter_vocab,  # noqa: E402
                             load_plan, sample_words, save_plan, seed_for)
from ringkey.session import Params, Runner  # noqa: E402
from ringkey.sync import estimate_offset  # noqa: E402


def test_hangul():
    assert word_to_keys("한국") == "gksrnr"
    assert word_to_consonant_keys("한국") == "gsrr"
    assert word_to_keys("값") == "rkqt" and word_to_consonant_keys("값") == "rqt"
    assert word_to_keys("읽") == "dlfr"
    assert word_to_keys("의") == "dml" and word_to_keys("과") == "rhk"
    assert not is_typable("꽃") and not is_typable("있다") and not is_typable("얘") and not is_typable("abc")
    assert is_typable("바보")


def test_prompts():
    rng = np.random.default_rng(seed_for("S01", "s1", "wear0"))
    vocab = filter_vocab([(w, 1.0 / (i + 1)) for i, w in enumerate(
        "한국 사람 학교 하늘 바다 친구 시간 마음 생각 음악 도서관 나무 사랑 가족 여행 컴퓨터 사진 노래 대학교 병원".split())],
        min_cons=2)
    words = sample_words(vocab, 10, rng)
    plan = build_plan(rng=rng, words=words, n_blocks=50)
    c = expected_counts(plan)
    assert c == {"sync_start": 5, "calib": 10, "isolated": 700, "continuous": 10, "sync_end": 5}, c
    iso = [p for p in plan if p.phase == "isolated"]
    cnt = Counter(p.label for p in iso)
    assert set(cnt) == set(CONSONANTS) and all(v == 50 for v in cnt.values())
    assert all(a.label != b.label for a, b in zip(iso, iso[1:])), "연속 동일 자음"
    # 재현성 + 다른 착용은 다른 순서
    rng2 = np.random.default_rng(seed_for("S01", "s1", "wear0"))
    plan2 = build_plan(rng=rng2, words=sample_words(vocab, 10, rng2), n_blocks=50)
    assert [p.label for p in plan] == [p.label for p in plan2]
    assert seed_for("S01", "s1", "wear0") != seed_for("S01", "s1", "wear1")
    with tempfile.TemporaryDirectory() as d:
        save_plan(plan, Path(d) / "p.csv")
        assert [p.__dict__ for p in load_plan(Path(d) / "p.csv")] == [p.__dict__ for p in plan]
    assert sum(consonant_coverage(words).values()) == sum(len(word_to_consonant_keys(w)) for w in words)


def test_wordfreq_vocab():
    from ringkey.prompts import load_vocab
    v = filter_vocab(load_vocab(None, 20000))
    assert len(v) > 1000
    from ringkey.prompts import balance_words
    rng = np.random.default_rng(0)
    words = sample_words(v, 150, rng)
    words = balance_words(v, words, rng, min_count=15)
    cov = consonant_coverage(words)
    assert len(words) == 150 and len(set(words)) == 150
    assert min(cov.values()) >= 15, cov


class FakeIMU:
    def __init__(self, peak):
        self.peak = peak

    def recent_peak_gyro(self, t0, t1):
        return self.peak

    def last_sample_age_s(self):
        return 0.0


def _run_runner(script, imu_peak=80.0):
    """script: 시나리오 함수(runner, clock) — 가상 시계로 구동."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        rng = np.random.default_rng(1)
        plan = build_plan(("sync_start", "calib", "isolated", "continuous", "sync_end"), rng=rng,
                          words=["한국", "바보"], n_blocks=1, calib_n=2, sync_n=3)
        kl = KeyLog(d / "keys.csv")
        r = Runner(plan, Params(), kl, d / "trials.csv", imu=FakeIMU(imu_peak), seed=1)
        t = [10_000_000_000]
        r.begin(t[0])

        def adv(sec):
            t[0] += int(sec * 1e9)
            r.tick(t[0])

        def key(k, hold=0.08):
            r.on_key(True, k, t[0])
            adv(hold)
            r.on_key(False, k, t[0])

        script(r, adv, key, t)
        r.close()
        kl.close()
        import pandas as pd
        return r, pd.read_csv(d / "trials.csv"), pd.read_csv(d / "keys.csv")


def _drive_all(r, adv, key, t, wrong_first=True, ime=False):
    """플랜을 끝까지 진행. 개별 키 첫 시도는 일부러 틀림."""
    wrong_done = False
    guard = 0
    while not r.done and guard < 5000:
        guard += 1
        st = r.state
        if st == "info":
            key(" ")
        elif st == "iti":
            adv(0.1)
        elif st == "cue":
            p = r.cur
            if p.kind == "key":
                if wrong_first and p.phase == "isolated" and not wrong_done:
                    wrong_done = True
                    key("p" if p.expect != "p" else "o")
                else:
                    key(p.expect)
            else:
                for ch in p.expect:
                    key(ch)
                    adv(0.12)
            adv(0.6)
        else:
            adv(0.1)


def test_runner_full():
    r, tr, ks = _run_runner(lambda r, adv, key, t: _drive_all(r, adv, key, t))
    assert r.done and not r.aborted
    ok = tr[tr.status == "ok"]
    assert (ok.phase == "sync_start").sum() == 3 and (ok.phase == "sync_end").sum() == 3
    assert (ok.phase == "calib").sum() == 2
    assert (ok.phase == "isolated").sum() == 14
    assert (ok.phase == "continuous").sum() == 2
    assert (tr.status == "wrong").sum() == 1, "틀린 항목 1개 기록"
    wrong = tr[tr.status == "wrong"].iloc[0]
    assert ((tr.prompt_id == wrong.prompt_id) & (tr.status == "ok")).sum() == 1, "틀린 항목 재시도됨"
    assert tr.gyro_peak_dps.notna().all()
    # 키 로그: 큐 중 키는 trial_id>0, 그 외 0
    assert (ks[ks.trial_id > 0].event == "down").sum() >= len(ok)
    assert ks.host_ns.is_monotonic_increasing
    # 단어: expect와 typed 일치
    w = ok[ok.phase == "continuous"].iloc[0]
    assert w.typed == w.expect == "gksrnr "


def test_runner_timeout_and_warn():
    def sc(r, adv, key, t):
        # info 통과 후 큐가 뜨면 아무것도 안 누르고 시간초과
        while r.state == "info":
            key(" ")
        for _ in range(100):
            adv(0.1)
            if r.state == "cue":
                break
        adv(7.0)
        assert r.status_counts["timeout"] == 1
        r.abort()
    r, tr, _ = _run_runner(sc)
    assert (tr.status == "timeout").sum() == 1

    # 센서 신호 약함 경고
    def sc2(r, adv, key, t):
        _drive_all(r, adv, key, t, wrong_first=False)
    r, tr, _ = _run_runner(sc2, imu_peak=3.0)
    assert r.low_streak >= 3 and r.low_signal_warned, (r.low_streak, r.low_signal_warned)
    assert (tr.gyro_peak_dps.astype(float) < 15).all()


def test_runner_ime_flag():
    def sc(r, adv, key, t):
        while r.state == "info":
            key(" ")
        for _ in range(100):
            adv(0.1)
            if r.state == "cue":
                break
        key("ㅂ")
        assert r.ime_suspected and "한글" in r.view.warn
        r.abort()
    _run_runner(sc)


def test_sync_estimate():
    rng = np.random.default_rng(0)
    fs, dur = 119.0, 120.0
    t_dev = np.arange(0, dur, 1 / fs) + 1000.0
    true_off = 5000.123
    acc = rng.normal(0, 0.004, (len(t_dev), 3)) + np.array([0, 0, 1.0])
    key_dev = np.array([20.0, 21.7, 23.3, 25.1, 26.6])
    for k in key_dev:
        i = int(np.argmin(np.abs(t_dev - (1000 + k))))
        acc[i] += [0.9, -0.5, 1.4]
        acc[i + 1] += [-0.6, 0.3, -0.9]
    # 호스트 도착시각: 장치시각 + 오프셋 + USB 지연(0~12ms 지터)
    host = t_dev + true_off + rng.uniform(0.001, 0.012, len(t_dev))
    keys = key_dev + 1000 + true_off + 0.004  # 키 이벤트 지연 4ms
    r = estimate_offset(host, t_dev, acc, keys)
    assert r is not None and r.n_used == 5
    assert abs(r.offset_s - (true_off + 0.004)) < 0.012, r.offset_s
    assert r.mad_ms < 10


class FakeSerial:
    """'G'를 받으면 D 줄을 ~119Hz로 내보내는 가짜 Arduino(OS 무관). micros 래핑 + 손실 + 쓰레기 줄 포함."""

    def __init__(self, port, baud, timeout=0.05, n=400, t0=(1 << 32) - 1_000_000, drop_at=100, drop_n=3):
        self.timeout = timeout
        self._buf = bytearray()
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._args = (n, t0, drop_at, drop_n)
        self._th = None

    # pyserial 인터페이스 일부
    @property
    def in_waiting(self):
        with self._cv:
            return len(self._buf)

    def reset_input_buffer(self):
        with self._cv:
            self._buf.clear()

    def write(self, data):
        for c in data:
            if c == ord("I"):
                self._push(b"H,ringkey-1.0,119.0,119.0,20.0\n")
            elif c == ord("G") and self._th is None:
                self._th = threading.Thread(target=self._gen, daemon=True)
                self._th.start()
            elif c == ord("X"):
                self._stop.set()
        return len(data)

    def flush(self):
        pass

    def read(self, n=1):
        with self._cv:
            if not self._buf:
                self._cv.wait(self.timeout)
            out = bytes(self._buf[:n])
            del self._buf[:n]
            return out

    def close(self):
        self._stop.set()

    def _push(self, b):
        with self._cv:
            self._buf.extend(b)
            self._cv.notify_all()

    def _gen(self):
        n, t0, drop_at, drop_n = self._args
        seq = 0
        for i in range(n):
            if self._stop.is_set():
                return
            if i == drop_at:
                seq += drop_n
            t = (t0 + int(i * 1e6 / 119)) % (1 << 32)
            self._push(f"D,{seq},{t},1,2,1000,10,-20,30,100,200,300,{i % 6 == 0:d}\n".encode())
            if i == 50:
                self._push(b"D,garbage\n")
            seq += 1
            time.sleep(1 / 119)


def test_imu_reader_fake_serial():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "imu.csv"
        r = IMUReader("FAKE", out, serial_factory=FakeSerial)
        r.start(timeout=5)
        time.sleep(4.2)
        r.stop()
        import pandas as pd
        df = pd.read_csv(out)
    s = r.stats()
    assert s["n"] >= 300, s
    assert s["dropped"] == 3 and s["bad_lines"] == 1, s
    assert any(line.startswith("H,") for line in s["fw_info"])
    t = df["t_us"].to_numpy()
    assert (np.diff(t) > 0).all(), "micros 래핑이 풀리지 않음"
    assert t[-1] > (1 << 32), "래핑 후 값이 4294967296을 넘어야 함"
    assert abs(df.az.iloc[0] - 1.0) < 1e-9 and abs(df.gx.iloc[0] - 0.01) < 1e-9 and abs(df.my.iloc[0] - 0.2) < 1e-9
    assert abs(s["rate_hz"] - 119) < 5, s["rate_hz"]
    assert df.host_ns.is_monotonic_increasing


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                import traceback
                print("FAIL", name, "->", repr(e))
                traceback.print_exc()
    sys.exit(1 if fails else 0)
