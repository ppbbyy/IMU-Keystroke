#!/usr/bin/env python3
"""파일럿/점검: 센서 연결, 실제 샘플링 레이트, 손실, 정지 노이즈, 자력계 상태 확인. 스트림을 N초 기록해 요약한다.

예) python scripts/check_device.py --seconds 10          # 센서를 책상에 정지시킨 채
    python scripts/check_device.py --seconds 10 --tap    # 중간에 스페이스를 치듯 탁탁 쳐서 스파이크 확인
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ringkey.imu_stream import IMUReader, find_port  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out", help="CSV로 저장(선택)")
    ap.add_argument("--tap", action="store_true", help="탭 스파이크를 표시")
    a = ap.parse_args()
    port = a.port or find_port()
    if not port:
        sys.exit("포트를 찾지 못했습니다. --port 로 지정하세요.")

    import tempfile
    out = a.out or str(Path(tempfile.gettempdir()) / "check_device.csv")
    r = IMUReader(port, out)
    r.start()
    print(f"{port} 에서 {a.seconds:.0f}초 기록 중…", flush=True)
    time.sleep(a.seconds)
    r.stop()

    import pandas as pd
    d = pd.read_csv(out)
    s = r.stats()
    med, sd = IMUReader.dt_jitter_ms(d["t_us"].to_numpy())
    print("펌웨어:", s["fw_info"])
    print(f"샘플 {s['n']} · 실측 {s['rate_hz']:.1f}Hz · dt 중앙값 {med:.2f}ms (표준편차 {sd:.2f}ms) · "
          f"손실 {s['dropped']} · 형식오류 {s['bad_lines']}")
    acc = d[["ax", "ay", "az"]].to_numpy()
    gyr = d[["gx", "gy", "gz"]].to_numpy()
    print(f"정지 가속도 크기 평균 {np.linalg.norm(acc, axis=1).mean():.3f} g (정지 시 ≈1.0)")
    print(f"자이로 평균(바이어스) {np.round(gyr.mean(0), 2)} dps · 표준편차 {np.round(gyr.std(0), 2)} dps")
    mag_rate = d["mag_new"].sum() / max(1e-9, s["duration_s"])
    mag = d[["mx", "my", "mz"]].to_numpy()
    print(f"자력계 갱신 {mag_rate:.1f}Hz · 크기 평균 {np.linalg.norm(mag, axis=1).mean():.1f} uT · "
          f"ΔB 범위 {np.round(np.ptp(mag, axis=0), 1)} uT")
    if a.tap:
        e = np.concatenate([[0], np.linalg.norm(np.diff(acc, axis=0), axis=1)])
        thr = 6 * np.median(e) + 1e-6
        idx = np.where((e > thr) & (e >= np.roll(e, 1)) & (e >= np.roll(e, -1)))[0]
        t = (d["t_us"].to_numpy() - d["t_us"].iloc[0]) / 1e6
        print(f"탭 스파이크 후보 {len(idx)}개 (임계 {thr:.3f} g): 시각 " + ", ".join(f"{t[i]:.2f}s" for i in idx[:20]))
    if s["rate_hz"] < 80 or s["drop_frac"] > 0.01:
        print("경고: 레이트가 낮거나 손실이 1%를 넘습니다. USB 케이블/허브, 다른 시리얼 프로그램을 확인하세요.")


if __name__ == "__main__":
    main()
