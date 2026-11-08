#!/usr/bin/env python3
"""장치 점검: 연결, 실측 레이트, 끊김, 정지 노이즈, 자이로 바이어스, 자력계, 시계 핑 품질.

  python scripts/check_device.py                 # 10초. 처음엔 링을 책상에 가만히
  python scripts/check_device.py --tap           # 중간에 F를 몇 번 '톡톡' -> 타격 스파이크 확인
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ringko.clock import fit_clock  # noqa: E402
from ringko.dataset import gaps, jerk  # noqa: E402
from ringko.device import Device, find_port  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--tap", action="store_true")
    a = ap.parse_args()
    port = a.port or find_port()
    if not port:
        sys.exit("Arduino 포트를 찾지 못했습니다. USB 연결을 확인하거나 --port COM3 처럼 지정하세요.")

    out = Path(tempfile.mkdtemp(prefix="ringko_check_"))
    dev = Device(port, out)
    dev.start()
    print(f"{port} · 펌웨어 {dev.header} · {a.seconds:.0f}초 기록", flush=True)
    if dev.header.get("hfxo") != 1:
        print("경고: 크리스털(HFXO) 꺼짐 — firmware/ring_imu (fw-2)를 다시 업로드하세요.", flush=True)
    if a.tap:
        print("처음 3초는 가만히, 그 다음 F를 5번 정도 톡톡 치세요.", flush=True)
    time.sleep(a.seconds)
    dev.stop()

    s = dev.stats()
    imu = pd.read_csv(out / "imu.csv")
    mag = pd.read_csv(out / "mag.csv")
    ping = pd.read_csv(out / "ping.csv")
    odr = dev.header["odr_hz"]
    ng, nmiss = gaps(imu.t_us.to_numpy())
    dt = np.diff(imu.t_us.to_numpy()) / 1e3
    print(f"\nIMU {s['n_imu']}샘플 · 실측 {s['rate_hz']:.1f}Hz (공칭 {odr:g}) · dt 중앙 {np.median(dt):.2f}ms "
          f"(표준편차 {dt.std():.2f}) · 끊김 {ng}회/누락 {nmiss} · 형식오류 {s['bad_lines']}")
    # 가장 조용한 1초 구간(자이로 최대값 최소)을 정지 구간으로 사용
    gn = np.linalg.norm(imu[["gx", "gy", "gz"]].to_numpy(), axis=1)
    w = max(1, int(round(1e6 / np.median(np.diff(imu.t_us.to_numpy())))))
    peaks = pd.Series(gn).rolling(w).max().to_numpy()[w - 1:]
    i0 = int(np.nanargmin(peaks)) if len(peaks) else 0
    st = imu.iloc[i0:i0 + w]
    t_rel = (st.t_us.iloc[0] - imu.t_us.iloc[0]) / 1e6
    an = np.linalg.norm(st[["ax", "ay", "az"]].to_numpy(), axis=1)
    print(f"실제 ODR {1e6 / np.median(dt * 1e3):.1f}Hz (칩 발진기 오차, 정상 범위면 문제없음)")
    print(f"정지(가장 조용한 1초, {t_rel:.1f}s~): |a| {an.mean():.4f}±{an.std():.4f} g (≈1.0) · "
          f"중력 방향 {np.round(st[['ax', 'ay', 'az']].mean().to_numpy(), 3)} g · 자이로 최대 {peaks[i0]:.1f} dps · "
          f"자이로 바이어스 {np.round(st[['gx', 'gy', 'gz']].mean().to_numpy(), 2)} dps · "
          f"노이즈 {np.round(st[['gx', 'gy', 'gz']].std().to_numpy(), 2)} dps")
    if len(mag) > 1:
        rate_m = (len(mag) - 1) / ((mag.t_us.iloc[-1] - mag.t_us.iloc[0]) / 1e6)
        mn = np.linalg.norm(mag[["mx", "my", "mz"]].to_numpy(), axis=1)
        print(f"자력계 {rate_m:.1f}Hz · |B| {mn.mean():.1f} uT (지구자기장 25~65) · "
              f"축별 변동폭 {np.round(np.ptp(mag[['mx', 'my', 'mz']].to_numpy(), axis=0), 1)} uT")
    if len(ping) >= 5:
        c = fit_clock(ping.send_ns, ping.recv_ns, ping.t_us)
        print(f"핑 {c['n_pings']}개 · RTT 최소 {c['rtt_ms_min']:.2f} / 중앙 {c['rtt_ms_median']:.2f} ms · "
              f"드리프트 {c['drift_ppm']:.0f}ppm · 시계 잔차 {c['resid_ms_std']:.2f}ms")
        # micros()는 32kHz RC 기반(보드에 32k 크리스털 없음). HFXO로 보정되면 수백 ppm의 '일정한' 드리프트만 남고
        # 시계 모델이 흡수한다. HFXO가 없으면 ~1.6%(16000ppm)에 흔들림까지 생긴다(파일럿 실측).
        if abs(c["drift_ppm"]) > 2000:
            print("경고: 장치 시계 드리프트가 매우 큽니다 — HFXO가 안 켜졌을 수 있습니다.")
    else:
        print(f"핑 {len(ping)}개(부족) — 펌웨어가 'P'에 응답하는지 확인")
    if a.tap:
        j = jerk(imu)
        thr = max(0.15, 8 * np.median(j))
        pk = np.where((j > thr) & (j >= np.roll(j, 1)) & (j >= np.roll(j, -1)))[0]
        keep = [i for n, i in enumerate(pk) if n == 0 or imu.t_us.iloc[i] - imu.t_us.iloc[pk[n - 1]] > 150_000]
        tt = (imu.t_us.to_numpy() - imu.t_us.iloc[0]) / 1e6
        print(f"타격 스파이크 {len(keep)}개 (임계 {thr:.2f} g): " + ", ".join(f"{tt[i]:.2f}s({j[i]:.2f}g)" for i in keep[:15]))
        print(f"최대 |a| {np.abs(imu[['ax', 'ay', 'az']].to_numpy()).max():.2f} g · "
              f"최대 |gyro| {np.abs(imu[['gx', 'gy', 'gz']].to_numpy()).max():.0f} dps")
    if abs(s["rate_hz"] - odr) / odr > 0.10 or nmiss > 0.005 * len(imu):
        print("경고: 레이트 부족/누락 많음 — USB 허브를 빼고 직접 연결, 다른 시리얼 프로그램 종료")
    if peaks[i0] > 10:
        print("참고: 1초 이상 가만히 있던 구간이 없어 정지 노이즈를 정확히 못 쟀습니다.")
    print(f"\n원시 파일: {out}")


if __name__ == "__main__":
    main()
