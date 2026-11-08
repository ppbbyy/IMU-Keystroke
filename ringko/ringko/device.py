"""IMU 장치 리더.

- 백그라운드 스레드가 시리얼 줄을 읽어 imu.csv / mag.csv / ping.csv 로 저장한다.
- 1초마다 'P' 핑을 보내 (호스트 송신시각, 수신시각, 장치 micros) 를 기록 -> clock.py 가 장치↔호스트 시계를 맞춘다.
  (스파이크 기반 동기화보다 정확하고, 사용자 동작이 필요 없다)
- 최근 2초 자이로 크기를 메모리에 유지 -> 수집 화면이 "손가락이 F 위에서 정지했는가"를 판단(is_still).
"""
from __future__ import annotations

import collections
import math
import threading
import time
from pathlib import Path

IMU_COLUMNS = ["seq", "t_us", "ax", "ay", "az", "gx", "gy", "gz"]   # g, deg/s
MAG_COLUMNS = ["t_us", "mx", "my", "mz"]                           # uT
PING_COLUMNS = ["send_ns", "recv_ns", "t_us"]
ARDUINO_VID = 0x2341
_WRAP = 1 << 32


def find_port() -> str | None:
    from serial.tools import list_ports
    ports = list(list_ports.comports())
    for p in ports:
        if p.vid == ARDUINO_VID:
            return p.device
    return None


class Unwrapper:
    """uint32 micros 래핑 해제(모든 줄 종류가 같은 카운터를 공유)."""

    def __init__(self):
        self.base = 0
        self.last: int | None = None

    def __call__(self, t: int) -> int:
        if self.last is not None and t < self.last and self.last - t > _WRAP // 2:
            self.base += _WRAP
        self.last = t
        return t + self.base


class Device:
    def __init__(self, port: str, out_dir: str | Path | None = None, *, baud: int = 115200,
                 ping_interval_s: float = 1.0, serial_factory=None):
        self.port, self.baud = port, baud
        self.out_dir = Path(out_dir) if out_dir else None
        self.ping_interval_s = ping_interval_s
        self._factory = serial_factory
        self.header: dict | None = None
        self.messages: list[str] = []
        self.error: Exception | None = None
        self.n_imu = self.n_mag = self.n_ping = 0
        self.bad_lines = 0
        self.seq_gaps = 0
        self.first_t_us: int | None = None
        self.last_t_us: int | None = None
        self.last_rx_ns: int | None = None
        self._unwrap = Unwrapper()
        self._last_seq: int | None = None
        self._ping_sent: int | None = None
        self._last_ping = 0.0
        self._recent: collections.deque = collections.deque(maxlen=600)  # (t_us, |gyro| dps)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ser = None
        self._fh: dict = {}
        self._scale = (0.0, 0.0, 0.0)

    # ------------------------------------------------------------------ 수명주기
    def start(self, timeout_s: float = 5.0) -> None:
        import serial
        self._ser = (self._factory or serial.Serial)(self.port, self.baud, timeout=0.02)
        self._ser.write(b"X")
        time.sleep(0.05)
        self._ser.reset_input_buffer()
        self._ser.write(b"I")
        self._ser.flush()
        self._wait_header(timeout_s)
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            for name, cols in (("imu", IMU_COLUMNS), ("mag", MAG_COLUMNS), ("ping", PING_COLUMNS)):
                fh = open(self.out_dir / f"{name}.csv", "w", encoding="utf-8", newline="", buffering=1 << 16)
                fh.write(",".join(cols) + "\n")
                self._fh[name] = fh
        self._ser.write(b"G")
        self._ser.flush()
        self._thread = threading.Thread(target=self._run, name="imu-reader", daemon=True)
        self._thread.start()
        t0 = time.time()
        while self.n_imu == 0:
            if self.error:
                raise RuntimeError(f"시리얼 오류: {self.error}")
            if time.time() - t0 > timeout_s:
                self.stop()
                raise RuntimeError("IMU 데이터가 오지 않습니다(헤더는 받음). 펌웨어/센서를 확인하세요.")
            time.sleep(0.01)

    def _wait_header(self, timeout_s: float) -> None:
        buf = bytearray()
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            buf.extend(self._ser.read(max(1, self._ser.in_waiting)))
            while b"\n" in buf:
                i = buf.index(b"\n")
                line = bytes(buf[:i]).strip().decode("ascii", "replace")
                del buf[:i + 1]
                if line.startswith("H,"):
                    p = line.split(",")
                    self.header = {"fw": p[1], "odr_hz": float(p[2]), "acc_mg_per_lsb": float(p[3]),
                                   "gyro_mdps_per_lsb": float(p[4]), "mag_mgauss_per_lsb": float(p[5])}
                    for extra in p[6:]:
                        k, _, v = extra.partition("=")
                        self.header[k] = int(v) if v.isdigit() else v
                    self._scale = (self.header["acc_mg_per_lsb"] / 1000.0,      # -> g
                                   self.header["gyro_mdps_per_lsb"] / 1000.0,   # -> dps
                                   self.header["mag_mgauss_per_lsb"] / 10.0)    # mgauss -> uT
                    return
                if line.startswith("E,"):
                    self.messages.append(line)
                    raise RuntimeError(f"장치 오류: {line}")
        self._ser.close()
        raise RuntimeError("장치 헤더를 받지 못했습니다. ring_imu 펌웨어가 올라가 있는지, "
                           "Arduino IDE 시리얼 모니터가 포트를 잡고 있지 않은지 확인하세요.")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._ser:
            try:
                self._ser.write(b"X")
                self._ser.flush()
                self._ser.close()
            except Exception:
                pass
            self._ser = None
        for fh in self._fh.values():
            fh.close()
        self._fh.clear()

    # ------------------------------------------------------------------ 스레드
    def _run(self) -> None:
        import serial
        buf = bytearray()
        last_flush = time.time()
        while not self._stop.is_set():
            try:
                chunk = self._ser.read(max(1, self._ser.in_waiting))
            except (serial.SerialException, OSError) as e:
                self.error = e
                return
            now = time.perf_counter_ns()
            if chunk:
                self.last_rx_ns = now
                buf.extend(chunk)
                while True:
                    i = buf.find(b"\n")
                    if i < 0:
                        break
                    line = bytes(buf[:i]).strip()
                    del buf[:i + 1]
                    if line:
                        self._handle(line, now)
            self._maybe_ping()
            if time.time() - last_flush > 1.0:
                for fh in self._fh.values():
                    fh.flush()
                last_flush = time.time()

    def _maybe_ping(self) -> None:
        t = time.time()
        if self._ping_sent is not None and t - self._last_ping > 0.5:
            self._ping_sent = None  # 응답 유실
        if self._ping_sent is None and t - self._last_ping >= self.ping_interval_s:
            self._last_ping = t
            self._ping_sent = time.perf_counter_ns()
            try:
                self._ser.write(b"P")
            except Exception as e:  # noqa: BLE001
                self.error = e

    def _handle(self, line: bytes, now: int) -> None:
        kind = line[:2]
        try:
            if kind == b"D,":
                p = line.split(b",")
                if len(p) != 9:
                    raise ValueError
                seq = int(p[1])
                t = self._unwrap(int(p[2]))
                sa, sg, _ = self._scale
                ax, ay, az = (int(v) * sa for v in p[3:6])
                gx, gy, gz = (int(v) * sg for v in p[6:9])
                if self._last_seq is not None and seq != self._last_seq + 1:
                    self.seq_gaps += 1
                self._last_seq = seq
                if self.first_t_us is None:
                    self.first_t_us = t
                self.last_t_us = t
                with self._lock:
                    self._recent.append((t, math.sqrt(gx * gx + gy * gy + gz * gz)))
                if "imu" in self._fh:
                    self._fh["imu"].write(f"{seq},{t},{ax:.5f},{ay:.5f},{az:.5f},{gx:.3f},{gy:.3f},{gz:.3f}\n")
                self.n_imu += 1
            elif kind == b"M,":
                p = line.split(b",")
                if len(p) != 5:
                    raise ValueError
                t = self._unwrap(int(p[1]))
                sm = self._scale[2]
                mx, my, mz = (int(v) * sm for v in p[2:5])
                if "mag" in self._fh:
                    self._fh["mag"].write(f"{t},{mx:.3f},{my:.3f},{mz:.3f}\n")
                self.n_mag += 1
            elif kind == b"P,":
                t = self._unwrap(int(line[2:]))
                if self._ping_sent is not None:
                    if "ping" in self._fh:
                        self._fh["ping"].write(f"{self._ping_sent},{now},{t}\n")
                    self.n_ping += 1
                    self._ping_sent = None
            elif kind in (b"H,", b"E,"):
                self.messages.append(line.decode("ascii", "replace"))
            else:
                raise ValueError
        except ValueError:
            self.bad_lines += 1

    # ------------------------------------------------------------------ 조회(GUI/세션용)
    def is_still(self, window_s: float = 0.3, gyro_thr_dps: float = 15.0) -> bool:
        """최근 window_s 동안 자이로 크기가 모두 임계 미만이면 True(샘플 80% 이상 있어야 함)."""
        with self._lock:
            if not self._recent:
                return False
            t_end = self._recent[-1][0]
            vals = [g for t, g in self._recent if t >= t_end - window_s * 1e6]
        need = 0.8 * window_s * (self.header or {}).get("odr_hz", 238.0)
        return len(vals) >= need and max(vals) < gyro_thr_dps

    def recent_gyro_peak(self, window_s: float) -> float:
        with self._lock:
            if not self._recent:
                return float("nan")
            t_end = self._recent[-1][0]
            vals = [g for t, g in self._recent if t >= t_end - window_s * 1e6]
        return max(vals) if vals else float("nan")

    def silence_s(self) -> float:
        if self.last_rx_ns is None:
            return float("inf")
        return (time.perf_counter_ns() - self.last_rx_ns) / 1e9

    def stats(self) -> dict:
        span = (self.last_t_us - self.first_t_us) / 1e6 if self.n_imu > 1 else 0.0
        return {"n_imu": self.n_imu, "n_mag": self.n_mag, "n_ping": self.n_ping,
                "duration_s": span, "rate_hz": (self.n_imu - 1) / span if span > 0 else float("nan"),
                "seq_gaps": self.seq_gaps, "bad_lines": self.bad_lines,
                "header": self.header, "messages": list(self.messages)}
