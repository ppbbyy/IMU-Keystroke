"""Arduino 시리얼 스트림 리더: 호스트 시각(perf_counter_ns) 부착, micros 래핑 해제, 손실 검출, CSV 저장."""
from __future__ import annotations

import collections
import math
import threading
import time
from pathlib import Path

import numpy as np

IMU_COLUMNS = ["host_ns", "seq", "t_us_raw", "t_us", "ax", "ay", "az",
               "gx", "gy", "gz", "mx", "my", "mz", "mag_new"]
ARDUINO_VID = 0x2341
_WRAP = 1 << 32


def find_port() -> str | None:
    """Arduino(VID 0x2341)로 보이는 시리얼 포트. 없으면 None."""
    from serial.tools import list_ports
    cands = [p.device for p in list_ports.comports() if p.vid == ARDUINO_VID]
    return cands[0] if cands else None


class IMUReader:
    def __init__(self, port: str, out_csv: str | Path | None = None, baud: int = 115200,
                 buffer_s: float = 12.0, serial_factory=None):
        self.port, self.baud = port, baud
        self._serial_factory = serial_factory  # 테스트용: (port, baud, timeout=) -> serial 유사 객체
        self.out_csv = Path(out_csv) if out_csv else None
        self.info: list[str] = []          # H,/E, 줄
        self.error: Exception | None = None
        self.n = 0
        self.dropped = 0
        self.bad_lines = 0
        self.first_t_us: int | None = None
        self.last_t_us: int | None = None
        self.first_host_ns: int | None = None
        self.last_host_ns: int | None = None
        self._last_seq: int | None = None
        self._last_raw: int | None = None
        self._wrap = 0
        self._ser = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._recent: collections.deque = collections.deque(maxlen=int(buffer_s * 250))
        self._fh = None
        self._last_flush = 0.0

    # ------------------------------------------------------------------ 수명주기
    def start(self, timeout: float = 5.0) -> None:
        import serial
        factory = self._serial_factory or serial.Serial
        self._ser = factory(self.port, self.baud, timeout=0.05)
        self._ser.reset_input_buffer()
        if self.out_csv:
            self._fh = open(self.out_csv, "w", encoding="utf-8", buffering=1 << 16)
            self._fh.write(",".join(IMU_COLUMNS) + "\n")
        self._ser.write(b"IG")  # I: 헤더 재요청(부팅 때 나간 헤더는 포트를 열기 전이라 놓침), G: 스트리밍 시작
        self._ser.flush()
        self._thread = threading.Thread(target=self._run, name="imu-reader", daemon=True)
        self._thread.start()
        t0 = time.time()
        while self.n == 0:
            if self.error:
                raise RuntimeError(f"시리얼 오류: {self.error}")
            if time.time() - t0 > timeout:
                self.stop()
                raise RuntimeError("센서에서 데이터가 오지 않습니다. 포트/펌웨어/케이블을 확인하세요.")
            time.sleep(0.02)

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
        if self._fh:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    # ------------------------------------------------------------------ 스레드
    def _run(self) -> None:
        import serial
        buf = bytearray()
        while not self._stop.is_set():
            try:
                chunk = self._ser.read(max(1, self._ser.in_waiting))
            except (serial.SerialException, OSError) as e:
                self.error = e
                return
            if not chunk:
                continue
            now = time.perf_counter_ns()  # 같은 chunk의 줄은 같은 도착시각(오프셋은 sync.py가 보정)
            buf.extend(chunk)
            while True:
                i = buf.find(b"\n")
                if i < 0:
                    break
                line = bytes(buf[:i]).strip()
                del buf[:i + 1]
                if line:
                    self._handle(line, now)
            if self._fh and time.time() - self._last_flush > 1.0:
                self._fh.flush()
                self._last_flush = time.time()

    def _handle(self, line: bytes, host_ns: int) -> None:
        if line[:2] != b"D,":
            if line[:2] in (b"H,", b"E,"):
                self.info.append(line.decode("ascii", "replace"))
            return
        p = line.split(b",")
        if len(p) != 13:
            self.bad_lines += 1
            return
        try:
            seq, t_raw = int(p[1]), int(p[2])
            ax, ay, az = (int(v) / 1000.0 for v in p[3:6])      # g
            gx, gy, gz = (int(v) / 1000.0 for v in p[6:9])      # deg/s
            mx, my, mz = (int(v) / 1000.0 for v in p[9:12])     # uT
            mag_new = int(p[12])
        except ValueError:
            self.bad_lines += 1
            return
        if self._last_raw is not None and t_raw < self._last_raw and self._last_raw - t_raw > _WRAP // 2:
            self._wrap += _WRAP
        self._last_raw = t_raw
        t_us = t_raw + self._wrap
        if self._last_seq is not None and seq > self._last_seq + 1:
            self.dropped += seq - self._last_seq - 1
        self._last_seq = seq
        if self.first_t_us is None:
            self.first_t_us, self.first_host_ns = t_us, host_ns
        self.last_t_us, self.last_host_ns = t_us, host_ns
        gn = math.sqrt(gx * gx + gy * gy + gz * gz)
        with self._lock:
            self._recent.append((host_ns, gn))
        if self._fh:
            self._fh.write(f"{host_ns},{seq},{t_raw},{t_us},{ax:.4f},{ay:.4f},{az:.4f},"
                           f"{gx:.3f},{gy:.3f},{gz:.3f},{mx:.3f},{my:.3f},{mz:.3f},{mag_new}\n")
        self.n += 1

    # ------------------------------------------------------------------ 조회
    def recent_peak_gyro(self, host_t0_ns: int, host_t1_ns: int) -> float:
        """[t0,t1] 구간(호스트 도착시각 기준, 근사)의 자이로 크기 최댓값 [deg/s]. 데이터 없으면 nan."""
        with self._lock:
            v = [g for t, g in self._recent if host_t0_ns <= t <= host_t1_ns]
        return max(v) if v else float("nan")

    def last_sample_age_s(self) -> float:
        if self.last_host_ns is None:
            return float("inf")
        return (time.perf_counter_ns() - self.last_host_ns) / 1e9

    def stats(self) -> dict:
        span = ((self.last_t_us - self.first_t_us) / 1e6) if self.n > 1 else 0.0
        return {
            "n": self.n, "dropped": self.dropped, "bad_lines": self.bad_lines,
            "duration_s": span, "rate_hz": (self.n - 1) / span if span > 0 else float("nan"),
            "drop_frac": self.dropped / max(1, self.n + self.dropped),
            "fw_info": list(self.info),
        }

    @staticmethod
    def dt_jitter_ms(t_us: np.ndarray) -> tuple[float, float]:
        dt = np.diff(t_us) / 1e3
        return float(np.median(dt)), float(np.std(dt))
