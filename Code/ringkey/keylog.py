"""키보드 이벤트 로그(keys.csv). 시각은 IMU와 같은 perf_counter_ns 시계."""
from __future__ import annotations

from pathlib import Path

KEY_COLUMNS = ["host_ns", "event", "key", "trial_id"]


class KeyLog:
    def __init__(self, path: str | Path):
        self._fh = open(path, "w", encoding="utf-8", buffering=1 << 14)
        self._fh.write(",".join(KEY_COLUMNS) + "\n")
        self.n = 0

    def write(self, host_ns: int, event: str, key: str, trial_id: int) -> None:
        k = key if key not in (",", '"') else {"," : "comma", '"': "quote"}[key]
        k = "space" if k == " " else k
        self._fh.write(f"{host_ns},{event},{k},{trial_id}\n")
        self.n += 1
        if self.n % 20 == 0:
            self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()
