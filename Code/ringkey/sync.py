"""IMU(장치 시계) <-> 키보드 로그(호스트 시계) 오프셋 추정.

세션 시작/끝에 스페이스바를 세게 5번 치는 구간(sync_start / sync_end)을 쓴다.
키를 칠 때 손가락이 건반에 부딪히는 순간 가속도에 날카로운 스파이크가 생기므로,
각 키 시각 근처에서 가속도 변화량(‖Δa‖)의 피크를 찾아 offset = key_time - peak_time 의 중앙값을 취한다.
Tk 이벤트 지연 등 일정한 지연은 같은 경로를 지나는 sync 키에 똑같이 들어가므로 상쇄된다.

결과 모델:  host_key_s = t_dev_s + offset(t_dev_s)   (offset은 시작/끝 두 점으로 선형 보간)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SyncResult:
    offset_s: float
    t_center_dev_s: float   # 이 오프셋이 대표하는 장치 시각
    n_used: int
    n_keys: int
    mad_ms: float
    per_key_ms: list[float]
    prominence: list[float]


def _motion_energy(t: np.ndarray, acc: np.ndarray) -> np.ndarray:
    da = np.linalg.norm(np.diff(acc, axis=0), axis=1)
    e = np.concatenate([[0.0], da])
    return np.convolve(e, np.ones(3) / 3, mode="same")


def estimate_offset(host_s: np.ndarray, t_dev_s: np.ndarray, acc: np.ndarray, key_s: np.ndarray,
                    search_s: float = 0.35, min_prominence: float = 4.0) -> SyncResult | None:
    """host_s: 각 IMU 줄의 호스트 도착시각, t_dev_s: 장치 시각, acc: (N,3) [g], key_s: 키 호스트 시각들."""
    energy = _motion_energy(t_dev_s, acc)
    noise = float(np.median(energy)) + 1e-9
    offs, peaks_t, proms = [], [], []
    for k in key_s:
        near = np.abs(host_s - k) < 15.0
        if near.sum() < 50:
            continue
        coarse = np.percentile(host_s[near] - t_dev_s[near], 5)  # 도착 지연의 하한 포락선
        center = k - coarse
        m = np.where(np.abs(t_dev_s - center) <= search_s)[0]
        if len(m) < 5:
            continue
        i = m[int(np.argmax(energy[m]))]
        prom = float(energy[i] / noise)
        tp = float(t_dev_s[i])
        if 0 < i < len(t_dev_s) - 1:  # 포물선 보간
            y0, y1, y2 = energy[i - 1], energy[i], energy[i + 1]
            den = y0 - 2 * y1 + y2
            if den < 0:
                tp += 0.5 * (y0 - y2) / den * float(np.median(np.diff(t_dev_s[i - 1:i + 2])))
        proms.append(prom)
        if prom >= min_prominence:
            offs.append(k - tp)
            peaks_t.append(tp)
    if not offs:
        return None
    offs_a = np.asarray(offs)
    med = float(np.median(offs_a))
    mad = float(np.median(np.abs(offs_a - med)))
    return SyncResult(med, float(np.mean(peaks_t)), len(offs), len(key_s), mad * 1e3,
                      [float(o * 1e3) for o in (offs_a - med)], proms)


def analyze_run(run_dir: str | Path, max_mad_ms: float = 15.0) -> dict:
    """run 디렉터리의 imu.csv/trials.csv로 시계 모델을 만들어 dict 반환(meta.json['sync']에 저장)."""
    run_dir = Path(run_dir)
    imu = pd.read_csv(run_dir / "imu.csv")
    trials = pd.read_csv(run_dir / "trials.csv")
    host_s = imu["host_ns"].to_numpy() / 1e9
    t_dev = imu["t_us"].to_numpy() / 1e6
    acc = imu[["ax", "ay", "az"]].to_numpy()

    res: dict[str, SyncResult] = {}
    for ph in ("sync_start", "sync_end"):
        sel = trials[(trials["phase"] == ph) & (trials["status"] == "ok")]
        keys = pd.to_numeric(sel["first_key_ns"], errors="coerce").dropna().to_numpy() / 1e9
        if len(keys) >= 3:
            r = estimate_offset(host_s, t_dev, acc, keys)
            if r is not None:
                res[ph] = r

    out: dict = {"groups": {k: asdict(v) for k, v in res.items()}, "ok": False, "model": None, "warnings": []}
    if not res:
        out["warnings"].append("동기화 구간에서 스파이크를 찾지 못함 — 시계 오프셋 추정 실패")
        return out
    for ph, r in res.items():
        if r.mad_ms > max_mad_ms:
            out["warnings"].append(f"{ph}: 키별 오프셋 산포 MAD {r.mad_ms:.1f}ms > {max_mad_ms}ms")
        if r.n_used < 3:
            out["warnings"].append(f"{ph}: 유효 sync 키 {r.n_used}개(<3)")
    a = res.get("sync_start")
    b = res.get("sync_end")
    if a and b and b.t_center_dev_s > a.t_center_dev_s:
        slope = (b.offset_s - a.offset_s) / (b.t_center_dev_s - a.t_center_dev_s)
        out["model"] = {"offset0_s": a.offset_s, "t0_dev_s": a.t_center_dev_s, "slope": slope}
        out["drift_ppm"] = slope * 1e6
        if abs(b.offset_s - a.offset_s) * 1e3 > 30:
            out["warnings"].append(f"시작/끝 오프셋 차이 {abs(b.offset_s - a.offset_s) * 1e3:.0f}ms: 시계 드리프트/이상")
    else:
        only = a or b
        out["model"] = {"offset0_s": only.offset_s, "t0_dev_s": only.t_center_dev_s, "slope": 0.0}
        out["warnings"].append("시작/끝 sync 중 하나뿐: 드리프트 미추정(slope=0)")
    out["ok"] = not any("MAD" in w or "실패" in w or "(<3)" in w for w in out["warnings"])
    return out


def dev_to_host_s(t_dev_s, model: dict):
    """장치 시각(초) -> 키보드 로그와 같은 호스트 시계(초)."""
    t = np.asarray(t_dev_s, dtype=float)
    return t + model["offset0_s"] + model["slope"] * (t - model["t0_dev_s"])


def save_sync(run_dir: str | Path, result: dict) -> None:
    (Path(run_dir) / "sync.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
