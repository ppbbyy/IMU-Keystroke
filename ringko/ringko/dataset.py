"""수집 세션 로드 + 품질 점검 + 시도별 윈도우 추출.

세션 폴더: imu.csv mag.csv ping.csv keys.csv trials.csv plan.csv meta.json
로드하면 IMU/자력계에 host_ns(키 로그와 같은 시계) 열이 붙는다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .clock import dev_to_host_ns, fit_clock
from .hangul import CONSONANTS


@dataclass
class SessionData:
    path: Path
    meta: dict
    imu: pd.DataFrame
    mag: pd.DataFrame
    ping: pd.DataFrame
    keys: pd.DataFrame
    trials: pd.DataFrame
    clock: dict


def load_session(path: str | Path) -> SessionData:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    imu = pd.read_csv(path / "imu.csv")
    mag = pd.read_csv(path / "mag.csv")
    ping = pd.read_csv(path / "ping.csv")
    keys = pd.read_csv(path / "keys.csv", keep_default_na=False, dtype={"key": str})
    trials = pd.read_csv(path / "trials.csv", keep_default_na=False, dtype={"typed": str, "expect": str})
    for c in ("first_key_ns",):
        trials[c] = pd.to_numeric(trials[c], errors="coerce")
    clock = fit_clock(ping.send_ns, ping.recv_ns, ping.t_us)
    imu["host_ns"] = dev_to_host_ns(imu.t_us, clock)
    if len(mag):
        mag["host_ns"] = dev_to_host_ns(mag.t_us, clock)
    return SessionData(path, meta, imu, mag, ping, keys, trials, clock)


# --------------------------------------------------------------------------- 신호 도우미
def jerk(imu: pd.DataFrame) -> np.ndarray:
    """가속도 변화량 |Δa| [g/sample]. 키 타격 순간 날카롭게 튄다."""
    a = imu[["ax", "ay", "az"]].to_numpy()
    return np.concatenate([[0.0], np.linalg.norm(np.diff(a, axis=0), axis=1)])


def gaps(t_us: np.ndarray) -> tuple[int, int]:
    """(끊김 구간 수, 추정 누락 샘플 수). 간격이 실측 주기(중앙값)의 1.5배를 넘으면 끊김.
    LSM9DS1 실제 ODR은 공칭(238Hz)과 수 % 다르므로(예: 229Hz) 실측 주기를 쓴다."""
    dt = np.diff(np.asarray(t_us, dtype=float))
    if not len(dt):
        return 0, 0
    period = float(np.median(dt))
    big = dt > 1.5 * period
    return int(big.sum()), int(np.round(dt[big] / period - 1).sum())


def impact_lags_ms(sd: SessionData, phases=("calib", "isolated"), search_ms=(-150.0, 100.0)) -> np.ndarray:
    """정상 시도마다 (키다운 시각 기준) 가속도 타격 피크까지의 시간차[ms]. 음수 = 피크가 키 이벤트보다 먼저."""
    t = sd.imu.host_ns.to_numpy()
    j = jerk(sd.imu)
    out = []
    sel = sd.trials[(sd.trials.status == "ok") & sd.trials.phase.isin(phases)]
    for k in sel.first_key_ns.dropna().to_numpy():
        lo, hi = np.searchsorted(t, [k + search_ms[0] * 1e6, k + search_ms[1] * 1e6])
        if hi - lo < 5:
            continue
        i = lo + int(np.argmax(j[lo:hi]))
        out.append((t[i] - k) / 1e6)
    return np.asarray(out)


def trial_windows(sd: SessionData, phase: str = "isolated", pre_s: float = 0.8, post_s: float = 0.8,
                  only_ok: bool = True) -> list[dict]:
    """시도별 IMU 구간. 기준 시각 = 첫 키다운. 단어는 마지막 입력(SPACE)까지 + post_s."""
    t = sd.imu.host_ns.to_numpy()
    rows = sd.trials[sd.trials.phase == phase]
    if only_ok:
        rows = rows[rows.status == "ok"]
    out = []
    for _, r in rows.iterrows():
        if np.isnan(r.first_key_ns):
            continue
        t0 = r.first_key_ns - pre_s * 1e9
        t1 = (r.end_ns if r.kind == "word" else r.first_key_ns) + post_s * 1e9
        lo, hi = np.searchsorted(t, [t0, t1])
        win = sd.imu.iloc[lo:hi].copy()
        win["t_rel_s"] = (win.host_ns - r.first_key_ns) / 1e9
        kd = sd.keys[(sd.keys.trial_id == r.trial_id) & (sd.keys.event == "down")]
        out.append({"trial": r, "imu": win, "key_down_rel_s": ((kd.host_ns - r.first_key_ns) / 1e9).to_numpy(),
                    "keys": kd.key.tolist()})
    return out


# --------------------------------------------------------------------------- 품질 보고
def quality_report(sd: SessionData) -> tuple[list[str], list[str]]:
    L, W = [], []
    m, c = sd.meta, sd.clock
    odr = (m.get("device") or {}).get("odr_hz", 238.0)
    t = sd.imu.t_us.to_numpy()
    dur = (t[-1] - t[0]) / 1e6
    rate = (len(t) - 1) / dur if dur > 0 else float("nan")
    ng, nmiss = gaps(t)
    L.append(f"세션: {sd.path}")
    L.append(f"피험자 {m.get('subject')} · 착용 {m.get('wear')} · 위치 {m.get('site')} · 완료 {m.get('complete')}")
    L.append(f"IMU {len(t)}샘플 · {dur / 60:.1f}분 · 실측 {rate:.1f}Hz(공칭 {odr:g}, 실제 ODR "
             f"{1e6 / np.median(np.diff(t)):.1f}) · 끊김 {ng}회/누락 약 {nmiss}샘플 ({nmiss / max(1, len(t) + nmiss):.2%})")
    L.append(f"자력계 {len(sd.mag)}샘플 · 핑 {c['n_pings']}개(사용 {c['n_used']}) · RTT 중앙 {c['rtt_ms_median']:.2f}ms · "
             f"시계 잔차 {c['resid_ms_std']:.2f}ms(최대 {c['resid_ms_max']:.2f}) · 드리프트 {c['drift_ppm']:.0f}ppm · "
             f"흔들림 {c['wander_ms']:.1f}ms")
    L.append(f"펌웨어 {(m.get('device') or {}).get('fw')} · HFXO {(m.get('device') or {}).get('hfxo')} · "
             f"키 입력 {m.get('keyboard_source')}")
    if (m.get("device") or {}).get("hfxo") != 1:
        W.append("HFXO 꺼짐: 장치 시계가 부정확(구간 보정으로 맞추지만 fw-2 업로드 권장)")
    if m.get("keyboard_source") not in (None, "hook"):
        W.append("키 입력이 Tk 이벤트: 한글 IME 상태면 키다운이 깨질 수 있음")
    if abs(rate - odr) / odr > 0.10:
        W.append(f"실측 레이트 {rate:.1f}Hz가 공칭 {odr:g}Hz와 10% 이상 다름")
    if nmiss / max(1, len(t)) > 0.005:
        W.append("누락 샘플 0.5% 초과: USB 허브/케이블/다른 프로그램 확인")
    if c["resid_ms_max"] > 3.0:
        W.append(f"시계 잔차 최대 {c['resid_ms_max']:.1f}ms > 3ms")
    if c["wander_ms"] > 5.0:
        W.append(f"장치 시계 흔들림 {c['wander_ms']:.1f}ms (구간 보정으로 맞췄지만 HFXO 확인)")

    a = sd.imu[["ax", "ay", "az"]].abs().to_numpy()
    sat = (a > 7.9).any(axis=1).mean()
    g = sd.imu[["gx", "gy", "gz"]].abs().to_numpy()
    gsat = (g > 1990).any(axis=1).mean()
    L.append(f"포화: 가속도 {sat:.3%} · 자이로 {gsat:.3%}")
    if sat > 0.001 or gsat > 0.001:
        W.append("센서 포화 샘플 존재(±8g/±2000dps): 타격이 매우 강함")

    tr = sd.trials
    ok = tr[tr.status == "ok"]
    L.append("시도 상태: " + ", ".join(f"{k} {v}" for k, v in tr.status.value_counts().items()))
    L.append("단계별 정상: " + ", ".join(f"{k} {v}" for k, v in ok.phase.value_counts().items()))
    iso = ok[ok.phase == "isolated"]
    if len(iso):
        cnt = iso.target.value_counts()
        L.append("자음별 정상 개별 키: " + " ".join(f"{c_}{int(cnt.get(c_, 0))}" for c_ in CONSONANTS))
        keyed = tr[tr.kind.isin(["key", "word"])]
        L.append(f"cue 시점 정지 확인 비율: {keyed.still_at_cue.astype(int).mean():.1%}")
        if keyed.still_at_cue.astype(int).mean() < 0.8:
            W.append("cue 시점 정지 확인이 80% 미만: F 복귀 후 잠깐 멈추도록 안내 필요")
    lag = impact_lags_ms(sd)
    if len(lag):
        L.append(f"키다운 대비 타격 피크: 중앙 {np.median(lag):+.1f}ms · IQR {np.subtract(*np.percentile(lag, [75, 25])):.1f}ms"
                 f" (n={len(lag)})")
        if np.subtract(*np.percentile(lag, [75, 25])) > 40:
            W.append("타격 피크-키 이벤트 간격 산포가 큼(IQR>40ms): 키 라벨 정렬 확인 필요")
    rest = tr[tr.phase == "rest"]
    for _, r in rest[rest.status == "ok"].iterrows():
        seg = sd.imu[(sd.imu.host_ns >= r.cue_ns) & (sd.imu.host_ns <= r.end_ns)]
        if len(seg):
            an = np.linalg.norm(seg[["ax", "ay", "az"]].to_numpy(), axis=1)
            L.append(f"정지 구간(trial {r.trial_id}): |a| {an.mean():.3f}±{an.std():.4f} g · "
                     f"자이로 평균 {np.round(seg[['gx', 'gy', 'gz']].mean().to_numpy(), 2)} dps")
    if m.get("ime_suspected"):
        W.append("한글 입력 상태 키 입력 흔적: 해당 부분 라벨 확인")
    return L, W
