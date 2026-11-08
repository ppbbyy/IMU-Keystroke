"""장치 시계(micros) -> 호스트 시계(perf_counter_ns) 변환 모델.

핑 한 번 = (send_ns, recv_ns, t_us). 장치가 'P'를 받은 순간은 [send, recv] 안에 있으므로
왕복시간(RTT)이 짧은 핑일수록 정확하다(파일럿 실측 RTT ~0.8ms).

모델 = 전역 직선 + 구간별 보정
  1) RTT 하위 분위 핑으로 host = a + b·(t - t0) 직선 적합(b ≈ 1000 ns/us, 차이 = 발진기 드리프트)
  2) 직선 잔차를 이웃 핑들에 대한 국소 직선 적합으로 매끄럽게 만들어 매듭(knot)으로 두고 선형 보간
발진기가 온도 등으로 흔들려도(파일럿: 내부 RC 사용 시 직선 잔차 최대 36ms) 따라간다.
"""
from __future__ import annotations

import numpy as np


def _local_linear(t: np.ndarray, x: np.ndarray, k: int) -> np.ndarray:
    """각 점에서 이웃 k개(가장자리는 한쪽으로 채움)에 직선을 맞춰 그 점의 값을 추정. 기울기 구간에서도 치우침 없음."""
    n = len(t)
    out = np.empty(n)
    for i in range(n):
        lo = min(max(0, i - k // 2), max(0, n - k))
        tt, xx = t[lo:lo + k], x[lo:lo + k]
        if np.ptp(tt) == 0:
            out[i] = np.median(xx)
            continue
        b, a = np.polyfit(tt - t[i], xx, 1)
        out[i] = a
    return out


def fit_clock(send_ns, recv_ns, t_us, keep_quantile: float = 0.8, min_pings: int = 5,
              smooth_pings: int = 5) -> dict:
    # 기본값 근거(파일럿 실데이터, 짝수 핑으로 적합·홀수 핑으로 검증): 중앙 0.21ms, p95 0.57ms, 최대 0.85ms.
    # 직선만 쓰면 잔차 최대 36ms였음.
    send = np.asarray(send_ns, dtype=np.float64)
    recv = np.asarray(recv_ns, dtype=np.float64)
    t = np.asarray(t_us, dtype=np.float64)
    if len(t) < min_pings:
        raise ValueError(f"핑이 {len(t)}개뿐(최소 {min_pings})")
    order = np.argsort(t)
    send, recv, t = send[order], recv[order], t[order]
    rtt = recv - send
    thr = max(np.quantile(rtt, keep_quantile), rtt.min() + 0.5e6)
    m = rtt <= thr
    if m.sum() < min_pings:
        m = np.zeros(len(t), bool)
        m[np.argsort(rtt)[:min_pings]] = True
    tg, mid = t[m], (send[m] + recv[m]) / 2
    t0 = float(tg[0])
    if np.ptp(tg) < 1e6:
        b = 1000.0
        a = float(np.median(mid - b * (tg - t0)))
    else:
        b, a = np.polyfit(tg - t0, mid, 1)
    lin_resid = mid - (a + b * (tg - t0))
    knots = (_local_linear(tg, lin_resid, smooth_pings) if len(tg) >= smooth_pings
             else np.full(len(tg), np.median(lin_resid)))
    resid = lin_resid - np.interp(tg, tg, knots)
    return {"t0_us": t0, "a_ns": float(a), "b_ns_per_us": float(b),
            "knot_t_us": tg.tolist(), "knot_off_ns": knots.tolist(),
            "drift_ppm": float((b / 1000.0 - 1.0) * 1e6),
            "n_pings": int(len(t)), "n_used": int(m.sum()),
            "rtt_ms_median": float(np.median(rtt) / 1e6), "rtt_ms_min": float(rtt.min() / 1e6),
            "linear_resid_ms_max": float(np.abs(lin_resid).max() / 1e6),
            "wander_ms": float(np.ptp(knots) / 1e6),
            "resid_ms_std": float(np.std(resid) / 1e6), "resid_ms_max": float(np.abs(resid).max() / 1e6),
            "method": "ping-linear+knots"}


def dev_to_host_ns(t_us, model: dict) -> np.ndarray:
    t = np.asarray(t_us, dtype=np.float64)
    base = model["a_ns"] + model["b_ns_per_us"] * (t - model["t0_us"])
    if model.get("knot_t_us"):
        base = base + np.interp(t, model["knot_t_us"], model["knot_off_ns"])
    return base
