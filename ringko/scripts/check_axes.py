#!/usr/bin/env python3
"""자이로 축 부호 검증: 정지 구간 쌍 사이의 중력 방향 변화(가속도계 실측) vs 자이로 적분 예측.

  python scripts/check_axes.py <세션폴더>

8가지 부호 조합의 예측 오차를 출력. 'axes' 단계(손을 여러 방향으로 돌리고 멈추기)가 있으면 세 축 모두 판정되고,
없으면 중력 축(이 착용에서는 거의 x) 둘레 회전은 판정할 수 없다. 결과는 ringko/frames.py GYRO_SIGN과 비교.
"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ringko.dataset import load_session  # noqa: E402
from ringko.frames import GYRO_SIGN  # noqa: E402


def rot(v):
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sd = load_session(sys.argv[1])
    t = sd.imu.t_us.values / 1e6
    A = sd.imu[["ax", "ay", "az"]].values
    W = np.radians(sd.imu[["gx", "gy", "gz"]].values)
    gn = np.degrees(np.linalg.norm(W, axis=1))
    an = np.linalg.norm(A, axis=1)
    n = 23  # 약 0.1초
    gmax = np.lib.stride_tricks.sliding_window_view(np.r_[np.full(n, 1e9), gn], n + 1).max(1)
    aptp = np.ptp(np.lib.stride_tricks.sliding_window_view(np.r_[np.full(n, an[0]), an], n + 1), axis=1)
    still = (gmax < 3) & (aptp < 0.01)
    segs, i = [], 0
    while i < len(t):
        if still[i]:
            j = i
            while j < len(t) and still[j]:
                j += 1
            segs.append((i, j))
            i = j
        else:
            i += 1
    bias = np.median(W[still], axis=0)
    pairs = []
    for ia, (a0, a1) in enumerate(segs):
        for b0, b1 in segs[ia + 1:ia + 4]:
            ga, gb = A[a0:a1].mean(0), A[b0:b1].mean(0)
            ga, gb = ga / np.linalg.norm(ga), gb / np.linalg.norm(gb)
            ang = np.degrees(np.arccos(np.clip(ga @ gb, -1, 1)))
            if ang > 1.0 and t[b0] - t[a1 - 1] < 10:
                pairs.append((a1 - 1, b0, ga, gb, ang))
    if not pairs:
        sys.exit("회전이 있는 정지 구간 쌍이 없습니다.")
    gdirs = np.array([A[a0:a1].mean(0) / np.linalg.norm(A[a0:a1].mean(0)) for a0, a1 in segs])
    print(f"정지 구간 {len(segs)}개 · 1° 이상 회전한 쌍 {len(pairs)}개 · 회전 중앙 {np.median([p[4] for p in pairs]):.1f}° · "
          f"정지 중 중력 방향 분산(축별 std) {np.round(gdirs.std(0), 3)}")
    res = []
    for signs in itertools.product([1, -1], repeat=3):
        s = np.array(signs, float)
        errs = []
        for i0, i1, ga, gb, _ in pairs:
            R = np.eye(3)
            for i in range(i0, i1):
                R = R @ rot(s * (W[i] - bias) * (t[i + 1] - t[i]))
            errs.append(np.degrees(np.arccos(np.clip((R.T @ ga) @ gb, -1, 1))))
        res.append((float(np.median(errs)), signs))
    res.sort()
    for e, s in res:
        mark = "  <- frames.GYRO_SIGN" if tuple(s) == tuple(GYRO_SIGN.astype(int)) else ""
        print(f"부호 {s}: 중력 예측 오차 중앙 {e:.2f}°{mark}")
    best, second = res[0], res[1]
    if second[0] - best[0] < 0.2:
        print(f"주의: 1·2위 차이 {second[0] - best[0]:.2f}° — 판정 불충분(다른 축 둘레 회전이 부족)")


if __name__ == "__main__":
    main()
