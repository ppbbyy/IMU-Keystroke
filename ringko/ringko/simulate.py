"""검증용 센서 조건 모의 — 실제 데이터(SessionData)의 원시 IMU를 물리적으로 일관되게 바꾼다.

실측 조건(같은 센서·같은 노트북, 재착용 w2)은 데이터가 있지만 '다른 센서', '부착 방향 크게 변경'은 아직 없으므로
알고리즘의 불변성을 먼저 모의로 확인한다. 모의는 실측을 대신하지 못한다(docs/consonant3d.md 8장 프로토콜 참고).

  mount   : 센서를 다르게 붙임 = body 좌표계 고정 회전 R (가속도·각속도 모두 회전; 원시 자이로 부호 규약 유지)
  sensor  : 다른 IMU 흉내 = 축 순서/부호 규약 변경(가속도 오른손 유지, 자이로 부호 규약 +1), 배율 오차, 바이어스,
            추가 잡음, 출력 주기 변경(리샘플)
타격 검출(기존 코드)도 바뀐 데이터를 그대로 보므로, 검출이 받는 영향까지 함께 측정된다.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .frames import GYRO_SIGN
from .motion import rotvec_to_mat


def _set(sd, acc_g, gyr_raw_dps):
    sd.imu[["ax", "ay", "az"]] = acc_g
    sd.imu[["gx", "gy", "gz"]] = gyr_raw_dps
    return sd


def mount_rotation(sd, angles_deg=(25.0, -35.0, 120.0), seed=None):
    """센서 부착 방향 변경. angles_deg: 회전 벡터[deg](seed를 주면 무작위 방향, 크기 = |angles_deg|)."""
    sd = copy.deepcopy(sd)
    v = np.radians(np.asarray(angles_deg, float))
    if seed is not None:
        rng = np.random.default_rng(seed)
        u = rng.normal(size=3)
        v = u / np.linalg.norm(u) * np.linalg.norm(v)
    R = rotvec_to_mat(v)
    acc = sd.imu[["ax", "ay", "az"]].to_numpy() @ R.T
    w_phys = sd.imu[["gx", "gy", "gz"]].to_numpy() * GYRO_SIGN
    return _set(sd, acc, (w_phys @ R.T) * GYRO_SIGN)


def other_sensor(sd, odr_hz: float | None = 119.0, acc_scale=(1.03, 0.98, 1.02), acc_bias_g=(0.02, -0.015, 0.01),
                 gyro_scale=(0.97, 1.02, 1.01), gyro_bias_dps=(1.0, -0.8, 0.5), noise_g=0.004, noise_dps=0.3,
                 perm=(2, 0, 1), seed=0):
    """다른 IMU: 축 순서 perm(가속도·자이로 같은 순열 — 오른손 유지하도록 순환 순열), 자이로 부호 규약 +1,
    축별 배율·바이어스·잡음, 출력 주기 odr_hz(None이면 그대로)."""
    sd = copy.deepcopy(sd)
    rng = np.random.default_rng(seed)
    acc = sd.imu[["ax", "ay", "az"]].to_numpy()[:, perm]
    w = (sd.imu[["gx", "gy", "gz"]].to_numpy() * GYRO_SIGN)[:, perm]       # 물리(오른손) 각속도
    acc = acc * acc_scale + acc_bias_g + rng.normal(0, noise_g, acc.shape)
    w = w * gyro_scale + gyro_bias_dps + rng.normal(0, noise_dps, w.shape)
    sd = _set(sd, acc, w)                                                    # 부호 규약 +1 (원시 = 물리)
    if odr_hz:
        im = sd.imu
        t = im.t_us.to_numpy(float)
        tq = np.arange(t[0], t[-1], 1e6 / odr_hz)
        new = {"seq": np.arange(len(tq)), "t_us": tq}
        for c in ["ax", "ay", "az", "gx", "gy", "gz", "host_ns"]:
            new[c] = np.interp(tq, t, im[c].to_numpy(float))
        sd.imu = pd.DataFrame(new)
    return sd


SENSOR_GYRO_SIGN = np.array([1.0, 1.0, 1.0])   # other_sensor의 원시 자이로 부호 규약(정답; 알고리즘은 auto로 추정해야 함)
