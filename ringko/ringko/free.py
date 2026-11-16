"""자유 입력: 제시 단어·키 로그 없이 IMU만으로 단어 경계를 찾아 기존 파이프라인에 넘긴다.

입력 규칙(수집과 같음): 왼손 검지를 F 위에 두고 잠깐 멈춤 → 단어(자음 = 왼손 검지, 모음·SPACE = 오른손) → F로 돌아와 멈춤.
  단어 사이에는 F 위에서 0.5초 이상 멈춘다. (SPACE는 오른손이 치므로 왼손 링은 볼 수 없다 — 단어의 끝은 'F 위 정지'로 안다.)

segment_words   자이로 크기가 still_dps 아래로 min_still_s 이상 유지되는 구간 = 정지. 정지와 정지 사이 = 단어 후보.
virtual_session 단어 후보를 trials 표의 '가상 시도'(phase=continuous, status=ok, cue_ns = 시작 정지 안, end_ns = 끝 정지 시작)로
                넣은 SessionData 사본을 만든다. 그러면 타격 검출(features.detect_presses / refine_presses, 수정 없음)과
                track3d / infer3d가 수집 세션과 똑같이 돈다. 타격이 하나도 없는 후보(손만 움직인 구간)는 추론 단계에서 빠진다.
typed_words     (검증용) 키 로그가 있으면 SPACE로 끊어 실제로 친 단어와 그 시각을 복원한다. 추론에는 쓰지 않는다.

한계
  · 단어 중간에 왼손이 오래 멈추면(모음을 여러 개 치는 동안 등) 그 지점에서 단어가 둘로 잘린다.
  · 멈춘 곳이 F인지 다른 키 위인지는 정지만으로는 모른다 → 규칙('단어가 끝나면 F로')을 지켜야 한다.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .hangul import LEFT_KEYS, compose, is_typable

TRIAL_COLS = ["trial_id", "uid", "phase", "kind", "target", "expect", "attempt", "settle_ns", "cue_ns",
              "still_at_cue", "first_key_ns", "end_ns", "typed", "status"]


def segment_words(t: np.ndarray, gyr_dps: np.ndarray, still_dps: float = 15.0, min_still_s: float = 0.4,
                  min_word_s: float = 0.12, max_word_s: float = 12.0, lead_s: float = 0.15) -> pd.DataFrame:
    """t[s], gyr_dps (N,3) 자이로[dps](부호 무관). 반환 열: start(움직임 시작), end(끝 정지 시작), cue(= start − lead_s,
    시작 정지 안), still_before, still_after [s]."""
    gn = np.linalg.norm(gyr_dps, axis=1)
    still = gn < still_dps
    d = np.diff(np.r_[0, still.astype(np.int8), 0])
    a, b = np.where(d == 1)[0], np.where(d == -1)[0]            # 정지 런 [a, b)
    keep = (t[np.minimum(b, len(t) - 1)] - t[a]) >= min_still_s
    a, b = a[keep], b[keep]
    rows = []
    for k in range(len(a) - 1):
        s, e = b[k], a[k + 1]                                    # 움직임 [s, e)
        dur = t[e] - t[s]
        if not (min_word_s <= dur <= max_word_s):
            continue
        sb = t[b[k] - 1] - t[a[k]]
        rows.append({"start": t[s], "end": t[e], "cue": max(t[s] - lead_s, t[a[k]]), "still_before": sb,
                     "still_after": t[min(b[k + 1], len(t) - 1)] - t[a[k + 1]]})
    return pd.DataFrame(rows, columns=["start", "end", "cue", "still_before", "still_after"])


def longest_still(t, gyr_dps, still_dps: float = 15.0):
    """가장 긴 정지 런 (t0, t1)[s] — 세션 기준 자세(rest)로 쓴다."""
    still = np.linalg.norm(gyr_dps, axis=1) < still_dps
    d = np.diff(np.r_[0, still.astype(np.int8), 0])
    a, b = np.where(d == 1)[0], np.where(d == -1)[0]
    if not len(a):
        return float(t[0]), float(t[min(len(t) - 1, 200)])
    i = int(np.argmax(t[np.minimum(b, len(t) - 1)] - t[a]))
    return float(t[a[i]]), float(t[min(b[i], len(t) - 1)])


def virtual_session(sd, t_range: tuple[float, float] | None = None, **seg_kw):
    """sd의 사본: IMU만으로 찾은 단어 후보가 trials(가상 시도)에 들어 있고 keys는 비어 있다.
    t_range (t0, t1)[s, host]: 그 시간 범위의 IMU만 본다(수집 세션의 연속 타이핑 구간만 검증할 때).
    반환 (가상 SessionData, 단어 후보 표(segment_words + trial_id))."""
    v = copy.copy(sd)
    imu = sd.imu
    if t_range is not None:
        m = (imu.host_ns >= t_range[0] * 1e9) & (imu.host_ns <= t_range[1] * 1e9)
        imu = imu[m.to_numpy()].reset_index(drop=True)
    v.imu = imu
    t = imu.host_ns.to_numpy() / 1e9
    g = imu[["gx", "gy", "gz"]].to_numpy()
    seg = segment_words(t, g, **seg_kw)
    seg.insert(0, "trial_id", np.arange(1, len(seg) + 1))
    r0, r1 = longest_still(t, g)
    rows = [{"trial_id": 0, "uid": 0, "phase": "rest", "kind": "rest", "target": "", "expect": "", "attempt": 1,
             "settle_ns": int(r0 * 1e9), "cue_ns": int(r0 * 1e9), "still_at_cue": 1, "first_key_ns": np.nan,
             "end_ns": int(r1 * 1e9), "typed": "", "status": "ok"}]
    for x in seg.itertuples():
        rows.append({"trial_id": int(x.trial_id), "uid": int(x.trial_id), "phase": "continuous", "kind": "word",
                     "target": "", "expect": "", "attempt": 1, "settle_ns": int(x.cue * 1e9), "cue_ns": int(x.cue * 1e9),
                     "still_at_cue": 1, "first_key_ns": np.nan, "end_ns": int(x.end * 1e9), "typed": "", "status": "ok"})
    v.trials = pd.DataFrame(rows, columns=TRIAL_COLS)
    v.keys = pd.DataFrame({"host_ns": pd.Series(dtype="int64"), "event": pd.Series(dtype=str),
                           "key": pd.Series(dtype=str), "trial_id": pd.Series(dtype="int64")})
    return v, seg


def plausible_word(track, pitch_mm: float = 19.0, max_u: float = 6.0, max_closure_mm: float = 400.0) -> bool:
    """추적 결과가 '키보드 위 단어'로 볼 만한가. 왼손 자음 키는 F에서 최대 3.4u — 추정 위치가 max_u(키 단위)를 넘거나
    F 복귀 닫힘 오차가 터무니없이 크면 손을 옮기거나 쉬는 동작이다. 그런 후보는 정렬 추정과 결과에서 뺀다."""
    if not np.isfinite(track.P).all():
        return False
    if np.linalg.norm(track.P[:, :2], axis=1).max() > max_u * pitch_mm:
        return False
    return not (track.has_return and np.linalg.norm(track.closure) > max_closure_mm)


# --------------------------------------------------------------------------- 검증용: 키 로그 → 친 단어
def typed_words(keys: pd.DataFrame) -> pd.DataFrame:
    """키 로그(down)를 SPACE로 끊어 친 단어를 복원. 열: word(조합한 한글), keys, t_first/t_last(왼손 키 첫/끝 시각[s]),
    n_left(자음 타격 수), ok(입력 가능한 한글 단어인가). 왼손 키가 없는 덩어리는 버린다."""
    kd = keys[keys.event == "down"].sort_values("host_ns")
    rows, cur = [], []

    def close():
        left = [(ns, k) for ns, k in cur if k in LEFT_KEYS]
        if left:
            ks = "".join(k for _, k in cur if len(k) == 1)
            w = compose(ks)
            rows.append({"word": w, "keys": ks, "t_first": left[0][0] / 1e9, "t_last": left[-1][0] / 1e9,
                         "n_left": len(left), "ok": is_typable(w)})
        cur.clear()

    for ns, k in zip(kd.host_ns.to_numpy(), kd.key.to_numpy()):
        if k in ("space", " ", "return", "enter"):
            close()
        else:
            cur.append((int(ns), str(k)))
    close()
    return pd.DataFrame(rows, columns=["word", "keys", "t_first", "t_last", "n_left", "ok"])


def match_truth(seg: pd.DataFrame, truth: pd.DataFrame, tol_s: float = 0.15) -> pd.DataFrame:
    """단어 후보 ↔ 실제 단어(t_first, t_last) 짝짓기. seg에 열 추가: word(정확히 한 단어의 모든 자음 타격을 담으면 그 단어),
    n_truth(겹치는 실제 단어 수). 실제 단어 쪽 결과는 attrs['truth']: 열 seg_id(그 단어를 온전히 담은 후보, 없으면 -1), n_seg."""
    seg = seg.copy()
    words, ntr = [], []
    tr = truth.copy()
    tr["seg_id"], tr["n_seg"] = -1, 0
    for x in seg.itertuples():
        ov = truth[(truth.t_last >= x.start - tol_s) & (truth.t_first <= x.end + tol_s)]
        full = ov[(ov.t_first >= x.start - tol_s) & (ov.t_last <= x.end + tol_s)]
        ntr.append(len(ov))
        words.append(full.word.iloc[0] if len(ov) == 1 and len(full) == 1 else "")
        tr.loc[ov.index, "n_seg"] += 1
        if words[-1]:
            tr.loc[full.index, "seg_id"] = int(x.trial_id)
    seg["word"], seg["n_truth"] = words, ntr
    seg.attrs["truth"] = tr
    return seg


# --------------------------------------------------------------------------- 자유 입력 녹화(화면 없는 부분)
class FreeRecorder:
    """자유 입력 세션 파일: trials.csv(머리글만), keys.csv(검증용 키 로그; keylog=False면 머리글만), meta.json.
    IMU·자력계·핑은 device.Device가 같은 폴더에 쓴다. 제시 단어가 없으므로 시도(trial)도 없다."""

    def __init__(self, out_dir, meta: dict, keylog: bool = True):
        import csv
        import json
        from pathlib import Path

        from .session import KeyLog
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / "trials.csv", "w", encoding="utf-8", newline="") as fh:
            csv.writer(fh).writerow(TRIAL_COLS)
        self.keylog = KeyLog(self.dir / "keys.csv")
        self.log_keys = keylog
        self.meta = dict(meta, phases=["free"], keylog=bool(keylog), complete=False)
        self._json = json
        self.cur: list[str] = []          # 지금 치는 단어의 키
        self.words: list[str] = []        # 끝낸 단어(조합한 한글) — 화면 표시용
        self.n_keys = 0
        self.save_meta()

    def save_meta(self):
        (self.dir / "meta.json").write_text(self._json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def on_key(self, down: bool, key: str, now_ns: int, scan: int = -1, source: str = "") -> None:
        if not self.log_keys:
            return
        self.keylog.write(now_ns, "down" if down else "up", key, 0, scan, source)
        if not down:
            return
        self.n_keys += 1
        if key in (" ", "space", "return"):
            if self.cur:
                self.words.append(compose("".join(self.cur)))
                self.cur = []
        elif key == "backspace":
            self.cur = self.cur[:-1]
        elif len(key) == 1:
            self.cur.append(key)

    def close(self, **meta) -> None:
        self.keylog.close()
        self.meta.update(meta, complete=True, n_keys=self.n_keys)
        self.save_meta()
