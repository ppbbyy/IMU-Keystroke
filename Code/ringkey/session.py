"""수집 진행 상태기계(UI 비의존). GUI는 on_key/tick을 호출하고 view를 그리기만 한다.

흐름: info(스페이스로 진행) -> [iti(손 복귀 대기) -> cue(키 입력 대기)] 반복.
 - 틀린 입력/시간초과 항목은 큐 끝에 다시 넣는다(최대 max_attempts회).
 - 각 시도는 trials.csv에 한 줄. 시각은 모두 perf_counter_ns.
 - 키 입력 직후 IMU 자이로 피크로 센서 신호 유무를 즉석 점검한다(경고 표시용).
"""
from __future__ import annotations

import csv
import math
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .keylog import KeyLog
from .prompts import Prompt

TRIAL_COLUMNS = ["trial_id", "prompt_id", "phase", "block", "label", "expect", "attempt",
                 "cue_ns", "first_key_ns", "end_ns", "typed", "status", "gyro_peak_dps"]
MODIFIERS = {"shift_l", "shift_r", "control_l", "control_r", "alt_l", "alt_r", "caps_lock",
             "meta_l", "meta_r", "super_l", "super_r", "win_l", "win_r", "num_lock",
             "iso_level3_shift", "mode_switch", "command", "option"}


def canon_key(keysym: str) -> str:
    """Tk keysym -> 정규 키 이름. 문자키는 소문자 한 글자, 스페이스는 ' ', 나머지는 소문자 이름."""
    if keysym == "space":
        return " "
    if len(keysym) == 1:
        return keysym.lower()
    return keysym.lower()


def looks_like_ime(key: str) -> bool:
    return (len(key) == 1 and ord(key) > 127) or key.startswith("hangul")


@dataclass
class Params:
    iti_range: tuple[float, float] = (1.4, 2.0)  # 입력 후 다음 큐까지(손 복귀 시간) [s]
    first_iti: float = 1.0                       # info 직후 첫 큐까지 [s]
    key_timeout: float = 6.0
    word_timeout: float = 25.0
    max_attempts: int = 3
    check_after_s: float = 0.45
    pre_s: float = 0.3
    post_s: float = 0.4
    low_gyro_dps: float = 15.0                   # 이보다 작으면 "센서 신호 약함"
    low_streak_warn: int = 3


@dataclass
class View:
    kind: str = "info"       # info | iti | cue | end
    title: str = ""
    text: str = ""
    sub: str = ""
    progress: float = 0.0
    status: str = ""
    warn: str = ""


class Runner:
    def __init__(self, plan: list[Prompt], params: Params, keylog: KeyLog,
                 trials_path: str | Path, imu=None, seed: int = 0):
        self.params = params
        self.keylog = keylog
        self.imu = imu
        self.rng = np.random.default_rng(seed + 12345)
        self.queue: deque[Prompt] = deque(plan)
        self.total = sum(1 for p in plan if p.kind != "info")
        self.ok_by_phase: Counter = Counter()
        self.status_counts: Counter = Counter()
        self.attempts: Counter = Counter()
        self.n_ok = 0
        self.trial_id = 0
        self.stray_keys = 0
        self.ime_suspected = False
        self.low_streak = 0
        self.low_signal_warned = False   # 런 중 "센서 신호 약함" 경고가 한 번이라도 떴는가
        self.pending: list[tuple[int, dict, int, int]] = []
        self.state = "init"
        self.cur: Prompt | None = None
        self.cue_ns = 0
        self.cue_at = 0
        self._typed: list[str] = []
        self._first_ns = 0
        self._after_info = True
        self.done = False
        self.aborted = False
        self.view = View()
        self._fh = open(trials_path, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._fh, fieldnames=TRIAL_COLUMNS)
        self._w.writeheader()

    # ------------------------------------------------------------------ 외부 API
    def begin(self, now: int) -> None:
        self._advance(now)

    def on_key(self, down: bool, key: str, now: int) -> None:
        tid = self.trial_id if self.state == "cue" else 0
        self.keylog.write(now, "down" if down else "up", key, tid)
        if not down:
            return
        if looks_like_ime(key):
            self.ime_suspected = True
            self._set_warn("한글 입력 상태입니다. 한/영 키로 영문(QWERTY)으로 바꾸세요.")
        if key in MODIFIERS:
            return
        if self.state == "info":
            if key == " ":
                self._advance(now)
        elif self.state == "cue":
            self._on_cue_key(key, now)
        else:  # iti 등: 큐 밖 입력
            self.stray_keys += 1

    def tick(self, now: int) -> None:
        if self.state == "iti" and now >= self.cue_at:
            self._start_cue(now)
        elif self.state == "cue":
            limit = self.params.word_timeout if self.cur.kind == "word" else self.params.key_timeout
            if (now - self.cue_ns) / 1e9 > limit:
                self._finish_trial("timeout", now)
        self._resolve_pending(now)
        if self.state == "finishing" and not self.pending:
            self.done = True
            self.view = View("end", "완료", "수고하셨습니다", "", 1.0, self._status_line())

    def abort(self) -> None:
        self.aborted = True
        self.close()

    def close(self) -> None:
        for _, row, _, _ in self.pending:
            row["gyro_peak_dps"] = "nan"
            self._w.writerow(row)
        self.pending.clear()
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    # ------------------------------------------------------------------ 내부
    def _status_line(self) -> str:
        c = self.status_counts
        return (f"완료 {self.n_ok}/{self.total}  | 재시도 대기 {sum(1 for p in self.queue if p.kind != 'info')}"
                f"  | 오입력 {c['wrong'] + c['bad_word']}  시간초과 {c['timeout']}")

    def _set_warn(self, msg: str) -> None:
        self.view.warn = msg

    def _advance(self, now: int) -> None:
        if not self.queue:
            self.state = "finishing"
            return
        p = self.queue.popleft()
        self.cur = p
        if p.kind == "info":
            self._after_info = True
            self.state = "info"
            self.view = View("info", p.phase if p.phase != "info" else "안내", p.label,
                             "[SPACE]를 눌러 계속", self.n_ok / max(1, self.total), self._status_line())
            return
        dur = self.params.first_iti if self._after_info else float(self.rng.uniform(*self.params.iti_range))
        self._after_info = False
        self.cue_at = now + int(dur * 1e9)
        self.state = "iti"
        self.view = View("iti", p.phase, "F 위로 복귀", "손을 멈추고 기다리세요",
                         self.n_ok / max(1, self.total), self._status_line(), self.view.warn)

    def _start_cue(self, now: int) -> None:
        p = self.cur
        self.trial_id += 1
        self.cue_ns = now
        self._typed = []
        self._first_ns = 0
        self.state = "cue"
        self.view = View("cue", p.phase, p.label, p.hint, self.n_ok / max(1, self.total),
                         self._status_line(), self.view.warn)

    def _on_cue_key(self, key: str, now: int) -> None:
        p = self.cur
        if not self._first_ns:
            self._first_ns = now
        if p.kind == "key":
            self._typed = [key]
            self._finish_trial("ok" if key == p.expect else "wrong", now)
            return
        self._typed.append("<BS>" if key == "backspace" else key)
        if key == " ":
            ok = "".join(self._typed) == p.expect
            self._finish_trial("ok" if ok else "bad_word", now)

    def _finish_trial(self, status: str, now: int) -> None:
        p = self.cur
        self.attempts[p.prompt_id] += 1
        att = self.attempts[p.prompt_id]
        typed = "".join(self._typed)
        row = {"trial_id": self.trial_id, "prompt_id": p.prompt_id, "phase": p.phase, "block": p.block,
               "label": p.label, "expect": p.expect, "attempt": att, "cue_ns": self.cue_ns,
               "first_key_ns": self._first_ns or "", "end_ns": now if self._first_ns else "",
               "typed": typed, "status": status, "gyro_peak_dps": ""}
        self.status_counts[status] += 1
        if status == "ok":
            self.n_ok += 1
            self.ok_by_phase[p.phase] += 1
        elif att < self.params.max_attempts:
            self.queue.append(p)  # 나중에 다시
        if self._first_ns and self.imu is not None:
            t0 = self._first_ns - int(self.params.pre_s * 1e9)
            t1 = now + int(self.params.post_s * 1e9)
            self.pending.append((now + int(self.params.check_after_s * 1e9), row, t0, t1))
        else:
            row["gyro_peak_dps"] = "nan"
            self._w.writerow(row)
        self._advance(now)

    def _resolve_pending(self, now: int) -> None:
        keep = []
        for due, row, t0, t1 in self.pending:
            if now < due:
                keep.append((due, row, t0, t1))
                continue
            peak = self.imu.recent_peak_gyro(t0, t1)
            row["gyro_peak_dps"] = "nan" if math.isnan(peak) else f"{peak:.1f}"
            self._w.writerow(row)
            if row["status"] == "ok" and row["phase"] in ("isolated", "calib"):
                if math.isnan(peak) or peak < self.params.low_gyro_dps:
                    self.low_streak += 1
                else:
                    self.low_streak = 0
                if self.low_streak >= self.params.low_streak_warn:
                    self.low_signal_warned = True
                    self._set_warn("센서 신호가 약합니다. 링 착용/케이블/전원을 확인하세요.")
                elif not self.ime_suspected:
                    self._set_warn("")
        self.pending = keep
        if self.imu is not None and self.state in ("cue", "iti") and self.imu.last_sample_age_s() > 1.0:
            self._set_warn("센서 데이터가 멈췄습니다(1초 이상). 연결을 확인하세요.")
