"""수집 진행 상태기계(UI 비의존). GUI는 on_key()/tick()을 부르고 view를 그리기만 한다.

시도(trial) 흐름
  key/word : settle(손가락이 F 위에서 멈출 때까지 대기) -> cue(표시, 입력 대기) -> 기록 -> 다음
             settle은 "정지 기준점"을 매 시도 확보하기 위한 것(Stage 1 body→world 변환의 정지 가정).
             장치가 정지를 확인하면 바로, 최대 settle_max_s 뒤에는 정지 미확인 플래그와 함께 cue를 띄운다.
  rest     : SPACE -> rest_s초 정지 측정. 자이로가 임계를 넘으면 다시.
  틀림/시간초과/움직임 -> 큐 끝에 재삽입(max_attempts까지).
시각은 모두 perf_counter_ns(키 로그, IMU 핑과 같은 시계).
"""
from __future__ import annotations

import csv
import math
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

from .plan import Item

TRIAL_COLUMNS = ["trial_id", "uid", "phase", "kind", "target", "expect", "attempt", "settle_ns", "cue_ns",
                 "still_at_cue", "first_key_ns", "end_ns", "typed", "status"]
KEY_COLUMNS = ["host_ns", "event", "key", "trial_id", "scan", "source"]
MODIFIERS = {"shift_l", "shift_r", "control_l", "control_r", "alt_l", "alt_r", "caps_lock", "win_l", "win_r",
             "super_l", "super_r", "meta_l", "meta_r", "num_lock", "hangul", "hangul_hanja", "kana", "hanja"}


def canon_key(keysym: str) -> str:
    if keysym == "space":
        return " "
    return keysym.lower()


def is_ime_key(key: str) -> bool:
    return (len(key) == 1 and ord(key) > 127) or key.startswith("hangul_") or key in ("process",)


@dataclass
class Params:
    settle_still_s: float = 0.30
    settle_gyro_dps: float = 15.0
    settle_max_s: float = 4.0
    min_gap_s: float = 0.5        # 이전 입력 후 다음 cue까지 최소 간격
    key_timeout_s: float = 5.0
    word_timeout_s: float = 25.0
    rest_s: float = 5.0
    rest_gyro_dps: float = 20.0
    max_attempts: int = 3


@dataclass
class View:
    kind: str = "info"   # info | settle | cue | rest | done
    title: str = ""
    big: str = ""
    sub: str = ""
    warn: str = ""
    status: str = ""
    progress: float = 0.0


class KeyLog:
    def __init__(self, path):
        self._fh = open(path, "w", encoding="utf-8", newline="")
        self._w = csv.writer(self._fh)
        self._w.writerow(KEY_COLUMNS)

    def write(self, host_ns: int, event: str, key: str, trial_id: int, scan: int = -1, source: str = "") -> None:
        self._w.writerow([host_ns, event, "space" if key == " " else key, trial_id, scan, source])

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


class Session:
    def __init__(self, plan: list[Item], params: Params, out_dir: str | Path, device=None):
        out_dir = Path(out_dir)
        self.p = params
        self.dev = device
        self.queue: deque[Item] = deque(plan)
        self.total = sum(1 for it in plan if it.kind != "info")
        self.keylog = KeyLog(out_dir / "keys.csv")
        self._fh = open(out_dir / "trials.csv", "w", encoding="utf-8", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=TRIAL_COLUMNS)
        self._w.writeheader()
        self.attempts: Counter = Counter()
        self.status_counts: Counter = Counter()
        self.ok_by_phase: Counter = Counter()
        self.n_ok = 0
        self.trial_id = 0
        self.stray_keys = 0
        self.unsettled = 0
        self.ime_suspected = False
        self.state = "init"
        self.cur: Item | None = None
        self.settle_ns = 0
        self.cue_ns = 0
        self.still_at_cue = False
        self.first_key_ns = 0
        self.last_end_ns = 0
        self.rest_end_ns = 0
        self.typed: list[str] = []
        self.done = False
        self.aborted = False
        self.view = View()
        self._warn = ""

    # ------------------------------------------------------------------ API
    def start(self, now: int) -> None:
        self._next(now)

    def on_key(self, down: bool, key: str, now: int, scan: int = -1, source: str = "") -> None:
        tid = self.trial_id if self.state in ("cue", "rest") else 0
        self.keylog.write(now, "down" if down else "up", key, tid, scan, source)
        if not down or key in MODIFIERS:
            return
        if is_ime_key(key):
            self.ime_suspected = True
            self._warn = "한글 입력 상태입니다! 한/영 키로 영문으로 바꾸세요."
            self._refresh()
            return
        if self.state == "info":
            if key == " ":
                self._next(now)
        elif self.state == "rest_ready":
            if key == " ":
                self._begin_rest(now)
        elif self.state == "cue":
            self._cue_key(key, now)
        else:
            self.stray_keys += 1

    def tick(self, now: int) -> None:
        if self.state == "settle":
            ready = now >= self.last_end_ns + int(self.p.min_gap_s * 1e9)
            still = self.dev is None or self.dev.is_still(self.p.settle_still_s, self.p.settle_gyro_dps)
            if ready and still:
                self._cue(now, True)
            elif (now - self.settle_ns) / 1e9 > self.p.settle_max_s:
                self.unsettled += 1
                self._cue(now, False)
        elif self.state == "cue":
            limit = self.p.word_timeout_s if self.cur.kind == "word" else self.p.key_timeout_s
            if (now - self.cue_ns) / 1e9 > limit:
                self._finish("timeout", now)
        elif self.state == "rest":
            if now >= self.rest_end_ns:
                if self.cur.kind == "timed":
                    self._finish("ok", now)
                    return
                peak = self.dev.recent_gyro_peak(self.p.rest_s) if self.dev is not None else 0.0
                ok = not math.isnan(peak) and peak < self.p.rest_gyro_dps
                self._finish("ok" if ok else "moved", now)
            else:
                self.view.big = f"{(self.rest_end_ns - now) / 1e9:.1f}"
        if self.dev is not None and self.state in ("settle", "cue", "rest") and self.dev.silence_s() > 1.0:
            self._warn = "센서 데이터가 1초 이상 끊겼습니다. 연결을 확인하세요."
            self._refresh()

    def abort(self) -> None:
        self.aborted = True
        self.close()

    def close(self) -> None:
        self.keylog.close()
        if not self._fh.closed:
            self._fh.close()

    # ------------------------------------------------------------------ 내부
    def _status(self) -> str:
        c = self.status_counts
        return (f"완료 {self.n_ok}/{self.total} · 틀림 {c['wrong'] + c['bad_word']} · "
                f"시간초과 {c['timeout']} · 정지실패 {c['moved']}")

    def _refresh(self) -> None:
        self.view.warn = self._warn
        self.view.status = self._status()
        self.view.progress = self.n_ok / max(1, self.total)

    def _set_view(self, kind, title, big, sub) -> None:
        self.view = View(kind, title, big, sub)
        self._refresh()

    def _next(self, now: int) -> None:
        if not self.queue:
            self.state = "done"
            self.done = True
            self._set_view("done", "완료", "수고하셨습니다", "")
            return
        it = self.cur = self.queue.popleft()
        title = {"isolated": "개별 키", "continuous": "연속 타이핑", "calib": "F 보정", "rest": "정지",
                 "axes": "축 확인"}.get(it.phase, "")
        if it.kind == "info":
            self.state = "info"
            self._set_view("info", title or "안내", it.target, "오른손으로 SPACE")
        elif it.kind == "rest":
            self.state = "rest_ready"
            self._set_view("info", "정지", "F 위에서 멈춤", f"준비되면 오른손으로 SPACE → {self.p.rest_s:g}초")
        elif it.kind == "timed":
            self.state = "rest_ready"
            self._set_view("info", title, "준비", f"오른손으로 SPACE → {it.target}초 동안 {it.hint}")
        else:
            self.state = "settle"
            self.settle_ns = now
            self._set_view("settle", title, "", "F 위로 돌아와 멈추세요")

    def _cue(self, now: int, still: bool) -> None:
        self.trial_id += 1
        self.state = "cue"
        self.cue_ns = now
        self.still_at_cue = still
        self.first_key_ns = 0
        self.typed = []
        self._warn = "" if still or self.ime_suspected else "손가락이 덜 멈췄습니다(기록은 계속)"
        it = self.cur
        self._set_view("cue", self.view.title, it.target, it.hint)

    def _begin_rest(self, now: int) -> None:
        self.trial_id += 1
        self.state = "rest"
        self.settle_ns = self.cue_ns = now
        self.still_at_cue = True
        self.first_key_ns = 0
        self.typed = []
        if self.cur.kind == "timed":
            dur = float(self.cur.target)
            self._set_view("rest", "축 확인", f"{dur:.1f}", self.cur.hint)
        else:
            dur = self.p.rest_s
            self._set_view("rest", "정지 측정", f"{dur:.1f}", "움직이지 마세요")
        self.rest_end_ns = now + int(dur * 1e9)

    def _cue_key(self, key: str, now: int) -> None:
        it = self.cur
        if not self.first_key_ns:
            self.first_key_ns = now
        if it.kind == "key":
            self.typed = [key]
            self._finish("ok" if key == it.expect else "wrong", now)
            return
        self.typed.append("<BS>" if key == "backspace" else key if len(key) == 1 else f"<{key}>")
        if key == " ":
            self._finish("ok" if "".join(self.typed) == it.expect else "bad_word", now)

    def _finish(self, status: str, now: int) -> None:
        it = self.cur
        self.attempts[it.uid] += 1
        att = self.attempts[it.uid]
        self._w.writerow({
            "trial_id": self.trial_id, "uid": it.uid, "phase": it.phase, "kind": it.kind, "target": it.target,
            "expect": it.expect, "attempt": att, "settle_ns": self.settle_ns, "cue_ns": self.cue_ns,
            "still_at_cue": int(self.still_at_cue), "first_key_ns": self.first_key_ns or "",
            "end_ns": now, "typed": "".join(self.typed), "status": status})
        self._fh.flush()
        self.status_counts[status] += 1
        if status == "ok":
            self.n_ok += 1
            self.ok_by_phase[it.phase] += 1
            if not self.ime_suspected:
                self._warn = ""
        else:
            if att < self.p.max_attempts and it.kind in ("rest", "timed"):
                self.queue.appendleft(it)  # 정지 측정은 바로 다시
            elif att < self.p.max_attempts:
                pos = len(self.queue)
                # 끝으로 보내되, 마지막 '정지/종료 안내' 앞에 넣는다
                while pos > 0 and self.queue[pos - 1].phase in ("rest", "outro", "axes"):
                    pos -= 1
                self.queue.insert(pos, it)
            msg = {"wrong": f"틀림: '{''.join(self.typed).upper()}' 입력", "bad_word": "단어가 다릅니다",
                   "timeout": "시간 초과", "moved": "정지 중 움직임 — 다시 측정"}[status]
            self._warn = msg + (" (나중에 다시)" if att < self.p.max_attempts else " (건너뜀)")
        self.last_end_ns = now
        self._next(now)
