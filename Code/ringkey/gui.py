"""tkinter 수집 화면. 터미널 에코/IME 간섭을 피하려고 창에 포커스를 두고 키 이벤트를 직접 받는다."""
from __future__ import annotations

import time
import tkinter as tk

from .session import Runner, canon_key

_BG = {"info": "#1e2230", "iti": "#2b2f3d", "cue": "#12664a", "end": "#1e2230"}


class CollectApp:
    def __init__(self, runner: Runner, imu, title: str):
        self.runner, self.imu = runner, imu
        self.down: set[str] = set()
        self._esc_t = 0.0
        self._last_view = None
        self._last_stats = 0.0
        r = self.root = tk.Tk()
        r.title(title)
        r.geometry("1100x640")
        r.configure(bg=_BG["info"])
        self.l_title = tk.Label(r, text="", font=("", 18), fg="#9aa4bf", bg=_BG["info"])
        self.l_text = tk.Label(r, text="", font=("", 96, "bold"), fg="white", bg=_BG["info"],
                               wraplength=1000, justify="center")
        self.l_sub = tk.Label(r, text="", font=("", 26), fg="#cfd6ea", bg=_BG["info"])
        self.l_warn = tk.Label(r, text="", font=("", 20, "bold"), fg="#ffb347", bg=_BG["info"],
                               wraplength=1000)
        self.l_stat = tk.Label(r, text="", font=("", 14), fg="#9aa4bf", bg=_BG["info"])
        self.l_imu = tk.Label(r, text="", font=("", 12), fg="#7a839d", bg=_BG["info"])
        self.bar = tk.Canvas(r, height=10, bg="#3a4056", highlightthickness=0)
        self.l_title.pack(pady=(24, 0))
        self.l_text.pack(expand=True)
        self.l_sub.pack()
        self.l_warn.pack(pady=10)
        self.bar.pack(fill="x", padx=40, pady=(10, 4))
        self.l_stat.pack()
        self.l_imu.pack(pady=(0, 10))
        r.bind("<KeyPress>", self._on_down)
        r.bind("<KeyRelease>", self._on_up)
        r.protocol("WM_DELETE_WINDOW", self._abort)
        r.focus_force()

    def run(self) -> bool:
        """창을 띄우고 끝날 때까지 실행. 정상 완료면 True, 중단이면 False."""
        self.runner.begin(time.perf_counter_ns())
        self.root.after(10, self._tick)
        self.root.mainloop()
        return self.runner.done and not self.runner.aborted

    def _on_down(self, e) -> None:
        now = time.perf_counter_ns()
        key = canon_key(e.keysym)
        if key == "escape":
            if time.time() - self._esc_t < 1.0:
                self._abort()
            self._esc_t = time.time()
            return
        if key in self.down:  # 키 반복 무시
            return
        self.down.add(key)
        self.runner.on_key(True, key, now)

    def _on_up(self, e) -> None:
        now = time.perf_counter_ns()
        key = canon_key(e.keysym)
        self.down.discard(key)
        self.runner.on_key(False, key, now)

    def _abort(self) -> None:
        self.runner.aborted = True
        self.root.quit()

    def _tick(self) -> None:
        now = time.perf_counter_ns()
        self.runner.tick(now)
        v = self.runner.view
        sig = (v.kind, v.title, v.text, v.sub, v.warn, v.status, round(v.progress, 3))
        if sig != self._last_view:
            self._last_view = sig
            bg = _BG.get(v.kind, _BG["info"])
            self.root.configure(bg=bg)
            for w in (self.l_title, self.l_text, self.l_sub, self.l_warn, self.l_stat, self.l_imu):
                w.configure(bg=bg)
            self.l_title.configure(text=v.title)
            self.l_text.configure(text=v.text, font=("", 96 if len(v.text) <= 4 else 64, "bold"))
            self.l_sub.configure(text=v.sub)
            self.l_warn.configure(text=v.warn)
            self.l_stat.configure(text=v.status)
            self.bar.delete("all")
            self.bar.create_rectangle(0, 0, self.bar.winfo_width() * v.progress, 10, fill="#4cc38a", width=0)
        if self.imu is not None and time.time() - self._last_stats > 0.5:
            self._last_stats = time.time()
            s = self.imu.stats()
            self.l_imu.configure(text=f"IMU {s['rate_hz']:.0f}Hz · 샘플 {s['n']} · 손실 {s['dropped']}"
                                      f"  |  중단: Esc 두 번")
        if self.runner.done:
            self.root.after(1200, self.root.quit)
            return
        self.root.after(10, self._tick)
