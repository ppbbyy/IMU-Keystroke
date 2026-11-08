"""tkinter 수집 화면.

키 입력: Windows에서는 저수준 키보드 훅(winkeys.KeyHook)으로 물리 키를 받는다 -> 한글 IME 상태와 무관.
수집 창이 맨 앞일 때의 키만 받는다. 그 외 OS(또는 keyboard="tk")에서는 Tk 키 이벤트를 쓴다.
"""
from __future__ import annotations

import sys
import time
import tkinter as tk

from .session import Session, canon_key

COLORS = {"info": "#1e2230", "settle": "#3a3320", "cue": "#12664a", "rest": "#203a5a", "done": "#1e2230"}
FONT = "Malgun Gothic"


class CollectWindow:
    def __init__(self, session: Session, device, title: str, keyboard: str = "auto"):
        self.s, self.dev = session, device
        self.down: set[str] = set()
        self._esc_t = 0.0
        self._sig = None
        self._last_stat = 0.0
        self.hook = None
        self.blurred = 0
        r = self.root = tk.Tk()
        r.title(title)
        r.geometry("1100x660")
        self.w_title = tk.Label(r, font=(FONT, 18), fg="#9aa4bf")
        self.w_big = tk.Label(r, font=(FONT, 110, "bold"), fg="white", wraplength=1040, justify="center")
        self.w_sub = tk.Label(r, font=(FONT, 26), fg="#cfd6ea")
        self.w_warn = tk.Label(r, font=(FONT, 20, "bold"), fg="#ffb347", wraplength=1040)
        self.bar = tk.Canvas(r, height=10, bg="#3a4056", highlightthickness=0)
        self.w_stat = tk.Label(r, font=(FONT, 14), fg="#9aa4bf")
        self.w_dev = tk.Label(r, font=(FONT, 12), fg="#7a839d")
        self.w_title.pack(pady=(20, 0))
        self.w_big.pack(expand=True)
        self.w_sub.pack()
        self.w_warn.pack(pady=8)
        self.bar.pack(fill="x", padx=40, pady=(8, 4))
        self.w_stat.pack()
        self.w_dev.pack(pady=(0, 10))
        self._labels = (self.w_title, self.w_big, self.w_sub, self.w_warn, self.w_stat, self.w_dev)
        r.protocol("WM_DELETE_WINDOW", self._abort)
        r.update()
        use_hook = keyboard == "hook" or (keyboard == "auto" and sys.platform == "win32")
        if use_hook:
            from .winkeys import KeyHook, disable_ime
            hwnd = int(r.wm_frame(), 16)
            disable_ime(hwnd)
            disable_ime(r.winfo_id())
            self.hook = KeyHook(hwnd)
            self.hook.start()
            self.source = "hook"
        else:
            r.bind("<KeyPress>", lambda e: self._key(True, canon_key(e.keysym), time.perf_counter_ns(), -1))
            r.bind("<KeyRelease>", lambda e: self._key(False, canon_key(e.keysym), time.perf_counter_ns(), -1))
            self.source = "tk"
        r.after(200, r.focus_force)

    def run(self) -> bool:
        self.s.start(time.perf_counter_ns())
        self.root.after(5, self._tick)
        try:
            self.root.mainloop()
        finally:
            if self.hook:
                self.hook.stop()
        try:
            self.root.destroy()
        except tk.TclError:
            pass
        return self.s.done and not self.s.aborted

    def _key(self, down: bool, key: str, now: int, scan: int) -> None:
        if key == "escape":
            if down:
                if time.time() - self._esc_t < 1.0:
                    self._abort()
                self._esc_t = time.time()
            return
        if down:
            if key in self.down:  # 자동 반복 무시
                return
            self.down.add(key)
        else:
            self.down.discard(key)
        self.s.on_key(down, key, now, scan, self.source)

    def _abort(self) -> None:
        self.s.aborted = True
        self.root.quit()

    def _tick(self) -> None:
        if self.hook:
            for now, down, key, scan, fg, _inj in self.hook.drain():
                if fg:
                    self._key(down, key, now, scan)
                elif down:
                    self.blurred += 1
        self.s.tick(time.perf_counter_ns())
        v = self.s.view
        sig = (v.kind, v.title, v.big, v.sub, v.warn, v.status, round(v.progress, 3))
        if sig != self._sig:
            self._sig = sig
            bg = COLORS.get(v.kind, COLORS["info"])
            self.root.configure(bg=bg)
            for w in self._labels:
                w.configure(bg=bg)
            size = 110 if len(v.big) <= 3 else 72 if len(v.big) <= 6 else 26
            self.w_title.configure(text=v.title)
            self.w_big.configure(text=v.big, font=(FONT, size, "bold"))
            self.w_sub.configure(text=v.sub)
            self.w_warn.configure(text=v.warn)
            self.w_stat.configure(text=v.status)
            self.bar.delete("all")
            self.bar.create_rectangle(0, 0, self.bar.winfo_width() * v.progress, 10, fill="#4cc38a", width=0)
        if self.dev is not None and time.time() - self._last_stat > 0.5:
            self._last_stat = time.time()
            st = self.dev.stats()
            still = "정지" if self.dev.is_still(0.3, 15.0) else "움직임"
            focus = "" if not self.hook or self._is_front() else "  |  ⚠ 이 창을 클릭해 맨 앞으로"
            self.w_dev.configure(text=f"IMU {st['rate_hz']:.0f}Hz · 핑 {st['n_ping']} · 손가락 {still} · "
                                      f"키입력 {self.source}{focus}   |   중단: Esc 두 번")
        if self.s.done:
            self.root.after(1500, self.root.quit)
            return
        self.root.after(5, self._tick)

    def _is_front(self) -> bool:
        try:
            import ctypes
            u = ctypes.windll.user32
            u.GetAncestor.restype = ctypes.c_void_p
            u.GetForegroundWindow.restype = ctypes.c_void_p
            return u.GetAncestor(u.GetForegroundWindow(), 2) == u.GetAncestor(ctypes.c_void_p(self.hook.hwnd), 2)
        except Exception:  # noqa: BLE001
            return True
