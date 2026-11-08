"""Windows 저수준 키보드 훅(WH_KEYBOARD_LL).

왜: 한글 IME가 켜져 있으면 Tk가 받는 키다운이 엉뚱한 글자로, 수 초 늦게 들어온다(파일럿 실측:
'd'를 누르면 키업은 'd'로 즉시, 키다운은 다음 cue 때 'g'로). 저수준 훅은 IME보다 먼저
물리 키의 가상키(VK)/스캔코드를 받으므로 입력기 상태와 무관하고, 타임스탬프도 Tk 이벤트 큐를 거치지 않는다.

이벤트: (host_ns, down, key, scan, foreground)
  key: 'a'..'z', '0'..'9', ' ', 'backspace', 'escape', 그 외 'vk<번호>'
  foreground: 그 순간 수집 창이 맨 앞이었는가
"""
from __future__ import annotations

import ctypes
import queue
import sys
import threading
import time
from ctypes import wintypes

WH_KEYBOARD_LL = 13
WM_KEYDOWN, WM_KEYUP, WM_SYSKEYDOWN, WM_SYSKEYUP = 0x100, 0x101, 0x104, 0x105
WM_QUIT = 0x12
LLKHF_INJECTED = 0x10
_SPECIAL = {0x20: " ", 0x08: "backspace", 0x1B: "escape", 0x0D: "return", 0x09: "tab",
            0x10: "shift", 0xA0: "shift_l", 0xA1: "shift_r", 0x11: "control", 0xA2: "control_l",
            0xA3: "control_r", 0x12: "alt", 0xA4: "alt_l", 0xA5: "alt_r", 0x14: "caps_lock",
            0x15: "hangul", 0x19: "hanja", 0x5B: "win_l", 0x5C: "win_r"}


def vk_name(vk: int) -> str:
    if 0x41 <= vk <= 0x5A:
        return chr(vk).lower()
    if 0x30 <= vk <= 0x39:
        return chr(vk)
    return _SPECIAL.get(vk, f"vk{vk}")


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("vkCode", wintypes.DWORD), ("scanCode", wintypes.DWORD), ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class KeyHook:
    def __init__(self, hwnd: int | None = None, swallow_injected: bool = False):
        """hwnd: 수집 창(맨 앞 여부 판정용). swallow_injected: 테스트용, 주입한 키가 다른 창으로 새지 않게 차단."""
        if sys.platform != "win32":
            raise OSError("KeyHook은 Windows 전용")
        self.hwnd = hwnd
        self.swallow_injected = swallow_injected
        self.events: queue.Queue = queue.Queue()
        self._tid = None
        self._ready = threading.Event()
        self._err: Exception | None = None
        self._thread: threading.Thread | None = None
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        u = self._user32
        self._HOOKPROC = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
        u.SetWindowsHookExW.argtypes = (ctypes.c_int, self._HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD)
        u.SetWindowsHookExW.restype = wintypes.HHOOK
        u.CallNextHookEx.argtypes = (wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
        u.CallNextHookEx.restype = ctypes.c_ssize_t
        u.UnhookWindowsHookEx.argtypes = (wintypes.HHOOK,)
        u.GetMessageW.argtypes = (ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT)
        u.PostThreadMessageW.argtypes = (wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
        u.GetForegroundWindow.restype = wintypes.HWND
        u.GetAncestor.argtypes = (wintypes.HWND, wintypes.UINT)
        u.GetAncestor.restype = wintypes.HWND
        self._kernel32.GetModuleHandleW.restype = wintypes.HMODULE
        self._proc = self._HOOKPROC(self._callback)  # 참조 유지(GC 방지)
        self._hook = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="key-hook", daemon=True)
        self._thread.start()
        self._ready.wait(3.0)
        if self._err:
            raise self._err
        if not self._hook:
            raise RuntimeError("키보드 훅 설치 실패")

    def stop(self) -> None:
        if self._tid:
            self._user32.PostThreadMessageW(self._tid, WM_QUIT, 0, 0)
        if self._thread:
            self._thread.join(2.0)

    def _run(self) -> None:
        u = self._user32
        self._tid = self._kernel32.GetCurrentThreadId()
        self._hook = u.SetWindowsHookExW(WH_KEYBOARD_LL, self._proc, self._kernel32.GetModuleHandleW(None), 0)
        if not self._hook:
            self._err = ctypes.WinError(ctypes.get_last_error())
        self._ready.set()
        if not self._hook:
            return
        msg = wintypes.MSG()
        while u.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            pass
        u.UnhookWindowsHookEx(self._hook)

    def _callback(self, code, wparam, lparam):
        now = time.perf_counter_ns()
        if code >= 0:
            k = ctypes.cast(lparam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
            down = wparam in (WM_KEYDOWN, WM_SYSKEYDOWN)
            fg = True
            if self.hwnd:
                top = self._user32.GetAncestor(self._user32.GetForegroundWindow(), 2)  # GA_ROOT
                fg = bool(top) and top == self._user32.GetAncestor(self.hwnd, 2)
            injected = bool(k.flags & LLKHF_INJECTED)
            self.events.put((now, down, vk_name(k.vkCode), int(k.scanCode), fg, injected))
            if injected and self.swallow_injected:
                return 1
        return self._user32.CallNextHookEx(self._hook, code, wparam, lparam)

    def drain(self) -> list[tuple]:
        out = []
        while True:
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                return out


def disable_ime(hwnd: int) -> bool:
    """창(과 자식 창)의 IME를 끈다. 실패해도 훅이 있으므로 치명적이지 않다."""
    if sys.platform != "win32":
        return False
    try:
        imm = ctypes.WinDLL("imm32")
        imm.ImmAssociateContextEx.argtypes = (wintypes.HWND, wintypes.HANDLE, wintypes.DWORD)
        return bool(imm.ImmAssociateContextEx(hwnd, None, 0x1))  # IACE_CHILDREN, hIMC=NULL -> 비활성
    except Exception:  # noqa: BLE001
        return False
