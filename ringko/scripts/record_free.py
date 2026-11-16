#!/usr/bin/env python3
"""자유 입력 녹화: 제시 단어 없이 치고 싶은 단어를 친다. 추론은 scripts/decode_free.py (IMU만 사용).

  python scripts/record_free.py --subject S01 --wear 4                # 키 로그도 함께(나중에 채점 가능)
  python scripts/record_free.py --subject S01 --wear 4 --no-keylog    # IMU만 기록(키를 전혀 저장하지 않음)
  python scripts/decode_free.py data/free/S01/<세션> --flip -1 [--add-words 내단어.txt]

치는 법 (수집과 같은 규칙)
  1) 왼손 검지를 F 위에 두고 0.5초쯤 멈춘다.
  2) 단어를 친다: 자음 = 왼손 검지, 모음·SPACE = 오른손. 쌍자음·ㅒ·ㅖ가 든 단어는 치지 않는다.
  3) 검지를 F로 되돌려 0.5초 이상 멈춘다 → 다음 단어.
  입력기는 영문 상태. 끝내려면 Esc 두 번. 시작할 때 5초쯤 F 위에서 가만히 있으면 기준 자세가 잘 잡힌다.
키 로그는 '실제로 친 단어'를 복원해 채점하는 데만 쓰이고 추론에는 쓰이지 않는다.
저장: <out>/<subject>/<YYYYmmdd_HHMMSS>_w<wear>_free/  (imu.csv mag.csv ping.csv keys.csv trials.csv meta.json)
"""
import argparse
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ringko import __version__  # noqa: E402
from ringko.device import Device, find_port  # noqa: E402
from ringko.free import FreeRecorder  # noqa: E402
from ringko.hangul import compose  # noqa: E402
from ringko.session import canon_key  # noqa: E402

FONT = "Malgun Gothic"


def run_window(rec: FreeRecorder, dev: Device, title: str) -> str:
    """녹화 창. 반환: 키 입력 출처(hook | tk)."""
    import tkinter as tk
    r = tk.Tk()
    r.title(title)
    r.geometry("1000x520")
    r.configure(bg="#1e2230")
    top = tk.Label(r, font=(FONT, 16), fg="#9aa4bf", bg="#1e2230",
                   text="자유 입력 — F 위에서 멈춤 → 단어(자음 = 왼손 검지) → SPACE → F로 돌아와 멈춤     끝: Esc 두 번")
    state = tk.Label(r, font=(FONT, 40, "bold"), fg="white", bg="#1e2230")
    typed = tk.Label(r, font=(FONT, 26), fg="#cfd6ea", bg="#1e2230", wraplength=940, justify="center")
    stat = tk.Label(r, font=(FONT, 12), fg="#7a839d", bg="#1e2230")
    top.pack(pady=(18, 0))
    state.pack(expand=True)
    typed.pack(pady=10)
    stat.pack(pady=(0, 10))
    hook, down, esc_t = None, set(), [0.0]
    r.update()

    def key(is_down, k, now, scan):
        if k == "escape":
            if is_down:
                if time.time() - esc_t[0] < 1.0:
                    r.quit()
                esc_t[0] = time.time()
            return
        if is_down:
            if k in down:
                return
            down.add(k)
        else:
            down.discard(k)
        rec.on_key(is_down, k, now, scan, source)

    if sys.platform == "win32":
        from ringko.winkeys import KeyHook, disable_ime
        hwnd = int(r.wm_frame(), 16)
        disable_ime(hwnd)
        disable_ime(r.winfo_id())
        hook = KeyHook(hwnd)
        hook.start()
        source = "hook"
    else:
        r.bind("<KeyPress>", lambda e: key(True, canon_key(e.keysym), time.perf_counter_ns(), -1))
        r.bind("<KeyRelease>", lambda e: key(False, canon_key(e.keysym), time.perf_counter_ns(), -1))
        source = "tk"
    r.protocol("WM_DELETE_WINDOW", r.quit)
    r.after(200, r.focus_force)

    def tick():
        if hook:
            for now, is_down, k, scan, fg, _inj in hook.drain():
                if fg:
                    key(is_down, k, now, scan)
        still = dev.is_still(0.4, 15.0)
        state.configure(text="F 위 정지 ✓ — 단어를 치세요" if still else "입력 중 …", fg="#4cc38a" if still else "white")
        if rec.log_keys:
            typed.configure(text=" ".join(rec.words[-8:] + ([compose("".join(rec.cur)) + "▁"] if rec.cur else [])))
        else:
            typed.configure(text="(키 로그 없음: IMU만 기록 중)")
        st = dev.stats()
        stat.configure(text=f"IMU {st['rate_hz']:.0f}Hz · 핑 {st['n_ping']} · 단어 {len(rec.words)} · 키입력 {source}")
        r.after(30, tick)

    r.after(30, tick)
    try:
        r.mainloop()
    finally:
        if hook:
            hook.stop()
    try:
        r.destroy()
    except tk.TclError:
        pass
    return source


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--wear", type=int, required=True)
    ap.add_argument("--no-keylog", action="store_true", help="키를 저장하지 않음(IMU만)")
    ap.add_argument("--port")
    ap.add_argument("--out", default="data/free")
    ap.add_argument("--keyboard", default="")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    port = a.port or find_port()
    if not port:
        sys.exit("Arduino 포트를 찾지 못했습니다. --port COM3 처럼 지정하세요.")
    d = Path(a.out) / a.subject / f"{datetime.now():%Y%m%d_%H%M%S}_w{a.wear}_free"
    meta = {"subject": a.subject, "wear": a.wear, "site": "middle_phalanx", "hand": "left", "finger": "index",
            "keyboard": a.keyboard, "layout": "dubeolsik", "notes": a.notes, "port": port, "software": __version__,
            "python": platform.python_version(), "platform": platform.platform(),
            "start": datetime.now().astimezone().isoformat()}
    rec = FreeRecorder(d, meta, keylog=not a.no_keylog)
    dev = Device(port, d)
    dev.start()
    rec.meta["device"] = dev.header
    rec.save_meta()
    source = "none"
    try:
        source = run_window(rec, dev, f"ringko 자유 입력 {a.subject} wear{a.wear}")
    finally:
        time.sleep(1.2)   # 마지막 핑 확보
        dev.stop()
        rec.close(end=datetime.now().astimezone().isoformat(), keyboard_source=source, device_stats=dev.stats())
    print("저장:", d)
    print(f"추론: python scripts/decode_free.py {d} --flip -1")


if __name__ == "__main__":
    main()
