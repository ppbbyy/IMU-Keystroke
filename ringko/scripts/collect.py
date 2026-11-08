#!/usr/bin/env python3
"""데이터 수집. 한 번 실행 = 한 세션(한 착용 안에서 연속 기록).

  python scripts/collect.py --subject S01 --wear 1                       # 전체(정지, F보정, 개별 키, 연속, 정지)
  python scripts/collect.py --subject S01 --wear 1 --phases rest,calib,isolated
  python scripts/collect.py --subject P00 --wear 1 --reps 3 --words 15 --out data/pilot   # 파일럿

--wear: 링을 새로 낄 때마다 1, 2, 3... (착용 간 일반화 평가용)
저장: <out>/<subject>/<YYYYmmdd_HHMMSS>_w<wear>_<phases>/
"""
import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ringko import __version__  # noqa: E402
from ringko.dataset import load_session, quality_report  # noqa: E402
from ringko.device import Device, find_port  # noqa: E402
from ringko.plan import (PHASES, build_plan, consonant_counts, counts, estimate_minutes, filter_vocab,  # noqa: E402
                         load_vocab, save_plan, seed_for, select_words)
from ringko.session import Params, Session  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True, help="피험자 ID(S01 등). 실명 금지")
    ap.add_argument("--wear", type=int, required=True, help="착용 번호(링을 새로 낄 때마다 증가)")
    ap.add_argument("--phases", default=",".join(PHASES), help=f"쉼표 구분: {','.join(PHASES)}")
    ap.add_argument("--reps", type=int, default=40, help="개별 키: 자음당 반복 수")
    ap.add_argument("--words", type=int, default=100, help="연속 타이핑 단어 수")
    ap.add_argument("--site", default="middle_phalanx", choices=["middle_phalanx", "proximal_phalanx"])
    ap.add_argument("--port")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--vocab", help="단어[ 빈도] 목록 파일(없으면 wordfreq)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keyboard", default="", help="노트북/키보드 모델")
    ap.add_argument("--notes", default="")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()

    phases = tuple(p.strip() for p in a.phases.split(",") if p.strip())
    bad = set(phases) - set(PHASES)
    if bad:
        sys.exit(f"알 수 없는 단계: {bad}")
    port = a.port or find_port()
    if not port:
        sys.exit("Arduino 포트를 찾지 못했습니다. --port COM3 처럼 지정하세요.")

    seed = seed_for(a.subject, a.wear, a.seed)
    rng = np.random.default_rng(seed)
    words = None
    if "continuous" in phases:
        words = select_words(filter_vocab(load_vocab(a.vocab)), a.words, rng)
    params = Params()
    plan = build_plan(rng, phases, reps=a.reps, words=words, rest_s=params.rest_s)
    c = counts(plan)
    print(f"[{a.subject} wear{a.wear} {a.site}] {c} · 예상 약 {estimate_minutes(plan)}분 · 포트 {port}")
    if words:
        print("단어 예:", " ".join(words[:12]), "…")
        print("단어 내 자음 등장:", consonant_counts(words))
    if not a.yes and input("시작할까요? [Enter=시작 / q=취소] ").strip().lower() == "q":
        return

    sess_dir = Path(a.out) / a.subject / f"{datetime.now():%Y%m%d_%H%M%S}_w{a.wear}_{'-'.join(phases)}"
    sess_dir.mkdir(parents=True, exist_ok=False)
    save_plan(plan, sess_dir / "plan.csv")
    meta = {"subject": a.subject, "wear": a.wear, "site": a.site, "hand": "left", "finger": "index",
            "phases": phases, "reps": a.reps, "n_words": len(words or []), "seed": seed, "keyboard": a.keyboard,
            "layout": "dubeolsik", "notes": a.notes, "port": port, "software": __version__,
            "python": platform.python_version(), "platform": platform.platform(),
            "params": vars(params), "start": datetime.now().astimezone().isoformat(), "complete": False}

    def save_meta():
        (sess_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    save_meta()
    dev = Device(port, sess_dir)
    dev.start()
    meta["device"] = dev.header
    save_meta()
    if dev.header.get("hfxo") != 1:
        print("경고: 장치 크리스털(HFXO)이 꺼져 있습니다(펌웨어 fw-2 이상 필요). 시계가 ~1.6% 틀리고 흔들립니다.")
    session = Session(plan, params, sess_dir, device=dev)
    completed = False
    win = None
    try:
        from ringko.gui import CollectWindow
        win = CollectWindow(session, dev, f"ringko {a.subject} wear{a.wear}")
        meta["keyboard_source"] = win.source
        completed = win.run()
    finally:
        time.sleep(1.2)  # 마지막 핑 확보
        dev.stop()
        session.close()

    meta.update({"end": datetime.now().astimezone().isoformat(), "complete": bool(completed),
                 "ok_by_phase": dict(session.ok_by_phase), "status_counts": dict(session.status_counts),
                 "stray_keys": session.stray_keys, "unsettled_cues": session.unsettled,
                 "ime_suspected": session.ime_suspected, "device_stats": dev.stats(),
                 "keys_while_not_front": win.blurred if win else None})
    save_meta()
    print("\n=== 요약 ===")
    print("저장:", sess_dir)
    print("상태:", "완료" if completed else "중단(불완전)")
    try:
        lines, warns = quality_report(load_session(sess_dir))
        print("\n".join(lines))
        for w in warns:
            print("경고:", w)
    except Exception as e:  # noqa: BLE001
        print("품질 보고 실패:", repr(e))


if __name__ == "__main__":
    main()
