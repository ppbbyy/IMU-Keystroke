#!/usr/bin/env python3
"""데이터 수집 실행기.

한 번 실행 = 한 착용(wear)의 한 '런'. 런 안에서 IMU를 계속 기록하고
[sync 시작] -> [F 보정 10회] -> [개별 키 14자음 x 50] -> [연속 타이핑 N단어] -> [sync 끝] 순서로 진행한다.

예)  python scripts/collect.py --subject S01 --session s1 --wear wear0 --rot 0
     python scripts/collect.py --subject S01 --session s1 --wear wear1 --rot 90 --only isolated
저장: data/raw/<subject>/<session>/<wear>_rot<deg>/run_<시각>_<only>/
      imu.csv keys.csv trials.csv prompts.csv sync.json meta.json
"""
import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_prompts import make  # noqa: E402

from ringkey import __version__  # noqa: E402
from ringkey.imu_stream import IMUReader, find_port  # noqa: E402
from ringkey.keylog import KeyLog  # noqa: E402
from ringkey.prompts import expected_counts, estimated_minutes, load_plan, save_plan, seed_for  # noqa: E402
from ringkey.session import Params, Runner  # noqa: E402
from ringkey.sync import analyze_run, save_sync  # noqa: E402

ONLY = {
    "all": ("sync_start", "calib", "isolated", "continuous", "sync_end"),
    "calib": ("sync_start", "calib", "sync_end"),
    "isolated": ("sync_start", "calib", "isolated", "sync_end"),
    "continuous": ("sync_start", "calib", "continuous", "sync_end"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True, help="피험자 ID(예: S01). 실명/개인정보 금지")
    ap.add_argument("--session", required=True, choices=["s1", "s2", "s3"])
    ap.add_argument("--wear", required=True, choices=["wear0", "wear1"])
    ap.add_argument("--rot", type=int, default=0, choices=[0, 90, 180, 270], help="링 회전 조건(도)")
    ap.add_argument("--site", default="ring", choices=["ring", "middle_phalanx"], help="부착 위치")
    ap.add_argument("--only", default="all", choices=list(ONLY), help="수집 단계 선택(sync는 항상 포함)")
    ap.add_argument("--port", help="시리얼 포트(미지정 시 Arduino 자동 탐색)")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--blocks", type=int, default=50)
    ap.add_argument("--words", type=int, default=150)
    ap.add_argument("--vocab")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompts", help="미리 만든 prompts CSV 사용(지정 시 단계/개수 옵션 무시)")
    ap.add_argument("--keyboard", default="", help="사용한 노트북/키보드 모델(메모)")
    ap.add_argument("--notes", default="")
    ap.add_argument("--yes", action="store_true", help="요약 확인 질문 생략")
    a = ap.parse_args()

    port = a.port or find_port()
    if not port:
        sys.exit("Arduino 포트를 찾지 못했습니다. --port 로 지정하세요 (예: COM5, /dev/ttyACM0, /dev/cu.usbmodem*).")

    # ---- 플랜
    if a.prompts:
        plan, seed = load_plan(a.prompts), None
    else:
        plan, _, seed = make(a.subject, a.session, a.wear, phases=ONLY[a.only], n_blocks=a.blocks,
                             n_words=a.words, vocab_path=a.vocab, alpha=a.alpha, base_seed=a.seed)
    counts = expected_counts(plan)
    print(f"[{a.subject} {a.session} {a.wear} rot={a.rot}] 항목: {counts} · 예상 약 {estimated_minutes(plan)}분")
    print(f"포트: {port}")
    if not a.yes and input("시작할까요? [Enter=시작 / q=취소] ").strip().lower() == "q":
        return

    # ---- 디렉터리
    base = Path(a.out) / a.subject / a.session / f"{a.wear}_rot{a.rot}"
    run_dir = base / f"run_{datetime.now():%Y%m%d_%H%M%S}_{a.only}"
    run_dir.mkdir(parents=True, exist_ok=False)
    save_plan(plan, run_dir / "prompts.csv")

    meta = {
        "subject": a.subject, "session": a.session, "wear": a.wear, "rot_deg": a.rot, "site": a.site,
        "only": a.only, "keyboard": a.keyboard, "notes": a.notes, "port": port,
        "expected": counts, "seed": seed, "software": __version__, "python": platform.python_version(),
        "platform": platform.platform(), "start_iso": datetime.now().astimezone().isoformat(),
        "clock": "perf_counter_ns (IMU host_ns, keys, trials 공통)", "complete": False,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 장비/로거 시작
    imu = IMUReader(port, run_dir / "imu.csv")
    keylog = KeyLog(run_dir / "keys.csv")
    runner = Runner(plan, Params(), keylog, run_dir / "trials.csv", imu=imu, seed=seed or 0)
    imu.start()
    completed = False
    try:
        from ringkey.gui import CollectApp
        app = CollectApp(runner, imu, f"{a.subject} {a.session} {a.wear} rot{a.rot}")
        completed = app.run()
    finally:
        time.sleep(0.2)
        imu.stop()
        runner.close()
        keylog.close()

    # ---- 사후 처리
    stats = imu.stats()
    sync = analyze_run(run_dir) if completed else {"ok": False, "warnings": ["중단된 런: 동기화 분석 생략"]}
    save_sync(run_dir, sync)
    meta.update({
        "end_iso": datetime.now().astimezone().isoformat(), "complete": bool(completed),
        "ok_by_phase": dict(runner.ok_by_phase), "status_counts": dict(runner.status_counts),
        "stray_keys": runner.stray_keys, "ime_suspected": runner.ime_suspected,
        "low_signal_warned": runner.low_signal_warned,
        "imu": stats, "sync_ok": sync.get("ok", False),
    })
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 요약 ===")
    print("저장 위치:", run_dir)
    print("상태:", "완료" if completed else "중단(불완전)")
    print("정상 입력:", dict(runner.ok_by_phase), "/ 필요:", counts)
    print(f"IMU {stats['n']}샘플 · {stats['rate_hz']:.1f}Hz · 손실 {stats['dropped']}({stats['drop_frac']:.2%})")
    for w in sync.get("warnings", []):
        print("경고:", w)
    if runner.low_signal_warned:
        print("경고: 수집 중 '센서 신호 약함' 경고가 떴습니다. trials.csv의 gyro_peak_dps가 낮은 구간을 확인하세요.")
    if runner.ime_suspected:
        print("경고: 한글 입력 상태에서 키가 눌린 흔적이 있습니다. 해당 런은 재수집을 고려하세요.")
    if completed and sync.get("ok"):
        g = sync["groups"]
        print("동기화: " + ", ".join(f"{k} MAD {v['mad_ms']:.1f}ms" for k, v in g.items())
              + (f", 드리프트 {sync.get('drift_ppm', 0):.1f}ppm" if "drift_ppm" in sync else ""))


if __name__ == "__main__":
    main()
