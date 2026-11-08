#!/usr/bin/env python3
"""수집된 세션 품질 보고.

  python scripts/report.py data/pilot/P00/20261103_160000_w1_rest-calib-isolated-continuous
  python scripts/report.py data/raw            # 하위의 모든 세션
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ringko.dataset import load_session, quality_report  # noqa: E402


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    root = Path(sys.argv[1])
    dirs = [root] if (root / "meta.json").exists() else sorted(p.parent for p in root.rglob("meta.json"))
    if not dirs:
        sys.exit(f"세션이 없습니다: {root}")
    for d in dirs:
        try:
            lines, warns = quality_report(load_session(d))
        except Exception as e:  # noqa: BLE001
            print(f"\n[{d}] 로드 실패: {e!r}")
            continue
        print("\n" + "\n".join(lines))
        for w in warns:
            print("경고:", w)


if __name__ == "__main__":
    main()
