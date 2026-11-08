#!/usr/bin/env python3
"""수집 현황 보고: python scripts/audit_data.py [data/raw]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ringkey.audit import report  # noqa: E402

if __name__ == "__main__":
    print(report(sys.argv[1] if len(sys.argv) > 1 else "data/raw"))
