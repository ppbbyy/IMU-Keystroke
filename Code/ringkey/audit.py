"""수집 현황 점검: PROTOCOL.md 기준(피험자 >=5, 세션 3, 착용 2, 보정 10, 개별 키 14x50, 연속 150단어)을
얼마나 채웠는지 표로 보고하고 경고를 낸다. 불완전/중단 런은 집계에서 제외한다."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .hangul_keys import CONSONANTS

NEED = {"calib": 10, "isolated": 14 * 50, "continuous": 150}
MIN_SUBJECTS = 5


def scan(root: str | Path) -> dict:
    root = Path(root)
    table: dict = defaultdict(lambda: defaultdict(lambda: {"ok": defaultdict(int), "runs": 0, "bad_runs": 0,
                                                           "sync_bad": 0, "drop": [], "rate": [], "rots": set()}))
    for meta_path in sorted(root.glob("*/*/*/run_*/meta.json")):
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        cell = table[m["subject"]][(m["session"], m["wear"])]
        cell["rots"].add(m.get("rot_deg", 0))
        if not m.get("complete"):
            cell["bad_runs"] += 1
            continue
        cell["runs"] += 1
        for ph, n in m.get("ok_by_phase", {}).items():
            cell["ok"][ph] += n
        if not m.get("sync_ok"):
            cell["sync_bad"] += 1
        if m.get("ime_suspected"):
            cell["sync_bad"] += 1
        cell["drop"].append(m["imu"]["drop_frac"])
        cell["rate"].append(m["imu"]["rate_hz"])
    return table


def report(root: str | Path) -> str:
    table = scan(root)
    lines, warns = [], []
    subjects_complete = 0
    for subj in sorted(table):
        lines.append(f"\n[{subj}]")
        complete = True
        for sess in ("s1", "s2", "s3"):
            for wear in ("wear0", "wear1"):
                c = table[subj].get((sess, wear))
                if c is None:
                    lines.append(f"  {sess} {wear}: (없음)")
                    complete = False
                    continue
                ok = c["ok"]
                prog = " ".join(f"{p}={ok.get(p, 0)}/{n}" for p, n in NEED.items())
                extra = []
                if c["bad_runs"]:
                    extra.append(f"중단런 {c['bad_runs']}")
                if c["sync_bad"]:
                    extra.append(f"sync/IME 문제 {c['sync_bad']}")
                if c["drop"] and max(c["drop"]) > 0.01:
                    extra.append(f"손실 최대 {max(c['drop']):.1%}")
                lines.append(f"  {sess} {wear} rot{sorted(c['rots'])}: {prog}" + (f"  ⚠ {', '.join(extra)}" if extra else ""))
                if any(ok.get(p, 0) < n for p, n in NEED.items()):
                    complete = False
        subjects_complete += complete
    if subjects_complete < MIN_SUBJECTS:
        warns.append(f"완전히 수집된 피험자 {subjects_complete}명 < {MIN_SUBJECTS}명: LOSO/일반화 주장 불가(PROTOCOL 3절)")
    if not table:
        warns.append("수집된 런이 없음")
    head = "수집 현황 (PROTOCOL.md 기준)\n" + "\n".join(f"경고: {w}" for w in warns)
    return head + "\n" + "\n".join(lines)


def isolated_balance(root: str | Path, subject: str, session: str, wear: str) -> dict[str, int]:
    """한 (피험자, 세션, 착용)의 자음별 정상 개별 키 trial 수."""
    import pandas as pd
    cnt = {c: 0 for c in CONSONANTS}
    for p in Path(root).glob(f"{subject}/{session}/{wear}_rot*/run_*/trials.csv"):
        d = pd.read_csv(p)
        d = d[(d["phase"] == "isolated") & (d["status"] == "ok")]
        for lab in d["label"]:
            if lab in cnt:
                cnt[lab] += 1
    return cnt
