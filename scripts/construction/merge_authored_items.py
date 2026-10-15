#!/usr/bin/env python3
"""Merge the authored item batches into one file for the assembly gate.

Usage: python3 scripts/construction/merge_authored_items.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from khat.paths import ROOT, LOCAL
OUTDIR = LOCAL / "construction" / "authoring" / "out"
DEST = LOCAL / "construction" / "authored_items.json"


def main() -> int:
    merged = []
    bad = []
    if OUTDIR.exists():
        for p in sorted(OUTDIR.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                bad.append((p.name, str(exc)))
                continue
            if isinstance(data, list):
                merged.extend(data)
            else:
                bad.append((p.name, "not a list"))
    DEST.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
    fam = Counter(r.get("item", {}).get("task_type", "?") for r in merged)
    sup = sum(1 for r in merged if r.get("reviewer_supported") and r.get("reviewer_unambiguous"))
    print(f"=== MERGE agent raw: {len(merged)} candidates from {OUTDIR.name}/ ===")
    for k, v in sorted(fam.items()):
        print(f"   {k:32s} {v}")
    print(f"   passed (supported & unambiguous): {sup}")
    if bad:
        print(f"   WARN unreadable files: {bad}")
    print(f"   -> {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
