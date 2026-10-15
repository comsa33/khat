#!/usr/bin/env python3
"""Write the dev and test_public split files with the cross-split leakage check.

Usage: python3 scripts/construction/make_splits.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from khat.paths import ROOT, LOCAL
SRC = LOCAL / "construction" / "qa_realistic_candidates.jsonl"
DEV = LOCAL / "construction" / "qa_dev.jsonl"
TPUB = LOCAL / "construction" / "qa_test_public.jsonl"
PART_B_Q = LOCAL / "construction" / "qa_test_part_b_questions.jsonl"
PART_B_A = LOCAL / "construction" / "qa_test_part_b_answers.jsonl"


def mask(item: dict) -> dict:
    out = dict(item)
    out["answer"] = "[HELD OUT]"
    out.pop("gold_predicate", None)
    out.pop("row_ids", None)
    ev = dict(out.get("evaluation", {}))
    ev = {"metric": ev.get("metric", "")}
    out["evaluation"] = ev
    out["answer_masked"] = True
    return out


def write(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main() -> int:
    rows = [json.loads(l) for l in SRC.open(encoding="utf-8") if l.strip()]
    dev = [r for r in rows if r.get("split") == "dev"]
    tpub = [r for r in rows if r.get("split") == "test_public"]
    thid = [r for r in rows if r.get("split") == "test_hidden"]

    write(DEV, dev)
    write(TPUB, tpub)
    if thid:
        write(PART_B_Q, [mask(r) for r in thid])
        write(PART_B_A, thid)

    import re
    def anns(r):
        s = set(r.get("announcement_ids", []) or [])
        for p in r.get("page_ids", []) or []:
            m = re.match(r"(.+)-p\d{3}$", p)
            if m:
                s.add(m.group(1))
        return s
    split_of_ann = {}
    leak = []
    for r in rows:
        for a in anns(r):
            sp = r.get("split")
            if a in split_of_ann and split_of_ann[a] != sp:
                leak.append((a, split_of_ann[a], sp))
            split_of_ann.setdefault(a, sp)

    print("=== splits ===")
    print(f"  dev: {len(dev)} -> {DEV.name}")
    print(f"  test_public: {len(tpub)} -> {TPUB.name}")
    print(f"  test part b: {len(thid)} ({'merged into test_public' if not thid else 'masked -> ' + PART_B_Q.name})")
    leak_anns = {a for a, _, _ in leak}
    print(f"  cross-split announcement leakage: {len(leak_anns)}" + (f"  !! {sorted(leak)[:3]}" if leak else ""))
    return 1 if leak else 0


if __name__ == "__main__":
    raise SystemExit(main())
