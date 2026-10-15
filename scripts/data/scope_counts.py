#!/usr/bin/env python3
"""Item counts of the evaluation scope (split x family) and the items of the oversized mixed bundle.

Usage: python3 scripts/data/scope_counts.py [--exclude-family F] [--exclude-ids FILE]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from khat.paths import ROOT as DEFAULT_ROOT
OVERSIZED = "mix_multiprovider_256k"


def read_id_file(path: str | None) -> set[str]:
    """qa_ids listed one per line in `path` ('#' starts a comment); empty set when path is None."""
    if not path:
        return set()
    ids = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.add(line)
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DEFAULT_ROOT))
    ap.add_argument("--exclude-family", default="")
    ap.add_argument("--exclude-ids", default=None, metavar="FILE",
                    help="file of qa_ids to drop (one per line, '#' comments), "
                         "e.g. data/exclusions/eval.txt; default: none")
    args = ap.parse_args()
    root = Path(args.dataset)
    xf = {s for s in args.exclude_family.split(",") if s}
    xi = read_id_file(args.exclude_ids)
    rows = []
    for sp in ("dev", "test_public"):
        for line in (root / "data" / f"qa_{sp}.jsonl").open(encoding="utf-8"):
            if line.strip():
                rows.append(json.loads(line))
        for line in (root / "data" / "auxiliary" / "legal_reasoning.jsonl").open(encoding="utf-8"):
            if line.strip() and json.loads(line)["split"] == sp:
                rows.append(json.loads(line))
    kept = [r for r in rows if r["task_type"] not in xf and r["qa_id"] not in xi]
    print(f"all items {len(rows)}; excluded families {sorted(xf) or '-'}; excluded ids {len(xi)}; in scope {len(kept)}")
    print("split:", dict(Counter(r["split"] for r in kept)))
    fam = Counter((r["task_type"], r["split"]) for r in kept)
    print(f"{'family':32} {'dev':>5} {'test':>5} {'all':>5}")
    for f in sorted({r["task_type"] for r in kept}, key=lambda f: -sum(v for (g, _), v in fam.items() if g == f)):
        d, t = fam[(f, "dev")], fam[(f, "test_public")]
        print(f"{f:32} {d:>5} {t:>5} {d + t:>5}")
    print("context tier (test_public):", dict(sorted(Counter(r.get("context_tier") for r in kept if r["split"] == "test_public").items(), key=str)))
    over = [r for r in kept if r.get("bundle_id") == OVERSIZED]
    print(f"{OVERSIZED} items in scope: {len(over)}  split {dict(Counter(r['split'] for r in over))}  "
          f"family {dict(Counter(r['task_type'] for r in over))}")
    print("  test_public ids:", " ".join(sorted(r["qa_id"] for r in over if r["split"] == "test_public")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
