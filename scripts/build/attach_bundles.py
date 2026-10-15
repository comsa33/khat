#!/usr/bin/env python3
"""Attach bundle_id, context_tier and evidence_position to the construction QA file from the bundle manifest.

Usage: python3 scripts/build/attach_bundles.py
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from khat.paths import ROOT, LOCAL
QA = LOCAL / "construction" / "qa_realistic_candidates.jsonl"
MANIFEST = LOCAL / "bundles" / "manifest.jsonl"

MULTI_FAMILIES = {"multi_document_comparison", "cross_document_legal_reasoning",
                  "region_comparison", "provider_comparison", "cross_source_aggregation"}
CELL_FAMILIES = {"eligibility_reasoning", "schedule_reasoning", "table_numeric_reasoning"}


def tier_int(t: str) -> int:
    return int(t.replace("k", ""))


def main() -> int:
    bundles = [json.loads(l) for l in MANIFEST.open(encoding="utf-8") if l.strip()]
    page_pos = defaultdict(list)
    bundle_pages = {}
    for b in bundles:
        ids = set()
        for c in b["component_positions"]:
            if c["type"] == "lh_page":
                page_pos[c["id"]].append((c["position_band"], b))
                ids.add(c["id"])
        bundle_pages[b["bundle_id"]] = ids
    mix = [b for b in bundles if b["bundle_id"].startswith("mix_")]

    def band_bundle(pid, want):
        cand = [b for (band, b) in page_pos.get(pid, []) if band == want]
        if not cand:
            return None
        return sorted(cand, key=lambda x: tier_int(x["context_tier"]))[0 if want == "early" else -1]

    def mix_for(pids):
        for b in sorted(mix, key=lambda x: -tier_int(x["context_tier"])):
            if all(p in bundle_pages[b["bundle_id"]] for p in pids):
                return b
        return None

    def any_bundle(pid):
        pos = page_pos.get(pid, [])
        if not pos:
            return None, None
        band, b = sorted(pos, key=lambda x: tier_int(x[1]["context_tier"]))[len(pos) // 2]
        return b, band

    rows = [json.loads(l) for l in QA.open(encoding="utf-8") if l.strip()]
    attached = 0
    for it in rows:
        pids = it.get("page_ids") or []
        if not pids:
            continue
        tt = it["task_type"]
        b, pos = None, None
        if tt == "long_context_retrieval":
            b = band_bundle(pids[0], "early"); pos = "early"
        elif tt == "long_distance_retrieval":
            b = band_bundle(pids[0], "late"); pos = "late"
        elif tt in MULTI_FAMILIES:
            b = mix_for(pids)
            pos = "multi"
        elif tt in CELL_FAMILIES:
            b, pos = any_bundle(pids[0])
        else:
            b, pos = any_bundle(pids[0])
        if b:
            it["bundle_id"] = b["bundle_id"]
            it["context_tier"] = b["context_tier"]
            it["evidence_position"] = pos
            attached += 1
        else:
            for k in ("bundle_id", "context_tier", "evidence_position"):
                it.pop(k, None)

    QA.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    from collections import Counter
    bb = [r for r in rows if r.get("bundle_id")]
    nonlh = sum(1 for r in bb if r.get("provider") and "공공데이터" not in r.get("provider", "")
                and not r.get("provider", "").startswith("복수") and r.get("provider") != "한국토지주택공사")
    print(f"=== attach bundles: {attached} bundle-bearing QA ===")
    print("  tiers:", dict(Counter(r.get("context_tier") for r in bb)))
    print("  positions:", dict(Counter(r.get("evidence_position") for r in bb)))
    print(f"  non-LH-provider bundle-bearing: {nonlh}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
