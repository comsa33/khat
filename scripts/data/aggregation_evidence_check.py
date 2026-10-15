#!/usr/bin/env python3
"""Evidence check of the 81 aggregation items: the gold predicate recomputed on the table rows equals the answer, and every matched row occurs verbatim in the 512k bundle.

Usage: python3 scripts/data/aggregation_evidence_check.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from khat.paths import DATA, LOCAL


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import table_rows as C

    ids = [json.loads(l)["qa_id"] for l in (DATA / "item_sets/sample_agg81.jsonl").open() if l.strip()]
    tp = {}
    for l in (DATA / "qa_test_public.jsonl").open(encoding="utf-8"):
        if l.strip():
            d = json.loads(l)
            tp[d["qa_id"]] = d
    bundle = (LOCAL / "bundles/mix_multiprovider_512k.txt").read_text(encoding="utf-8")
    lines = set(bundle.splitlines())

    hug = {r["_row_id"]: r for r in C.hug_rows()}
    molit = {r["_row_id"]: r for r in C.molit_rows()}

    def rendered(source: str, rid: str) -> str:
        if source == "hug-sale-history":
            r = hug[rid]
            return " | ".join([r.get("_query_area_name", ""), r.get("_query_year", ""), r.get("BSU_NM", ""),
                               str(r.get("TOT_HOCO", "")), str(r.get("GNRL_SILT_HOCO", ""))])
        r = molit[rid]
        return (f"{r['_row_id']} | {r['_lawd_name']} {r['umdNm']} | {r['aptNm']} | "
                f"{r['excluUseAr']}㎡ {r['dealAmount']}만원 | {r['_deal_ymd']}")

    n_ok = n_val = n_rows = 0
    fails = []
    for q in ids:
        item = tp[q]
        pred = item["gold_predicate"]
        val, rids = C.recompute(pred)
        gold_num = re.sub(r"[^0-9.]", "", str(item["answer"]))
        val_ok = gold_num != "" and abs(float(val) - float(gold_num)) < 1e-6
        missing = [rid for rid in rids if rendered(pred["source"], rid) not in lines]
        n_val += val_ok
        n_rows += not missing
        if val_ok and not missing:
            n_ok += 1
        else:
            fails.append((q, val, item["answer"], len(rids), len(missing)))
    print(f"items={len(ids)}  predicate==answer: {n_val}/{len(ids)}  all matched rows in bundle: {n_rows}/{len(ids)}  both: {n_ok}/{len(ids)}")
    print("matched rows per item: min/max =", min(len(C.recompute(tp[q]['gold_predicate'])[1]) for q in ids),
          max(len(C.recompute(tp[q]['gold_predicate'])[1]) for q in ids))
    for f in fails:
        print("FAIL", f)
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
