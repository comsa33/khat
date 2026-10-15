"""Matching table rows of the aggregation items in the 512k bundle.
"""
from __future__ import annotations

import collections
import re

import rag_lib as lib
import table_rows as C

AGG_BUNDLE = "mix_multiprovider_512k"
_hug = {r["_row_id"]: r for r in C.hug_rows()}
_molit = {r["_row_id"]: r for r in C.molit_rows()}


def rendered(source: str, rid: str) -> str:
    if source == "hug-sale-history":
        r = _hug[rid]
        return " | ".join([r.get("_query_area_name", ""), r.get("_query_year", ""), r.get("BSU_NM", ""),
                           str(r.get("TOT_HOCO", "")), str(r.get("GNRL_SILT_HOCO", ""))])
    r = _molit[rid]
    return (f"{r['_row_id']} | {r['_lawd_name']} {r['umdNm']} | {r['aptNm']} | "
            f"{r['excluUseAr']}㎡ {r['dealAmount']}만원 | {r['_deal_ymd']}")


def agg_items(qa: dict) -> list:
    ids = [r["qa_id"] for r in lib.load_jsonl(lib.REL / "data" / "item_sets" / "agg81.jsonl")]
    return [qa[i] for i in ids]


def build(qa: dict) -> tuple:
    """-> (items, checks). items: list of dicts {qa_id, source, filter, lines(set of rendered lines)}."""
    bundle_lines = (lib.BUNDLES / f"{AGG_BUNDLE}.txt").read_text(encoding="utf-8").splitlines()
    line_count = collections.Counter(bundle_lines)
    items, checks = [], collections.Counter()
    for rec in agg_items(qa):
        pred = rec["gold_predicate"]
        val, rids = C.recompute(pred)
        gold = re.sub(r"[^0-9.]", "", str(rec["answer"]))
        checks["value_eq_answer"] += bool(gold) and abs(float(val) - float(gold)) < 1e-6
        if pred["op"] == "count":
            checks["count_items"] += 1
            checks["count_eq_nrows"] += int(float(gold)) == len(rids)
        lines = [rendered(pred["source"], rid) for rid in rids]
        checks["rows_total"] += len(lines)
        checks["rows_in_bundle"] += sum(line_count[l] >= 1 for l in lines)
        checks["rows_line_unique_in_bundle"] += sum(line_count[l] == 1 for l in lines)
        checks["items_with_duplicate_rendered_lines"] += len(set(lines)) != len(lines)
        items.append({"qa_id": rec["qa_id"], "source": pred["source"], "filter": pred["filter"],
                      "op": pred["op"], "lines": set(lines), "n_rows": len(rids), "rec": rec})
    checks["items"] = len(items)
    return items, dict(checks)


_chunk_line_sets: dict = {}


def chunk_line_sets(size: int) -> list:
    if size not in _chunk_line_sets:
        _chunk_line_sets[size] = [set(txt.split("\n")) for _, txt in lib.chunks_for(AGG_BUNDLE, size)]
    return _chunk_line_sets[size]


def covered(item: dict, idx: list, size: int) -> int:
    cls = chunk_line_sets(size)
    got = set()
    for i in idx:
        got |= item["lines"] & cls[i]
    return len(got)


def min_chunks_needed(item: dict, size: int) -> int:
    """Number of distinct chunks that contain at least one matched row (rows are never split)."""
    return sum(1 for s in chunk_line_sets(size) if item["lines"] & s)
