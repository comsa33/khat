#!/usr/bin/env python3
"""Access to the HUG and MOLIT table rows and bundle manifests used by the benchmark scripts.
"""
from __future__ import annotations

import json
import statistics
from functools import lru_cache
from pathlib import Path

from khat.paths import ROOT, DATA, LOCAL
PROC = LOCAL

HUG = "hug-sale-history"
MOLIT = "molit-apt-trade-detail"
LH = "lh-sale-announcements"


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def _to_int(s):
    try:
        return int(str(s).replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def _to_float(s):
    try:
        return float(str(s).replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


# ---- normalized numeric accessors ---------------------------------------------
MOLIT_NUM = {
    "dealAmount_manwon": lambda r: _to_int(r.get("dealAmount")),
    "excluUseAr_m2": lambda r: _to_float(r.get("excluUseAr")),
    "floor": lambda r: _to_int(r.get("floor")),
    "buildYear": lambda r: _to_int(r.get("buildYear")),
}
HUG_NUM = {
    "TOT_HOCO": lambda r: _to_int(r.get("TOT_HOCO")),
    "GNRL_SILT_HOCO": lambda r: _to_int(r.get("GNRL_SILT_HOCO")),
    "MOTA_HOCO": lambda r: _to_int(r.get("MOTA_HOCO")),
    "LEAS_GUAR_HOCO": lambda r: _to_int(r.get("LEAS_GUAR_HOCO")),
}


@lru_cache(maxsize=None)
def hug_rows() -> list[dict]:
    rows = _load_jsonl(PROC / HUG / "rows.jsonl")
    return [r for r in rows if not r.get("ERROR_CODE") and r.get("_row_id")]


@lru_cache(maxsize=None)
def molit_rows() -> list[dict]:
    return _load_jsonl(DATA / "tables/molit_apt_trade_rows.jsonl")


@lru_cache(maxsize=None)
def lh_pages() -> list[dict]:
    return _load_jsonl(PROC / LH / "document_pages.jsonl")


@lru_cache(maxsize=None)
def lh_facts() -> list[dict]:
    return _load_jsonl(PROC / LH / "numeric_facts.jsonl")


@lru_cache(maxsize=None)
def rows_by_id(source: str) -> dict:
    rows = hug_rows() if source == HUG else molit_rows()
    return {r["_row_id"]: r for r in rows}


@lru_cache(maxsize=None)
def page_ids() -> set:
    return {p["page_id"] for p in lh_pages()}


def _num(source: str, row: dict, field: str):
    table = MOLIT_NUM if source == MOLIT else HUG_NUM
    if field in table:
        return table[field](row)
    return _to_float(row.get(field))


def _match(source: str, row: dict, filt: dict) -> bool:
    for k, cond in filt.items():
        if isinstance(cond, dict) and ("min" in cond or "max" in cond):
            v = _num(source, row, k)
            if v is None:
                return False
            if "min" in cond and v < cond["min"]:
                return False
            if "max" in cond and v >= cond["max"]:
                return False
        else:
            if str(row.get(k, "")) != str(cond):
                return False
    return True


def select(predicate: dict) -> list[dict]:
    source = predicate["source"]
    rows = hug_rows() if source == HUG else molit_rows()
    filt = predicate.get("filter", {})
    return [r for r in rows if _match(source, r, filt)]


def recompute(predicate: dict):
    """Return (value, matching_row_ids). value is a number (agg) or str (argmax/argmin) or None."""
    source = predicate["source"]
    op = predicate["op"]
    sel = select(predicate)
    ids = [r["_row_id"] for r in sel]
    if op == "count":
        return len(sel), ids
    field = predicate["field"]
    vals = [(_num(source, r, field), r) for r in sel]
    vals = [(v, r) for v, r in vals if v is not None]
    if not vals:
        return None, ids
    nums = [v for v, _ in vals]
    if op == "avg":
        out = round(statistics.mean(nums), predicate.get("round", 0))
        return (int(out) if predicate.get("round", 0) == 0 else out), ids
    if op == "sum":
        return sum(nums), ids
    if op == "min":
        return min(nums), ids
    if op == "max":
        return max(nums), ids
    if op == "median":
        out = round(statistics.median(nums), predicate.get("round", 0))
        return (int(out) if predicate.get("round", 0) == 0 else out), ids
    if op in ("argmax", "argmin"):
        rf = predicate["return_field"]
        chosen = (max if op == "argmax" else min)(vals, key=lambda x: x[0])[1]
        return str(chosen.get(rf, "")), ids
    raise ValueError(f"unknown op {op}")


BUNDLES_DIR = LOCAL / "bundles"


@lru_cache(maxsize=None)
def bundles() -> list[dict]:
    mf = BUNDLES_DIR / "manifest.jsonl"
    return _load_jsonl(mf) if mf.exists() else []


@lru_cache(maxsize=None)
def page_bundle_positions() -> dict:
    """page_id -> list of {bundle_id, context_tier, position_band} (LH pages present in each bundle)."""
    out: dict[str, list] = {}
    for b in bundles():
        for comp in b["components"]:
            if comp["type"] == "lh_page":
                out.setdefault(comp["id"], []).append(
                    {"bundle_id": b["bundle_id"], "context_tier": b["context_tier"],
                     "position_band": comp["position_band"]})
    return out


def predicate_human(predicate: dict) -> str:
    """Short human-readable summary of a predicate for QA copyright_note / debugging."""
    f = "&".join(f"{k}={v}" for k, v in predicate.get("filter", {}).items())
    return f"{predicate['source']}[{f}].{predicate['op']}({predicate.get('field','*')})"
