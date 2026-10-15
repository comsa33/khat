#!/usr/bin/env python3
"""Announcement provider, region and housing-type metadata, table cells and grounding helpers.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

import source_text as V2
import table_rows as C
import lh_announcements as Q

PROVIDER_LH = "한국토지주택공사"

ANN_SPLIT = {
    "lh-daedong2-b1-public-sale-20251017": ("test_hidden", ["ood_region"]),
    "lh-announce-0000061016-public-sale-2025q4": ("test_hidden", ["ood_region"]),
    "lh-announce-0000061086-public-sale-2026q2": ("test_public", ["ood_year"]),
    "lh-siheung-hajung-a1-newlywed-hope-20260430": ("test_public", []),
}


def ann_split(ann_id: str) -> str:
    return ANN_SPLIT.get(ann_id, ("dev", []))[0]


def ann_split_tags(ann_id: str) -> list:
    return ANN_SPLIT.get(ann_id, ("dev", []))[1]


def ann_provider(ann_id: str) -> str:
    return PROVIDER_LH


def ann_housing_type(ann_id: str) -> str:
    return Q.ann_meta().get(ann_id, {}).get("housing_type_from_audit", "") or "공공분양"


def ann_meta_fields(ann_id: str) -> dict:
    return {
        "provider": ann_provider(ann_id),
        "region_sido": Q.ann_sido(ann_id) or "",
        "region_sigungu": Q.ann_sigungu(ann_id) or "",
        "housing_type": ann_housing_type(ann_id),
        "split": ann_split(ann_id),
    }


# ----------------------------------------------------------------- table cells
@lru_cache(maxsize=None)
def ann_cells(ann_id: str) -> list:
    return C._load_jsonl(Q.V04_DIR / ann_id / "table_cells.jsonl")


@lru_cache(maxsize=None)
def ann_tables(ann_id: str) -> list:
    return C._load_jsonl(Q.V04_DIR / ann_id / "tables.jsonl")


@lru_cache(maxsize=None)
def all_cell_ids() -> frozenset:
    out = set()
    for a in Q.announcement_ids():
        for c in ann_cells(a):
            out.add(f"{c['table_id']}#r{c['row_index']}c{c['col_index']}")
    return frozenset(out)


@lru_cache(maxsize=None)
def all_table_ids() -> frozenset:
    out = set()
    for a in Q.announcement_ids():
        for t in ann_tables(a):
            out.add(t["table_id"])
    return frozenset(out)


def cell_id(c: dict) -> str:
    return f"{c['table_id']}#r{c['row_index']}c{c['col_index']}"


def page_of_table(table_id: str) -> str:
    return re.sub(r"-t\d{2}$", "", table_id)


# ----------------------------------------------------------------- announcement-of-page (for split mapping)
def ann_of_page(pid: str) -> str | None:
    m = re.match(r"(.+)-p\d{3}$", pid)
    return m.group(1) if m else None


def item_announcements(item: dict) -> set:
    anns = set(item.get("announcement_ids", []) or [])
    for pid in item.get("page_ids", []) or []:
        a = ann_of_page(pid)
        if a:
            anns.add(a)
    return anns


def item_split(item: dict):
    """Announcement-level split for a QA item.
    - no announcement (MOLIT/HUG-only): 'dev'
    - one shared split across all cited announcements: that split
    - mixed splits (multi-doc spanning split boundaries): None  -> caller DROPS it, so no split leakage
      and test_hidden answers never appear in another split.
    """
    anns = item_announcements(item)
    splits = {ann_split(a) for a in anns}
    if not splits:
        return "dev"
    if len(splits) == 1:
        return next(iter(splits))
    return None
