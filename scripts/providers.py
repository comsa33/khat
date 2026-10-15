#!/usr/bin/env python3
"""Announcement registry across providers: pages, cells, tables, region, housing type, split, grounding.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

import source_text as V2
import table_rows as C
import lh_announcements as Q
import announcement_tables as F5

PROC = Q.PROC
AUDIT = Q.AUDIT
PROVIDERS_INDEX = AUDIT / "providers.json"

PROVIDER_NAME = {
    "lh-sale-announcements-v04": "한국토지주택공사",
    "sh-announcements": "서울주택도시공사", "gh-announcements": "경기주택도시공사",
    "ih-announcements": "인천도시공사", "jpdc-announcements": "제주특별자치도개발공사",
}
NEW_PROVIDER_SPLIT = {"ih-announcements": ("test_hidden", ["ood_provider"]),
                      "jpdc-announcements": ("test_hidden", ["ood_provider"]),
                      "sh-announcements": ("dev", []), "gh-announcements": ("dev", [])}

LOC_RE = re.compile(r"(?:공급위치|건설위치|분양위치|대지위치|소재지|위\s*치)\s*[:：]\s*([^\n]+)")
HOUSING_KW = [
    ("토지임대부", "토지임대부"), ("이익공유형", "이익공유형분양"), ("나눔형", "나눔형분양"),
    ("신혼희망", "신혼희망타운"), ("행복주택", "행복주택"), ("전세임대", "전세임대"),
    ("국민임대", "국민임대"), ("영구임대", "영구임대"), ("공공임대", "공공임대"),
    ("분양전환", "분양전환공공임대"), ("민영주택", "민영주택"), ("국민주택", "국민주택"),
    ("공공분양", "공공분양"), ("분양", "분양주택"), ("임대", "임대주택"),
]


@lru_cache(maxsize=None)
def _new_reports() -> list:
    if not PROVIDERS_INDEX.exists():
        return []
    idx = json.loads(PROVIDERS_INDEX.read_text(encoding="utf-8"))
    return [r for r in idx.get("reports", []) if not r.get("error") and r.get("pages", 0) > 0]


@lru_cache(maxsize=None)
def registry() -> dict:
    """announcement_id -> {source_id, provider, dir(Path|None for LH)}."""
    reg = {}
    for a in Q.announcement_ids():
        reg[a] = {"source_id": Q.LH_V04, "provider": PROVIDER_NAME[Q.LH_V04], "dir": Q.V04_DIR / a}
    for r in _new_reports():
        a = r["announcement_id"]
        src = r["source_id"]
        reg[a] = {"source_id": src, "provider": PROVIDER_NAME.get(src, r.get("provider", "")),
                  "dir": PROC / src / a}
    return reg


def announcement_ids() -> list:
    return sorted(registry().keys())


def source_of(ann_id: str) -> str:
    return registry().get(ann_id, {}).get("source_id", "")


def provider_of(ann_id: str) -> str:
    return registry().get(ann_id, {}).get("provider", "")


@lru_cache(maxsize=None)
def pages(ann_id: str) -> list:
    info = registry().get(ann_id)
    if not info:
        return []
    return C._load_jsonl(info["dir"] / "document_pages.jsonl")


@lru_cache(maxsize=None)
def cells(ann_id: str) -> list:
    info = registry().get(ann_id)
    if not info:
        return []
    return C._load_jsonl(info["dir"] / "table_cells.jsonl")


@lru_cache(maxsize=None)
def tables(ann_id: str) -> list:
    info = registry().get(ann_id)
    if not info:
        return []
    return C._load_jsonl(info["dir"] / "tables.jsonl")


@lru_cache(maxsize=None)
def page_by_id() -> dict:
    out = {}
    for a in announcement_ids():
        for p in pages(a):
            out[p["page_id"]] = p
    return out


def page_text(pid: str) -> str:
    p = page_by_id().get(pid)
    return p["text"] if p else ""


@lru_cache(maxsize=None)
def all_page_ids() -> frozenset:
    return frozenset(page_by_id().keys())


@lru_cache(maxsize=None)
def all_cell_ids() -> frozenset:
    out = set()
    for a in announcement_ids():
        for c in cells(a):
            out.add(f"{c['table_id']}#r{c['row_index']}c{c['col_index']}")
    return frozenset(out)


@lru_cache(maxsize=None)
def all_table_ids() -> frozenset:
    out = set()
    for a in announcement_ids():
        for t in tables(a):
            out.add(t["table_id"])
    return frozenset(out)


def cell_id(c: dict) -> str:
    return f"{c['table_id']}#r{c['row_index']}c{c['col_index']}"


def ann_of_page(pid: str):
    m = re.match(r"(.+)-p\d{3}$", pid)
    a = m.group(1) if m else None
    return a if a in registry() else None


def page_id_for(ann_id: str, page_no: int) -> str:
    return f"{ann_id}-p{page_no:03d}"


def grounded(term: str, page_ids: list) -> bool:
    nt = V2.norm(term)
    if not nt:
        return False
    return any(nt in V2.norm(page_text(pid)) for pid in page_ids)


# ---------------- region / type / split ----------------
@lru_cache(maxsize=None)
def supply_location(ann_id: str) -> str:
    for p in pages(ann_id)[:4]:
        m = LOC_RE.search(p["text"])
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()
    return ""


def _first_sido(text: str):
    """The sido token appearing EARLIEST by position (not by SIDO-list order)."""
    best, bestpos = None, len(text) + 1
    for s in Q.SIDO:
        i = text.find(s)
        if 0 <= i < bestpos:
            best, bestpos = s, i
    return best


@lru_cache(maxsize=None)
def _report(ann_id: str) -> dict:
    return next((r for r in _new_reports() if r["announcement_id"] == ann_id), {})


PROVIDER_DEFAULT_SIDO = {
    "서울주택도시공사": "서울특별시", "경기주택도시공사": "경기도", "인천도시공사": "인천광역시",
    "제주특별자치도개발공사": "제주특별자치도", "부산도시공사": "부산광역시", "광주광역시도시공사": "광주광역시",
    "대전도시공사": "대전광역시", "대구도시개발공사": "대구광역시", "충북개발공사": "충청북도",
    "충남개발공사": "충청남도", "전북개발공사": "전라북도", "전남개발공사": "전라남도",
    "경북개발공사": "경상북도", "경남개발공사": "경상남도", "강원개발공사": "강원특별자치도",
    "울산도시공사": "울산광역시",
}


@lru_cache(maxsize=None)
def region_sido(ann_id: str):
    m = _report(ann_id).get("manifest_region_sido")
    if m:
        return _first_sido(m) or m
    s = _first_sido(supply_location(ann_id))
    if s:
        return s
    s = _first_sido(" ".join(p["text"] for p in pages(ann_id)[:2]))
    if s:
        return s
    return PROVIDER_DEFAULT_SIDO.get(provider_of(ann_id))


@lru_cache(maxsize=None)
def region_sigungu(ann_id: str):
    sido = region_sido(ann_id)
    if not sido:
        return None
    mf = _report(ann_id).get("manifest_region_sigungu")
    if mf:
        return mf if sido in mf else f"{sido} {mf}".strip()
    src = supply_location(ann_id)
    if sido in src:
        after = src.split(sido, 1)[1]
        m = re.search(r"\s*([가-힣]{2,}(?:시|군|구))", after)
        if m:
            return f"{sido} {m.group(1)}"
    m = re.search(r"([가-힣]{2,}(?:시|군|구))", src)
    if m:
        return f"{sido} {m.group(1)}"
    return sido


def short_region(ann_id: str) -> str:
    sg = region_sigungu(ann_id)
    if sg and " " in sg:
        return sg.split()[-1]
    return region_sido(ann_id) or ""


@lru_cache(maxsize=None)
def housing_type(ann_id: str) -> str:
    src = source_of(ann_id)
    if src == Q.LH_V04:
        return F5.ann_housing_type(ann_id)
    mf = _report(ann_id).get("manifest_housing_type")
    if mf:
        return mf
    rep = _report(ann_id)
    blob = (rep.get("title", "") + " " + (pages(ann_id)[0]["text"][:400] if pages(ann_id) else ""))
    for kw, label in HOUSING_KW:
        if kw in blob:
            return label
    return "주택공급"


HOLDOUT_SOURCES: set = {"dgdc-announcements", "cbdc-announcements"}


def split_of(ann_id: str) -> str:
    src = source_of(ann_id)
    if src == Q.LH_V04:
        return F5.ann_split(ann_id)
    if src in NEW_PROVIDER_SPLIT:
        return NEW_PROVIDER_SPLIT[src][0]
    return "test_hidden" if src in HOLDOUT_SOURCES else "dev"


def split_tags(ann_id: str) -> list:
    src = source_of(ann_id)
    if src == Q.LH_V04:
        return F5.ann_split_tags(ann_id)
    if src in NEW_PROVIDER_SPLIT:
        tags = list(NEW_PROVIDER_SPLIT[src][1])
    else:
        tags = ["ood_provider"] if src in HOLDOUT_SOURCES else []
    sido = region_sido(ann_id)
    if sido and sido not in ("서울특별시", "경기도") and "ood_region" not in tags:
        tags.append("ood_region")
    return tags


def meta_fields(ann_id: str) -> dict:
    return {"provider": provider_of(ann_id), "region_sido": region_sido(ann_id) or "",
            "region_sigungu": region_sigungu(ann_id) or "", "housing_type": housing_type(ann_id),
            "split": split_of(ann_id)}
