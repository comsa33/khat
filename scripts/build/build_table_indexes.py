#!/usr/bin/env python3
"""Compact indexes of the single-announcement LH pages and the HUG and MOLIT rows used by the item builders.

Usage: python3 scripts/build/build_table_indexes.py
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import table_rows as C

AUDIT = C.LOCAL / "index"


def snippet(s: str, n: int = 140) -> str:
    return re.sub(r"\s+", " ", s).strip()[:n]


def build_lh() -> dict:
    pages = C.lh_pages()
    facts = C.lh_facts()
    facts_by_page = defaultdict(list)
    for f in facts:
        facts_by_page[f["page_no"]].append(f)
    vt_counts = Counter(f["value_text"] for f in facts)
    idx = []
    for p in sorted(pages, key=lambda x: x["page_no"]):
        first_line = next((ln.strip() for ln in p["text"].splitlines() if ln.strip()), "")
        idx.append({
            "page_id": p["page_id"], "page_no": p["page_no"], "locator": p["locator"],
            "char_count": p["char_count"], "heading_guess": snippet(first_line, 60),
            "facts": [{"fact_id": f["fact_id"], "value_text": f["value_text"],
                       "value_unique_in_doc": vt_counts[f["value_text"]] == 1,
                       "snippet": snippet(f["local_snippet"], 160)}
                      for f in facts_by_page.get(p["page_no"], [])],
        })
    return {"source_id": C.LH, "n_pages": len(pages), "n_facts": len(facts),
            "announcement_id": pages[0]["announcement_id"] if pages else None, "pages": idx}


def build_hug() -> dict:
    rows = C.hug_rows()
    by_region_year = Counter((r.get("_query_area_name"), r.get("_query_year")) for r in rows)
    regions = sorted({r.get("_query_area_name") for r in rows})
    years = sorted({r.get("_query_year") for r in rows})
    sample_sites = defaultdict(list)
    for r in rows:
        if len(sample_sites[r.get("_query_area_name")]) < 3:
            sample_sites[r.get("_query_area_name")].append(snippet(r.get("BSU_NM", ""), 50))
    slices = [{"region": k[0], "year": k[1], "count": v} for k, v in sorted(by_region_year.items())]
    return {"source_id": C.HUG, "n_rows": len(rows), "regions": regions, "years": years,
            "numeric_fields": list(C.HUG_NUM.keys()),
            "date_fields": ["COLL_ANNO_APVL_DT", "SILT_OPEN_DT", "GUAR_FRST_ISSE_DT"],
            "region_year_slices": slices, "sample_business_sites": dict(sample_sites)}


def build_molit() -> dict:
    rows = C.molit_rows()
    by_dm = Counter((r["_lawd_name"], r["_deal_ymd"]) for r in rows)
    districts = sorted({r["_lawd_name"] for r in rows})
    months = sorted({r["_deal_ymd"] for r in rows})
    top_apts = defaultdict(Counter)
    for r in rows:
        top_apts[r["_lawd_name"]][r["aptNm"]] += 1
    slices = [{"district": k[0], "deal_ymd": k[1], "count": v} for k, v in sorted(by_dm.items())]
    return {"source_id": C.MOLIT, "n_rows": len(rows), "districts": districts, "months": months,
            "numeric_fields": list(C.MOLIT_NUM.keys()),
            "note": "dealAmount_manwon=거래금액(만원), excluUseAr_m2=전용면적(㎡), floor=층, buildYear=건축년도",
            "district_month_slices": slices,
            "top_apts_per_district": {d: c.most_common(8) for d, c in top_apts.items()}}


def main() -> int:
    AUDIT.mkdir(parents=True, exist_ok=True)
    for name, data in [("index_lh.json", build_lh()),
                       ("index_hug.json", build_hug()),
                       ("index_molit.json", build_molit())]:
        (AUDIT / name).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  wrote {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
