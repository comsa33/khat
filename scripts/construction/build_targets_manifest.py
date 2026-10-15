#!/usr/bin/env python3
"""Announcement target manifest (acquired announcements and backlog) at data/construction/announcement_targets.jsonl.

Usage: python3 scripts/construction/build_targets_manifest.py
"""
from __future__ import annotations

import json
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lh_announcements as Q
import announcement_tables as F

from khat.paths import ROOT, LOCAL
SUMMARY = LOCAL / "audit" / "lh-announcements" / "summary.json"
OUT = ROOT / "data" / "construction" / "announcement_targets.jsonl"

LICENSE_LH = "Official LH 입주자모집공고 page; statute-like official notice posture, raw kept internal."
REDIST = "Release source URL + locators + QA only; do not redistribute raw PDF/HWP."

BACKLOG_VERIFIED = [
    ("SH 서울주택도시공사", "https://www.i-sh.co.kr/"),
    ("GH 경기주택도시공사", "https://www.gh.or.kr/gh/index.do"),
    ("iH 인천도시공사", "https://www.ih.co.kr/"),
    ("부산도시공사", "https://www.bmc.busan.kr/"),
    ("제주개발공사", "https://www.jpdc.co.kr/"),
]
BACKLOG_UNVERIFIED = ["대구도시개발공사", "대전도시공사", "광주도시공사", "충북개발공사"]


def lh_rows() -> list[dict]:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    rows = []
    for r in summary["results"]:
        if r.get("downloaded_file_count", 0) <= 0:
            continue
        ann = r["announcement_id"]
        pdf = next((d for d in r["downloaded_files"] if d["content_disposition_filename"].lower().endswith(".pdf")), None)
        meta = Q.ann_meta().get(ann, {})
        rows.append({
            "announcement_id": ann,
            "provider": "한국토지주택공사",
            "official_page_url": r["official_page_url"],
            "attachment_url": pdf["download_url_without_session"] if pdf else "",
            "attachment_filename": pdf["content_disposition_filename"] if pdf else "",
            "document_format": "pdf",
            "region_sido": Q.ann_sido(ann) or "",
            "region_sigungu": Q.ann_sigungu(ann) or "",
            "announcement_date": meta.get("announcement_date_from_audit", ""),
            "housing_type": F.ann_housing_type(ann),
            "program_type": F.ann_housing_type(ann),
            "source_license_basis": LICENSE_LH,
            "redistribution_policy": REDIST,
            "acquisition_status": "acquired",
            "split": F.ann_split(ann),
            "notes": "v0.4/v0.5 acquired; extracted pages+facts+table cells internal.",
        })
    return rows


def backlog_rows() -> list[dict]:
    rows = []
    for prov, url in BACKLOG_VERIFIED:
        rows.append({
            "announcement_id": "", "provider": prov, "official_page_url": url,
            "attachment_url": "", "attachment_filename": "", "document_format": "",
            "region_sido": "", "region_sigungu": "", "announcement_date": "",
            "housing_type": "", "program_type": "",
            "source_license_basis": "Official public housing provider portal (verify per-source license at acquisition).",
            "redistribution_policy": REDIST,
            "acquisition_status": "backlog_portal_verified",
            "split": "",
            "notes": "Portal reachable (HTTP 200) this session; resolve specific 입주자모집공고 list + attachment URLs in acquisition phase.",
        })
    for prov in BACKLOG_UNVERIFIED:
        rows.append({
            "announcement_id": "", "provider": prov, "official_page_url": "",
            "attachment_url": "", "attachment_filename": "", "document_format": "",
            "region_sido": "", "region_sigungu": "", "announcement_date": "",
            "housing_type": "", "program_type": "",
            "source_license_basis": "Official local public housing provider (find official portal at acquisition).",
            "redistribution_policy": REDIST,
            "acquisition_status": "needs_official_url",
            "split": "",
            "notes": "Official portal URL not verified this session; a human/next phase must locate the official provider page.",
        })
    return rows


def main() -> int:
    rows = lh_rows() + backlog_rows()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    acq = sum(1 for r in rows if r["acquisition_status"] == "acquired")
    print(f"=== target manifest: {len(rows)} rows ({acq} acquired, {len(rows)-acq} backlog) -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
