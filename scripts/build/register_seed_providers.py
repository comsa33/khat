#!/usr/bin/env python3
"""Add the SH, GH, iH and JPDC source rows to data/source_manifest.jsonl (skips rows already present).

Usage: python3 scripts/build/register_seed_providers.py
"""
from __future__ import annotations

import json
from pathlib import Path

from khat.paths import ROOT, LOCAL
MANIFEST = ROOT / "data" / "source_manifest.jsonl"

NEW_SOURCES = [
    {"source_id": "sh-announcements", "title": "서울주택도시공사(SH) 입주자모집공고", "provider": "서울주택도시공사",
     "access_url": "https://www.i-sh.co.kr/app/lay2/program/S48T1581C1617/www/brd/m_244/list.do"},
    {"source_id": "gh-announcements", "title": "경기주택도시공사(GH) 분양/임대 입주자모집공고", "provider": "경기주택도시공사",
     "access_url": "https://gh.or.kr/gh/announcement-of-salerental001.do"},
    {"source_id": "ih-announcements", "title": "인천도시공사(iH) 입주자모집공고", "provider": "인천도시공사",
     "access_url": "https://www.ih.co.kr/main/bbs/bbsMsgList.do?bcd=notice"},
    {"source_id": "jpdc-announcements", "title": "제주개발공사(JPDC) 주택 입주자모집공고", "provider": "제주특별자치도개발공사",
     "access_url": "https://www.jpdc.co.kr/"},
]
COMMON = {
    "source_type": "official_public_announcement_pages",
    "domain": "housing_announcement",
    "license_basis": "Official public housing provider announcement page; official-notice posture, raw kept internal.",
    "license_status": "public_official_announcement_internal_use",
    "redistribution_policy": "Release source URL + locators + QA only; do not redistribute raw PDF/HWP.",
    "corpus_inclusion": "manifest_only",
    "retrieved_at": "2026-06-05",
    "notes": "v0.5 source expansion (provider diversity). Raw under workspace_local/raw/<provider>-announcements/.",
}


def main() -> int:
    rows = [json.loads(l) for l in MANIFEST.open(encoding="utf-8") if l.strip()]
    have = {r["source_id"] for r in rows}
    added = 0
    for s in NEW_SOURCES:
        if s["source_id"] in have:
            continue
        rows.append({**s, **COMMON})
        added += 1
    with MANIFEST.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"registered {added} new provider sources (total {len(rows)} sources)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
