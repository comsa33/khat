#!/usr/bin/env python3
"""Add a source row to data/source_manifest.jsonl for every provider in the ingested index (skips rows already present).

Usage: python3 scripts/build/register_provider_sources.py
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from urllib.parse import urlsplit

from khat.paths import ROOT, LOCAL
MANIFEST = ROOT / "data" / "source_manifest.jsonl"
INDEX = LOCAL / "index" / "providers.json"

COMMON = {
    "source_type": "official_public_announcement_pages",
    "domain": "housing_announcement",
    "license_basis": "Official public housing provider (지방공사/도시공사) announcement page; official-notice posture, raw kept internal.",
    "license_status": "public_official_announcement_internal_use",
    "redistribution_policy": "Release source URL + locators + QA only; do not redistribute raw PDF/HWP/HWPX.",
    "corpus_inclusion": "manifest_only",
    "retrieved_at": "2026-06-05",
    "notes": "v0.6 source expansion (provider/region diversity). Raw under workspace_local/raw/.",
}


def host_of(url: str) -> str:
    if not url:
        return ""
    s = urlsplit(url)
    return f"{s.scheme}://{s.netloc}/" if s.netloc else ""


def main() -> int:
    rows = [json.loads(l) for l in MANIFEST.open(encoding="utf-8") if l.strip()]
    have = {r["source_id"] for r in rows}
    idx = json.loads(INDEX.read_text(encoding="utf-8")) if INDEX.exists() else {"reports": []}
    by_src = OrderedDict()
    for r in idx.get("reports", []):
        if r.get("error"):
            continue
        sid = r["source_id"]
        if sid not in by_src:
            by_src[sid] = {"provider": r.get("provider", ""), "page_url": r.get("page_url", "")}
        elif not by_src[sid]["page_url"] and r.get("page_url"):
            by_src[sid]["page_url"] = r["page_url"]
    added = 0
    for sid, info in by_src.items():
        if sid in have:
            continue
        rows.append({
            "source_id": sid,
            "title": f"{info['provider']} 입주자모집공고",
            "provider": info["provider"],
            "access_url": host_of(info["page_url"]) or info["page_url"],
            **COMMON,
        })
        added += 1
    with MANIFEST.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"registered {added} new provider sources (total {len(rows)} sources)")
    for sid, info in by_src.items():
        if sid not in have:
            print(f"   + {sid}  ({info['provider']})  access_url={host_of(info['page_url'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
