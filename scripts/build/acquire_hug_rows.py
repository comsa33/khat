#!/usr/bin/env python3
"""Download HUG sale-history rows from the data.go.kr API (key from HUG_API_KEY) and normalize them.

Usage: python3 scripts/build/acquire_hug_rows.py [--years 2023 2024 2025] [--areas AREA ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlencode

import requests


from khat.paths import ROOT, LOCAL
RAW_DIR = LOCAL / "raw" / "hug-sale-history" / "real_rows"
PROC_DIR = LOCAL / "hug-sale-history"
AUDIT_DIR = LOCAL / "audit"

ENDPOINT = "https://www.khug.or.kr/infoDistributionHistory.do"
AREA_CODES = {
    "01": "서울특별시",
    "02": "부산광역시",
    "03": "대구광역시",
    "04": "인천광역시",
    "05": "광주광역시",
    "06": "대전광역시",
    "07": "경기도",
    "08": "강원도",
    "09": "충청북도",
    "10": "충청남도",
    "11": "전라북도",
    "12": "전라남도",
    "13": "경상북도",
    "14": "경상남도",
    "15": "제주특별자치도",
    "16": "울산광역시",
}


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def read_key() -> str:
    key = os.environ.get("HUG_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "Missing HUG API key. Set HUG_API_KEY."
        )
    return key


def parse_response(data: bytes) -> list[dict]:
    text = data.decode("utf-8-sig", errors="replace").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"HUG response was not JSON: {text[:200]!r}") from exc
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        if str(payload.get("ERROR_CODE", "")).strip():
            raise RuntimeError(f"HUG API error: {payload}")
        for key in ("data", "items", "response", "list", "rows"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
    raise RuntimeError(f"Unsupported HUG response shape: {type(payload).__name__}")


def call_api(session: requests.Session, api_key: str, year: str, area_code: str) -> tuple[bytes, list[dict]]:
    params = {"API_KEY": api_key, "STDD_YYM": year, "AREA_DCD": area_code}
    response = session.get(ENDPOINT, params=params, timeout=60)
    response.raise_for_status()
    data = response.content
    return data, parse_response(data)


def profile_rows(rows: list[dict]) -> dict:
    field_counts = Counter()
    non_empty = Counter()
    area_counts = Counter()
    year_counts = Counter()
    numeric_candidates: dict[str, int] = defaultdict(int)
    for row in rows:
        for key, value in row.items():
            field_counts[key] += 1
            if value not in ("", None):
                non_empty[key] += 1
                if isinstance(value, str) and value.replace(",", "").replace(".", "", 1).isdigit():
                    numeric_candidates[key] += 1
        area_counts[row.get("AREA_DCD_NM") or row.get("_area_name") or ""] += 1
        year_counts[str(row.get("_query_year", ""))] += 1
    return {
        "row_count": len(rows),
        "field_count": len(field_counts),
        "fields": sorted(field_counts),
        "non_empty_top": non_empty.most_common(30),
        "numeric_candidate_top": Counter(numeric_candidates).most_common(30),
        "area_counts": dict(sorted(area_counts.items())),
        "year_counts": dict(sorted(year_counts.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", default=["2023", "2024", "2025"])
    parser.add_argument("--areas", nargs="+", default=sorted(AREA_CODES))
    args = parser.parse_args()

    api_key = read_key()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (KR-Housing-LongRAG-Bench dataset builder; research use)",
        "Accept": "application/json,text/plain,*/*",
    })

    all_rows: list[dict] = []
    calls: list[dict] = []
    for year in args.years:
        for area_code in args.areas:
            if area_code not in AREA_CODES:
                raise SystemExit(f"Unknown AREA_DCD: {area_code}")
            raw, rows = call_api(session, api_key, year, area_code)
            raw_name = f"stdd_yym_{year}_area_{area_code}.json"
            (RAW_DIR / raw_name).write_bytes(raw)
            digest = sha256_bytes(raw)
            for idx, row in enumerate(rows, 1):
                enriched = dict(row)
                enriched["_row_id"] = f"HUG-{year}-{area_code}-{idx:04d}"
                enriched["_query_year"] = year
                enriched["_query_area_dcd"] = area_code
                enriched["_query_area_name"] = AREA_CODES[area_code]
                all_rows.append(enriched)
            calls.append({
                "year": year,
                "area_code": area_code,
                "area_name": AREA_CODES[area_code],
                "url_without_key": f"{ENDPOINT}?{urlencode({'STDD_YYM': year, 'AREA_DCD': area_code})}",
                "raw_file": f"workspace_local/raw/hug-sale-history/real_rows/{raw_name}",
                "sha256": digest,
                "bytes": len(raw),
                "row_count": len(rows),
            })

    rows_path = PROC_DIR / "rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    profile = profile_rows(all_rows)
    profile_path = PROC_DIR / "field_profile_v0.3.json"
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    audit = {
        "source_id": "hug-sale-history",
        "batch": "v0.3",
        "provider": "주택도시보증공사",
        "endpoint": ENDPOINT,
        "query_params": ["API_KEY", "STDD_YYM", "AREA_DCD"],
        "secret_handling": "API key read locally and omitted from all outputs.",
        "license_observation": {
            "status": "usable_public_data_no_known_restriction",
            "basis": "Public Data Portal / provider open API entry; re-check before public packaging.",
        },
        "release_decision": "internal_rows_only_for_now_public_release_qa_and_row_locators",
        "processed_rows_file": "workspace_local/processed/hug-sale-history/rows_v0.3.jsonl",
        "field_profile_file": "workspace_local/processed/hug-sale-history/field_profile_v0.3.json",
        "row_count": len(all_rows),
        "calls": calls,
    }
    audit_path = AUDIT_DIR / "hug-sale-history-v0.3-real-rows.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"OK: acquired {len(all_rows)} HUG rows from {len(calls)} calls")
    print(f"rows: {rows_path}")
    print(f"profile: {profile_path}")
    print(f"audit: {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
