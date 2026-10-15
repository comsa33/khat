#!/usr/bin/env python3
"""Download MOLIT apartment trade rows from the data.go.kr API (key from DATA_GO_KR_SERVICE_KEY) and normalize them.

Usage: python3 scripts/build/acquire_molit_rows.py [--years 2025] [--lawd-codes CODE ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from urllib.parse import urlencode

import requests


from khat.paths import ROOT, LOCAL
RAW_DIR = LOCAL / "raw" / "molit-apt-trade-detail" / "real_rows"
PROC_DIR = LOCAL / "molit-apt-trade-detail"
AUDIT_DIR = LOCAL / "audit"

SOURCE_ID = "molit-apt-trade-detail"
ENDPOINT = "https://apis.data.go.kr/1613000/RTMSDataSvcAptTradeDev/getRTMSDataSvcAptTradeDev"

DEFAULT_LAWD_CODES = {
    "11110": "서울특별시 종로구",
    "11650": "서울특별시 서초구",
    "11680": "서울특별시 강남구",
    "11710": "서울특별시 송파구",
    "30110": "대전광역시 동구",
    "30200": "대전광역시 유성구",
}


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def read_key() -> str:
    key = os.environ.get("DATA_GO_KR_SERVICE_KEY", "").strip()
    if not key:
        raise SystemExit(
            "Missing data.go.kr key. Set DATA_GO_KR_SERVICE_KEY."
        )
    return key


def month_range(year: int) -> list[str]:
    return [f"{year}{month:02d}" for month in range(1, 13)]


def parse_items(xml_bytes: bytes) -> tuple[list[dict], dict]:
    root = ET.fromstring(xml_bytes)
    header = {
        "result_code": root.findtext(".//resultCode"),
        "result_msg": root.findtext(".//resultMsg"),
        "total_count": root.findtext(".//totalCount"),
        "page_no": root.findtext(".//pageNo"),
        "num_of_rows": root.findtext(".//numOfRows"),
    }
    if header["result_code"] not in (None, "000", "00"):
        raise RuntimeError(f"MOLIT API error: {header}")
    rows = []
    for item in root.findall(".//item"):
        row = {child.tag: (child.text or "").strip() for child in list(item)}
        rows.append(row)
    return rows, header


def call_api(session: requests.Session, key: str, lawd_cd: str, deal_ymd: str, page_no: int, num_rows: int) -> tuple[bytes, list[dict], dict]:
    params = {
        "ServiceKey": key,
        "LAWD_CD": lawd_cd,
        "DEAL_YMD": deal_ymd,
        "pageNo": str(page_no),
        "numOfRows": str(num_rows),
    }
    response = session.get(ENDPOINT, params=params, timeout=60)
    response.raise_for_status()
    data = response.content
    rows, header = parse_items(data)
    return data, rows, header


def profile_rows(rows: list[dict]) -> dict:
    field_counts = Counter()
    non_empty = Counter()
    area_counts = Counter()
    month_counts = Counter()
    for row in rows:
        for key, value in row.items():
            field_counts[key] += 1
            if value not in ("", None):
                non_empty[key] += 1
        area_counts[row.get("_lawd_name", "")] += 1
        month_counts[row.get("_deal_ymd", "")] += 1
    return {
        "row_count": len(rows),
        "field_count": len(field_counts),
        "fields": sorted(field_counts),
        "non_empty_top": non_empty.most_common(40),
        "area_counts": dict(sorted(area_counts.items())),
        "month_counts": dict(sorted(month_counts.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=[2025])
    parser.add_argument("--lawd-codes", nargs="+", default=sorted(DEFAULT_LAWD_CODES))
    parser.add_argument("--num-rows", type=int, default=1000)
    args = parser.parse_args()

    key = read_key()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (KR-Housing-LongRAG-Bench dataset builder; research use)",
        "Accept": "application/xml,text/xml,*/*",
    })

    all_rows: list[dict] = []
    calls: list[dict] = []
    for year in args.years:
        for deal_ymd in month_range(year):
            for lawd_cd in args.lawd_codes:
                lawd_name = DEFAULT_LAWD_CODES.get(lawd_cd, lawd_cd)
                raw, rows, header = call_api(session, key, lawd_cd, deal_ymd, 1, args.num_rows)
                raw_name = f"deal_ymd_{deal_ymd}_lawd_{lawd_cd}.xml"
                (RAW_DIR / raw_name).write_bytes(raw)
                for idx, row in enumerate(rows, 1):
                    enriched = dict(row)
                    enriched["_row_id"] = f"MOLIT-APTTRADE-{deal_ymd}-{lawd_cd}-{idx:04d}"
                    enriched["_deal_ymd"] = deal_ymd
                    enriched["_lawd_cd"] = lawd_cd
                    enriched["_lawd_name"] = lawd_name
                    all_rows.append(enriched)
                calls.append({
                    "deal_ymd": deal_ymd,
                    "lawd_cd": lawd_cd,
                    "lawd_name": lawd_name,
                    "endpoint_without_key": ENDPOINT,
                    "params_without_key": {"LAWD_CD": lawd_cd, "DEAL_YMD": deal_ymd, "pageNo": 1, "numOfRows": args.num_rows},
                    "raw_file": f"workspace_local/raw/{SOURCE_ID}/real_rows/{raw_name}",
                    "sha256": sha256_bytes(raw),
                    "bytes": len(raw),
                    "row_count": len(rows),
                    "result_code": header.get("result_code"),
                    "result_msg": header.get("result_msg"),
                    "total_count": header.get("total_count"),
                })

    rows_path = PROC_DIR / "rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    profile = profile_rows(all_rows)
    profile_path = PROC_DIR / "field_profile_v0.3.json"
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    audit = {
        "source_id": SOURCE_ID,
        "batch": "v0.3",
        "provider": "국토교통부",
        "dataset_title": "국토교통부_아파트 매매 실거래가 상세 자료",
        "registry_url": "https://www.data.go.kr/data/15126468/openapi.do?recommendDataYn=Y",
        "endpoint": ENDPOINT,
        "query_params": ["ServiceKey", "LAWD_CD", "DEAL_YMD", "pageNo", "numOfRows"],
        "secret_handling": "API key read locally and omitted from all outputs.",
        "license_observation": {
            "status": "usable_public_data_no_known_restriction",
            "basis": "Public Data Portal entry indicates 이용허락범위 제한 없음; re-check before public packaging.",
        },
        "release_decision": "internal_rows_only_for_now_public_release_qa_and_row_locators",
        "processed_rows_file": f"workspace_local/processed/{SOURCE_ID}/rows_v0.3.jsonl",
        "field_profile_file": f"workspace_local/processed/{SOURCE_ID}/field_profile_v0.3.json",
        "row_count": len(all_rows),
        "calls": calls,
    }
    audit_path = AUDIT_DIR / f"{SOURCE_ID}-v0.3-real-rows.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"OK: acquired {len(all_rows)} MOLIT apartment trade-detail rows from {len(calls)} calls")
    print(f"rows: {rows_path}")
    print(f"profile: {profile_path}")
    print(f"audit: {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
