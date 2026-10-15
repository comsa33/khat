#!/usr/bin/env python3
"""Download the statute texts and the source metadata pages into local/raw, with a sha256 record per source.

Usage: python3 scripts/build/acquire_sources.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import requests

from khat.paths import ROOT, LOCAL
RAW = LOCAL / "raw"
AUDIT = LOCAL / "audit"
DOWNLOAD_DATE = "2026-06-04"

UA = {
    "User-Agent": "Mozilla/5.0 (KR-Housing-LongRAG-Bench dataset builder; research use)",
    "Referer": "https://www.law.go.kr/",
}

# ---- Sources -------------------------------------------------
STATUTES = [
    {"source_id": "law-housing-supply-rule", "lsiSeq": 171762,
     "title": "주택공급에 관한 규칙"},
    {"source_id": "law-public-housing-special-act-rule", "lsiSeq": 204719,
     "title": "공공주택 특별법 시행규칙"},
    {"source_id": "law-private-rental-housing-special-act", "lsiSeq": 174472,
     "title": "민간임대주택에 관한 특별법"},
]

HTML_PAGES = [
    {"source_id": "kogl-license-guide",
     "url": "https://www.kogl.or.kr/info/license.do",
     "kind": "policy_html",
     "license_status": "policy_reference"},
    {"source_id": "public-data-portal-use-policy",
     "url": "https://www.data.go.kr/en/ugs/selectPortalPolicyView.do",
     "kind": "policy_html",
     "license_status": "policy_reference"},
    {"source_id": "hug-sale-history",
     "url": "https://www.data.go.kr/data/15057686/openapi.do",
     "kind": "metadata_html",
     "license_status": "usable_public_data_no_known_restriction",
     "data_blocker": "Bulk records require a data.go.kr serviceKey bound to a registered IP. "
                     "No key available in this environment; only the API spec/metadata page was "
                     "acquired. Raw rows NOT downloaded."},
    {"source_id": "molit-apt-official-price-2025",
     "url": "https://www.data.go.kr/data/3073746/fileData.do",
     "kind": "metadata_html",
     "license_status": "usable_public_data_no_known_restriction",
     "data_blocker": "Full CSV is ~15.58M rows behind a portal file-download flow (session/agreement). "
                     "Only the dataset metadata page was acquired. Bulk CSV NOT downloaded."},
]


def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def write_audit(source_id: str, payload: dict) -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / f"{source_id}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def discover_efyd(lsiSeq: int) -> tuple[str, str]:
    """Return (efYd 'YYYYMMDD', human '[시행 ...]') from the lsInfoP shell page."""
    shell = requests.get(
        f"https://www.law.go.kr/LSW/lsInfoP.do?lsiSeq={lsiSeq}", headers=UA, timeout=30
    ).text
    m = re.search(r"\[시행\s*(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.", shell)
    if not m:
        m2 = re.search(r"efYd\s*=\s*'(\d{8})'", shell)
        if m2:
            d = m2.group(1)
            return d, f"[시행 {d[:4]}. {int(d[4:6])}. {int(d[6:8])}.]"
        raise RuntimeError(f"could not find efYd for lsiSeq={lsiSeq}")
    y, mo, da = m.group(1), int(m.group(2)), int(m.group(3))
    efyd = f"{y}{mo:02d}{da:02d}"
    return efyd, m.group(0) + "]"


def fetch_statute(s: dict) -> dict:
    sid = s["source_id"]
    seq = s["lsiSeq"]
    efyd, human = discover_efyd(seq)
    url = f"https://www.law.go.kr/LSW/lsInfoR.do?lsiSeq={seq}&chrClsCd=010202&efYd={efyd}"
    r = requests.get(url, headers=UA, timeout=120)
    body = r.content
    text = r.text
    n_art = len(re.findall(r"제\d+조(?:의\d+)?\s*[\(（]", text))
    if n_art == 0 or len(body) < 20000:
        raise RuntimeError(f"{sid}: body looks empty (bytes={len(body)}, articles={n_art})")
    out_dir = RAW / sid
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{sid}_lsInfoR_efYd{efyd}.html"
    (out_dir / fname).write_bytes(body)
    audit = {
        "source_id": sid,
        "downloaded_at": DOWNLOAD_DATE,
        "provider": "법제처 국가법령정보센터",
        "title": s["title"],
        "pinned_version": {"lsiSeq": seq, "efYd": efyd, "effective_date_human": human},
        "files": [{
            "filename": fname,
            "url": url,
            "registry_url": f"https://www.law.go.kr/LSW/lsInfoP.do?lsiSeq={seq}",
            "sha256": sha256_bytes(body),
            "bytes": len(body),
            "parsed_article_markers": n_art,
        }],
        "license_observation": {
            "status": "usable_statutory_text",
            "evidence_url": "https://www.law.go.kr/",
            "evidence_text_short": "Korean Copyright Act Art. 7 excludes laws/orders/ordinances/rules "
                                   "from copyright protection.",
        },
        "release_decision": "url_and_labels_only",
        "notes": "Full statute body stored internally only; public release carries locators + short answers.",
    }
    write_audit(sid, audit)
    return {"source_id": sid, "ok": True, "bytes": len(body), "articles": n_art,
            "efYd": efyd, "file": fname}


def fetch_html_page(p: dict) -> dict:
    sid = p["source_id"]
    r = requests.get(p["url"], headers=UA, timeout=60)
    body = r.content
    out_dir = RAW / sid
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{sid}.html"
    (out_dir / fname).write_bytes(body)
    is_meta = p["kind"] == "metadata_html"
    audit = {
        "source_id": sid,
        "downloaded_at": DOWNLOAD_DATE,
        "files": [{
            "filename": fname,
            "url": p["url"],
            "sha256": sha256_bytes(body),
            "bytes": len(body),
            "http_status": r.status_code,
        }],
        "license_observation": {
            "status": p["license_status"],
            "evidence_url": p["url"],
            "evidence_text_short": ("Public Data Portal entry / KOGL guide; factual policy or schema "
                                    "metadata only." if not is_meta
                                    else "Public Data Portal dataset metadata page (no bulk data)."),
        },
        "release_decision": "url_and_labels_only",
    }
    if is_meta:
        audit["data_acquisition_blocker"] = p["data_blocker"]
        audit["release_decision"] = "url_and_labels_only_metadata_only"
    write_audit(sid, audit)
    return {"source_id": sid, "ok": True, "bytes": len(body), "kind": p["kind"],
            "status": r.status_code, "file": fname}


def main() -> int:
    results = []
    for s in STATUTES:
        try:
            results.append(("statute", fetch_statute(s)))
        except Exception as e:
            results.append(("statute", {"source_id": s["source_id"], "ok": False, "error": repr(e)}))
    for p in HTML_PAGES:
        try:
            results.append((p["kind"], fetch_html_page(p)))
        except Exception as e:
            results.append((p["kind"], {"source_id": p["source_id"], "ok": False, "error": repr(e)}))

    print("=== ACQUISITION SUMMARY ===")
    ok = 0
    for kind, r in results:
        if r.get("ok"):
            ok += 1
            extra = (f"articles={r.get('articles')} efYd={r.get('efYd')}" if kind == "statute"
                     else f"http={r.get('status')}")
            print(f"  OK   [{kind:13s}] {r['source_id']:42s} {r['bytes']:>8}B  {extra}")
        else:
            print(f"  FAIL [{kind:13s}] {r['source_id']:42s} {r.get('error')}")
    print(f"--- {ok}/{len(results)} sources acquired; audit -> {AUDIT}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    raise SystemExit(main())
