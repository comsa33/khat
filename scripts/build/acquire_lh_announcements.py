#!/usr/bin/env python3
"""Download the LH announcement pages and attachments listed in the seed target manifest.

Usage: python3 scripts/build/acquire_lh_announcements.py [--manifest FILE] [--limit N] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
from email.message import Message
from pathlib import Path
from urllib.parse import unquote

import requests


from khat.paths import ROOT, LOCAL
DEFAULT_MANIFEST = ROOT / "data" / "construction" / "announcement_targets_seed.jsonl"
RAW_BASE = LOCAL / "raw" / "lh-sale-announcements-v04"
PROC_BASE = LOCAL / "lh-sale-announcements-v04"
AUDIT_BASE = LOCAL / "audit" / "lh-announcements"

SOURCE_ID = "lh-sale-announcements-v04"
ANNOUNCEMENT_LABEL_RE = re.compile(r"(입주자\s*모집\s*공고|입주자모집공고|모집공고문|공고문)", re.I)
ALLOWED_EXT_RE = re.compile(r"\.(pdf|hwp|hwpx)$", re.I)
EXCLUDE_LABEL_RE = re.compile(
    r"(팸플릿|팜플렛|개인정보|동의서|위임장|금융정보|자산보유|작성방법|신청서|별지|도면|평면|브로셔|카탈로그)",
    re.I,
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def safe_name(text: str, fallback: str) -> str:
    text = re.sub(r"[^\w.\-]+", "_", text.strip(), flags=re.UNICODE).strip("_")
    return text[:160] if text else fallback


def strip_tags(text: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def decode_content_disposition(value: str) -> str:
    if not value:
        return ""
    msg = Message()
    msg["content-disposition"] = value
    filename = msg.get_filename() or ""
    if not filename:
        m = re.search(r'filename\*?=(?:"([^"]+)"|([^;]+))', value)
        filename = (m.group(1) or m.group(2)).strip() if m else value
    filename = unquote(filename.strip("\"'"))
    try:
        filename = filename.encode("latin-1").decode("utf-8")
    except UnicodeError:
        pass
    return filename


def discover_file_downloader(page_text: str) -> list[dict]:
    """Return all fileDownLoad(file_id) links with visible labels."""
    links: list[dict] = []
    pattern = re.compile(
        r"<a[^>]+href=\"javascript:fileDownLoad\('([^']+)'\);\"[^>]*>(.*?)</a>",
        re.S | re.I,
    )
    for file_id, label_html in pattern.findall(page_text):
        label = strip_tags(label_html)
        links.append({"file_id": file_id, "label": label})
    return links


def is_announcement_attachment(label: str) -> bool:
    return bool(
        label
        and ALLOWED_EXT_RE.search(label)
        and ANNOUNCEMENT_LABEL_RE.search(label)
        and not EXCLUDE_LABEL_RE.search(label)
    )


def infer_extension(label: str, content_type: str, data: bytes) -> str:
    m = ALLOWED_EXT_RE.search(label)
    if m:
        return "." + m.group(1).lower()
    if data.startswith(b"%PDF"):
        return ".pdf"
    if data.startswith(bytes.fromhex("d0cf11e0")):
        return ".hwp"
    if "pdf" in content_type.lower():
        return ".pdf"
    return ".bin"


def acquire_target(session: requests.Session, target: dict, dry_run: bool = False) -> dict:
    ann_id = target["announcement_id"]
    url = target["official_page_url"]
    page_response = session.get(url, timeout=90)
    page_response.raise_for_status()
    page_bytes = page_response.content
    page_text = page_response.text
    visible_text = strip_tags(page_text)
    links = discover_file_downloader(page_text)
    selected = [link for link in links if is_announcement_attachment(link["label"])]

    raw_dir = RAW_BASE / ann_id
    proc_dir = PROC_BASE / ann_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    page_file = raw_dir / "official_page.html"
    visible_file = proc_dir / "official_page_visible_text.txt"
    if not dry_run:
        page_file.write_bytes(page_bytes)
        visible_file.write_text(visible_text, encoding="utf-8")

    downloaded = []
    for idx, link in enumerate(selected, 1):
        file_id = link["file_id"]
        dl_url = f"https://apply.lh.or.kr/lhapply/lhFile.do?fileid={file_id}"
        response = session.get(dl_url, timeout=180)
        response.raise_for_status()
        data = response.content
        content_type = response.headers.get("Content-Type", "")
        filename_header = decode_content_disposition(response.headers.get("Content-Disposition", ""))
        ext = infer_extension(link["label"] or filename_header, content_type, data)
        fname = f"{idx:02d}_{file_id}_{safe_name(link['label'], 'announcement')}{ext}"
        fname = re.sub(r"(\.(?:pdf|hwp|hwpx))\.(?:pdf|hwp|hwpx)$", r"\1", fname, flags=re.I)
        raw_path = raw_dir / fname
        if not dry_run:
            raw_path.write_bytes(data)
        downloaded.append({
            "file_id": file_id,
            "official_label": link["label"],
            "download_url_without_session": dl_url,
            "filename": f"workspace_local/raw/lh-sale-announcements-v04/{ann_id}/{fname}",
            "content_disposition_filename": filename_header,
            "content_type": content_type,
            "sha256": sha256_bytes(data),
            "bytes": len(data),
            "prefix_hex": data[:8].hex(),
        })

    return {
        "source_id": SOURCE_ID,
        "batch": "v0.4-source-expansion",
        "announcement_id": ann_id,
        "provider": target.get("provider", "한국토지주택공사"),
        "title_from_manifest": target.get("title", ""),
        "announcement_date_from_manifest": target.get("announcement_date", ""),
        "housing_type_from_manifest": target.get("housing_type", ""),
        "region_hint": target.get("region_hint", ""),
        "official_page_url": url,
        "page_file": f"workspace_local/raw/lh-sale-announcements-v04/{ann_id}/official_page.html",
        "page_sha256": sha256_bytes(page_bytes),
        "visible_text_file": f"workspace_local/processed/lh-sale-announcements-v04/{ann_id}/official_page_visible_text.txt",
        "all_file_links": links,
        "selected_file_links": selected,
        "downloaded_files": downloaded,
        "downloaded_file_count": len(downloaded),
        "license_observation": {
            "status": "public_official_announcement_internal_use",
            "basis": "Official LH public announcement page; keep raw files internal and release URL/locators only.",
        },
        "release_decision": "do_not_redistribute_raw_pdf_hwp_release_url_locators_and_qa_only",
        "notes": target.get("notes", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    targets = [t for t in load_jsonl(Path(args.manifest)) if t.get("include_for_v04", True)]
    if args.limit:
        targets = targets[: args.limit]

    RAW_BASE.mkdir(parents=True, exist_ok=True)
    PROC_BASE.mkdir(parents=True, exist_ok=True)
    AUDIT_BASE.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (KR-Housing-LongRAG-Bench dataset builder; research use)",
        "Referer": "https://apply.lh.or.kr/",
    })

    results = []
    for target in targets:
        try:
            audit = acquire_target(session, target, dry_run=args.dry_run)
            results.append(audit)
            audit_path = AUDIT_BASE / f"{target['announcement_id']}.json"
            if not args.dry_run:
                audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"OK {target['announcement_id']}: downloaded={audit['downloaded_file_count']} selected={len(audit['selected_file_links'])}")
        except Exception as exc:
            failed = {
                "announcement_id": target.get("announcement_id"),
                "official_page_url": target.get("official_page_url"),
                "ok": False,
                "error": repr(exc),
            }
            results.append(failed)
            print(f"FAIL {target.get('announcement_id')}: {exc}", file=sys.stderr)

    summary = {
        "batch": "v0.4-source-expansion",
        "target_count": len(targets),
        "ok_count": sum(1 for r in results if r.get("downloaded_file_count", 0) > 0),
        "zero_download_count": sum(1 for r in results if r.get("downloaded_file_count", 0) == 0),
        "failed_count": sum(1 for r in results if r.get("ok") is False),
        "downloaded_file_count": sum(r.get("downloaded_file_count", 0) for r in results),
        "results": results,
    }
    if not args.dry_run:
        (AUDIT_BASE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "results"}, ensure_ascii=False, indent=2))
    return 0 if summary["failed_count"] == 0 and summary["ok_count"] >= 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
