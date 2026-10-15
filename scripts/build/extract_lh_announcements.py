#!/usr/bin/env python3
"""Page text (pdftotext) and numeric facts of the LH announcement PDFs, with a merged index.

Usage: python3 scripts/build/extract_lh_announcements.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


from khat.paths import ROOT, LOCAL
RAW_BASE = LOCAL / "raw" / "lh-sale-announcements-v04"
PROC_BASE = LOCAL / "lh-sale-announcements-v04"
AUDIT_BASE = LOCAL / "audit" / "lh-announcements"
MERGED_INDEX = LOCAL / "index" / "lh_announcements.json"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def normalize_page_text(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def extract_numeric_facts(announcement_id: str, page_id: str, page_no: int, pdf_name: str, text: str) -> list[dict]:
    facts: list[dict] = []
    pattern = re.compile(r"\d[\d,]*(?:\.\d+)?\s*(?:세대|원|만원|천원|㎡|%|일|회|개월|년|호|층)")
    for idx, match in enumerate(pattern.finditer(text), 1):
        snippet_start = max(0, match.start() - 80)
        snippet_end = min(len(text), match.end() + 80)
        snippet = re.sub(r"\s+", " ", text[snippet_start:snippet_end]).strip()
        facts.append({
            "fact_id": f"{page_id}-num-{idx:03d}",
            "announcement_id": announcement_id,
            "page_id": page_id,
            "page_no": page_no,
            "value_text": match.group(0).strip(),
            "locator": f"{pdf_name}#page={page_no}",
            "local_snippet": snippet,
        })
    return facts


def pdf_page_count(pdf_path: Path) -> int | None:
    try:
        proc = subprocess.run(["pdfinfo", str(pdf_path)], check=True, capture_output=True, text=True)
    except Exception:
        return None
    m = re.search(r"^Pages:\s+(\d+)", proc.stdout, flags=re.M)
    return int(m.group(1)) if m else None


def extract_pdf(announcement_id: str, pdf_path: Path, audit: dict | None) -> dict:
    proc_dir = PROC_BASE / announcement_id
    proc_dir.mkdir(parents=True, exist_ok=True)
    txt_path = proc_dir / f"{pdf_path.stem}_pdf.txt"
    subprocess.run(["pdftotext", "-layout", str(pdf_path), str(txt_path)], check=True)
    raw_text = txt_path.read_text(encoding="utf-8", errors="replace")
    pages = [normalize_page_text(page) for page in raw_text.split("\f")]
    pages = [page for page in pages if page.strip()]

    page_rows = []
    numeric_rows = []
    for page_no, text in enumerate(pages, 1):
        page_id = f"{announcement_id}-p{page_no:03d}"
        locator = f"{pdf_path.name}#page={page_no}"
        row = {
            "source_id": "lh-sale-announcements-v04",
            "announcement_id": announcement_id,
            "page_id": page_id,
            "page_no": page_no,
            "locator": locator,
            "char_count": len(text),
            "text": text,
        }
        page_rows.append(row)
        numeric_rows.extend(extract_numeric_facts(announcement_id, page_id, page_no, pdf_path.name, text))

    pages_path = proc_dir / "document_pages.jsonl"
    with pages_path.open("w", encoding="utf-8") as f:
        for row in page_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    numeric_path = proc_dir / "numeric_facts.jsonl"
    with numeric_path.open("w", encoding="utf-8") as f:
        for row in numeric_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    report = {
        "source_id": "lh-sale-announcements-v04",
        "announcement_id": announcement_id,
        "title_from_audit": audit.get("title_from_manifest", "") if audit else "",
        "announcement_date_from_audit": audit.get("announcement_date_from_manifest", "") if audit else "",
        "housing_type_from_audit": audit.get("housing_type_from_manifest", "") if audit else "",
        "region_hint": audit.get("region_hint", "") if audit else "",
        "official_page_url": audit.get("official_page_url", "") if audit else "",
        "input_pdf": f"workspace_local/raw/lh-sale-announcements-v04/{announcement_id}/{pdf_path.name}",
        "text_file": f"workspace_local/processed/lh-sale-announcements-v04/{announcement_id}/{txt_path.name}",
        "document_pages_file": f"workspace_local/processed/lh-sale-announcements-v04/{announcement_id}/document_pages.jsonl",
        "numeric_facts_file": f"workspace_local/processed/lh-sale-announcements-v04/{announcement_id}/numeric_facts.jsonl",
        "pdfinfo_page_count": pdf_page_count(pdf_path),
        "extracted_page_count": len(page_rows),
        "char_count": sum(row["char_count"] for row in page_rows),
        "numeric_fact_count": len(numeric_rows),
        "extraction_tool": "pdftotext -layout",
    }
    (proc_dir / "extraction_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not RAW_BASE.exists():
        raise SystemExit(f"Missing raw announcement directory: {RAW_BASE}")
    PROC_BASE.mkdir(parents=True, exist_ok=True)

    reports: list[dict] = []
    target_dirs = [p for p in sorted(RAW_BASE.iterdir()) if p.is_dir()]
    if args.limit:
        target_dirs = target_dirs[: args.limit]

    for ann_dir in target_dirs:
        ann_id = ann_dir.name
        audit_path = AUDIT_BASE / f"{ann_id}.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
        pdfs = sorted(ann_dir.glob("*.pdf"))
        if not pdfs:
            print(f"SKIP {ann_id}: no PDF")
            continue
        report = extract_pdf(ann_id, pdfs[0], audit)
        reports.append(report)
        print(
            f"OK {ann_id}: pages={report['extracted_page_count']} "
            f"chars={report['char_count']} facts={report['numeric_fact_count']}"
        )

    merged_pages = []
    merged_facts = []
    for report in reports:
        ann_id = report["announcement_id"]
        proc_dir = PROC_BASE / ann_id
        merged_pages.extend(load_jsonl(proc_dir / "document_pages.jsonl"))
        merged_facts.extend(load_jsonl(proc_dir / "numeric_facts.jsonl"))

    index = {
        "source_id": "lh-sale-announcements-v04",
        "announcement_count": len(reports),
        "page_count": len(merged_pages),
        "char_count": sum(row.get("char_count", 0) for row in merged_pages),
        "numeric_fact_count": len(merged_facts),
        "announcements": reports,
    }
    MERGED_INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {MERGED_INDEX}")
    print(json.dumps({k: index[k] for k in ["announcement_count", "page_count", "char_count", "numeric_fact_count"]}, ensure_ascii=False))
    return 0 if reports else 1


if __name__ == "__main__":
    raise SystemExit(main())
