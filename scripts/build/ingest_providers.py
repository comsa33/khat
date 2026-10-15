#!/usr/bin/env python3
"""Ingest the downloaded announcements of the other providers (PDF, HWP, HWPX): pages, numeric facts, table cells, merged index.

Usage: python3 scripts/build/ingest_providers.py [--list | --row N]
"""
from __future__ import annotations

import json
import html
import os
import re
import shutil
import struct
import subprocess
import zlib
from pathlib import Path

import olefile
import fitz

SOFFICE = shutil.which("soffice") or shutil.which("libreoffice")

from khat.paths import ROOT, LOCAL
RAW = LOCAL / "raw"
PROC = LOCAL
AUDIT = LOCAL / "audit"
MERGED = LOCAL / "index" / "providers.json"


def raw_file(path: str) -> Path:
    """A manifest path (workspace_local/raw/...) resolved under LOCAL/raw."""
    return LOCAL / "raw" / Path(path).relative_to("workspace_local/raw")

MANIFESTS = [
    RAW / "sh-announcements" / "sh_download_manifest.jsonl",
    RAW / "non_lh_provider_download_manifest.jsonl",
    RAW / "provider_expansion_v06_manifest.jsonl",
]
PROVIDER_CODE = {
    "SH": "sh", "GH": "gh", "iH": "ih", "JPDC": "jpdc",
    "서울주택도시공사": "sh", "경기주택도시공사": "gh", "인천도시공사": "ih", "제주특별자치도개발공사": "jpdc",
    "BMC": "bmc", "부산도시공사": "bmc",
    "대구도시개발공사": "dgdc", "대구도시공사": "dgdc",
    "대전도시공사": "dtco", "광주도시공사": "gjuco", "광주광역시도시공사": "gjuco",
    "충북개발공사": "cbdc", "충청북도개발공사": "cbdc",
    "충남개발공사": "cndc", "충청남도개발공사": "cndc",
    "전북개발공사": "jbdc", "전라북도개발공사": "jbdc",
    "전남개발공사": "jndc", "전라남도개발공사": "jndc",
    "경북개발공사": "gbdc", "경상북도개발공사": "gbdc",
    "경남개발공사": "gndc", "경상남도개발공사": "gndc",
    "강원개발공사": "gwdc", "강원도개발공사": "gwdc",
    "울산도시공사": "udco", "세종도시교통공사": "sjc", "세종시도시공사": "sjc",
}


def _romanize_code(provider: str) -> str:
    """Stable short code for an unmapped provider: ascii letters/digits if any, else a char-sum hash."""
    asc = re.sub(r"[^a-zA-Z0-9]", "", provider).lower()
    if asc:
        return asc[:8]
    return "prov" + str(sum(ord(c) for c in provider) % 100000)


def provider_code(provider: str) -> str:
    return PROVIDER_CODE.get(provider) or _romanize_code(provider)


def provider_source(provider: str) -> str:
    return f"{provider_code(provider)}-announcements"

NUM_FACT_RE = re.compile(r"\d[\d,]*(?:\.\d+)?\s*(?:세대|원|만원|천원|억원|㎡|%|일|회|개월|년|호|층|세)")
UNIT_RE = re.compile(r"(만원|천원|억원|원|세대|㎡|％|%|평|개월|년|월|일|회|호|층|배|점|명|가구|동)")


def jl(p: Path) -> list:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()] if p.exists() else []


def clean_title(t: str) -> str:
    """Strip list-page JS/HTML residue from a scraped announcement title -> human-readable 공고 제목.

    Some provider list pages embed an onclick handler (getDetailView('seq');return false;">) in the
    visible-text title. Remove the handler + any HTML tag/attribute residue, unescape entities, and
    normalize whitespace. The released question title must read as the real announcement title.
    """
    if not t:
        return ""
    t = html.unescape(t)
    t = re.sub(r"getDetailView\s*\([^)]*\)", " ", t)
    t = re.sub(r"return\s+false", " ", t, flags=re.I)
    t = re.sub(r"on\w+\s*=\s*(\"[^\"]*\"|'[^']*'|\S+)", " ", t, flags=re.I)
    t = re.sub(r"javascript:", " ", t, flags=re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r'[;"\'<>]', " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def slug(s: str, n: int = 48) -> str:
    s = re.sub(r"\.(pdf|hwp|hwpx)$", "", s, flags=re.I)
    s = re.sub(r"[^0-9A-Za-z가-힣]+", "-", s).strip("-").lower()
    return s[:n] or "doc"


def norm_page(text: str) -> str:
    lines = [l.rstrip() for l in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


# ---------------- HWP text extraction (olefile + zlib, tag 67) ----------------
def hwp_text(path: Path) -> str:
    ole = olefile.OleFileIO(str(path))
    try:
        fh = ole.openstream("FileHeader").read()
        compressed = bool(fh[36] & 1)
        secs = sorted([s for s in ole.listdir() if s and s[0] == "BodyText"], key=lambda x: x[-1])
        chunks = []
        for s in secs:
            data = ole.openstream(s).read()
            if compressed:
                try:
                    data = zlib.decompress(data, -15)
                except zlib.error:
                    continue
            i, n = 0, len(data)
            while i + 4 <= n:
                hdr = struct.unpack("<I", data[i:i + 4])[0]
                i += 4
                tag = hdr & 0x3FF
                size = (hdr >> 20) & 0xFFF
                if size == 0xFFF:
                    if i + 4 > n:
                        break
                    size = struct.unpack("<I", data[i:i + 4])[0]
                    i += 4
                rec = data[i:i + size]
                i += size
                if tag == 67:
                    try:
                        chunks.append(rec.decode("utf-16le", "ignore"))
                    except Exception:
                        pass
        txt = "\n".join(chunks)
    finally:
        ole.close()
    txt = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", txt)
    return txt


def hwp_to_pdf(path: Path, workdir: Path) -> Path | None:
    """Convert HWP/HWPX -> PDF via LibreOffice (handles HWP 3.0/한글97 and HWP 5.0). Returns PDF path."""
    if not SOFFICE:
        return None
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [SOFFICE, "-env:UserInstallation=file:///tmp/lo_profile_khat", "--headless",
             "--convert-to", "pdf", "--outdir", str(workdir), str(path)],
            check=True, capture_output=True, timeout=240)
    except Exception:
        return None
    out = workdir / (path.stem + ".pdf")
    return out if out.exists() else None


def hwp_pages(path: Path, chars_per_page: int = 2600) -> list:
    txt = hwp_text(path)
    paras = [p.strip() for p in txt.split("\n")]
    pages, buf, size = [], [], 0
    for p in paras:
        if not p:
            continue
        buf.append(p)
        size += len(p)
        if size >= chars_per_page:
            pages.append("\n".join(buf))
            buf, size = [], 0
    if buf:
        pages.append("\n".join(buf))
    return pages or [txt]


# ---------------- PDF extraction ----------------
def pdf_pages(path: Path, proc_dir: Path) -> list:
    out_txt = proc_dir / (path.stem + "_pdf.txt")
    subprocess.run(["pdftotext", "-layout", str(path), str(out_txt)], check=True)
    raw = out_txt.read_text(encoding="utf-8", errors="replace")
    pages = [norm_page(p) for p in raw.split("\f")]
    return [p for p in pages if p.strip()]


def pdf_cells(path: Path, ann_id: str, n_pages: int) -> tuple[list, list]:
    doc = fitz.open(str(path))
    tables_rows, cell_rows = [], []
    use = doc.page_count == n_pages
    for pno in range(doc.page_count):
        page_no = pno + 1
        if not use and pno >= n_pages:
            break
        page_id = f"{ann_id}-p{page_no:03d}"
        try:
            found = doc[pno].find_tables()
        except Exception:
            continue
        for ti, tab in enumerate(found.tables, 1):
            try:
                grid = tab.extract()
            except Exception:
                continue
            if not grid or len(grid) < 2 or len(grid[0]) < 2:
                continue
            table_id = f"{page_id}-t{ti:02d}"
            n_rows = len(grid)
            n_cols = max(len(r) for r in grid)
            nonempty = sum(1 for r in grid for c in r if (c or "").strip())
            conf = round(nonempty / max(n_rows * n_cols, 1), 3)
            col_headers = [re.sub(r"\s+", " ", (grid[0][j] or "")).strip() if j < len(grid[0]) else "" for j in range(n_cols)]
            tables_rows.append({"table_id": table_id, "page_id": page_id, "page_no": page_no,
                                "n_rows": n_rows, "n_cols": n_cols, "col_headers": col_headers,
                                "bbox": [round(x, 1) for x in tab.bbox], "source_format": "pdf", "confidence": conf})
            for ri, row in enumerate(grid):
                rh = re.sub(r"\s+", " ", (row[0] or "")).strip() if row else ""
                for ci in range(n_cols):
                    ct = re.sub(r"\s+", " ", (row[ci] if ci < len(row) else "") or "").strip()
                    if not ct:
                        continue
                    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", ct)
                    val = None
                    if m:
                        try:
                            val = float(m.group(0).replace(",", ""))
                            val = int(val) if val == int(val) else val
                        except ValueError:
                            val = None
                    u = UNIT_RE.search(ct)
                    cconf = 1.0 if len(ct) <= 40 else (0.6 if len(ct) <= 120 else 0.3)
                    cell_rows.append({"table_id": table_id, "page_id": page_id, "row_index": ri, "col_index": ci,
                                      "row_header": rh, "col_header": col_headers[ci] if ci < len(col_headers) else "",
                                      "cell_text": ct, "normalized_value": val, "unit": (u.group(1) if u else None),
                                      "bbox": None, "source_format": "pdf", "confidence": round(min(conf, cconf), 3)})
    doc.close()
    return tables_rows, cell_rows


def numeric_facts(ann_id: str, page_id: str, page_no: int, src: str, text: str) -> list:
    out = []
    for idx, m in enumerate(NUM_FACT_RE.finditer(text), 1):
        a, b = max(0, m.start() - 80), min(len(text), m.end() + 80)
        out.append({"fact_id": f"{page_id}-num-{idx:03d}", "announcement_id": ann_id, "page_id": page_id,
                    "page_no": page_no, "value_text": m.group(0).strip(),
                    "locator": f"{src}#page={page_no}", "local_snippet": re.sub(r"\s+", " ", text[a:b]).strip()})
    return out


# ---------------- driver ----------------
def collect_rows() -> list:
    rows = []
    seen_ann = {}
    for mf in MANIFESTS:
        for r in jl(mf):
            if r.get("status") != "downloaded":
                continue
            path = r.get("path")
            if not path or not raw_file(path).exists():
                continue
            prov = r.get("provider")
            if not prov:
                continue
            src = provider_source(prov)
            code = provider_code(prov)
            stem = Path(path).stem
            base = f"{code}-{slug(stem)}"
            ann_id = base
            k = 2
            while ann_id in seen_ann and seen_ann[ann_id] != path:
                ann_id = f"{base}-{k}"
                k += 1
            seen_ann[ann_id] = path
            rows.append({"provider": prov, "source_id": src, "announcement_id": ann_id,
                         "path": path, "title": clean_title(r.get("title", "")), "page_url": r.get("page_url", ""),
                         "download_url": r.get("download_url", ""),
                         "region_sido": r.get("region_sido", ""), "region_sigungu": r.get("region_sigungu", ""),
                         "housing_type": r.get("housing_type", "")})
    return rows


def ingest(row: dict) -> dict:
    path = raw_file(row["path"])
    ann_id = row["announcement_id"]
    src = row["source_id"]
    proc_dir = PROC / src / ann_id
    proc_dir.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext == ".pdf":
        pages = pdf_pages(path, proc_dir)
        tables_rows, cell_rows = pdf_cells(path, ann_id, len(pages))
        fmt = "pdf"
    elif ext in (".hwp", ".hwpx"):
        pdf = hwp_to_pdf(path, proc_dir / "_lo")
        if pdf is not None:
            pages = pdf_pages(pdf, proc_dir)
            tables_rows, cell_rows = pdf_cells(pdf, ann_id, len(pages))
            fmt = f"{ext[1:]}_via_soffice"
            shutil.rmtree(proc_dir / "_lo", ignore_errors=True)
        else:
            tables_rows, cell_rows = [], []
            fmt = ext[1:] + "_textonly"
            if ext == ".hwp":
                pages = hwp_pages(path)
            else:
                import zipfile
                texts = []
                with zipfile.ZipFile(path) as z:
                    for nm in z.namelist():
                        if re.search(r"Contents/section\d+\.xml$", nm):
                            texts.append(re.sub(r"<[^>]+>", " ", z.read(nm).decode("utf-8", "ignore")))
                big = re.sub(r"\s+", " ", " ".join(texts))
                pages = [big[i:i + 2600] for i in range(0, len(big), 2600)] or [big]
    else:
        return {"announcement_id": ann_id, "error": f"unsupported ext {ext}", "pages": 0}

    page_rows, fact_rows = [], []
    for pno, text in enumerate(pages, 1):
        page_id = f"{ann_id}-p{pno:03d}"
        page_rows.append({"source_id": src, "announcement_id": ann_id, "page_id": page_id, "page_no": pno,
                          "locator": f"{path.name}#page={pno}", "char_count": len(text), "text": text})
        fact_rows.extend(numeric_facts(ann_id, page_id, pno, path.name, text))

    with (proc_dir / "document_pages.jsonl").open("w", encoding="utf-8") as f:
        for r in page_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (proc_dir / "numeric_facts.jsonl").open("w", encoding="utf-8") as f:
        for r in fact_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (proc_dir / "tables.jsonl").open("w", encoding="utf-8") as f:
        for r in tables_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (proc_dir / "table_cells.jsonl").open("w", encoding="utf-8") as f:
        for r in cell_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {"source_id": src, "provider": row["provider"], "announcement_id": ann_id,
              "title": row["title"], "page_url": row["page_url"], "document_format": fmt,
              "input_file": row["path"], "pages": len(page_rows), "chars": sum(p["char_count"] for p in page_rows),
              "numeric_facts": len(fact_rows), "tables": len(tables_rows), "cells": len(cell_rows),
              "manifest_region_sido": row.get("region_sido", ""), "manifest_region_sigungu": row.get("region_sigungu", ""),
              "manifest_housing_type": row.get("housing_type", "")}
    (proc_dir / "extraction_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_merged_index() -> dict:
    """Rebuild the merged index from the per-announcement extraction_report.json files (no extraction)."""
    from collections import Counter
    reports = []
    src_dirs = sorted(d for d in PROC.glob("*-announcements")
                      if d.is_dir() and not d.name.startswith("lh-"))
    for base in src_dirs:
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            rp = d / "extraction_report.json"
            if rp.exists():
                try:
                    reports.append(json.loads(rp.read_text(encoding="utf-8")))
                except json.JSONDecodeError:
                    reports.append({"announcement_id": d.name, "error": "bad extraction_report.json"})
    by_prov = Counter(r.get("provider") for r in reports if not r.get("error"))
    idx = {"providers": dict(by_prov), "announcement_count": len([r for r in reports if not r.get("error")]),
           "total_pages": sum(r.get("pages", 0) for r in reports),
           "total_cells": sum(r.get("cells", 0) for r in reports),
           "errors": [r for r in reports if r.get("error")], "reports": reports}
    MERGED.write_text(json.dumps(idx, ensure_ascii=False, indent=1), encoding="utf-8")
    return idx


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Memory-safe per-file provider ingestion.")
    ap.add_argument("--list", action="store_true", help="list manifest rows as 'index\\tprovider\\tann_id'")
    ap.add_argument("--row", type=int, default=None, help="process ONLY this manifest row index (isolated process)")
    ap.add_argument("--provider", default=None, help="restrict to one provider code (SH/GH/iH/JPDC)")
    ap.add_argument("--merge-index", action="store_true", help="rebuild merged index from existing reports only")
    args = ap.parse_args()

    if args.merge_index:
        idx = build_merged_index()
        print(f"=== MERGE-INDEX: {idx['announcement_count']} announcements, providers={idx['providers']}, "
              f"pages={idx['total_pages']}, cells={idx['total_cells']}, errors={len(idx['errors'])} -> {MERGED}")
        return 0

    rows = collect_rows()
    if args.provider:
        rows = [r for r in rows if r["provider"] == args.provider]
    if args.list:
        for i, r in enumerate(rows):
            print(f"{i}\t{r['provider']}\t{r['announcement_id']}")
        return 0

    if args.row is not None:
        all_rows = collect_rows()
        if not (0 <= args.row < len(all_rows)):
            print(f"row {args.row} out of range 0..{len(all_rows)-1}")
            return 2
        row = all_rows[args.row]
        try:
            rep = ingest(row)
            print(f"OK row {args.row}: {rep.get('provider')} {row['announcement_id'][:42]} "
                  f"fmt={rep.get('document_format','?')} pages={rep.get('pages',0)} cells={rep.get('cells',0)}")
        except Exception as exc:
            print(f"FAIL row {args.row}: {row['announcement_id']} {exc!r}")
            return 1
        return 0

    reports = []
    for row in rows:
        try:
            rep = ingest(row)
        except Exception as exc:
            rep = {"announcement_id": row["announcement_id"], "provider": row["provider"], "error": repr(exc)}
        reports.append(rep)
        print(f"  {rep.get('provider', row['provider']):5s} {row['announcement_id'][:40]:40s} "
              f"fmt={rep.get('document_format', '?'):4s} pages={rep.get('pages', 0):3d} cells={rep.get('cells', 0):4d} "
              f"{'ERR ' + rep['error'][:40] if rep.get('error') else ''}")
    idx = build_merged_index()
    print(f"=== INGEST: {idx['announcement_count']} announcements, providers={idx['providers']}, "
          f"pages={idx['total_pages']}, cells={idx['total_cells']}, errors={len(idx['errors'])} -> {MERGED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
