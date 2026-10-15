#!/usr/bin/env python3
"""Download the SH announcement attachments found by the board search and write the SH download manifest.

Usage: python3 scripts/build/acquire_sh_announcements.py
"""
import html
import json
import re
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener
from http.cookiejar import CookieJar

from khat.paths import LOCAL


BASE = "https://www.i-sh.co.kr"
LIST_URL = BASE + "/app/lay2/program/S48T1581C1617/www/brd/m_244/list.do"
VIEW_URL = BASE + "/app/lay2/program/S48T1581C1617/www/brd/m_244/view.do"
DOWN_URL = BASE + "/com/file/innoFD.do"
OUT = LOCAL / "raw" / "sh-announcements"
MANIFEST = OUT / "sh_download_manifest.jsonl"

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
KEYWORD = "입주자 모집공고"
INCLUDE = ("입주자 모집공고", "입주자모집공고", "입주자 모집 공고", "모집공고문", "공고문")
EXCLUDE = ("위임장", "개인정보", "금융정보", "자산보유", "확인각서", "평면도", "예시", "제출서류", "신청서", "서식", "동의서")


def req(opener, url, data=None):
    body = None
    headers = {"User-Agent": UA}
    if data is not None:
        body = urlencode(data).encode("utf-8")
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    return opener.open(Request(url, data=body, headers=headers), timeout=30).read()


def text(opener, url, data=None):
    return req(opener, url, data).decode("utf-8", errors="replace")


def safe_name(value):
    value = re.sub(r"\s+", "_", value.strip())
    value = re.sub(r"[^\w가-힣().&+-]+", "_", value)
    return value[:140].strip("_")


def parse_rows(page_html):
    rows = []
    for m in re.finditer(r"getDetailView\('(?P<seq>\d+)'\).*?</a>", page_html, flags=re.S):
        seq = m.group("seq")
        block = m.group(0)
        title = re.sub(r"<[^>]+>", " ", block)
        title = html.unescape(re.sub(r"\s+", " ", title)).strip()
        rows.append({"seq": seq, "title": title})
    return rows


def parse_downlist(view_html):
    m = re.search(r"initParam\.downList\s*=\s*(\[.*?\]);", view_html, flags=re.S)
    if not m:
        return []
    return json.loads(m.group(1))


def wanted(files):
    keep = []
    for item in files:
        name = item.get("oriFileNm", "")
        lower = name.lower()
        if not lower.endswith((".pdf", ".hwp", ".hwpx")):
            continue
        if any(term in name for term in EXCLUDE):
            continue
        if any(term in name for term in INCLUDE):
            keep.append(item)
    if keep:
        return keep
    return [
        item for item in files
        if item.get("oriFileNm", "").lower().endswith((".pdf", ".hwp", ".hwpx"))
        and "공고" in item.get("oriFileNm", "")
        and not any(term in item.get("oriFileNm", "") for term in EXCLUDE)
    ]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    opener = build_opener(HTTPCookieProcessor(CookieJar()))

    seen = {}
    for page in (1, 2):
        page_html = text(opener, LIST_URL, {
            "page": str(page),
            "seq": "",
            "multi_itm_seq": "1",
            "multi_itm_seqsStr": "",
            "srchTp": "0",
            "srchWord": KEYWORD,
        })
        for row in parse_rows(page_html):
            seen[row["seq"]] = row

    manifest_rows = []
    for row in seen.values():
        seq = row["seq"]
        view_url = VIEW_URL + "?" + urlencode({"multi_itm_seq": "1", "seq": seq})
        view_html = text(opener, view_url)
        files = wanted(parse_downlist(view_html))
        if not files:
            manifest_rows.append({**row, "provider": "SH", "status": "no_wanted_attachment", "page_url": view_url})
            continue

        target_dir = OUT / f"seq-{seq}-{safe_name(row['title'])[:48]}"
        target_dir.mkdir(parents=True, exist_ok=True)
        for item in files:
            name = item["oriFileNm"]
            ext = Path(name).suffix.lower() or ".bin"
            out_path = target_dir / f"SH_{seq}_{safe_name(name)}{ext if not safe_name(name).lower().endswith(ext) else ''}"
            down_url = DOWN_URL + "?" + urlencode({
                "brdId": item["brdId"],
                "seq": item["seq"],
                "fileTp": item["fileTp"],
                "fileSeq": item["fileSeq"],
            })
            data = req(opener, down_url)
            out_path.write_bytes(data)
            manifest_rows.append({
                **row,
                "provider": "SH",
                "status": "downloaded",
                "page_url": view_url,
                "download_url": down_url,
                "filename": name,
                "path": "workspace_local/raw/" + out_path.relative_to(LOCAL / "raw").as_posix(),
                "bytes": len(data),
            })
            time.sleep(0.2)

    with MANIFEST.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"rows={len(manifest_rows)} manifest={MANIFEST}")


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    main()
