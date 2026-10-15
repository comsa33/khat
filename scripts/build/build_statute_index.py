#!/usr/bin/env python3
"""Per-statute article indexes (label, title, short snippet, locator).

Usage: python3 scripts/build/build_statute_index.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source_text import ROOT, STATUTE_TITLES, load_jsonl, PROC
from khat.paths import LOCAL

OUT = LOCAL / "index"


def snippet(text: str, n: int = 200) -> str:
    t = re.sub(r"<개정[^>]*>", "", text)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:n]


def main() -> int:
    for sid, title in STATUTE_TITLES.items():
        rows = load_jsonl(PROC / sid / "document_pages.jsonl")
        idx = [{
            "article_label": r["article_label"],
            "title": r["title"],
            "locator": r["locator"],
            "n_hang": r["n_hang"], "n_ho": r["n_ho"],
            "snippet": snippet(r["text"]),
        } for r in rows if r["unit_type"] == "article"]
        (OUT / f"index_{sid}.json").write_text(
            json.dumps({"source_id": sid, "title": title, "n_articles": len(idx), "articles": idx},
                       ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  index_{sid}.json: {len(idx)} articles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
