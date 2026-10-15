#!/usr/bin/env python3
"""Statute and source text of the rebuilt corpus (local/) for grounding checks.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from khat.paths import ROOT, LOCAL
PROC = LOCAL

STATUTE_TITLES = {
    "law-housing-supply-rule": "주택공급에 관한 규칙",
    "law-public-housing-special-act-rule": "공공주택 특별법 시행규칙",
    "law-private-rental-housing-special-act": "민간임대주택에 관한 특별법",
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


@lru_cache(maxsize=None)
def articles(source_id: str) -> dict:
    """Map article_label -> page record (with full text) for a statute source."""
    rows = load_jsonl(PROC / source_id / "document_pages.jsonl")
    return {r["article_label"]: r for r in rows}


@lru_cache(maxsize=None)
def full_text(source_id: str) -> str:
    rows = load_jsonl(PROC / source_id / "document_pages.jsonl")
    if rows:
        return "\n".join(r["text"] for r in rows)
    meta = PROC / source_id / "metadata.json"
    if meta.exists():
        return meta.read_text(encoding="utf-8")
    return ""


def article_text(source_id: str, article_label: str) -> str:
    rec = articles(source_id).get(article_label)
    return rec["text"] if rec else ""


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def term_in_source(source_id: str, term: str, article_label: str | None = None) -> bool:
    """True if `term` appears (whitespace-insensitive) in the cited article, else anywhere in source."""
    nt = norm(term)
    if not nt:
        return False
    if article_label:
        at = article_text(source_id, article_label)
        if at and nt in norm(at):
            return True
    return nt in norm(full_text(source_id))


def term_absent_from_sources(term: str, source_ids: list[str]) -> bool:
    nt = norm(term)
    return all(nt not in norm(full_text(sid)) for sid in source_ids)
