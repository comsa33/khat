#!/usr/bin/env python3
"""Source packets (announcement pages and statute articles) and announcement pairs for the authored items.

Usage: python3 scripts/construction/build_authoring_packets.py
"""
from __future__ import annotations

import json
import re

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import source_text as V2
import lh_announcements as Q

OUTDIR = Q.LOCAL / "construction" / "authoring"
LEAD = 8
STATUTE_KEYWORDS = ["입주자저축", "청약통장", "순위", "1순위", "가입", "납입", "소득", "자산", "거주",
                    "특별공급", "일반공급", "재당첨", "전매", "거주의무", "무주택", "세대구성원",
                    "우선공급", "당첨자", "공급신청", "공급대상", "분양전환"]


def relevant_articles(sid: str, limit: int = 12) -> list:
    rows = V2.load_jsonl(Q.PROC / sid / "document_pages.jsonl")
    arts = [r for r in rows if r.get("unit_type") == "article"]
    scored = []
    for r in arts:
        text = r.get("text", "")
        score = sum(text.count(k) for k in STATUTE_KEYWORDS)
        if score > 0:
            scored.append((score, r))
    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored[:limit]]


def build_packet(ann_id: str) -> str:
    meta = Q.ann_meta().get(ann_id, {})
    title = meta.get("title_from_audit") or ann_id
    region = Q.supply_location(ann_id)
    lines = [f"# Authoring packet — {ann_id}", "",
             f"- title: {title}", f"- announcement_id: {ann_id}",
             f"- supply_location (verbatim on p.1): {region}",
             f"- source_id for LH pages: {Q.LH_V04}", "",
             "## LH announcement lead pages (cite page_id; copy gold_terms verbatim, <=40 chars)", ""]
    for p in Q.ann_pages(ann_id):
        if p["page_no"] > LEAD:
            break
        body = re.sub(r"\n{3,}", "\n\n", p["text"]).strip()
        lines.append(f"### page_id: {p['page_id']}  (p.{p['page_no']}, locator: {p['locator']})")
        lines.append(body)
        lines.append("")
    lines.append("## Relevant statute articles (cite the law source_id + article label)")
    lines.append("")
    for sid in Q.STATUTES:
        title_ko = V2.STATUTE_TITLES.get(sid, sid)
        lines.append(f"### statute source_id: {sid}  ({title_ko})")
        for r in relevant_articles(sid):
            lines.append(f"- {r['article_label']}({r.get('title','')}): {re.sub(chr(10),' ',r['text'])[:600]}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    anns = list(Q.announcement_ids())
    for a in anns:
        (OUTDIR / f"{a}.md").write_text(build_packet(a), encoding="utf-8")
    pairs = []
    for i in range(len(anns)):
        pairs.append([anns[i], anns[(i + 1) % len(anns)]])
    pairs.append([anns[0], anns[5]])
    pairs.append([anns[2], anns[7]])
    (OUTDIR / "_pairs.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(anns)} packets + {len(pairs)} pairs -> {OUTDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
