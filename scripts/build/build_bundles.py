#!/usr/bin/env python3
"""Build the long-context bundles (32k to 512k) of every announcement and the multi-provider mix, with a metadata manifest.

Usage: PYTHONHASHSEED=1 python3 scripts/build/build_bundles.py [--with-row-ids] [--out DIR]
(MOLIT row identifiers are left out by default; --with-row-ids with PYTHONHASHSEED=1 rebuilds the bundles of the reported runs)
"""
from __future__ import annotations

import json
from collections import Counter

import tiktoken

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import source_text as V2
import table_rows as C
import providers as P

ENC = tiktoken.get_encoding("cl100k_base")
BUNDLES = C.LOCAL / "bundles"
WITH_ROW_IDS = False
LEAD = 6
TIERS = {"32k": 32000, "64k": 64000, "128k": 128000, "256k": 256000, "512k": 512000}
EARLY_TIERS = ["32k", "64k"]
LATE_TIERS = ["128k", "256k"]


def _rid(r: dict) -> str:
    return f"{r['_row_id']} | " if WITH_ROW_IDS else ""


def toks(s: str) -> int:
    return len(ENC.encode(s))


def band(frac: float) -> str:
    return "early" if frac < 0.34 else ("middle" if frac < 0.67 else "late")


def ann_components(ann_id: str, max_page=None) -> list:
    out = []
    for p in sorted(P.pages(ann_id), key=lambda x: x["page_no"]):
        if max_page and p["page_no"] > max_page:
            continue
        t = f"[공고 {P.provider_of(ann_id)} {p['page_id']} (p.{p['page_no']})]\n{p['text']}\n"
        out.append({"type": "lh_page", "id": p["page_id"], "ann": ann_id, "text": t, "tokens": toks(t)})
    return out


def law_components() -> list:
    out = []
    for sid in P.Q.STATUTES if hasattr(P, "Q") else ("law-housing-supply-rule", "law-public-housing-special-act-rule", "law-private-rental-housing-special-act"):
        for r in V2.load_jsonl(C.PROC / sid / "document_pages.jsonl"):
            if r.get("unit_type") != "article":
                continue
            t = f"[법령 {sid} {r['article_label']}({r.get('title','')})]\n{r['text']}\n"
            out.append({"type": "law_article", "id": f"{sid}::{r['article_label']}", "ann": None, "text": t, "tokens": toks(t)})
    return out


def row_block_components() -> list:
    out, buf = [], []

    def flush(kind, lines):
        if not lines:
            return
        t = f"[{kind} rows]\n" + "\n".join(lines) + "\n"
        out.append({"type": f"{kind.lower()}_rows", "id": f"{kind.lower()}_block_{len(out)}", "ann": None, "text": t, "tokens": toks(t)})

    for r in C.molit_rows():
        buf.append(_rid(r) + f"{r['_lawd_name']} {r['umdNm']} | {r['aptNm']} | {r['excluUseAr']}㎡ {r['dealAmount']}만원 | {r['_deal_ymd']}")
        if len(buf) >= 60:
            flush("MOLIT", buf); buf = []
    flush("MOLIT", buf)
    return out


def hug_block_component() -> dict:
    """The complete HUG (주택도시보증공사) sale-history table as ONE guaranteed component.

    `cross_source_aggregation` questions aggregate over this table (count / avg TOT_HOCO by 지역·연도), so the
    bundle must embed it COMPLETELY (a partial/padding-truncated copy would change the count). Included
    verbatim and in full in the multi-provider mix bundle, never as a paddable distractor.
    """
    rows = sorted(C.hug_rows(), key=lambda r: (r.get("_query_area_name", ""), r.get("_query_year", ""), r.get("_row_id", "")))
    lines = [
        "===== [참고자료] HUG 분양이력 자료 (주택도시보증공사, 분양보증 발급 사업장) =====",
        f"총 {len(rows)}개 사업장. 각 행 = 분양 사업장 1건. ‘분양 사업장 건수’는 해당 지역·연도 행의 개수이고, "
        "‘평균 총세대수(TOT_HOCO)’는 그 행들의 총세대수 평균입니다.",
        "지역 | 연도 | 사업장명 | 총세대수(TOT_HOCO) | 일반분양세대(GNRL_SILT_HOCO)",
    ]
    for r in rows:
        lines.append(" | ".join([
            r.get("_query_area_name", ""), r.get("_query_year", ""),
            r.get("BSU_NM", ""), str(r.get("TOT_HOCO", "")), str(r.get("GNRL_SILT_HOCO", "")),
        ]))
    lines.append("===== [참고자료 끝] =====")
    text = "\n".join(lines) + "\n"
    return {"type": "hug_sale_history_rows", "id": "hug_sale_history_block", "ann": None,
            "text": text, "tokens": toks(text)}


def _needed_512k_refs():
    """Scan the released QA for evidence the 512k mix bundle's items require: the specific statute
    articles (cross_document_legal_reasoning) and MOLIT apt-trade-detail (lawd, ymd) slices
    (cross_source_aggregation). The 512k bundle must embed these or those items are unanswerable
    in the full-context regime."""
    import re
    arts, molit = set(), set()
    for fn in ("qa_dev.jsonl", "qa_test_public.jsonl", "auxiliary/legal_reasoning.jsonl"):
        p = C.ROOT / "data" / fn
        if not p.exists():
            continue
        for d in V2.load_jsonl(p):
            if str(d.get("context_tier")) != "512k":
                continue
            ev = d.get("evidence") or []
            if isinstance(ev, str):
                try:
                    ev = json.loads(ev)
                except Exception:
                    ev = []
            for e in ev:
                sid, loc = e.get("source_id", ""), e.get("locator", "")
                if sid.startswith("law-"):
                    m = re.search(r"제[0-9]+조(의[0-9]+)?", loc)
                    if m:
                        arts.add((sid, m.group(0)))
                if "molit-apt-trade-detail[" in loc:
                    m = re.search(r"_lawd_name=([^&\]]+).*?_deal_ymd=([0-9]+)", loc)
                    if m:
                        molit.add((m.group(1).strip(), m.group(2)))
    return arts, molit


def needed_law_components(arts) -> list:
    """The specific statute articles the 512k items cite, as guaranteed components (not paddable)."""
    out = []
    by_src = {}
    for sid, art in arts:
        by_src.setdefault(sid, set()).add(art)
    for sid, want in by_src.items():
        for r in V2.load_jsonl(C.PROC / sid / "document_pages.jsonl"):
            if r.get("unit_type") == "article" and r["article_label"] in want:
                t = f"[법령 {sid} {r['article_label']}({r.get('title','')})]\n{r['text']}\n"
                out.append({"type": "law_article", "id": f"{sid}::{r['article_label']}",
                            "ann": None, "text": t, "tokens": toks(t)})
    return out


def needed_molit_component(molit):
    """The MOLIT apt-trade-detail rows for the (lawd, ymd) slices the 512k cross_source items
    aggregate over, embedded COMPLETE as one guaranteed block (like the HUG table) so counts/avgs
    are recomputable from context."""
    if not molit:
        return None
    rows = [r for r in C.molit_rows() if (r["_lawd_name"], str(r["_deal_ymd"])) in molit]
    rows.sort(key=lambda r: (r["_lawd_name"], str(r["_deal_ymd"]), r["_row_id"]))
    lines = [
        "===== [참고자료] MOLIT 아파트 매매 실거래 상세 (국토교통부) =====",
        f"총 {len(rows)}건. 각 행 = 매매 1건. '거래 건수'는 해당 시군구·거래연월 행의 개수이고, "
        "'평균 거래금액'은 그 행들의 거래금액(만원) 평균입니다.",
        ("행ID | " if WITH_ROW_IDS else "") + "시군구 읍면동 | 단지 | 전용면적 거래금액 | 거래연월",
    ]
    for r in rows:
        lines.append(_rid(r) + f"{r['_lawd_name']} {r['umdNm']} | {r['aptNm']} | "
                     f"{r['excluUseAr']}㎡ {r['dealAmount']}만원 | {r['_deal_ymd']}")
    lines.append("===== [참고자료 끝] =====")
    text = "\n".join(lines) + "\n"
    return {"type": "molit_trade_rows", "id": "molit_trade_detail_block", "ann": None,
            "text": text, "tokens": toks(text)}


DISTRACTORS: list = []


def assemble(bundle_id, tier, template, ordered_parts, target, no_pad=False):
    parts = [dict(c) for c in ordered_parts]
    cur = sum(c["tokens"] for c in parts)
    di = 0
    guard = 0
    while not no_pad and cur < target and guard < 200000 and DISTRACTORS:
        if di >= len(DISTRACTORS):
            di = 0
        c = DISTRACTORS[di]; di += 1; guard += 1
        parts.append(dict(c)); cur += c["tokens"]
    text_parts, comps = [], []
    char_off, tok_off = 0, 0
    for c in parts:
        t = c["text"]
        frac = (tok_off + c["tokens"] / 2) / max(cur, 1)
        comp = {"type": c["type"], "id": c["id"], "position_band": band(frac)}
        if c.get("ann"):
            comp["ann"] = c["ann"]
        comps.append(comp)
        text_parts.append(t)
        char_off += len(t); tok_off += c["tokens"]
    (BUNDLES / f"{bundle_id}.txt").write_text("".join(text_parts), encoding="utf-8")
    anns = sorted({c["ann"] for c in parts if c.get("ann")})
    providers = sorted({P.provider_of(a) for a in anns})
    src_mix = Counter(P.source_of(a) for a in anns)
    return {"bundle_id": bundle_id, "context_tier": tier, "template": template,
            "providers": providers, "announcement_ids": anns, "source_mix": dict(src_mix),
            "token_estimate": cur, "n_components": len(comps),
            "component_positions": comps}


def build():
    global DISTRACTORS
    BUNDLES.mkdir(parents=True, exist_ok=True)
    laws = law_components()
    rows = row_block_components()
    DISTRACTORS = laws + rows
    manifest = []
    anns = list(P.announcement_ids())

    for ann_id in anns:
        comps = ann_components(ann_id)
        if not comps:
            continue
        for tier in EARLY_TIERS:
            target = TIERS[tier]
            lead, cur = [], 0
            for c in comps:
                if cur + c["tokens"] > target * 0.9:
                    break
                lead.append(c); cur += c["tokens"]
            manifest.append(assemble(f"{ann_id}__early_{tier}", tier, "announcement_focus_early", lead, target))
        for tier in LATE_TIERS:
            target = TIERS[tier]
            ann_cap = target * 0.40
            lead, atok = [], 0
            for c in reversed(comps):
                if atok + c["tokens"] > ann_cap:
                    break
                lead.insert(0, c); atok += c["tokens"]
            pre, cur, di = [], 0, 0
            while cur < target - atok - 2000 and di < len(DISTRACTORS):
                pre.append(DISTRACTORS[di]); cur += DISTRACTORS[di]["tokens"]; di += 1
            parts = pre + lead
            manifest.append(assemble(f"{ann_id}__late_{tier}", tier, "announcement_focus_late", parts, sum(c["tokens"] for c in parts), no_pad=True))

    by_prov = {}
    for a in anns:
        by_prov.setdefault(P.source_of(a), []).append(a)
    interleaved = []
    i = 0
    while True:
        added = False
        for src in sorted(by_prov):
            if i < len(by_prov[src]):
                interleaved.extend(ann_components(by_prov[src][i], max_page=LEAD))
                added = True
        if not added:
            break
        i += 1
    hug = hug_block_component()
    _arts, _molit = _needed_512k_refs()
    ev = [hug] + needed_law_components(_arts)
    _mc = needed_molit_component(_molit)
    if _mc:
        ev.append(_mc)
    gtok = sum(c["tokens"] for c in ev)
    il, cur = [], gtok
    for c in interleaved:
        if cur + c["tokens"] > TIERS["512k"]:
            break
        il.append(c)
        cur += c["tokens"]
    manifest.append(assemble("mix_multiprovider_512k", "512k", "multiprovider_evidence_complete",
                             ev + il, TIERS["512k"], no_pad=True))
    manifest.append(assemble("mix_multiprovider_256k", "256k", "multiprovider_law_heavy", laws + interleaved, TIERS["256k"]))
    manifest.append(assemble("mix_multiprovider_32k", "32k", "multiprovider_compact", interleaved[:14], TIERS["32k"]))

    with (BUNDLES / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    tiers = Counter(m["context_tier"] for m in manifest)
    pages_band = {}
    for m in manifest:
        for c in m["component_positions"]:
            if c["type"] == "lh_page":
                pages_band.setdefault(c["id"], set()).add(c["position_band"])
    print(f"=== BUNDLES: {len(manifest)} bundles ; tiers {dict(tiers)} ===")
    print(f"  LH-pages covered: {len(pages_band)} ; early {sum(1 for v in pages_band.values() if 'early' in v)} ; late {sum(1 for v in pages_band.values() if 'late' in v)}")
    print(f"  multi-provider mix bundles: providers per mix = {[len(m['providers']) for m in manifest if m['bundle_id'].startswith('mix_')]}")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build the long-context bundles and their manifest.")
    ap.add_argument("--with-row-ids", action="store_true",
                    help="keep the MOLIT row identifiers in the row blocks (the bundles of the reported runs)")
    ap.add_argument("--out", type=Path, default=BUNDLES, help="output directory (default: local/bundles)")
    args = ap.parse_args()
    WITH_ROW_IDS, BUNDLES = args.with_row_ids, args.out
    raise SystemExit(build())
