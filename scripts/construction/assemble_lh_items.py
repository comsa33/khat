#!/usr/bin/env python3
"""Gate the authored items (review flags, verbatim grounding) and assemble them with the deterministic LH items.

Usage: python3 scripts/construction/assemble_lh_items.py
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import source_text as V2
import table_rows as C
import lh_announcements as Q


def _rot(key: str, n: int) -> int:
    return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % max(n, 1)

CONS = C.LOCAL / "construction"
DET = CONS / "lh_items.jsonl"
AGENT = CONS / "authored_items.json"
OUT = C.LOCAL / "construction" / "candidates_lh.jsonl"
PROV = CONS / "candidates_lh_provenance.jsonl"
DROP = CONS / "candidates_lh_dropped.json"

PUBLIC = ["qa_id", "split", "task_type", "question", "answer", "answer_type", "source_ids",
          "required_capabilities", "evidence", "evaluation", "copyright_note",
          "bundle_id", "context_tier", "evidence_position", "row_ids", "page_ids",
          "announcement_ids", "gold_predicate"]


def ann_of(pid: str) -> str | None:
    import re
    m = re.match(r"(lh-[a-z0-9-]+)-p\d{3}$", pid)
    return m.group(1) if m else None


# ----------------------------------------------------------------- agent grounding gate
def grounded_agent(it: dict) -> tuple[bool, str]:
    declared = set(it.get("source_ids", []))
    for e in it.get("evidence", []):
        if e.get("source_id") not in declared:
            return False, f"evidence source_id not in source_ids: {e.get('source_id')!r}"
        if not e.get("locator"):
            return False, "evidence missing locator"
    for pid in it.get("page_ids", []):
        if pid not in Q.v04_page_ids():
            return False, f"page_id missing {pid}"
    corp = [Q.page_text(pid) for pid in it.get("page_ids", [])]
    for s in {e["source_id"] for e in it.get("evidence", [])} | set(it.get("source_ids", [])):
        if s in Q.STATUTES:
            corp.append(Q.statute_text(s))
    ans = it.get("answer", "")
    ev = it.get("evaluation", {})
    for term in ev.get("gold_terms", []):
        nt = V2.norm(term)
        if nt and not (any(nt in V2.norm(c) for c in corp) or nt in V2.norm(ans)):
            return False, f"gold_term not grounded: {term!r}"
    for n in ev.get("gold_numbers", []):
        if n and not (any(n in c for c in corp) or n in ans.replace(",", "")):
            return False, f"gold_number not grounded: {n!r}"
    tt = it.get("task_type")
    srcs = set(it.get("source_ids", []))
    if tt == "cross_document_legal_reasoning":
        if not (any(s in Q.STATUTES for s in srcs) and Q.LH_V04 in srcs):
            return False, "cross_document_legal must cite LH-v04 + a statute"
    if tt == "multi_document_comparison":
        anns = {ann_of(p) for p in it.get("page_ids", [])} - {None}
        if len(anns) < 2:
            return False, "multi_document_comparison must cite >=2 announcements"
    return True, "ok"


# ----------------------------------------------------------------- bundle attachment
def _bundles_for_page(pid: str):
    return Q.page_bundle_positions_v04().get(pid, [])


def _find_band_bundle(pid: str, band: str, rot_key: str = ""):
    cands = [b for b in _bundles_for_page(pid) if b["position_band"] == band]
    if not cands:
        return None
    order = sorted(cands, key=lambda x: int(x["context_tier"].replace("k", "")))
    return order[_rot(rot_key or pid, len(order))]


def _find_multi_bundle(page_ids: list, need_law: bool):
    """A bundle containing all cited page_ids (and a law_article if need_law). Prefer largest tier."""
    best = None
    for b in sorted(Q.bundles_v04(), key=lambda x: -int(x["context_tier"].replace("k", ""))):
        ids = {c["id"] for c in b["components"]}
        if not all(p in ids for p in page_ids):
            continue
        if need_law and not any(c["type"] == "law_article" for c in b["components"]):
            continue
        best = b
        break
    return best


def attach_bundle(it: dict) -> bool:
    """Attach bundle fields. Return False if a required context bundle cannot be found (caller drops)."""
    tt = it.get("task_type", "")
    pids = it.get("page_ids", [])
    if tt == "long_context_retrieval" and pids:
        b = _find_band_bundle(pids[0], "early", rot_key=it.get("question", "") + pids[0])
        if not b:
            return False
        it["bundle_id"], it["context_tier"], it["evidence_position"] = b["bundle_id"], b["context_tier"], "early"
        return True
    if tt == "long_distance_retrieval" and pids:
        b = _find_band_bundle(pids[0], "late", rot_key=it.get("question", "") + pids[0])
        if not b:
            return False
        it["bundle_id"], it["context_tier"], it["evidence_position"] = b["bundle_id"], b["context_tier"], "late"
        return True
    if tt in ("cross_document_legal_reasoning", "multi_document_comparison") and pids:
        b = _find_multi_bundle(pids, need_law=(tt == "cross_document_legal_reasoning"))
        if not b:
            return False
        it["bundle_id"], it["context_tier"], it["evidence_position"] = b["bundle_id"], b["context_tier"], "multi"
        return True
    if tt == "cross_source_aggregation" and pids:
        positions = _bundles_for_page(pids[0])
        pick = next((p for p in positions if p["bundle_id"] == "mix_table_128k"), None)
        if pick is None and positions:
            pick = sorted(positions, key=lambda x: int(x["context_tier"].replace("k", "")))[len(positions) // 2]
        if pick:
            it["bundle_id"], it["context_tier"], it["evidence_position"] = pick["bundle_id"], pick["context_tier"], pick["position_band"]
        return True
    return True


# ----------------------------------------------------------------- public projection
def public_only(it: dict, qid: str) -> dict:
    out = {"qa_id": qid, "split": it.get("split", "dev")}
    for f in PUBLIC:
        if f in ("qa_id", "split"):
            continue
        if f in it and it[f] not in (None, [], {}):
            out[f] = it[f]
    out["evidence"] = [{"source_id": e["source_id"], "locator": e["locator"]} for e in it.get("evidence", [])]
    ev = it.get("evaluation", {})
    nev = {"metric": ev.get("metric", "contains_all")}
    if ev.get("gold_terms"):
        nev["gold_terms"] = ev["gold_terms"]
    if ev.get("gold_numbers"):
        nev["gold_numbers"] = ev["gold_numbers"]
    out["evaluation"] = nev
    return out


def main() -> int:
    final, prov, dropped = [], [], []
    seq = 0
    seen_q = set()

    det = [json.loads(l) for l in DET.open(encoding="utf-8") if l.strip()] if DET.exists() else []
    for it in det:
        if it["question"] in seen_q:
            continue
        if not attach_bundle(it):
            dropped.append({"origin": "deterministic", "task_type": it.get("task_type"),
                            "reason": "no bundle band coverage", "q": it.get("question", "")[:70]})
            continue
        seen_q.add(it["question"])
        seq += 1
        qid = f"krhlrb_v04_{seq:03d}"
        final.append(public_only(it, qid))
        prov.append({"qa_id": qid, "origin": "deterministic", "task_type": it["task_type"],
                     "verification": "verified_by_script"})

    n_det = len(final)
    if AGENT.exists():
        for row in json.loads(AGENT.read_text(encoding="utf-8")):
            it = row["item"]
            if it.get("question") in seen_q:
                dropped.append({"origin": "agent", "reason": "duplicate question", "q": it.get("question", "")[:70]})
                continue
            sup, unamb = row.get("reviewer_supported"), row.get("reviewer_unambiguous")
            ok, why = grounded_agent(it)
            if not (sup and unamb and ok):
                dropped.append({"origin": f"agent:{row.get('batch')}", "task_type": it.get("task_type"),
                                "supported": sup, "unambiguous": unamb, "grounding": why,
                                "q": it.get("question", "")[:70]})
                continue
            if not attach_bundle(it):
                dropped.append({"origin": f"agent:{row.get('batch')}", "task_type": it.get("task_type"),
                                "reason": "no bundle coverage for cited pages", "q": it.get("question", "")[:70]})
                continue
            seen_q.add(it["question"])
            seq += 1
            qid = f"krhlrb_v04_{seq:03d}"
            final.append(public_only(it, qid))
            prov.append({"qa_id": qid, "origin": f"agent:{row.get('batch')}", "task_type": it["task_type"],
                         "verification": "reviewer+grounded", "reviewer_reason": row.get("reviewer_reason", "")})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for it in final:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with PROV.open("w", encoding="utf-8") as f:
        for p in prov:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    DROP.write_text(json.dumps(dropped, ensure_ascii=False, indent=2), encoding="utf-8")

    fam = Counter(it["task_type"] for it in final)
    tiers = Counter(it.get("context_tier", "—") for it in final)
    pos = Counter(it.get("evidence_position", "—") for it in final)
    print("=== ASSEMBLE ===")
    print(f"  deterministic={n_det}  agent_kept={len(final)-n_det}  dropped={len(dropped)}  TOTAL={len(final)}")
    for k, v in sorted(fam.items()):
        print(f"    {k:32s} {v}")
    print("  context_tier:", dict(tiers))
    print("  evidence_position:", dict(pos))
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
