#!/usr/bin/env python3
"""Deterministic verification of a QA file against the rebuilt corpus: sources, pages, rows, predicates, grounding, split leakage. Writes local/audit/verification_report_<file>.json.

Usage: python3 scripts/data/verify_qa.py --qa data/construction/qa_realistic_candidates.jsonl
"""
from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import source_text as V2
import table_rows as V3
import lh_announcements as V4
import announcement_tables as V5
import providers as P5
from khat.paths import ROOT, DATA, LOCAL

AUDIT = LOCAL / "audit"


def internal_corpus_present() -> bool:
    """True only if the rebuilt corpus under local/ is populated; grounding is skipped otherwise."""
    proc = LOCAL
    if not proc.exists():
        return False
    return any(p.is_file() and p.name != ".gitkeep" for p in proc.rglob("*"))

UNANS_MARKERS = ["확정할 수 없", "답할 수 없", "알 수 없", "unanswerable", "근거가 없", "근거 부재", "없음", "제공된 자료"]
REQUIRED = ["qa_id", "split", "task_type", "question", "answer", "answer_type",
            "source_ids", "required_capabilities", "evidence", "evaluation", "copyright_note"]
TIERS = {"32k", "64k", "128k", "256k", "512k"}
POSITIONS = {"early", "middle", "late", "multi"}


@lru_cache(maxsize=None)
def resolvable_ids() -> frozenset:
    ids = set()
    for fn in ("source_manifest.jsonl", "excluded_sources.jsonl"):
        for l in (DATA / fn).open(encoding="utf-8"):
            if l.strip():
                ids.add(json.loads(l)["source_id"])
    return frozenset(ids)


# ----------------------------------------------------------------- item checks
@lru_cache(maxsize=None)
def all_bundle_ids() -> frozenset:
    ids = set(V4.bundle_ids_v04()) | {b["bundle_id"] for b in V3.bundles()}
    v06 = LOCAL / "bundles" / "manifest.jsonl"
    if v06.exists():
        ids |= {json.loads(l)["bundle_id"] for l in v06.open(encoding="utf-8") if l.strip()}
    return frozenset(ids)


SPLITS = {"dev", "test_public"}


def _p5_corpus(item: dict) -> list:
    """Grounding corpus across ALL providers: cited page text (provider-agnostic) + cited statute text."""
    corp = [P5.page_text(pid) for pid in item.get("page_ids", [])]
    for s in {e["source_id"] for e in item.get("evidence", [])} | set(item.get("source_ids", [])):
        if s in V4.STATUTES:
            corp.append(V2.full_text(s))
    return [c for c in corp if c]


def check_item(item: dict, ids: frozenset) -> dict:
    """Self-contained, provider-aware check of the released items (LH + SH/GH/iH/JPDC)."""
    errs = []
    qid = item.get("qa_id")
    for f in REQUIRED:
        if f not in item:
            errs.append(f"missing {f}")
    for fld in ("provider", "region_sido", "housing_type", "split"):
        if not item.get(fld):
            errs.append(f"missing {fld}")
    if item.get("split") and item["split"] not in SPLITS:
        errs.append(f"bad split {item.get('split')}")
    for s in item.get("source_ids", []):
        if s not in ids:
            errs.append(f"unknown source_id {s}")
    for e in item.get("evidence", []):
        if e.get("source_id") not in ids:
            errs.append(f"unknown evidence source_id {e.get('source_id')}")
        if not e.get("locator"):
            errs.append("evidence missing locator")
    tier = item.get("context_tier")
    if tier is not None and tier not in TIERS:
        errs.append(f"bad context_tier {tier}")
    pos = item.get("evidence_position")
    if pos is not None and pos not in POSITIONS:
        errs.append(f"bad evidence_position {pos}")
    bid = item.get("bundle_id")
    if bid and bid not in all_bundle_ids():
        errs.append(f"unknown bundle_id {bid}")
    for pid in item.get("page_ids", []):
        if pid not in P5.all_page_ids():
            errs.append(f"page_id not found: {pid}")
    for cid in item.get("cell_ids", []):
        if cid not in P5.all_cell_ids():
            errs.append(f"cell_id not found: {cid}")
    for tid in item.get("table_ids", []):
        if tid not in P5.all_table_ids():
            errs.append(f"table_id not found: {tid}")

    pred = item.get("gold_predicate")
    matched_ids = None
    if pred:
        try:
            val, matched_ids = V3.recompute(pred)
        except Exception as exc:
            errs.append(f"predicate error: {exc!r}")
            val = None
        if val is None:
            errs.append("predicate produced no value")
        else:
            ev = item.get("evaluation", {})
            sval = str(val)
            if pred["op"] in ("argmax", "argmin"):
                if sval not in (ev.get("gold_terms", []) + [item.get("answer", "")]) and sval not in item.get("answer", ""):
                    errs.append(f"argmax/min value {sval!r} not in answer/gold_terms")
            else:
                golds = [str(g) for g in ev.get("gold_numbers", [])]
                if sval not in golds and sval not in item.get("answer", "").replace(",", ""):
                    errs.append(f"predicate value {sval} != gold_numbers {golds}")
    for rid in item.get("row_ids", []):
        src = V3.HUG if rid.startswith("HUG-") else (V3.MOLIT if rid.startswith("MOLIT-") else None)
        if src is None or rid not in V3.rows_by_id(src):
            errs.append(f"row_id not found: {rid}")
        elif matched_ids is not None and rid not in matched_ids:
            errs.append(f"row_id {rid} not in predicate match set")

    ev = item.get("evaluation", {})
    metric = ev.get("metric", "")
    if metric == "boolean_and_reason":
        if not any(m in item.get("answer", "") for m in UNANS_MARKERS):
            errs.append("answerability lacks unanswerable marker")
    elif not pred:
        corp = _p5_corpus(item)
        ans = item.get("answer", "")
        for t in ev.get("gold_terms", []):
            nt = V2.norm(t)
            if nt and not (any(nt in V2.norm(c) for c in corp) or nt in V2.norm(ans)):
                errs.append(f"gold_term not grounded: {t!r}")
        for n in ev.get("gold_numbers", []):
            if n and not (any(n in c for c in corp) or n in ans.replace(",", "")):
                errs.append(f"gold_number not grounded: {n!r}")
    elif item.get("page_ids"):
        corp = _p5_corpus(item)
        ans = item.get("answer", "")
        for t in ev.get("gold_terms", []):
            nt = V2.norm(t)
            if nt and not nt.isdigit() and not (any(nt in V2.norm(c) for c in corp) or nt in V2.norm(ans)):
                errs.append(f"cross-source region term not grounded in cited page: {t!r}")

    tt = item.get("task_type")
    srcs = set(item.get("source_ids", []))
    anns = {m.group(1) for p in item.get("page_ids", []) if (m := re.match(r"(.+)-p\d{3}$", p))}
    if tt == "cross_document_legal_reasoning" and not (any(s in V4.STATUTES for s in srcs) and V4.LH_V04 in srcs):
        errs.append("cross_document_legal must cite LH-v04 + a statute")
    if tt == "cross_source_aggregation" and len(srcs) < 2:
        errs.append("cross_source_aggregation must cite >=2 sources")
    if tt == "multi_document_comparison" and len(anns) < 2:
        errs.append("multi_document_comparison must cite >=2 announcements")
    if tt == "provider_comparison" and len(srcs) < 2:
        errs.append("provider_comparison must cite >=2 provider sources")
    if tt == "region_comparison" and len(anns) < 2:
        errs.append("region_comparison must cite >=2 announcements")

    return {"qa_id": qid, "task_type": tt, "status": "FAIL" if errs else "verified", "errors": errs}


def check_split_leakage(path: Path) -> bool:
    """Dataset-level: no announcement may appear in more than one EVALUATION split."""
    if not path.exists():
        return True
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    eval_splits = {"test_public"}
    ann_splits: dict = {}
    for r in rows:
        for a in V5.item_announcements(r):
            ann_splits.setdefault(a, set()).add(r.get("split"))
    leaks = {a: sorted(s) for a, s in ann_splits.items() if len({x for x in s if x in eval_splits}) > 1}
    if leaks:
        print(f"  SPLIT LEAKAGE: {len(leaks)} announcements in >1 eval split")
        for a, s in list(leaks.items())[:10]:
            print(f"    {a}: {s}")
        return False
    print(f"  split-leakage check: OK ({len(ann_splits)} announcements, none in >1 eval split)")
    return True


def run_file(path: Path, checker, ids: frozenset, label: str) -> bool:
    if not path.exists():
        print(f"  ({path.name} absent — skipped)")
        return True
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    reports = [checker(r, ids) for r in rows]
    failed = [r for r in reports if r["status"] == "FAIL"]
    fams = Counter(r["task_type"] for r in rows)
    report = {"file": path.name, "total": len(rows), "verified": len(rows) - len(failed),
              "failed": len(failed), "by_task_family": dict(fams),
              "failures": failed}
    (AUDIT / f"verification_report_{label}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"== {path.name}: total={len(rows)} verified={report['verified']} failed={len(failed)} ==")
    for k, v in sorted(fams.items()):
        print(f"    {k}: {v}")
    for r in failed[:40]:
        print(f"    FAIL {r['qa_id']} [{r['task_type']}]: {'; '.join(r['errors'][:3])}")
    return not failed


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True, help="QA file to verify (path relative to the repository root)")
    args = ap.parse_args()
    AUDIT.mkdir(parents=True, exist_ok=True)
    if not internal_corpus_present():
        print("verify_qa: rebuilt corpus absent (local/ not populated) —")
        print("  deep grounding/recompute verification SKIPPED (clean checkout / CI).")
        print("  Schema + source-id resolution are covered by validate_dataset.py; rebuild the corpus")
        print("  locally (scripts/build/) to run the full grounding checks.")
        return 0
    ids = resolvable_ids()
    if args.qa:
        path = ROOT / args.qa
        label = path.stem.replace(".", "_")
        ok = run_file(path, check_item, ids, label)
        ok_leak = check_split_leakage(path)
        return 0 if (ok and ok_leak) else 1


if __name__ == "__main__":
    raise SystemExit(main())
