"""Shared paths and helpers of the closed-book, thinking-on and oracle builders: gold, scope, locator lines without row identifiers.
"""
import json, re, sys
from pathlib import Path
from khat.paths import ROOT as V, LOCAL
B = LOCAL / "runs"
RO = LOCAL / "runs"
OUT = LOCAL / "runs"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_baseline import build_prompt

LEGAL = "cross_document_legal_reasoning"
GPT = ["gpt-4.1-mini", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.5"]
OPEN = ["glm-5.2", "minimax-m3"]

def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def write(p, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

GOLD = {r["qa_id"]: r for f in ("qa_test_public.jsonl", "qa_dev.jsonl", "auxiliary/legal_reasoning.jsonl") for r in load(V / "data" / f)}
FIX = json.load(open(V / "data/errata/test_review_gold_fixes.json", encoding="utf-8"))
EXCL = set(FIX["exclude_from_evaluation"])

def inscope(q):
    g = GOLD[q]
    return g.get("task_type") != LEGAL and q not in EXCL

COUNT_LEAK_FIELDS = ["row_ids"]
DATE_LEAK_FIELDS = ["page_ids", "bundle_id", "table_ids", "cell_ids"]

def count_leak(rec):
    g = GOLD[rec["qa_id"]]; gp = g.get("gold_predicate") or {}
    rids = (rec.get("context_spec") or {}).get("row_ids") or []
    nums = {x.replace(",", "") for x in re.findall(r"\d[\d,]*", str(g.get("answer", "")))}
    return gp.get("op") == "count" and 0 < len(rids) <= 20 and str(len(rids)) in nums

DATE_LEAK_IDS = {h["qa_id"] for h in json.load(open(V / "data/item_sets/cb_date_leak_hits.json"))}

def strip(rec, fields):
    r = json.loads(json.dumps(rec))
    spec = r.get("context_spec") or {}
    for k in fields:
        spec.pop(k, None)
    r["context_spec"] = spec
    r.pop("prompt", None)
    return r

def fix_record(rec):
    """Return (stripped record, removed fields) for a leaking CB record, or (None, []) if no leak."""
    fields = []
    if count_leak(rec):
        fields += COUNT_LEAK_FIELDS
    if rec["qa_id"] in DATE_LEAK_IDS:
        fields += DATE_LEAK_FIELDS
    if not fields:
        return None, []
    r = strip(rec, fields)
    r["prompt"] = build_prompt(r)
    r["removed_fields"] = fields
    return r, fields
