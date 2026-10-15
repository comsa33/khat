"""Run-to-run variance of full context (three runs of glm-5.2 and minimax-m3 on the 19-item and 82-item sets). Every item of the set counts in each run: a missing verdict and an empty first-call answer are scored wrong.

Usage: python3 scripts/analysis/run_variance.py results/verdicts/gpt-4.1-mini results/verdicts/gpt-4.1-mini_original_gold
"""
import json, os, sys, statistics
from pathlib import Path
from khat.paths import DATA
d1, d23 = sys.argv[1], sys.argv[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "judges"))
from panel_verdicts import FIRST_CALL
FIX = json.load(open(DATA / "errata/test_review_gold_fixes.json"))
XI = set(FIX["exclude_from_evaluation"])
CORRECTED = set(FIX["accept_any"]) | set(FIX["replace"])
gold = {}
for l in open(DATA / "qa_test_public.jsonl", encoding="utf-8"):
    if l.strip():
        d = json.loads(l)
        if d.get("task_type") != "cross_document_legal_reasoning" and d["qa_id"] not in XI:
            gold[d["qa_id"]] = d
def load(tag, vd):
    p = Path(vd) / f"{tag}.jsonl"
    return {json.loads(l)["qa_id"]: bool(json.loads(l)["judge_correct"]) for l in p.open(encoding="utf-8") if l.strip()}
for m in ["minimax-m3", "glm-5.2"]:
    for setname, r1tags, var in [("native", [f"full_context_{m}"], "sample"), ("tier", [f"full_context_512k_rest_{m}", f"full_context_512k_sample_{m}"], "tier")]:
        r2 = load(f"full_context_512k_{var}_run2_{m}", d23)
        items = {q for q in r2 if q in gold}
        r1 = {}
        for t in r1tags:
            r1.update(load(t, d1))
        empty1 = {"krhlrb_v05_" + q for q in FIRST_CALL.get(("agg512k", m), [])} if setname == "tier" else set()
        r1 = {q: v and q not in empty1 for q, v in r1.items()}
        runs = [r1, r2, load(f"full_context_512k_{var}_run3_{m}", d23)]
        cells, accs = [], []
        for r in runs:
            ids = sorted(items)
            h = sum(r.get(q, False) for q in ids); cells.append(f"{h}/{len(ids)}={100*h/len(ids):.1f}%"); accs.append(100*h/len(ids))
        miss = sorted(items - set(runs[2]))
        print(f"{m:11} {setname:6} n={len(items)}  " + "  ".join(cells) + f"  mean={statistics.mean(accs):.1f} SD={statistics.stdev(accs):.1f}"
              f"  overlap_corrected={sorted(items & CORRECTED)} overlap_excl={sorted(set(r2) & XI)} r3_missing={miss}")
