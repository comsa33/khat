"""Correct counts on the gold-corrected test_public items per model and setting, claude-sonnet-5-5 (gpt-4.1-mini) judge.

Usage: python3 scripts/tables/corrected_gold_cells.py
"""
import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "score"))
from verdict_sets import *
fix=[r["qa_id"] for r in csv.DictReader(open(R/"data/errata/test_review.csv")) if r["split"]=="test_public" and r["status"] in ("gold_accept_any","gold_replaced")]
print("corrected items", len(fix))
for m in MODELS[:6]:
    cells=[]
    for s in SETTINGS:
        p=V[s][m]; sec=verdicts(s,m,"gpt-4.1-mini")
        qs=[q for q in fix if q in p]
        cells.append(f"{s} {sum(p[q] for q in qs)} ({sum(sec[q] for q in qs)}) of {len(qs)}")
    print(m, " | ".join(cells))
