"""Upper bound of the accept-any items' contribution to each model's accuracy.

Usage: python3 scripts/analysis/answer_set_bound.py
"""
import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "score"))
from verdict_sets import *
acc_any=[r["qa_id"] for r in csv.DictReader(open(R/"data/errata/test_review.csv")) if r["status"]=="gold_accept_any"]
print(len(acc_any), len(set(acc_any)&set(TEST)))
for m in MODELS:
    out=[]
    for s in ("bm25","dense","fc"):
        qs=items(s,V[s][m]); a=[q for q in qs if q in acc_any]; k=sum(V[s][m][q] for q in a)
        out.append(f"{s} {k}/{len(a)} of n={len(qs)} -> <= {100*k/len(qs):.2f}")
    print(m, " | ".join(out))
