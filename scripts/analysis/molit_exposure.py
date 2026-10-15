"""Per-judge verdicts on the MOLIT count or average items whose rows carry identifiers (full context, dense k=50, dense top-5).

Usage: python3 scripts/analysis/molit_exposure.py [count3|avg3|count3+avg3]
"""
import sys, re
from pathlib import Path
HERE = Path(__file__).resolve().parent
src = (HERE.parent / "judges/judge_agreement.py").read_text(encoding="utf-8")
sys.path.insert(0, str(HERE.parent / "judges"))
SETS = {"count3": ["0425", "0428", "0431"], "avg3": ["0426", "0429", "0432"], "count3+avg3": ["0425", "0428", "0431", "0426", "0429", "0432"]}
WHICH = sys.argv[1] if len(sys.argv) > 1 else "count3"
DROP = ["krhlrb_v05_" + x for x in SETS[WHICH]]
head = src.split("# ------------------------------------------------------------------ 1. accuracy, agreement, 200 sample", 1)[0]
g = {"__name__": "molit3", "__file__": str(HERE.parent / "judges/judge_agreement.py")}
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    exec(head, g)
V, agg81, JN, FULLWIN = g["V"], g["agg81"], g["JN"], g["FULLWIN"]
print(f"== (1) correct on {' / '.join(q[-4:] for q in DROP)} (1 = correct), per judge; total of {len(DROP)}")
for s, lab in (("fc", "FC 512k"), ("k50", "dense k=50"), ("dense", "dense top-5")):
    print(f"-- {lab}")
    for m in FULLWIN:
        cells = []
        for jn in JN:
            v = agg81(V, jn, m, s)
            cells.append(f"{jn} " + "".join(str(int(v[q])) for q in DROP) + f" ({sum(v[q] for q in DROP)})")
        print(f"  {m}: " + " | ".join(cells))
