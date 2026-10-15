"""Full-context errors per model by family and tier, and the tiers where the gpt-4.1-mini judge differs.

Usage: python3 scripts/analysis/full_context_errors.py
"""
from collections import Counter
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "score"))
from verdict_sets import *
for m in MODELS:
    v=V["fc"][m]; qs=items("fc",v); err=[q for q in qs if not v[q]]
    agg=sum(1 for q in err if G[q]["task_type"]=="cross_source_aggregation" and G[q]["context_tier"]=="512k")
    c=Counter((G[q]["task_type"],G[q]["context_tier"]) for q in err if not (G[q]["task_type"]=="cross_source_aggregation" and G[q]["context_tier"]=="512k"))
    print(m,len(err),agg,dict(c))
    for judge in ["gpt-4.1-mini"]:
        s=verdicts("fc",m,judge)
        out=[]
        for t in ("32k","64k","128k","256k","512k"):
            t_=[q for q in qs if G[q]["context_tier"]==t]
            if t_:
                a=sum(v[q] for q in t_); b=sum(s[q] for q in t_)
                if a!=b: out.append(f"{t} {b}/{len(t_)}={100*b/len(t_):.0f}")
        print("   gpt-4.1-mini differs:", out)
print([ ["%.1f"%x for x in wilson(k,n)] for k,n in [(19,19),(3,19),(20,20),(19,20)]])
