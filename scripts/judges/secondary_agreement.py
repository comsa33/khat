"""Agreement and Cohen kappa of the claude-sonnet-5-5 and gpt-4.1-mini judges per model and setting.

Usage: python3 scripts/judges/secondary_agreement.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "score"))
from verdict_sets import *
def kappa(a,b):
    n=len(a); po=sum(x==y for x,y in zip(a,b))/n
    pa=sum(a)/n; pb=sum(b)/n; pe=pa*pb+(1-pa)*(1-pb)
    return po,(po-pe)/(1-pe) if pe<1 else 1.0
for m in MODELS:
    A=[];B=[]; per=[]
    for s in SETTINGS:
        p=V[s][m]; sc=verdicts(s,m,"gpt-4.1-mini"); qs=items(s,p)
        a=[p[q] for q in qs]; b=[sc[q] for q in qs]; A+=a; B+=b
        po,k=kappa(a,b); per.append(f"{s} {100*po:.1f}/{k:.2f}")
    po,k=kappa(A,B); print(m, f"all {100*po:.1f} k {k:.3f}", per)
