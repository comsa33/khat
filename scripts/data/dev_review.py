"""Dev review: annotator agreement, adjudicated errors by family, error types and actions.

Usage: python3 scripts/data/dev_review.py
"""
import sys, json, csv, itertools
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "score"))
from verdict_sets import R, jl
D={r["qa_id"]:r for r in jl(R/"data/qa_dev.jsonl") + [x for x in jl(R/"data/auxiliary/legal_reasoning.jsonl") if x["split"] == "dev"]}
lab=jl(R/"results/human_labels/dev_review/annotator_labels.jsonl"); adj={r["qa_id"]:r for r in jl(R/"results/human_labels/dev_review/adjudication.jsonl")}
by=defaultdict(dict)
for r in lab: by[r["qa_id"]][r["annotator"]]=r
print("reviewed",len(by), "blocks", len({r["block"] for r in lab}), Counter(len(d) for d in by.values()))
flag=lambda r: r["verdict"]!="no_error"
print("verdict values", Counter(r["verdict"] for r in lab))
n1=sum(1 for d in by.values() if any(r["verdict"]=="error" for r in d.values())); n2=sum(1 for d in by.values() if all(r["verdict"]=="error" for r in d.values()))
print("flagged>=1",n1,"both",n2, "adjudicated", len(adj), Counter(r["final_error"] for r in adj.values()), Counter(r["n_flag"] for r in adj.values()))
err={q for q,r in adj.items() if r["final_error"]=="yes"}
fam=Counter(D[q]["task_type"] for q in by); fe=Counter(D[q]["task_type"] for q in err)
for f in fam: print(f, fam[f], fe[f])
print("types", Counter(adj[q].get("final_type") for q in err), "actions", Counter(adj[q].get("action") for q in err))
def route(r):
    st=r.get("question_style"); rr=r.get("rewrite_rationale","") or ""
    return st
print("question_style among reviewed", Counter(D[q].get("question_style") for q in by))
anns=sorted({a for d in by.values() for a in d})
for a,b in itertools.combinations(anns,2):
    qs=[q for q,d in by.items() if a in d and b in d]
    x=[by[q][a]["verdict"]=="error" for q in qs]; y=[by[q][b]["verdict"]=="error" for q in qs]
    n=len(qs); po=sum(i==j for i,j in zip(x,y))/n; p1=sum(x)/n; p2=sum(y)/n; pe=p1*p2+(1-p1)*(1-p2)
    k=(po-pe)/(1-pe) if pe<1 else 1.0
    print(a,b,n,f"{100*po:.1f} k={k:.3f} pabak={2*po-1:.3f}")
