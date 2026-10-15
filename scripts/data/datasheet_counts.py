"""Dataset counts, errata, test_public review agreement and the Wilson intervals of the datasheet.

Usage: python3 scripts/data/datasheet_counts.py
"""
import sys, json, csv, ast, itertools
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "score"))
from verdict_sets import wilson, R, jl
L=lambda x: ast.literal_eval(x) if isinstance(x,str) and x[:1] in "[{" else x
A={}
for s in ["dev","test_public"]:
    for r in jl(R/f"data/qa_{s}.jsonl") + [x for x in jl(R/"data/auxiliary/legal_reasoning.jsonl") if x["split"] == s]: A[r["qa_id"]]=r
print("total",len(A), Counter(r["split"] for r in A.values()))
fam=Counter(r["task_type"] for r in A.values()); print(fam)
pred=[q for q,r in A.items() if L(r.get("gold_predicate")) not in (None,"None","",{})]
print("predicate",len(pred), Counter(A[q]["task_type"] for q in pred))
anns=set(); prov=set(); sido=set()
for r in A.values():
    for a in L(r["announcement_ids"]): anns.add(a)
    if L(r["announcement_ids"]): prov.add(r["provider"]); sido.add(r["region_sido"])
print("announcements",len(anns),"providers",len(prov),sorted(prov),"sido",len(sido),sorted(sido))
print("tiers", Counter(r.get("context_tier") for r in A.values()))
print("clusters", len({r["cluster_id"] for r in A.values()}))
print("tags", Counter(t for r in A.values() for t in L(r["split_tags"])))
print("metric", Counter((L(r["evaluation"]) or {}).get("metric") for r in A.values()))
ans=[q for q,r in A.items() if r["task_type"]=="answerability_detection"]
print("answerability",len(ans),"with bundle",sum(1 for q in ans if A[q].get("bundle_id")))
e2=list(csv.DictReader(open(R/"data/errata/test_review.csv"))); e7=list(csv.DictReader(open(R/"data/errata/dev_review.csv")))
print("test review status", Counter((r["split"],r["status"]) for r in e2))
print("test review types", Counter((r["errata_type"],r["status"]) for r in e2 if r["split"]=="test_public"))
for t in set(r["errata_type"] for r in e2 if r["split"]=="test_public"):
    print("  ",t, sorted(r["qa_id"][-4:] for r in e2 if r["split"]=="test_public" and r["errata_type"]==t))
print("dev review status", Counter(r["status"] for r in e7), "types", Counter(r["errata_type"] for r in e7 if r["status"]!="flag"))
lab=jl(R/"results/human_labels/test_review/annotator_labels.jsonl"); adj={r["qa_id"]:r for r in jl(R/"results/human_labels/test_review/adjudication.jsonl")}
by=defaultdict(dict); strat={}
for r in lab: by[r["qa_id"]][r["annotator"]]=r; strat[r["qa_id"]]=r["stratum"]
print("reviewed",len(by), Counter(strat.values()), Counter(r["annotator"] for r in lab))
nflag=Counter(sum(v["error_label"] for v in d.values()) for d in by.values()); print("nflag dist",nflag)
final=[q for q,r in adj.items() if r["final_error"]=="yes"]; print("final errors",len(final), Counter(strat[q] for q in final))
maj={q for q,d in by.items() if sum(v["error_label"] for v in d.values())>=2}
print("majority",len(maj),"all3",sum(1 for d in by.values() if sum(v["error_label"] for v in d.values())==3),"any",sum(1 for d in by.values() if any(v["error_label"] for v in d.values())))
print("maj-only", sorted(q[-4:] for q in maj-set(final)), "final-only", sorted(q[-4:] for q in set(final)-maj))
unnat={q for q,d in by.items() if sum(v["q4_natural"]=="no" for v in d.values())>=2}
EX=set(l.strip() for l in open(R/"data/exclusions/eval.txt") if l.strip() and not l.startswith("#"))
print("unnatural",len(unnat),Counter(strat[q] for q in unnat),"in eval 342",sum(1 for q in unnat if q not in EX))
ann=sorted({a for d in by.values() for a in d})
def kappa(x,y):
    n=len(x); po=sum(a==b for a,b in zip(x,y))/n; p1=sum(x)/n; p2=sum(y)/n; pe=p1*p2+(1-p1)*(1-p2); return po,(po-pe)/(1-pe),2*po-1
for a,b in itertools.combinations(ann,2):
    qs=[q for q in by if a in by[q] and b in by[q]]
    x=[by[q][a]["error_label"] for q in qs]; y=[by[q][b]["error_label"] for q in qs]
    po,k,pab=kappa(x,y); print(a,b,len(qs),f"{100*po:.1f} k={k:.3f} pabak={pab:.3f}")
for k,n in [(25,179),(4,168),(29,347),(7,100),(120,1534),(28,56),(5,20),(13,82),(3,25),(47,485),(2,27),(12,180),(10,474),(0,90),(0,60),(0,35),(47,496),(63,529),(10,509)]:
    lo,hi=wilson(k,n); print(f"wilson {k}/{n} {100*k/n:.1f} [{lo:.1f}, {hi:.1f}]")
