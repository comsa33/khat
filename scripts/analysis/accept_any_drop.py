"""Accuracy drop per model and setting if the accept-any items were scored against the original gold.

Usage: python3 scripts/analysis/accept_any_drop.py
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "score"))
from verdict_sets import *
AA=set(json.load(open(R/"data/errata/test_review_gold_fixes.json"))["accept_any"])
OLD = {
 "gpt-4.1-mini": {"cb": "closed_book_gpt-4.1-mini", "bm25": "bm25_gpt-4.1-mini", "dense": "dense_gpt-4.1-mini", "fc": "full_context_gpt-4.1-mini"},
 "gpt-5.4-mini": {"cb": "closed_book_gpt-5.4-mini", "bm25": "bm25_gpt-5.4-mini", "dense": "dense_gpt-5.4-mini", "fc": "full_context_gpt-5.4-mini"},
 "gpt-5.4-nano": {"cb": "closed_book_gpt-5.4-nano", "bm25": "bm25_gpt-5.4-nano", "dense": "dense_gpt-5.4-nano", "fc": "full_context_gpt-5.4-nano"},
 "gpt-5.5": {"cb": "closed_book_gpt-5.5", "bm25": "bm25_gpt-5.5", "dense": "dense_gpt-5.5", "fc": "full_context_gpt-5.5"},
 "minimax-m3": {"cb": "closed_book_minimax-m3", "bm25": "bm25_minimax-m3", "dense": "dense_minimax-m3", "fc": "full_context_minimax-m3"},
 "glm-5.2": {"cb": "closed_book_no_reasoning_glm-5.2", "bm25": "bm25_no_reasoning_glm-5.2", "dense": "dense_no_reasoning_glm-5.2", "fc": "full_context_glm-5.2"},
}
def ld(p):
    return {r["qa_id"]:bool(r["judge_correct"]) for r in jl(p)} if p.exists() else {}
res={}
for m,tags in OLD.items():
    for s,t in tags.items():
        st=ld(R/"results/verdicts/gpt-4.1-mini_original_gold"/f"{t}.jsonl")
        g=ld(R/"results/verdicts/gpt-4.1-mini"/f"{t}.jsonl"); c=ld(R/"results/verdicts/claude-sonnet-5-5"/f"{t}.jsonl")
        scope=set(items(s, V[s][m]))
        ids=[q for q in g if q in scope]
        aa=[q for q in ids if q in AA]
        if not ids: continue
        gd=sum(g[q] for q in aa)-sum(st.get(q,g[q]) for q in aa)
        cd=sum(c[q] for q in aa)-sum(c[q] and st.get(q,c[q]) for q in aa)
        res[(m,s)]=(len(ids),len(aa),100*gd/len(ids),100*cd/len(ids))
        print(m,s,"n",len(ids),"aa",len(aa),"sec drop %.2f"%(100*gd/len(ids)),"prim drop %.2f"%(100*cd/len(ids)))
for s in ["bm25","dense","fc","cb"]:
    v=[res[k] for k in res if k[1]==s]
    print(s,"sec range %.1f-%.1f"%(min(x[2] for x in v),max(x[2] for x in v)),"prim max %.1f"%max(x[3] for x in v))
