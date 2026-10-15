"""Recall@5 and accuracy of BM25 and bge-m3 by whether the question carries an announcement number or page marker.

Usage: python3 scripts/analysis/identifier_recall.py
"""
import csv, re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "score"))
from verdict_sets import *
rows={r["qa_id"]:r for r in csv.DictReader(open(R/"results/retrieval/nonagg_recall_test.csv")) if r["qa_id"] not in BEX}
ANN=re.compile(r"(?<!\d)\d{10}(?!\d)"); PG=re.compile(r"\(\s*p\.\s*\d+\s*\)")
S=items("bm25",V["bm25"]["gpt-5.5"])
RET=("long_context_retrieval","long_distance_retrieval")
def rep(name,qs):
    qs=[q for q in qs if q in rows]
    b=100*sum(float(rows[q]["bm25_1200_r@5"]) for q in qs)/len(qs); d=100*sum(float(rows[q]["dense_bge_1200_r@5"]) for q in qs)/len(qs)
    ab=sum(100*sum(V["bm25"][m][q] for q in qs)/len(qs) for m in MODELS)/8; ad=sum(100*sum(V["dense"][m][q] for q in qs)/len(qs) for m in MODELS)/8
    print(f"| {name} | {len(qs)} | {b:.1f} | {d:.1f} | {ab:.1f} | {ad:.1f} |")
print("| non-agg subset | n | BM25 r@5 | bge-m3 r@5 | BM25 acc mean8 | dense acc mean8 |")
rep("all non-agg",S)
rep("ann# present",[q for q in S if ANN.search(G[q]["question"])])
rep("ann# absent",[q for q in S if not ANN.search(G[q]["question"])])
rep("p.N present",[q for q in S if PG.search(G[q]["question"])])
rep("p.N absent",[q for q in S if not PG.search(G[q]["question"])])
rep("retrieval fam ann#",[q for q in S if G[q]["task_type"] in RET and ANN.search(G[q]["question"])])
rep("retrieval fam no ann#",[q for q in S if G[q]["task_type"] in RET and not ANN.search(G[q]["question"])])
rep("no identifier at all",[q for q in S if not ANN.search(G[q]["question"]) and not PG.search(G[q]["question"])])
