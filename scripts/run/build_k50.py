"""Dense top-50 prompts for the 81 aggregation items; first rebuilds the top-5 prompts and requires them to equal the dense run's.

Usage: python3 scripts/run/build_k50.py [K]   (needs the bge-m3 embedding cache local/cache/bge_m3_emb.pkl or Ollama bge-m3)
"""
import os, sys, json, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import rag_lib as lib
from rag_lib import R
K = int(sys.argv[1]) if len(sys.argv) > 1 else 50

qa = lib.test_qa()
agg = lib.load_jsonl(lib.BASEL / "rag_dense/agg81.jsonl")
tp = {r["qa_id"]: r for r in lib.load_jsonl(lib.BASEL / "rag_dense/tp386.jsonl")}
assert len(agg) == 81
bc, dc = {}, {}
same_agg = same_tp = 0
for s in agg:
    rec = qa[s["qa_id"]]; bid = s["bundle_id"]
    picked = R.retrieve("dense_bge", rec, lib.chunks_for(bid, 1200), 5, bid, bc, dc, s["per_page_max"])
    p = R.rag_prompt(rec, picked)
    same_agg += p == s["prompt"]
    same_tp += p == tp[s["qa_id"]]["prompt"]
lib.save_cache()
print(f"k=5 rebuild: identical to agg81 file {same_agg}/81, to tp386 file {same_tp}/81", flush=True)
assert same_agg == 81 and same_tp == 81, "k=5 rebuild not identical; stop"

out = []
for s in agg:
    rec = qa[s["qa_id"]]; bid = s["bundle_id"]
    picked = R.retrieve("dense_bge", rec, lib.chunks_for(bid, 1200), K, bid, bc, dc, s["per_page_max"])
    r = copy.deepcopy(s)
    r["prompt"] = R.rag_prompt(rec, picked)
    r["k"] = K; r["n_passages"] = len(picked)
    pages = []
    for pid, _ in picked:
        if pid not in pages: pages.append(pid)
    r["retrieved_page_ids"] = pages
    out.append(r)
lib.save_cache()
(lib.BASEL / "k50").mkdir(parents=True, exist_ok=True)
fn = lib.BASEL / "k50" / f"prompts_dense_k{K}_agg81.jsonl"
with open(fn, "w", encoding="utf-8") as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
chars = sorted(len(r["prompt"]) for r in out)
print(f"k={K}: wrote {fn.name}; n_passages {min(r['n_passages'] for r in out)}-{max(r['n_passages'] for r in out)}; prompt chars median {chars[40]} max {chars[-1]}")
