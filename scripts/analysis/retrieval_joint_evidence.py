"""Aggregation items whose top-k passages hold both the gold announcement page and all matching rows.

Usage: python3 scripts/analysis/retrieval_joint_evidence.py
"""
import json
import rag_lib as lib, rag_rows as rows
qa = lib.test_qa(); agg, _ = rows.build(qa); rk = lib.Ranker(); out = {}
ch = lib.chunks_for(rows.AGG_BUNDLE, 1200)
L = ["### T10. Aggregation items with BOTH the gold announcement page and all matched rows in top-k (chunk 1,200, original question; Wilson 95%)\n",
     "| retriever | k=5 | k=10 | k=20 | k=50 |", "|---|---|---|---|---|"]
for retr, name in (("bm25","BM25"),("dense_bge","bge-m3"),("rrf","RRF")):
    cells = []
    for k in (5,10,20,50):
        x = 0
        for it in agg:
            idx = rk.full(retr, it["rec"], rows.AGG_BUNDLE, 1200)[:k]
            x += set(it["rec"]["page_ids"]) <= {ch[i][0] for i in idx} and rows.covered(it, idx, 1200) == it["n_rows"]
        cells.append(lib.fmt_ci(x, len(agg))); out[f"{retr}|{k}"] = x
    L.append(f"| {name} | " + " | ".join(cells) + " |")
(lib.OUT / "supp3_tables.md").write_text("\n".join(L), encoding="utf-8")
json.dump(out, open(lib.OUT / "supp3.json", "w"), indent=1)
print("\n".join(L))
