"""Wilson intervals of the all-gold-pages counts, table-row share of the 512k bundle and top-k composition of the aggregation items.

Usage: python3 scripts/analysis/retrieval_coverage_extra.py   (after retrieval_coverage.py)
"""
from __future__ import annotations

import csv
import json

import rag_lib as lib
import rag_rows as rows
import table_rows as C

out = {}
L = []
items = list(csv.DictReader(open(lib.OUT / "nonagg_recall_items.csv", encoding="utf-8")))
L.append("### T1-CI. Non-agg items with all gold pages in top-k (chunk 1,200; Wilson 95%)\n")
L.append("| retriever | k=5 | k=10 | k=20 | k=50 |")
L.append("|---|---|---|---|---|")
for retr, name in (("bm25", "BM25"), ("dense_bge", "bge-m3"), ("rrf", "RRF")):
    cells = []
    for k in (5, 10, 20, 50):
        x = sum(float(r[f"{retr}_1200_r@{k}"]) == 1.0 for r in items)
        cells.append(lib.fmt_ci(x, len(items)))
        out[f"allgold|{retr}|{k}"] = x
    L.append(f"| {name} | " + " | ".join(cells) + " |")
L.append("")

lines = (lib.BUNDLES / f"{rows.AGG_BUNDLE}.txt").read_text(encoding="utf-8").splitlines()
lineset = set(lines)
hug_all = {rows.rendered("hug-sale-history", rid) for rid in rows._hug}
mol_all = {rows.rendered("molit-apt-trade-detail", rid) for rid in rows._molit}
hug_in = hug_all & lineset
mol_in = mol_all & lineset
qa = lib.test_qa()
agg, _ = rows.build(qa)
needed_hug = set().union(*[it["lines"] for it in agg if it["source"].startswith("hug")])
needed_mol = set().union(*[it["lines"] for it in agg if not it["source"].startswith("hug")])
L.append("### T7. HUG / MOLIT rows present in the 512k bundle\n")
L.append(f"- HUG rows in bundle: {len(hug_in)}; union of rows matched by the 72 HUG items: {len(needed_hug)}")
L.append(f"- MOLIT rows in bundle: {len(mol_in)}; union of rows matched by the 9 MOLIT items: {len(needed_mol)}")
for s in (600, 1200, 2400):
    cls = rows.chunk_line_sets(s)
    nh = sum(1 for c in cls if c & hug_in)
    nm = sum(1 for c in cls if c & mol_in)
    L.append(f"- chunk {s}: total chunks {len(cls)}, chunks with any HUG row {nh}, with any MOLIT row {nm}")
    out[f"table_chunks|{s}"] = {"total": len(cls), "hug": nh, "molit": nm}

rk = lib.Ranker()
cls = rows.chunk_line_sets(1200)
table_ch = [bool(c & (hug_in | mol_in)) for c in cls]
L.append("\n### T8. Top-k composition for aggregation items (chunk 1,200, original question; mean per item)\n")
L.append("| retriever | k | table-row chunks in top-k | chunks with own matched rows | own-row chunks needed (mean) |")
L.append("|---|---:|---:|---:|---:|")
for retr, name in (("bm25", "BM25"), ("dense_bge", "bge-m3"), ("rrf", "RRF")):
    for k in (5, 20, 50):
        t = o = need = 0
        for it in agg:
            idx = rk.full(retr, it["rec"], rows.AGG_BUNDLE, 1200)[:k]
            t += sum(table_ch[i] for i in idx)
            o += sum(bool(it["lines"] & cls[i]) for i in idx)
            need += rows.min_chunks_needed(it, 1200)
        n = len(agg)
        L.append(f"| {name} | {k} | {t / n:.1f} | {o / n:.1f} | {need / n:.1f} |")
        out[f"composition|{retr}|{k}"] = {"table": t / n, "own": o / n}
lib.save_cache()
(lib.OUT / "supp_tables.md").write_text("\n".join(L), encoding="utf-8")
json.dump(out, open(lib.OUT / "supp.json", "w"), indent=1)
print("\n".join(L))
