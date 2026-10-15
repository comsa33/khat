"""Retrieval without model calls: gold-page recall@k of the non-aggregation items and row coverage@k of the aggregation items for BM25, bge-m3 and RRF, chunk sizes and oracle queries (writes local/tables.md, local/results.json and per-item CSVs).

Usage: python3 scripts/analysis/retrieval_coverage.py
"""
from __future__ import annotations

import collections
import csv
import json

import rag_lib as lib
import rag_rows as rows

K_MAIN = [5, 10, 20, 50]
K_ALL = [2, 3, 5, 10, 20, 25, 40, 50, 100]
KMAX = max(K_ALL)
RETRS = ["bm25", "dense_bge", "rrf"]
NAME = {"bm25": "BM25", "dense_bge": "bge-m3", "rrf": "RRF(BM25+bge-m3)"}
SIZES = [600, 1200, 2400]

qa = lib.test_qa()
BEX = {l.strip() for l in open(lib.REL / "data/exclusions/bundle.txt", encoding="utf-8") if l.strip() and not l.startswith("#")}
scope = [r for r in lib.eval_scope(qa) if r["qa_id"] not in BEX]
agg_items, row_checks = rows.build(qa)
agg_ids = {it["qa_id"] for it in agg_items}
non_agg = [r for r in scope if r["qa_id"] not in agg_ids]
assert agg_ids <= {r["qa_id"] for r in scope}
print(f"scope={len(scope)} non-agg={len(non_agg)} agg={len(agg_items)}  row checks={row_checks}", flush=True)

rk = lib.Ranker()
tops: dict = {}


def top(retr, rec, size, query=None, variant="q"):
    key = (retr, size, rec["qa_id"], variant)
    if key not in tops:
        tops[key] = rk.full(retr, rec, rec["bundle_id"], size, query)[:KMAX]
    return tops[key]


runs = [(r, 1200) for r in RETRS] + [("bm25", 600), ("bm25", 2400)]
for retr, size in runs:
    print(f"== ranking {retr} @ {size}", flush=True)
    for i, rec in enumerate(non_agg + [it["rec"] for it in agg_items]):
        top(retr, rec, size)
        if (i + 1) % 100 == 0:
            print(f"   {i + 1}", flush=True)
            lib.save_cache()
    lib.save_cache()

# ---- step 1/3: gold-page recall on non-aggregation items ------------------------------------------
def page_recall(rec, idx, size):
    chunks = lib.chunks_for(rec["bundle_id"], size)
    got = {chunks[i][0] for i in idx}
    g = set(rec.get("page_ids") or [])
    return len(got & g) / len(g)


res = {"scope_n": len(scope), "non_agg_n": len(non_agg), "agg_n": len(agg_items), "row_checks": row_checks}
n_gold = collections.Counter(len(set(r.get("page_ids") or [])) for r in non_agg)
res["non_agg_gold_pages_per_item"] = dict(n_gold)
binary = set(n_gold) == {1}

recall = {}
item_rows = []
for retr, size in runs:
    for k in K_ALL:
        recall[(retr, size, k)] = [page_recall(r, top(retr, r, size)[:k], size) for r in non_agg]
for r in non_agg:
    row = {"qa_id": r["qa_id"], "task_type": r["task_type"], "context_tier": r["context_tier"],
           "bundle_id": r["bundle_id"]}
    for retr, size in runs:
        for k in K_MAIN:
            row[f"{retr}_{size}_r@{k}"] = page_recall(r, top(retr, r, size)[:k], size)
    item_rows.append(row)
with open(lib.OUT / "nonagg_recall_items.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(item_rows[0]))
    w.writeheader()
    w.writerows(item_rows)


def rec_cell(vals):
    n = len(vals)
    if binary:
        return lib.fmt_ci(int(round(sum(vals))), n)
    return f"{100 * sum(vals) / n:.1f}% (n={n})"


res["recall"] = {f"{r}|{s}|{k}": sum(v) / len(v) for (r, s, k), v in recall.items()}

# ---- step 2/3/4: row coverage on aggregation items ------------------------------------------------
def oracle_query(it):
    return it["rec"]["question"] + " " + " ".join(str(v) for v in it["filter"].values())


cov_runs = [(r, s, "q") for r, s in runs] + [(r, 1200, "oracle") for r in RETRS]
for retr, size, var in cov_runs:
    if var == "oracle":
        print(f"== ranking oracle-stage1 {retr} @ {size}", flush=True)
        for it in agg_items:
            top(retr, it["rec"], size, oracle_query(it), "oracle")
        lib.save_cache()

cov = {}
for retr, size, var in cov_runs:
    for k in K_ALL:
        cov[(retr, size, var, k)] = [
            (rows.covered(it, top(retr, it["rec"], size, oracle_query(it) if var == "oracle" else None, var)[:k], size),
             it["n_rows"], "HUG" if it["source"].startswith("hug") else "MOLIT")
            for it in agg_items]

agg_rows = []
for it in agg_items:
    row = {"qa_id": it["qa_id"], "source": it["source"], "op": it["op"], "n_rows": it["n_rows"],
           "filter": json.dumps(it["filter"], ensure_ascii=False)}
    for s in SIZES:
        row[f"min_chunks_{s}"] = rows.min_chunks_needed(it, s)
    for retr, size, var in cov_runs:
        q = oracle_query(it) if var == "oracle" else None
        for k in K_MAIN:
            row[f"{retr}_{size}_{var}_cov@{k}"] = rows.covered(it, top(retr, it["rec"], size, q, var)[:k], size)
    agg_rows.append(row)
with open(lib.OUT / "agg_coverage_items.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(agg_rows[0]))
    w.writeheader()
    w.writerows(agg_rows)


def cov_stats(lst, src=None):
    l = [x for x in lst if src is None or x[2] == src]
    n = len(l)
    mean_frac = sum(c / nd for c, nd, _ in l) / n
    pooled = sum(c for c, _, _ in l) / sum(nd for _, nd, _ in l)
    full = sum(c == nd for c, nd, _ in l)
    anyrow = sum(c > 0 for c, _, _ in l)
    return {"n": n, "mean_item_cov": mean_frac, "pooled_row_cov": pooled, "full": full, "any": anyrow}


res["coverage"] = {f"{r}|{s}|{v}|{k}|{src or 'ALL'}": cov_stats(lst, src)
                   for (r, s, v, k), lst in cov.items() for src in (None, "HUG", "MOLIT")}
res["avg_chunk_chars"] = {}
for s in SIZES:
    bids = sorted({r["bundle_id"] for r in scope})
    allc = [len(t) for b in bids for _, t in lib.chunks_for(b, s)]
    res["avg_chunk_chars"][s] = sum(allc) / len(allc)
json.dump(res, open(lib.OUT / "results.json", "w"), ensure_ascii=False, indent=1)

# ---- tables.md -----------------------------------------------------------------------------------
L = []
L.append(f"scope = {len(scope)} (non-aggregation {len(non_agg)}, aggregation {len(agg_items)}); "
         f"gold pages per non-agg item: {dict(n_gold)}; row checks: {row_checks}\n")
L.append("### T1. Gold-page recall@k, non-aggregation items (chunk 1,200; Wilson 95%)\n")
L.append("| retriever | " + " | ".join(f"k={k}" for k in K_MAIN) + " |")
L.append("|---|" + "---|" * len(K_MAIN))
for retr in RETRS:
    L.append(f"| {NAME[retr]} | " + " | ".join(rec_cell(recall[(retr, 1200, k)]) for k in K_MAIN) + " |")
L.append("")
L.append("### T1b. Recall@k by family and tier (chunk 1,200; % , k=5 / k=20)\n")
groups = [("family", "task_type"), ("tier", "context_tier")]
L.append("| group | n | " + " | ".join(f"{NAME[r]} @5 / @20" for r in RETRS) + " |")
L.append("|---|---:|" + "---|" * len(RETRS))
for gname, field in groups:
    vals = sorted({r[field] for r in non_agg}, key=lambda v: (str(v)))
    for v in vals:
        ix = [i for i, r in enumerate(non_agg) if r[field] == v]
        cells = []
        for retr in RETRS:
            a = [recall[(retr, 1200, 5)][i] for i in ix]
            b = [recall[(retr, 1200, 20)][i] for i in ix]
            cells.append(f"{100 * sum(a) / len(a):.0f} / {100 * sum(b) / len(b):.0f}")
        L.append(f"| {gname}: {v} | {len(ix)} | " + " | ".join(cells) + " |")
L.append("")


def cov_table(title, rowspec):
    L.append(title + "\n")
    L.append("| setting | src | n | " + " | ".join(f"k={k}: row cov / full cover" for k in K_MAIN) + " |")
    L.append("|---|---|---:|" + "---|" * len(K_MAIN))
    for label, (retr, size, var) in rowspec:
        for src in (None, "HUG", "MOLIT"):
            cells = []
            for k in K_MAIN:
                st = cov_stats(cov[(retr, size, var, k)], src)
                lo, hi = lib.wilson(st["full"], st["n"])
                cells.append(f"{100 * st['mean_item_cov']:.1f}% / {st['full']}/{st['n']} "
                             f"[{100 * lo:.0f}, {100 * hi:.0f}]")
            L.append(f"| {label} | {src or 'all'} | {cov_stats(cov[(retr, size, var, 5)], src)['n']} | "
                     + " | ".join(cells) + " |")
    L.append("")
    L.append("row cov = mean over items of (matched rows in top-k)/(matched rows); full cover = items with all "
             "matched rows in top-k, Wilson 95% in brackets.\n")


cov_table("### T2. Row coverage@k, aggregation items (chunk 1,200, original question)",
          [(NAME[r], (r, 1200, "q")) for r in RETRS])

L.append("### T3. Chunk size, BM25 only (non-agg recall@k; agg row coverage@k / full cover)\n")
L.append("avg chunk chars over scope bundles: " + ", ".join(f"{s}: {res['avg_chunk_chars'][s]:.0f}" for s in SIZES) + "\n")
L.append("| chunk | " + " | ".join(f"recall@{k}" for k in K_MAIN) + " | " +
         " | ".join(f"agg cov@{k}" for k in K_MAIN) + " |")
L.append("|---|" + "---|" * (2 * len(K_MAIN)))
for s in SIZES:
    rc = [f"{100 * sum(recall[('bm25', s, k)]) / len(non_agg):.1f}%" for k in K_MAIN]
    cc = []
    for k in K_MAIN:
        st = cov_stats(cov[("bm25", s, "q", k)])
        cc.append(f"{100 * st['mean_item_cov']:.1f}% ({st['full']}/{st['n']})")
    L.append(f"| {s} | " + " | ".join(rc) + " | " + " | ".join(cc) + " |")
L.append("")
L.append("Equal character budget (BM25): chunk×k with the same nominal chars.\n")
L.append("| budget (nominal chars) | 600×k | 1,200×k | 2,400×k | recall (600/1200/2400) | agg cov (600/1200/2400) | agg full (600/1200/2400) |")
L.append("|---|---|---|---|---|---|---|")
for k12 in (5, 10, 20, 50):
    k6, k24 = 2 * k12, k12 // 2
    if k6 not in K_ALL or k24 not in K_ALL:
        continue
    rc = [f"{100 * sum(recall[('bm25', s, kk)]) / len(non_agg):.1f}" for s, kk in ((600, k6), (1200, k12), (2400, k24))]
    sts = [cov_stats(cov[("bm25", s, "q", kk)]) for s, kk in ((600, k6), (1200, k12), (2400, k24))]
    L.append(f"| {1200 * k12:,} | k={k6} | k={k12} | k={k24} | " + " / ".join(rc) + " | "
             + " / ".join(f"{100 * st['mean_item_cov']:.1f}" for st in sts) + " | "
             + " / ".join(f"{st['full']}" for st in sts) + " |")
L.append("")

cov_table("### T4. Oracle stage-1 query (question + gold filter values: region & year / district & month), chunk 1,200",
          [(f"{NAME[r]} original", (r, 1200, "q")) for r in ("bm25", "rrf")] +
          [(f"{NAME[r]} oracle-stage1", (r, 1200, "oracle")) for r in RETRS])

L.append("### T5. Row coverage curve, full grid of k (mean item coverage %, all 81)\n")
L.append("| setting | " + " | ".join(f"k={k}" for k in K_ALL) + " |")
L.append("|---|" + "---|" * len(K_ALL))
for retr, size, var in cov_runs:
    L.append(f"| {NAME[retr]} {size} {var} | " +
             " | ".join(f"{100 * cov_stats(cov[(retr, size, var, k)])['mean_item_cov']:.1f}" for k in K_ALL) + " |")
L.append("")
L.append("### T6. Distinct chunks holding the matched rows (lower bound on k for full coverage)\n")
L.append("| chunk | HUG (72) min chunks: count | MOLIT (9) |")
L.append("|---|---|---|")
for s in SIZES:
    h = collections.Counter(r[f"min_chunks_{s}"] for r in agg_rows if r["source"].startswith("hug"))
    m = collections.Counter(r[f"min_chunks_{s}"] for r in agg_rows if not r["source"].startswith("hug"))
    L.append(f"| {s} | " + ", ".join(f"{a}:{b}" for a, b in sorted(h.items())) + " | "
             + ", ".join(f"{a}:{b}" for a, b in sorted(m.items())) + " |")
L.append("")
(lib.OUT / "tables.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
