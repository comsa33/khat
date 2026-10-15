"""Hit/miss accuracy and closed-book numbers under the primary judge.
"""
import csv, json, sys
from pathlib import Path
from khat.paths import DATA
H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / "judges"))
import panel_verdicts as E
V, _, _ = E.load("full"); G = E.G
rows = {r["qa_id"]: r for r in csv.DictReader(open(E.R / "results/retrieval/nonagg_recall_test.csv", encoding="utf-8"))}
CLS = {json.loads(l)["qa_id"]: json.loads(l) for l in open(DATA / "analysis/answer_classes.jsonl")}
pct = lambda k, n: f"{k}/{n} {100*k/n:.1f}"
for jn, jv in (("kimi", "kimi"), ("sonnet", "claude-sonnet-5-5")):
    print(f"== {jn}")
    for m in E.MODELS:
        v = V[jv][("bm25", m)]; na = [q for q in v if G[q]["task_type"] != "cross_source_aggregation"]
        hit = [q for q in na if float(rows[q]["bm25_1200_r@5"]) >= 1]; miss = [q for q in na if float(rows[q]["bm25_1200_r@5"]) == 0]
        vd = V[jv][("dense", m)]; dh = [q for q in na if float(rows[q]["dense_bge_1200_r@5"]) >= 1]; dm = [q for q in na if float(rows[q]["dense_bge_1200_r@5"]) == 0]
        cb = V[jv][("cb", m)]; sp = [q for q in cb if CLS.get(q, {}).get("S_prime")]
        ag = [q for q in cb if G[q]["task_type"] == "cross_source_aggregation"]
        print(f"  {m:18s} n_na {len(na)} BM25 hit {pct(sum(v[q] for q in hit), len(hit))} miss {pct(sum(v[q] for q in miss), len(miss))} | "
              f"dense hit {pct(sum(vd[q] for q in dh), len(dh))} miss {pct(sum(vd[q] for q in dm), len(dm))} | "
              f"CB S' {pct(sum(cb[q] for q in sp), len(sp))} | CB agg {pct(sum(cb[q] for q in ag), len(ag))}")
