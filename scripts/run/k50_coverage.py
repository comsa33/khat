"""Per-item coverage of the dense top-50 context: announcement page present, matched rows covered.

Usage: python3 scripts/run/k50_coverage.py   (after build_k50.py; writes local/runs/k50/coverage_k50.json)
"""
import os, sys, json
from pathlib import Path
from khat.paths import LOCAL
HERE = LOCAL / "runs" / "k50"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import rag_lib as lib, rag_rows as rows
from rag_lib import R
qa = lib.test_qa(); agg, _ = rows.build(qa); rk = lib.Ranker()
ch = lib.chunks_for(rows.AGG_BUNDLE, 1200)
P = {json.loads(l)["qa_id"]: json.loads(l) for l in open(HERE / "prompts_dense_k50_agg81.jsonl")}
bc, dc = {}, {}
per, same = {}, 0
for it in agg:
    q = it["rec"]["qa_id"]; idx = rk.full("dense_bge", it["rec"], rows.AGG_BUNDLE, 1200)[:50]
    picked = R.retrieve("dense_bge", it["rec"], ch, 50, rows.AGG_BUNDLE, bc, dc, 0)
    same += sorted(ch[i][1] for i in idx) == sorted(t for _, t in picked)
    page = set(it["rec"]["page_ids"]) <= {ch[i][0] for i in idx}
    cov = rows.covered(it, idx, 1200)
    per[q] = {"page": page, "rows_covered": cov, "n_rows": it["n_rows"], "full": page and cov == it["n_rows"],
              "op": it["rec"]["gold_predicate"]["op"]}
    assert P[q]["bundle_id"] == rows.AGG_BUNDLE
lib.save_cache()
print("ranker top-50 == prompt chunks:", same, "/", len(agg), "| full", sum(v["full"] for v in per.values()),
      "page", sum(v["page"] for v in per.values()), "rows-only", sum(v["rows_covered"] == v["n_rows"] for v in per.values()))
json.dump(per, open(HERE / "coverage_k50.json", "w"), indent=1)
