"""The 122 dev items without bundle evidence: aggregation rows present vs comparison items a closed-book model answers.

Usage: python3 scripts/analysis/unanswerable_classes.py
"""
import json, collections
from khat.paths import DATA
d = json.load(open(DATA / "analysis/no_bundle_evidence_items.json")); cbc = json.load(open(DATA / "analysis/no_bundle_evidence_cb_correct.json"))
anyc = {q for v in cbc.values() for q in v}; g55 = set(cbc["gpt-5.5"])
C = collections.Counter()
for x in d:
    stem = any(x["term_stem_in_q"].values())
    if x["fam"] == "cross_source_aggregation":
        k = "agg: rows present, region only via the announcement title" if x.get("n_rows_in_bundle_by_filter") else "agg: rows absent"
    else:
        k = "comparison: some CB model correct" if x["q"] in anyc else ("comparison: place stem in question" if stem else "comparison: no CB correct, no stem")
    C[(x["fam"][:12], x["nmiss"], k)] += 1
for k, v in sorted(C.items()): print(k, v)
cmp_ = [x for x in d if x["fam"] != "cross_source_aggregation"]
print("comparison items:", len(cmp_), "gpt-5.5 CB correct:", sum(x["q"] in g55 for x in cmp_), "any of 4 CB correct:", sum(x["q"] in anyc for x in cmp_))
print("by nmiss:", {n: (sum(1 for x in cmp_ if x["nmiss"] == n), sum(1 for x in cmp_ if x["nmiss"] == n and x["q"] in g55)) for n in (1, 2)})
agg = [x for x in d if x["fam"] == "cross_source_aggregation"]
print("agg: rows in bundle by filter > 0:", sum(1 for x in agg if x.get("n_rows_in_bundle_by_filter")), "| stem in q:", sum(any(x["term_stem_in_q"].values()) for x in agg), "| distinct announcements:", len({x["answer"] for x in agg}))
