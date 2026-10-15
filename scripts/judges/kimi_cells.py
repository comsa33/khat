"""kimi-k2.6 (primary judge) values: Table S33 kimi row, Table S34 K column and Table 7, with the first-call rule.

Usage: python3 scripts/judges/kimi_cells.py results/verdicts/kimi-k2.6
"""
import json, sys
from pathlib import Path
O = Path(sys.argv[1])
KIMI_FILE = {"cb": "closed_book", "fc": "full_context", "agg512k": "full_context_aggregation", "k50page": "k50_page", "openregion": "open_region"}
def load(s, m):
    return {r["qa_id"]: r["judge_correct"] for r in map(json.loads, open(O / f"{KIMI_FILE.get(s, s)}_{m}.jsonl")) }
import csv
from khat.paths import DATA
FIRST_CALL = {}
for _r in csv.DictReader(open(DATA / "first_call_items.csv", encoding="utf-8")):
    _s = {"closed_book": "cb", "full_context_aggregation": "agg512k"}.get(_r["setting"], _r["setting"])
    FIRST_CALL.setdefault((_s, _r["model"]), set()).add(_r["qa_id"][-4:])
def get(s, m):
    v = load(s, m)
    for q in v:
        if q[-4:] in FIRST_CALL.get((s, m), ()): v[q] = False
    return v
agg_ids = set(load("openregion", "gpt-5.5"))
assert len(agg_ids) == 81
def agg(m):
    fc, a = get("fc", m), get("agg512k", m)
    ids = {q: fc[q] for q in fc if q in agg_ids}; ids.update(a)
    assert set(ids) == agg_ids, (m, len(ids))
    return sum(ids.values())
print("S33 kimi row, correct/81 and %:")
for m in ["gpt-5.5", "glm-5.2", "minimax-m3", "gpt-4.1-mini", "claude-sonnet-5-5", "claude-haiku-5-5"]:
    k = agg(m); print(f"  {m}: {k}/81 = {100*k/81:.1f}")
print("S34 K column:")
for m in ["claude-sonnet-5-5", "claude-haiku-5-5"]:
    for s in ["cb", "bm25", "dense", "k50", "fc", "AGG", "openregion", "oracle"]:
        if s == "AGG": k, n = agg(m), 81
        else: v = get(s, m); k, n = sum(v.values()), len(v)
        print(f"  {m} {s}: {k}/{n} = {100*k/n:.1f}")
print("Table 7, plain accuracy (%) on each model's full-context sample:")
for m in ["gpt-4.1-mini", "gpt-5.5", "minimax-m3", "glm-5.2", "claude-sonnet-5-5", "claude-haiku-5-5", "gpt-5.4-mini", "gpt-5.4-nano"]:
    S = sorted(get("fc", m))
    print(f"  {m}: n={len(S)} | " + " | ".join(f"{s} {100*sum(get(s, m)[q] for q in S)/len(S):.1f}" for s in ["cb", "bm25", "dense", "fc"]))
