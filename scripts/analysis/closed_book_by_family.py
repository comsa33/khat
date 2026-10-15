"""Closed-book accuracy by task family on the 342 evaluated test_public items (claude-sonnet-5-5 judge).

Usage: python3 scripts/analysis/closed_book_by_family.py
"""
import sys, json, collections
from pathlib import Path
from khat.paths import ROOT as R
J = R / "results"
def v(*ps):
    out = {}
    for p in ps:
        for l in open(J / p):
            if l.strip():
                r = json.loads(l); out[r["qa_id"]] = bool(r["judge_correct"])
    return out
P2, P3 = "verdicts/claude-sonnet-5-5/", "verdicts/claude-sonnet-5-5/"
M = {
 "gpt-4.1-mini": [P2+"closed_book_gpt-4.1-mini.jsonl", P3+"closed_book_no_row_ids_gpt-4.1-mini_test_public.jsonl"],
 "gpt-5.4-mini": [P2+"closed_book_gpt-5.4-mini.jsonl", P3+"closed_book_no_row_ids_gpt-5.4-mini_test_public.jsonl"],
 "gpt-5.4-nano": [P2+"closed_book_gpt-5.4-nano.jsonl", P3+"closed_book_no_row_ids_gpt-5.4-nano_test_public.jsonl"],
 "gpt-5.5": [P2+"closed_book_gpt-5.5.jsonl", P3+"closed_book_second_call_gpt-5.5_test_public.jsonl", P3+"closed_book_no_row_ids_gpt-5.5_test_public.jsonl"],
 "minimax-m3": [P2+"closed_book_minimax-m3.jsonl", P3+"closed_book_no_row_ids_minimax-m3_test_public.jsonl"],
 "glm-5.2": [P3+"closed_book_glm-5.2.jsonl"],
 "claude-sonnet-5-5": [P3+"closed_book_claude-sonnet-5-5.jsonl"],
 "claude-haiku-5-5": [P3+"closed_book_claude-haiku-5-5.jsonl"],
}
ex = {l.strip() for l in open(R / "data/exclusions/eval_test_and_dev_sample.txt") if l.strip() and l[0] != "#"}
qa = [json.loads(l) for l in open(R / "data/qa_test_public.jsonl")]
items = [x for x in qa if x["task_type"] != "cross_document_legal_reasoning" and x["qa_id"] not in ex]
assert len(items) == 342
fam = collections.defaultdict(list)
for x in items: fam[x["task_type"]].append(x["qa_id"])
V = {m: v(*ps) for m, ps in M.items()}
for m in V: assert all(x["qa_id"] in V[m] for x in items), m
ms = list(M)
print("| family | n | " + " | ".join(ms) + " |"); print("|---|---:|" + "---:|" * len(ms))
for f in sorted(fam, key=lambda f: -len(fam[f])):
    qs = fam[f]
    print(f"| {f} | {len(qs)} | " + " | ".join(f"{sum(V[m][q] for q in qs)} ({100*sum(V[m][q] for q in qs)/len(qs):.0f}%)" for m in ms) + " |")
print("| all | 342 | " + " | ".join(f"{sum(V[m][x['qa_id']] for x in items)} ({100*sum(V[m][x['qa_id']] for x in items)/342:.0f}%)" for m in ms) + " |")
