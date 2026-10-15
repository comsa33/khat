#!/usr/bin/env python3
"""Closed-book accuracy by answer class.
"""
import json
import re
import sys
from pathlib import Path
from khat.paths import DATA

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "score"))
from verdict_sets import G, MODELS, TEST, V, verdicts, wilson, jl, R

CLS = {r["qa_id"]: r for r in jl(DATA / "analysis/answer_classes.jsonl")}
assert len(CLS) == 167
PRED_FILES = {
    "gpt-4.1-mini": ["closed_book_gpt-4.1-mini", "closed_book_no_row_ids_gpt-4.1-mini_test_public"],
    "gpt-5.4-mini": ["closed_book_gpt-5.4-mini", "closed_book_no_row_ids_gpt-5.4-mini_test_public"],
    "gpt-5.4-nano": ["closed_book_gpt-5.4-nano", "closed_book_no_row_ids_gpt-5.4-nano_test_public"],
    "gpt-5.5": ["closed_book_gpt-5.5", "closed_book_second_call_gpt-5.5_test_public", "closed_book_no_row_ids_gpt-5.5_test_public"],
    "minimax-m3": ["closed_book_minimax-m3", "closed_book_no_row_ids_minimax-m3_test_public"],
    "glm-5.2": ["closed_book_glm-5.2"],
    "claude-sonnet-5-5": ["closed_book_claude-sonnet-5-5"], "claude-haiku-5-5": ["closed_book_claude-haiku-5-5"]}
PRED = {}
for m, layers in PRED_FILES.items():
    d = {}
    for t in layers:
        d.update({r["qa_id"]: (r.get("prediction") or "") for r in jl(R / "results/predictions" / f"{t}.jsonl")})
    PRED[m] = d
ABST = re.compile(r"확정할 수 없|답할 수 없|알 수 없|정보가 없|unanswerable")
first = lambda s: re.split(r"[.!?。\n]", (s or "").strip(), maxsplit=1)[0][:80]
abst = lambda m, q: bool(ABST.search(first(PRED[m][q])))

print("check, abstention rate on the 342 (Section S3 rule):",
      ", ".join(f"{m} {100 * sum(abst(m, q) for q in TEST) / len(TEST):.1f}" for m in MODELS))
SEC = {m: verdicts("cb", m, "gpt-4.1-mini") for m in MODELS}
classes = {"L": lambda r: r["L"], "R": lambda r: r["R"], "N": lambda r: r["N"], "S'": lambda r: r["S_prime"],
           "S": lambda r: r["S"], "all": lambda r: True}
print("\nclass sizes:", {k: sum(f(r) for r in CLS.values()) for k, f in classes.items()})
for judge, VV in (("claude-sonnet-5-5", {m: V["cb"][m] for m in MODELS}), ("gpt-4.1-mini", SEC)):
    print(f"\n== closed-book accuracy by class, {judge} (correct/n, %)")
    print("model\t" + "\t".join(classes))
    for m in MODELS:
        cells = []
        for k, f in classes.items():
            qs = [q for q, r in CLS.items() if f(r)]
            c = sum(VV[m][q] for q in qs)
            cells.append(f"{c}/{len(qs)} ({100 * c / len(qs):.1f})" if qs else "-")
        print(m + "\t" + "\t".join(cells))
    sp = [q for q, r in CLS.items() if r["S_prime"]]
    acc = {m: sum(VV[m][q] for q in sp) for m in MODELS}
    lo, hi = wilson(acc["gpt-5.5"], len(sp))
    best = max((100 * acc[m] / len(sp), m) for m in MODELS if m != "gpt-5.5")
    print(f"outcome rule: gpt-5.5 S' {acc['gpt-5.5']}/{len(sp)} = {100 * acc['gpt-5.5'] / len(sp):.1f}%, "
          f"Wilson 95% [{lo:.1f}, {hi:.1f}]; highest other S' {best[0]:.1f}% ({best[1]}); "
          f"lower bound above it: {lo > best[0]} -> branch {'(i)' if lo > best[0] else '(ii)'}")
    print("S' abstention and accuracy on non-abstaining answers:")
    for m in MODELS:
        ab = [q for q in sp if abst(m, q)]
        na = [q for q in sp if not abst(m, q)]
        c = sum(VV[m][q] for q in na)
        print(f"  {m}: abstains on {len(ab)}/{len(sp)} ({100 * len(ab) / len(sp):.1f}%); "
              f"non-abstaining {c}/{len(na)}" + (f" ({100 * c / len(na):.1f}%)" if na else ""))
    print("  abstaining answers judged correct on S':", {m: sum(VV[m][q] for q in sp if abst(m, q)) for m in MODELS})

man = {}
for l in open(R / "data/source_manifest.jsonl", encoding="utf-8"):
    pass
import glob
dates = {}
for f in ("construction/announcement_targets.jsonl", "construction/announcement_targets_seed.jsonl", "source_manifest.jsonl"):
    for r in jl(R / "data" / f):
        if r.get("announcement_id") and r.get("announcement_date"):
            dates.setdefault(r["announcement_id"], r["announcement_date"])
from collections import Counter
sp_ann = Counter()
for q, r in CLS.items():
    if r["S_prime"]:
        for a in r["announcement_ids"]:
            sp_ann[(a, dates.get(a, "date not in manifest"))] += 1
print("\nS' items by announcement and date:")
for (a, d), n in sorted(sp_ann.items(), key=lambda x: x[0][1]):
    print(f"  {d}\t{n}\t{a}")
