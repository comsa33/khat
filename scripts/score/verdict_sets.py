"""Verdict files, item scope and scoring helpers of the four main settings (test_public), shared by score_test.py and the analysis scripts. V holds the verdicts of the claude-sonnet-5-5 judge without the first-call rule.
"""
import csv
import json

from khat.paths import ROOT as R
LEGAL = "cross_document_legal_reasoning"
P = "verdicts/{j}/"
MODELS = ["gpt-4.1-mini", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.5",
          "minimax-m3", "glm-5.2", "claude-sonnet-5-5", "claude-haiku-5-5"]
FILES = {
    "cb": {"gpt-4.1-mini": [P + "closed_book_gpt-4.1-mini", P + "closed_book_no_row_ids_gpt-4.1-mini_test_public"],
           "gpt-5.4-mini": [P + "closed_book_gpt-5.4-mini", P + "closed_book_no_row_ids_gpt-5.4-mini_test_public"],
           "gpt-5.4-nano": [P + "closed_book_gpt-5.4-nano", P + "closed_book_no_row_ids_gpt-5.4-nano_test_public"],
           "gpt-5.5": [P + "closed_book_gpt-5.5", P + "closed_book_second_call_gpt-5.5_test_public", P + "closed_book_no_row_ids_gpt-5.5_test_public"],
           "minimax-m3": [P + "closed_book_minimax-m3", P + "closed_book_no_row_ids_minimax-m3_test_public"],
           "glm-5.2": [P + "closed_book_glm-5.2"],
           "claude-sonnet-5-5": [P + "closed_book_claude-sonnet-5-5"], "claude-haiku-5-5": [P + "closed_book_claude-haiku-5-5"]},
    "bm25": {"gpt-4.1-mini": [P + "bm25_gpt-4.1-mini"], "gpt-5.4-mini": [P + "bm25_gpt-5.4-mini"],
             "gpt-5.4-nano": [P + "bm25_gpt-5.4-nano"], "gpt-5.5": [P + "bm25_gpt-5.5"],
             "minimax-m3": [P + "bm25_minimax-m3"], "glm-5.2": [P + "bm25_glm-5.2"],
             "claude-sonnet-5-5": [P + "bm25_claude-sonnet-5-5"], "claude-haiku-5-5": [P + "bm25_claude-haiku-5-5"]},
    "dense": {"gpt-4.1-mini": [P + "dense_gpt-4.1-mini"], "gpt-5.4-mini": [P + "dense_gpt-5.4-mini"],
              "gpt-5.4-nano": [P + "dense_gpt-5.4-nano"], "gpt-5.5": [P + "dense_gpt-5.5"],
              "minimax-m3": [P + "dense_minimax-m3"], "glm-5.2": [P + "dense_glm-5.2"],
              "claude-sonnet-5-5": [P + "dense_claude-sonnet-5-5"], "claude-haiku-5-5": [P + "dense_claude-haiku-5-5"]},
    "fc": {"gpt-4.1-mini": [P + "full_context_gpt-4.1-mini", P + "full_context_512k_sample_gpt-4.1-mini"],
           "gpt-5.4-mini": [P + "full_context_gpt-5.4-mini"], "gpt-5.4-nano": [P + "full_context_gpt-5.4-nano"],
           "gpt-5.5": [P + "full_context_gpt-5.5", P + "full_context_512k_sample_gpt-5.5"],
           "minimax-m3": [P + "full_context_minimax-m3", P + "full_context_512k_sample_minimax-m3"],
           "glm-5.2": [P + "full_context_glm-5.2", P + "full_context_512k_sample_glm-5.2"],
           "claude-sonnet-5-5": [P + "full_context_claude-sonnet-5-5"], "claude-haiku-5-5": [P + "full_context_claude-haiku-5-5"]},
}
SETTINGS = ("cb", "bm25", "dense", "fc")


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def ids(p):
    return {l.strip() for l in open(p, encoding="utf-8") if l.strip() and not l.startswith("#")}


FIRST_CALL = {}
for _r in csv.DictReader(open(R / "data/first_call_items.csv", encoding="utf-8")):
    FIRST_CALL.setdefault(({"closed_book": "cb"}.get(_r["setting"], _r["setting"]), _r["model"]), set()).add(_r["qa_id"])
FIRST_CALL_RULE = False


def verdicts(setting, model, judge="claude-sonnet-5-5"):
    v = {}
    for t in FILES[setting][model]:
        v.update({r["qa_id"]: bool(r["judge_correct"]) for r in jl(R / "results" / (t.format(j=judge) + ".jsonl"))})
    if FIRST_CALL_RULE:
        for q in FIRST_CALL.get((setting, model), ()):
            if q in v:
                v[q] = False
    return v


BEX = ids(R / "data/exclusions/bundle.txt")
G = {d["qa_id"]: d for d in jl(R / "data/qa_test_public.jsonl")}
EX = ids(R / "data/exclusions/eval.txt")
TEST = sorted(q for q, g in G.items() if g["task_type"] != LEGAL and q not in EX)
w = lambda q: float(G[q].get("cluster_weight", 1))


def items(setting, v):
    return [q for q in TEST if q in v and (setting == "cb" or q not in BEX)]


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return 100 * (c - h), 100 * (c + h)


def acc(v, qs):
    return 100 * sum(v[q] for q in qs) / len(qs), 100 * sum(w(q) for q in qs if v[q]) / sum(w(q) for q in qs)


V = {s: {m: verdicts(s, m) for m in MODELS} for s in SETTINGS}
