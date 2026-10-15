#!/usr/bin/env python3
"""kimi-k2.6 judge items for every reported prediction (question, gold and prediction as claude-sonnet-5-5 saw them), closed-book requests with the announcement number and page marker removed from the question, and the minimax-m3 512k items.

Usage: python3 scripts/judges/build_panel_inputs.py [--out local/runs/judge_panel]   (after the judge-test, closed-book and k=50 judge builders)
"""
import argparse
import collections
import glob
import json
import re
from pathlib import Path

from khat.paths import ROOT as REL, LOCAL
RUNS = LOCAL / "runs"
J2 = RUNS / "judge_test"
JN = RUNS / "judge_reruns"
E1 = RUNS / "cb_fix"
K50 = RUNS / "k50/judge"
K50P = RUNS / "k50page/judge"

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(RUNS / "judge_panel"))
ap.add_argument("--ann-repl", default="", help="string that replaces a ten-digit announcement number")
ap.add_argument("--page-repl", default="", help="string that replaces a page marker such as (p.1)")
ap.add_argument("--locator", choices=["keep", "mask"], default="keep",
                help="keep: locator lines unchanged; mask: also replace ten-digit numbers and -pNNN in locator lines")
ap.add_argument("--rescoring-scope", choices=["all", "new"], default="all",
                help="all: every reported prediction (10,722); new: only those without a kimi-k2.6 verdict (7,546)")
ap.add_argument("--cap-gpt5", type=int, default=16000)
ap.add_argument("--cap-gpt41", type=int, default=4000)
ap.add_argument("--cap-claude", type=int, default=16384)
ap.add_argument("--cap-glm", type=int, default=16384)
ap.add_argument("--cap-minimax", type=int, default=32768)
args = ap.parse_args()
OUT = Path(args.out)

jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
ids = lambda p: {l.strip() for l in open(p, encoding="utf-8") if l.strip() and not l.startswith("#")}
G = {r["qa_id"]: r for r in jl(REL / "data/qa_test_public.jsonl")}
EX = ids(REL / "data/exclusions/eval.txt")
BEX = ids(REL / "data/exclusions/bundle.txt")
TEST = sorted(q for q, g in G.items() if g["task_type"] != "cross_document_legal_reasoning" and q not in EX)

# ---------------------------------------------------------------- claude-sonnet-5-5 judge prompts
PAT = re.compile(r"\n\n\[질문\] (.*)\n\[정답\] (.*)\n\[모델답변\] (.*)\n\n판정:$", re.S)


def parse(p):
    m = PAT.search(p)
    return m.groups() if m else None


def msg_text(req):
    return req["params"]["messages"][0]["content"]


prompts = collections.defaultdict(dict)
src = {}
core = {r["custom_id"]: r["prompt"] for r in jl(J2 / "prompts.jsonl")}
for r in jl(J2 / "index.jsonl"):
    prompts[r["tag"]][r["qa_id"]] = core[r["custom_id"]]
gf = {r["custom_id"]: r["prompt"] for r in jl(J2 / "prompts_multi.jsonl")}
n_gf = 0
for r in jl(J2 / "index_multi.jsonl"):
    prompts[r["tag"]][r["qa_id"]] = gf[r["custom_id"]]
    n_gf += 1
jn = {r["custom_id"]: r["prompt"] for r in jl(JN / "prompts.jsonl")}
for r in jl(JN / "index.jsonl"):
    prompts[r["tag"]][r["qa_id"]] = jn[r["custom_id"]]
for d, letters in ((K50, "ABC"), (K50P, "A")):
    for L in letters:
        req = {r["custom_id"]: msg_text(r) for r in jl(d / f"claude_requests_{L}.jsonl")}
        for r in jl(d / f"index_{L}.jsonl"):
            prompts[r["tag"]][r["qa_id"]] = req[r["custom_id"]]

ALIAS = {}
for m in ("gpt-4.1-mini", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.5", "minimax-m3", "glm-5.2"):
    ALIAS[f"closed_book_no_row_ids_{m}_test_public"] = f"e1_{m}_test"
    ALIAS[f"oracle_agg81_{m}"] = f"e3oracle_{m}"
    ALIAS[f"openregion_agg81_{m}"] = f"e3region_{m}"
ALIAS.update({"closed_book_glm-5.2": "e2_glm-5.2_cb", "bm25_glm-5.2": "e2_glm-5.2_bm25",
              "dense_glm-5.2": "e2_glm-5.2_dense", "closed_book_second_call_gpt-5.5_test_public": "g55retry_test",
              "k50dense_minimax-m3": "k50dense_minimax-m3-thinkon"})
for m, short in (("claude-sonnet-5-5", "sonnet55"), ("claude-haiku-5-5", "haiku55")):
    for s in ("cb", "bm25", "dense", "fc", "agg512k"):
        ALIAS[f"{s}_{m}"] = f"{short}_{s}"
    ALIAS[f"openregion_agg81_{m}"] = f"{short}_e3region"
    ALIAS[f"oracle_agg81_{m}"] = f"{short}_e3oracle"

FILES = {
    "cb": {"gpt-4.1-mini": ["gpt-4.1-mini_cb", "closed_book_no_row_ids_gpt-4.1-mini_test_public"],
           "gpt-5.4-mini": ["gpt-5.4-mini_cb", "closed_book_no_row_ids_gpt-5.4-mini_test_public"],
           "gpt-5.4-nano": ["gpt-5.4-nano_cb", "closed_book_no_row_ids_gpt-5.4-nano_test_public"],
           "gpt-5.5": ["gpt-5.5_cb", "closed_book_second_call_gpt-5.5_test_public", "closed_book_no_row_ids_gpt-5.5_test_public"],
           "minimax-m3": ["cb_minimax-m3-cloud_tp", "closed_book_no_row_ids_minimax-m3_test_public"],
           "glm-5.2": ["closed_book_glm-5.2"]},
    "bm25": {"gpt-4.1-mini": ["gpt-4.1-mini_rag"], "gpt-5.4-mini": ["gpt-5.4-mini_rag"],
             "gpt-5.4-nano": ["gpt-5.4-nano_rag"], "gpt-5.5": ["gpt-5.5_rag"],
             "minimax-m3": ["rag_minimax-m3-cloud_tp"], "glm-5.2": ["bm25_glm-5.2"]},
    "dense": {m: [f"jf386_{m}_dense"] for m in ("gpt-4.1-mini", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.5", "minimax-m3")}
             | {"glm-5.2": ["dense_glm-5.2"]},
    "fc": {"gpt-4.1-mini": ["gpt-4.1-mini_fc", "fc_gpt-4.1-mini_test_public_fixed512k"],
           "gpt-5.4-mini": ["gpt-5.4-mini_fc"], "gpt-5.4-nano": ["gpt-5.4-nano_fc"],
           "gpt-5.5": ["g55fc", "fc_gpt-5.5_test_public_fixed512k"],
           "minimax-m3": ["fc_minimax-m3-cloud_tp", "fc_minimax-m3-cloud_tp_fixed"],
           "glm-5.2": ["fc_glm-5.2-cloud_tp", "fc_glm-5.2-cloud_tp_fixed"]},
    "openregion": {m: [f"openregion_agg81_{m}"] for m in ("gpt-4.1-mini", "gpt-5.5", "minimax-m3", "glm-5.2")},
    "oracle": {m: [f"oracle_agg81_{m}"] for m in ("gpt-4.1-mini", "gpt-5.5", "minimax-m3", "glm-5.2")},
    "k50": {m: [f"k50dense_{m}"] for m in ("gpt-4.1-mini", "gpt-5.5", "minimax-m3", "glm-5.2")},
    "k50page": {m: [f"k50page_{m}"] for m in ("gpt-4.1-mini", "gpt-5.5", "minimax-m3", "glm-5.2")},
    "agg512k": {"gpt-4.1-mini": ["fc_gpt-4.1-mini_test_public_512kexp"], "gpt-5.5": ["fc_gpt-5.5_test_public_512kexp"],
                "minimax-m3": ["fc_minimax-m3-cloud_tp_512kexp"], "glm-5.2": ["fc_glm-5.2-cloud_tp_512kexp"]},
}
CLAUDE = ("claude-sonnet-5-5", "claude-haiku-5-5")
for m in CLAUDE:
    for s in ("cb", "bm25", "dense", "fc", "agg512k"):
        FILES[s][m] = [f"{s}_{m}"]
    FILES["openregion"][m] = [f"openregion_agg81_{m}"]
    FILES["oracle"][m] = [f"oracle_agg81_{m}"]
    FILES["k50"][m] = [f"k50dense_{m}"]
    FILES["k50page"][m] = [f"k50page_{m}"]
VDIR = {"k50": "verdicts/claude-sonnet-5-5", "k50page": "verdicts/claude-sonnet-5-5"}
AGG_FC_PANEL = {"gpt-4.1-mini", "gpt-5.5", "minimax-m3", "glm-5.2"}
AGG = {q for q in TEST if G[q]["task_type"] == "cross_source_aggregation"}


RELEASED_NAME = {'agg512k_claude-haiku-5-5': 'full_context_aggregation_claude-haiku-5-5', 'agg512k_claude-sonnet-5-5': 'full_context_aggregation_claude-sonnet-5-5', 'cb_claude-haiku-5-5': 'closed_book_claude-haiku-5-5', 'cb_claude-sonnet-5-5': 'closed_book_claude-sonnet-5-5', 'cb_minimax-m3-cloud_tp': 'closed_book_minimax-m3', 'fc_claude-haiku-5-5': 'full_context_claude-haiku-5-5', 'fc_claude-sonnet-5-5': 'full_context_claude-sonnet-5-5', 'fc_glm-5.2-cloud_tp': 'full_context_glm-5.2', 'fc_glm-5.2-cloud_tp_512kexp': 'full_context_512k_rest_glm-5.2', 'fc_glm-5.2-cloud_tp_fixed': 'full_context_512k_sample_glm-5.2', 'fc_gpt-4.1-mini_test_public_512kexp': 'full_context_512k_rest_gpt-4.1-mini', 'fc_gpt-4.1-mini_test_public_fixed512k': 'full_context_512k_sample_gpt-4.1-mini', 'fc_gpt-5.5_test_public_512kexp': 'full_context_512k_rest_gpt-5.5', 'fc_gpt-5.5_test_public_fixed512k': 'full_context_512k_sample_gpt-5.5', 'fc_minimax-m3-cloud_tp': 'full_context_minimax-m3', 'fc_minimax-m3-cloud_tp_512kexp': 'full_context_512k_rest_minimax-m3', 'fc_minimax-m3-cloud_tp_fixed': 'full_context_512k_sample_minimax-m3', 'g55fc': 'full_context_gpt-5.5', 'gpt-4.1-mini_cb': 'closed_book_gpt-4.1-mini', 'gpt-4.1-mini_fc': 'full_context_gpt-4.1-mini', 'gpt-4.1-mini_rag': 'bm25_gpt-4.1-mini', 'gpt-5.4-mini_cb': 'closed_book_gpt-5.4-mini', 'gpt-5.4-mini_fc': 'full_context_gpt-5.4-mini', 'gpt-5.4-mini_rag': 'bm25_gpt-5.4-mini', 'gpt-5.4-nano_cb': 'closed_book_gpt-5.4-nano', 'gpt-5.4-nano_fc': 'full_context_gpt-5.4-nano', 'gpt-5.4-nano_rag': 'bm25_gpt-5.4-nano', 'gpt-5.5_cb': 'closed_book_gpt-5.5', 'gpt-5.5_rag': 'bm25_gpt-5.5', 'jf386_glm-5.2_dense': 'dense_no_reasoning_glm-5.2', 'jf386_gpt-4.1-mini_dense': 'dense_gpt-4.1-mini', 'jf386_gpt-5.4-mini_dense': 'dense_gpt-5.4-mini', 'jf386_gpt-5.4-nano_dense': 'dense_gpt-5.4-nano', 'jf386_gpt-5.5_dense': 'dense_gpt-5.5', 'jf386_minimax-m3_dense': 'dense_minimax-m3', 'k50dense_claude-haiku-5-5': 'k50_claude-haiku-5-5', 'k50dense_claude-sonnet-5-5': 'k50_claude-sonnet-5-5', 'k50dense_glm-5.2': 'k50_glm-5.2', 'k50dense_gpt-4.1-mini': 'k50_gpt-4.1-mini', 'k50dense_gpt-5.5': 'k50_gpt-5.5', 'k50dense_minimax-m3': 'k50_minimax-m3', 'k50dense_minimax-m3_reasoning-absent': 'k50_no_reasoning_minimax-m3', 'k50page_claude-haiku-5-5': 'k50_page_claude-haiku-5-5', 'k50page_claude-sonnet-5-5': 'k50_page_claude-sonnet-5-5', 'k50page_glm-5.2': 'k50_page_glm-5.2', 'k50page_gpt-4.1-mini': 'k50_page_gpt-4.1-mini', 'k50page_gpt-5.5': 'k50_page_gpt-5.5', 'k50page_minimax-m3': 'k50_page_minimax-m3', 'openregion_agg81_claude-haiku-5-5': 'open_region_claude-haiku-5-5', 'openregion_agg81_claude-sonnet-5-5': 'open_region_claude-sonnet-5-5', 'openregion_agg81_glm-5.2': 'open_region_glm-5.2', 'openregion_agg81_gpt-4.1-mini': 'open_region_gpt-4.1-mini', 'openregion_agg81_gpt-5.5': 'open_region_gpt-5.5', 'openregion_agg81_minimax-m3': 'open_region_minimax-m3', 'oracle_agg81_claude-haiku-5-5': 'oracle_claude-haiku-5-5', 'oracle_agg81_claude-sonnet-5-5': 'oracle_claude-sonnet-5-5', 'oracle_agg81_glm-5.2': 'oracle_glm-5.2', 'oracle_agg81_gpt-4.1-mini': 'oracle_gpt-4.1-mini', 'oracle_agg81_gpt-5.5': 'oracle_gpt-5.5', 'oracle_agg81_minimax-m3': 'oracle_minimax-m3', 'rag_minimax-m3-cloud_tp': 'bm25_minimax-m3'}
KIMI_FILE = {"cb": "closed_book", "fc": "full_context", "agg512k": "full_context_aggregation", "k50page": "k50_page", "openregion": "open_region"}


def released_verdict_ids(tag, setting):
    for d in ([VDIR[setting]] if setting in VDIR else
              ["verdicts/claude-sonnet-5-5", "verdicts/claude-sonnet-5-5"]):
        p = REL / "results" / d / f"{RELEASED_NAME.get(tag, tag)}.jsonl"
        if p.exists():
            return [r["qa_id"] for r in jl(p)]
    raise SystemExit(f"no verdict file for {tag}")


summary = collections.OrderedDict()
missing = collections.Counter()
items_dir = OUT / "items"
items_dir.mkdir(parents=True, exist_ok=True)
for setting, per in FILES.items():
    for m, layers in per.items():
        final = {}
        for tag in layers:
            for q in released_verdict_ids(tag, setting):
                final[q] = tag
        if setting == "cb":
            scope = [q for q in TEST if q in final]
        elif setting in ("bm25", "dense", "fc"):
            scope = [q for q in TEST if q in final and q not in BEX]
        else:
            scope = [q for q in TEST if q in final]
        if setting == "agg512k":
            fc_final = set()
            for tag in FILES["fc"][m]:
                fc_final |= set(released_verdict_ids(tag, "fc"))
            scope = [q for q in scope if q in AGG and q not in (fc_final - BEX)]
        if args.rescoring_scope == "new":
            if m in CLAUDE or setting == "agg512k":
                continue
            if setting == "fc" and m in AGG_FC_PANEL:
                scope = [q for q in scope if q not in AGG]
        rows = []
        for q in scope:
            jt = ALIAS.get(final[q], final[q])
            p = prompts.get(jt, {}).get(q)
            parts = parse(p) if p else None
            if not parts:
                missing[(setting, m)] += 1
                continue
            rows.append({"qa_id": q, "question": parts[0], "gold": parts[1], "prediction": parts[2],
                         "src_tag": RELEASED_NAME.get(final[q], final[q])})
        f = items_dir / f"{KIMI_FILE.get(setting, setting)}_{m}.jsonl"
        with open(f, "w", encoding="utf-8") as fo:
            for r in rows:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[f"items/{f.name}"] = len(rows)

# ---------------------------------------------------------------- closed-book without identifiers
ANN = re.compile(r"(?<!\d)\d{10}(?!\d)")
PG = re.compile(r"\(\s*p\.\s*\d+\s*\)")
cb = {r["qa_id"]: r["prompt"] for r in jl(RUNS / "cb_claude_prompts.jsonl") if r.get("regime") == "cb"}
fix = {r["qa_id"]: r["prompt"] for r in jl(E1 / "prompts_open_test.jsonl")}
mism = [q for q in fix if q in cb and cb[q] != fix[q]]
assert not mism, f"Claude CB prompts differ from the prompts without row identifiers for {mism[:3]}"
ident = [q for q in TEST if ANN.search(G[q]["question"]) or PG.search(G[q]["question"])]
QSEC = re.compile(r"(\[질문\]\n)(.*?)(\n\n\[근거 위치 정보\])", re.S)


def strip(prompt):
    def fq(m):
        qt = PG.sub(args.page_repl, ANN.sub(args.ann_repl, m.group(2)))
        return m.group(1) + qt + m.group(3)
    out, n = QSEC.subn(fq, prompt, count=1)
    assert n == 1
    if args.locator == "mask":
        head, loc = out.split("[근거 위치 정보]", 1)
        loc = re.sub(r"-p\d{3}\b", "-p???", ANN.sub(args.ann_repl or "##########", loc))
        out = head + "[근거 위치 정보]" + loc
    return out


CB_NOID = OUT / "cb_no_locator_ids"
CB_NOID.mkdir(parents=True, exist_ok=True)
stripped = {q: strip(cb[q]) for q in ident}
loc_check = sum(1 for q in ident if ANN.search(stripped[q].split("[근거 위치 정보]", 1)[1])
               or re.search(r"-p\d{3}\b", stripped[q].split("[근거 위치 정보]", 1)[1]))
assert all(not ANN.search(QSEC.search(stripped[q]).group(2)) and not PG.search(QSEC.search(stripped[q]).group(2))
           for q in ident)
SETTING = "cb_noid"
cid = lambda q, m: f"{q}__{m}__{SETTING}"
with open(CB_NOID / "index.jsonl", "w", encoding="utf-8") as fo:
    for q in ident:
        for m in ("gpt-4.1-mini", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.5", "claude-sonnet-5-5",
                  "claude-haiku-5-5", "minimax-m3", "glm-5.2"):
            fo.write(json.dumps({"custom_id": cid(q, m), "key": f"{q}|{m}|{SETTING}", "qa_id": q,
                                 "model": m, "setting": SETTING}, ensure_ascii=False) + "\n")


def openai_body(m, prompt):
    b = {"model": m, "messages": [{"role": "user", "content": prompt}]}
    if m == "gpt-4.1-mini":
        b.update({"max_tokens": args.cap_gpt41, "temperature": 0.0})
    else:
        b["max_completion_tokens"] = args.cap_gpt5
    return b


def claude_params(m, prompt):
    return {"model": m, "max_tokens": args.cap_claude,
            "thinking": {"type": "between_tools" if m == "claude-sonnet-5-5" else "disabled"},
            "messages": [{"role": "user", "content": prompt}]}


def write(path, rows):
    with open(path, "w", encoding="utf-8") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary[str(path.relative_to(OUT))] = len(rows)


for m in ("gpt-4.1-mini", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.5"):
    write(CB_NOID / f"openai_batch_{m}.jsonl",
          [{"custom_id": cid(q, m), "method": "POST", "url": "/v1/chat/completions",
            "body": openai_body(m, stripped[q])} for q in ident])
for m in ("claude-sonnet-5-5", "claude-haiku-5-5"):
    write(CB_NOID / f"anthropic_batch_{m}.jsonl", [{"custom_id": cid(q, m), "params": claude_params(m, stripped[q])} for q in ident])
for m in ("minimax-m3", "glm-5.2"):
    write(CB_NOID / f"ollama_prompts_{m}.jsonl",
          [{"qa_id": q, "split": "test_public", "regime": "cb", "prompt": stripped[q]} for q in ident])

# ---------------------------------------------------------------- minimax-m3 512k items
MMX512 = OUT / "minimax_512k"
MMX512.mkdir(parents=True, exist_ok=True)
four = {"krhlrb_v05_0435", "krhlrb_v05_0438", "krhlrb_v05_0442", "krhlrb_v05_1896"}
src4 = [r for r in jl(RUNS / "full_context/512kexp_tp.jsonl") if r["qa_id"] in four]
write(MMX512 / "prompts_4.jsonl", src4)

# ---------------------------------------------------------------- report
print("corrected-gold judge prompts loaded:", n_gf)
print("missing judge prompts:", dict(missing) or 0)
print("judge items:", sum(v for k, v in summary.items() if k.startswith("items/")))
print("closed-book items with an identifier in the question:", len(ident), "| locator still carries a number or page id:", loc_check,
      f"(--locator {args.locator})")
print("minimax-m3 512k items:", len(src4))
for k, v in summary.items():
    print(f"  {k}: {v}")
