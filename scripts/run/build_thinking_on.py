#!/usr/bin/env python3
"""Reasoning-on re-run inputs for glm-5.2 and minimax-m3: closed-book, BM25 and dense test prompts and the Ollama run commands.

Usage: python3 scripts/run/build_thinking_on.py   (writes local/runs/thinking_on/)
"""
import json, statistics as st
from rerun_common import *

E = OUT / "thinking_on"
E.mkdir(parents=True, exist_ok=True)
fixed = {r["qa_id"]: r for r in load(OUT / "cb_fix/prompts_open_test.jsonl")}
SRC = {"cb": RO / "closed_book/prompts_open_tp.jsonl",
       "bm25": B / "rag_bm25/tp386.jsonl",
       "dense": B / "rag_dense/tp386.jsonl"}
prompts = {}
for reg, p in SRC.items():
    rows = []
    for r in load(p):
        if r["split"] != "test_public" or not inscope(r["qa_id"]):
            continue
        if reg == "cb" and r["qa_id"] in fixed:
            r = fixed[r["qa_id"]]
        rows.append(r)
    prompts[reg] = rows
    write(E / f"prompts_{reg}.jsonl", rows)
    (E / f"items_{reg}.txt").write_text("\n".join(r["qa_id"] for r in rows) + "\n")

SET = {"glm-5.2:cloud": {"think": "on", "temperature": 0.0, "max_output_tokens": 8192, "num_ctx": 32768},
       "minimax-m3:cloud": {"think": "on", "temperature": 0.0, "max_output_tokens": 16384, "num_ctx": 32768}}
plan = {}
for M, s in SET.items():
    tag = M.replace(":", "-")
    plan[M] = {"settings": s, "cmds": {reg: (
        f"python3 scripts/run_baseline.py --provider ollama --model {M} --split test_public "
        f"--prompt-file local/runs/thinking_on/prompts_{reg}.jsonl --out local/runs/predictions/thinkon_{reg}_{tag}_tp.jsonl "
        f"--temperature 0 --max-output-tokens {s['max_output_tokens']} --num-ctx {s['num_ctx']} --think on --resume")
        for reg in SRC}}
json.dump(plan, open(E / "ollama_plan.json", "w"), ensure_ascii=False, indent=1)

print({k: len(v) for k, v in prompts.items()})
