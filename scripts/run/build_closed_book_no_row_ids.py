#!/usr/bin/env python3
"""Closed-book prompts without the locator fields that reveal answers, their OpenAI batch request files and the Ollama run commands.

Usage: python3 scripts/run/build_closed_book_no_row_ids.py   (writes local/runs/cb_fix/)
"""
import json, statistics as st
from rerun_common import *
from run_baseline import _openai_chat_kwargs

E = OUT / "cb_fix"
E.mkdir(parents=True, exist_ok=True)
base = {r["qa_id"]: r for r in load(B / "closed_book/prompts_main.jsonl")}
n285 = {r["qa_id"]: r for r in load(B / "closed_book/prompts_new285.jsonl")}
opn = {r["qa_id"]: r for r in load(RO / "closed_book/prompts_open_tp.jsonl")}
retry = {r["qa_id"] for r in load(RO / "closed_book/prompts_open_retry_tp.jsonl")}

items, gpt_test, gpt_dev, open_test = [], [], [], []
for src, dst, split in ((n285, gpt_test, "test_public"), (base, gpt_dev, "dev")):
    for q, rec in src.items():
        if rec["split"] != split or not inscope(q):
            continue
        fx, fields = fix_record(rec)
        if fx:
            dst.append(fx)
            items.append({"qa_id": q, "split": split, "task_type": GOLD[q]["task_type"],
                          "op": (GOLD[q].get("gold_predicate") or {}).get("op"),
                          "leak": "count_rowids" if "row_ids" in fields else "date_in_ids",
                          "removed_fields": fields, "gpt_source_prompt_file":
                          "closedbook_v09_prompts_new285.jsonl" if split == "test_public" else "closedbook_v09_prompts.jsonl"})
for q, rec in opn.items():
    if inscope(q):
        fx, _ = fix_record(rec)
        if fx:
            assert fx["prompt"] == next(r["prompt"] for r in gpt_test if r["qa_id"] == q)
            open_test.append(fx)
write(E / "items.jsonl", items); write(E / "prompts_gpt_test.jsonl", gpt_test)
write(E / "prompts_gpt_dev.jsonl", gpt_dev); write(E / "prompts_open_test.jsonl", open_test)

# ---- original generation settings
GPT_SET = {
    "gpt-4.1-mini": {"test_public": 4000, "dev": 256},
    "gpt-5.4-mini": {"test_public": 4000, "dev": 4000},
    "gpt-5.4-nano": {"test_public": 4000, "dev": 4000},
    "gpt-5.5": {"test_public": 4000, "dev": 4000},
}
for M, caps in GPT_SET.items():
    lines = []
    for rec in gpt_test + gpt_dev:
        body = _openai_chat_kwargs(M, rec["prompt"], 0.0, caps[rec["split"]], None)
        lines.append({"custom_id": f"cbfix__{rec['split']}__{rec['qa_id']}", "method": "POST",
                      "url": "/v1/chat/completions", "body": body})
    write(E / f"batch_input_{M}.jsonl", lines)

pf = "local/runs/cb_fix/prompts_open_test.jsonl"
ollama_plan = {
    "glm-5.2:cloud": {"temperature": 0.0, "max_output_tokens": 2048, "num_ctx": 8192, "think": "off",
                      "cmd": f"python3 scripts/run_baseline.py --provider ollama --model glm-5.2:cloud --split test_public --prompt-file {pf} --out local/runs/predictions/cbfix_glm-5.2-cloud_tp.jsonl --temperature 0 --max-output-tokens 2048 --num-ctx 8192 --think off --resume"},
    "minimax-m3:cloud": {"temperature": 0.0, "max_output_tokens": 2048, "num_ctx": 8192, "think": "on",
                         "retry_empty_with_max_output_tokens": 8192,
                         "items_originally_run_at_8192": sorted(q for q in retry if any(r['qa_id'] == q for r in open_test)),
                         "cmd": f"python3 scripts/run_baseline.py --provider ollama --model minimax-m3:cloud --split test_public --prompt-file {pf} --out local/runs/predictions/cbfix_minimax-m3-cloud_tp.jsonl --temperature 0 --max-output-tokens 2048 --num-ctx 8192 --think on --resume",
                         "note": "original protocol: first pass 2048; items with empty output re-run at 8192 (cb_retry_tp). 1879 was in the original retry set, so run it at 8192 directly. Think on: the original closed-book runs returned thinking although think=off was set."},
}
json.dump(ollama_plan, open(E / "ollama_plan.json", "w"), ensure_ascii=False, indent=1)

print("items", len(items), "gpt_test", len(gpt_test), "gpt_dev", len(gpt_dev), "open_test", len(open_test))
import collections
print(collections.Counter((i["split"], i["leak"], i["task_type"]) for i in items))
