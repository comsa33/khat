#!/usr/bin/env python3
"""Closed-book prompt records: question, task instruction and locator (bundle, pages, rows, tables), no document text.

Usage: python3 scripts/run/build_closed_book.py [--out local/runs/closed_book/locator_prompts.jsonl] [--inline-context]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from khat.paths import ROOT, LOCAL
BUNDLES = LOCAL / "bundles"

INSTRUCTION = {
    "table_numeric_reasoning": "주어진 자료(표/공공데이터)에서 요청한 수치를 계산해 간단히 답하라.",
    "cross_source_aggregation": "공고의 공급위치(지역)를 파악한 뒤 해당 지역의 공공데이터를 집계해 답하라.",
    "cross_document_legal_reasoning": "공고 내용과 관련 법령 조문을 함께 근거로 답하라.",
    "long_context_retrieval": "주어진 공고 본문에서 해당 값을 찾아 간단히 답하라.",
    "long_distance_retrieval": "긴 컨텍스트의 뒷부분에 위치한 해당 값을 찾아 답하라.",
    "eligibility_reasoning": "공고의 신청자격/소득·자산 기준을 근거로 답하라.",
    "schedule_reasoning": "공고의 청약/계약 일정에서 해당 일자를 답하라.",
    "multi_document_comparison": "두 공고를 비교해 각각의 값을 답하라.",
    "provider_comparison": "공급기관이 다른 두 공고를 비교해 답하라.",
    "region_comparison": "두 공고의 공급위치(지역)를 비교해 답하라.",
    "answerability_detection": "제공된 자료만으로 판단 가능한지 답하고, 불가하면 그 이유를 밝혀라.",
    "format_robustness": "제공된 표(형식 무관)에서 요청한 값을 답하라.",
    "correction_notice_reasoning": "공고가 최초/정정 공고인지 표기 근거와 함께 답하라.",
}


def context_spec(it: dict) -> dict:
    spec = {k: it.get(k) for k in ("bundle_id", "context_tier", "evidence_position",
                                   "page_ids", "source_ids", "row_ids", "table_ids", "cell_ids")
            if it.get(k)}
    pred = it.get("gold_predicate")
    if pred:
        spec["predicate_source"] = pred.get("source")
    if it.get("bundle_id"):
        spec["bundle_file"] = f"workspace_local/processed/bundles-v06/{it['bundle_id']}.txt"
        spec["retrieval_mode"] = "long_context_bundle"
    elif pred:
        spec["retrieval_mode"] = "table_tool_or_locator"
    else:
        spec["retrieval_mode"] = "locator"
    return spec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="data/construction/qa_realistic_candidates.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--inline-context", action="store_true",
                    help="embed bundle text into the prompt (output under local/)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (ROOT / args.qa).open(encoding="utf-8") if l.strip()]
    if args.out:
        out = ROOT / args.out
    elif args.inline_context:
        out = LOCAL / "runs" / "closed_book" / "inlined_prompts.jsonl"
    else:
        out = LOCAL / "runs" / "closed_book" / "locator_prompts.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    bundle_cache: dict = {}
    n_inlined = 0
    with out.open("w", encoding="utf-8") as f:
        for it in rows:
            spec = context_spec(it)
            instr = INSTRUCTION.get(it["task_type"], "주어진 근거 자료로 질문에 답하라.")
            rec = {"qa_id": it["qa_id"], "task_type": it["task_type"], "split": it.get("split"),
                   "question_style": it.get("question_style"), "answer_type": it.get("answer_type"),
                   "instruction": instr, "question": it["question"], "context_spec": spec}
            if args.inline_context and it.get("bundle_id"):
                bf = BUNDLES / f"{it['bundle_id']}.txt"
                if it["bundle_id"] not in bundle_cache:
                    bundle_cache[it["bundle_id"]] = bf.read_text(encoding="utf-8") if bf.exists() else ""
                ctx = bundle_cache[it["bundle_id"]]
                if ctx:
                    rec["prompt"] = f"{instr}\n\n[제공 컨텍스트]\n{ctx}\n\n[질문] {it['question']}"
                    n_inlined += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    note = f" ({n_inlined} prompts with bundle text)" if args.inline_context else " (locator only)"
    print(f"{len(rows)} prompt records -> {out}{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
