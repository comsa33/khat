"""Shared paths and helpers of the judge-input builders: judge prompt, gold string, Claude judge parameters.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from khat.paths import ROOT, LOCAL
BASELINES = LOCAL / "runs" / "baselines"
INITIAL_VERDICTS = LOCAL / "verdicts" / "gpt-4.1-mini_original_gold"
GOLD_FIXES = ROOT / "data" / "errata" / "test_review_gold_fixes.json"
LEGAL = "cross_document_legal_reasoning"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_judge import judge_prompt

CLAUDE_MODEL = "claude-sonnet-5-5"
CLAUDE_OUTPUT = {"format": {"type": "json_schema", "schema": {
    "type": "object", "properties": {"verdict": {"type": "string", "enum": ["YES", "NO"]}},
    "required": ["verdict"], "additionalProperties": False}}}
CLAUDE_PARAMS = {"model": CLAUDE_MODEL, "max_tokens": 32, "thinking": {"type": "between_tools"},
                 "output_config": CLAUDE_OUTPUT}


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def write_jsonl(p: Path, rows: list[dict]) -> None:
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def gold_fixes() -> dict:
    return json.load(open(GOLD_FIXES, encoding="utf-8"))


def multi_answer(answers: list[str]) -> str:
    """Gold text for an accept-any item, as shown to both judges."""
    return "다음 중 어느 하나와 일치하면 정답: " + " / ".join(f"({i}) {a}" for i, a in enumerate(answers, 1))


def judge_gold(item: dict, fix: dict, accept_any_format: str = "multi") -> str:
    """Gold string of the judge prompt; accept-any items as "A 또는 B" ("or") or as numbered alternatives ("multi")."""
    q = item["qa_id"]
    if q in fix["replace"]:
        return fix["replace"][q]
    if q in fix["accept_any"]:
        a = fix["accept_any"][q]
        return multi_answer(a) if accept_any_format == "multi" else " 또는 ".join(a)
    return str(item.get("answer", ""))
