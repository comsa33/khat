#!/usr/bin/env python3
"""Agreement and Cohen's kappa of each judge against the author labels on the 80-prediction validation sample.

Usage: python3 scripts/judges/judge_panel_kappa.py
"""
from __future__ import annotations

import json
from collections import Counter

from khat.paths import RESULTS

JV = RESULTS / "human_labels" / "judge_validation_80.jsonl"

JUDGES = [
    ("claude-sonnet-5-5", "Anthropic"),
    ("gpt-4.1-mini", "OpenAI"),
    ("gemma4:31b", "Google"),
    ("deepseek-v4.1-flash", "DeepSeek"),
    ("kimi-k2.6", "Moonshot"),
    ("mistral-large-3", "Mistral"),
]


def cohen_kappa(a: list[int], b: list[int]) -> tuple[float, float]:
    """Two binary raters -> (observed agreement, Cohen's kappa)."""
    n = len(a)
    if n == 0:
        return (0.0, 0.0)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    kappa = (po - pe) / (1 - pe) if pe != 1 else 1.0
    return (po, kappa)


def main() -> int:
    rows = [json.loads(l) for l in JV.open(encoding="utf-8")]
    human = {r["qa_id"]: 1 if r["author"] == "Y" else 0 for r in rows}

    print(f"{'judge':22} {'lab':>9} {'kappa':>6} {'agree':>7}")
    for disp, lab in JUDGES:
        hv = {r["qa_id"]: int(r["judges"][disp]) for r in rows}
        ids = [q for q in human if q in hv]
        po, k = cohen_kappa([human[q] for q in ids], [hv[q] for q in ids])
        flag = "" if len(ids) == 80 else f"  [human n={len(ids)}/80]"
        print(f"{disp:22} {lab:>9} {k:6.3f} {po*100:6.1f}%   {sum(human[q] == hv[q] for q in ids)}/{len(ids)}" + flag)

    print()
    for key in ("model", "setting", "split"):
        print(f"by {key}: " + ", ".join(f"{v} {n}" for v, n in Counter(r[key] for r in rows).most_common()))
    for a in ("A2", "A3"):
        print(f"{a}: unsure {sum(r[a] == 'unsure' for r in rows)}")
    both = [r for r in rows if "unsure" not in (r["A2"], r["A3"])]
    agree = [r for r in both if r["A2"] == r["A3"]]
    claude = [r for r in agree if (r["A2"] == "correct") == r["judges"]["claude-sonnet-5-5"]]
    print(f"labeled by both: {len(both)}; A2 = A3 on {len(agree)}; also equal to claude-sonnet-5-5 on {len(claude)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
