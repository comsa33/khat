#!/usr/bin/env python3
"""OpenAI batch request file of the gpt-4.1-mini judge for judge prompts (all core runs, or the corrected-gold items only).

Usage: python3 scripts/judges/build_judge_gpt.py PROMPTS INDEX OUT [--gold-corrected]
"""
from __future__ import annotations

import argparse
import json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts")
    ap.add_argument("index")
    ap.add_argument("out")
    ap.add_argument("--gold-corrected", action="store_true", help="keep only index rows with gold_fixed=true")
    ap.add_argument("--model", default="gpt-4.1-mini")
    a = ap.parse_args()
    idx = {}
    for line in open(a.index, encoding="utf-8"):
        d = json.loads(line)
        idx[d["custom_id"]] = d
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for line in open(a.prompts, encoding="utf-8"):
            p = json.loads(line)
            i = idx[p["custom_id"]]
            if i.get("set") != "core_or_agg" or (a.gold_corrected and not i.get("gold_fixed")):
                continue
            f.write(json.dumps({"custom_id": p["custom_id"], "method": "POST", "url": "/v1/chat/completions",
                                "body": {"model": a.model, "messages": [{"role": "user", "content": p["prompt"]}],
                                         "max_tokens": 4, "temperature": 0.0}}, ensure_ascii=False) + "\n")
            n += 1
    print("requests", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
