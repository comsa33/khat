#!/usr/bin/env python3
"""Full-context prompts: the whole bundle of each item of an item list, then the question.

Usage: python3 scripts/run/build_full_context.py --sample data/item_sets/sample_full_context_512k_test.jsonl --out local/runs/full_context/512k_fixed_tp.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from khat.paths import ROOT, DATA, LOCAL
FC_SAMPLE = DATA / "item_sets" / "sample_full_context_test.jsonl"
SPLIT_FILES = {
    "dev": ROOT / "data" / "qa_dev.jsonl",
    "test_public": ROOT / "data" / "qa_test_public.jsonl",
    "legal": ROOT / "data" / "auxiliary" / "legal_reasoning.jsonl",
}
BUNDLES = LOCAL / "bundles"
DEFAULT_OUT = LOCAL / "runs" / "full_context" / "groupA_tp.jsonl"
DEFAULT_INSTR = "주어진 근거 자료로 질문에 답하라."


def full_context_prompt(question: str, bundle_text: str, instr: str = DEFAULT_INSTR) -> str:
    """Identical shape to build_full_context_smoke.full_context_prompt()."""
    return (
        f"{instr}\n\n"
        f"[제공 문서] (아래 공고/번들 본문에서 근거를 찾으세요.)\n{bundle_text}\n\n"
        f"[질문]\n{question}\n\n"
        f"[지시] 위 문서 내용에 근거하여 답하세요. 문서에서 찾을 수 없으면 "
        f"'제공된 자료만으로는 확정할 수 없음'이라고 답하세요. 군더더기 없이 최종 답만 간단히 출력하세요."
    )


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="output JSONL (must be under local/)")
    ap.add_argument("--sample", default=None, help="item list JSONL (default: data/item_sets/sample_full_context_test.jsonl)")
    args = ap.parse_args()

    out = Path(args.out).resolve() if args.out else DEFAULT_OUT
    wl = LOCAL.resolve()
    if not out.is_relative_to(wl):
        raise SystemExit(f"--out must be under local/ (embeds bundle text). Got: {out}")

    sample = Path(args.sample).resolve() if args.sample else FC_SAMPLE
    if not sample.exists():
        raise SystemExit(f"missing {sample}")
    fc = load_jsonl(sample)

    q_by_id: dict[str, str] = {}
    for sp, f in SPLIT_FILES.items():
        for r in load_jsonl(f):
            q_by_id[r["qa_id"]] = r.get("question", "")

    written, skipped, total_chars = [], 0, 0
    for r in fc:
        qid, bid = r["qa_id"], r.get("bundle_id")
        bf = BUNDLES / f"{bid}.txt"
        if not bid or not bf.exists():
            print(f"  WARN: missing bundle for {qid} (bundle={bid}); skipping")
            skipped += 1
            continue
        question = q_by_id.get(qid, "")
        if not question:
            print(f"  WARN: no question text for {qid}; skipping")
            skipped += 1
            continue
        bundle_text = bf.read_text(encoding="utf-8")
        total_chars += len(bundle_text)
        written.append({
            "qa_id": qid,
            "split": r["split"],
            "prompt": full_context_prompt(question, bundle_text),
            "context_tier": r.get("context_tier"),
            "bundle_id": bid,
            "answer_type": r.get("answer_type"),
            "regime": "full_context",
        })

    if not written:
        raise SystemExit(
            f"no full-context prompts written — internal bundle text under "
            f"{BUNDLES} is absent. Rebuild bundles locally; cannot run without the corpus."
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for rec in written:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    by_split = collections.Counter(r["split"] for r in written)
    by_tier = collections.Counter(r["context_tier"] for r in written)
    est_tokens = int(total_chars / 2.45)
    print(f"[ok] wrote {out} ({len(written)} full-context prompts, skipped {skipped})")
    print(f"     by split: {dict(by_split)}")
    print(f"     by tier:  {dict(by_tier)}")
    print(f"     bundle chars total: {total_chars:,}  (~{est_tokens:,} input tokens @2.45 c/t)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
