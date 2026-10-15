#!/usr/bin/env python3
"""Dense top-k prompts (bge-m3 through Ollama /api/embed, k=5, 1200-character passages) for the items of an item list.

Usage: python3 scripts/run/build_dense.py --sample data/item_sets/sample_retrieval_test.jsonl --out local/runs/rag_dense/tp386.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval import split_chunks, retrieve, rag_prompt, OLLAMA_EMBED_MODEL, DENSE_BGE_LABEL

from khat.paths import ROOT, DATA, LOCAL
SAMPLE = DATA / "item_sets" / "sample_retrieval_test.jsonl"
SPLIT_FILES = {
    "dev": ROOT / "data" / "qa_dev.jsonl",
    "test_public": ROOT / "data" / "qa_test_public.jsonl",
    "legal": ROOT / "data" / "auxiliary" / "legal_reasoning.jsonl",
}
BUNDLES = LOCAL / "bundles"
DEFAULT_OUT = LOCAL / "runs" / "rag_dense" / "tp386.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def pages_of(passages: list) -> list[str]:
    """Unique page_ids among the retrieved passages, in reading order."""
    seen, out = set(), []
    for pid, _ in passages:
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5, help="top-k passages")
    ap.add_argument("--chunk-chars", type=int, default=1200, help="passage size for sub-page chunking")
    ap.add_argument("--per-page-max", type=int, default=0, help="cap chunks per page (page-diverse); 0=off")
    ap.add_argument("--out", default=None, help="output JSONL (must be under local/)")
    ap.add_argument("--sample", default=None, help="item list JSONL (default: data/item_sets/sample_retrieval_test.jsonl)")
    args = ap.parse_args()

    out = Path(args.out).resolve() if args.out else DEFAULT_OUT
    if not out.is_relative_to(LOCAL.resolve()):
        raise SystemExit(f"--out must be under local/ (embeds bundle text). Got: {out}")
    sample_path = (Path(args.sample) if args.sample else SAMPLE)
    if not sample_path.is_absolute():
        sample_path = ROOT / sample_path
    if not sample_path.exists():
        raise SystemExit(f"missing sample {sample_path}")

    sample = load_jsonl(sample_path)
    qa_by_id: dict[str, dict] = {}
    for f in SPLIT_FILES.values():
        for r in load_jsonl(f):
            qa_by_id[r["qa_id"]] = r

    chunks_cache: dict = {}
    bm25_cache: dict = {}
    dense_cache: dict = {}
    written, no_bundle, no_passage, recalls = [], 0, 0, []
    for s in sample:
        qid, bid = s["qa_id"], s.get("bundle_id")
        rec = qa_by_id.get(qid)
        if not rec:
            continue
        bf = BUNDLES / f"{bid}.txt" if bid else None
        if not bid or not bf.exists():
            no_bundle += 1
            continue
        if bid not in chunks_cache:
            chunks_cache[bid] = split_chunks(bf.read_text(encoding="utf-8"), args.chunk_chars)
        chunks = chunks_cache[bid]
        if not chunks:
            no_passage += 1
            continue
        picked = retrieve("dense_bge", rec, chunks, args.k, bid, bm25_cache, dense_cache, args.per_page_max)
        if not picked:
            no_passage += 1
            continue
        retrieved_pages = pages_of(picked)
        gold_pages = list(rec.get("page_ids") or [])
        if gold_pages:
            recall = len(set(retrieved_pages) & set(gold_pages)) / len(set(gold_pages))
            recalls.append(recall)
        written.append({
            "qa_id": qid,
            "split": s["split"],
            "prompt": rag_prompt(rec, picked),
            "retriever": "dense_bge",
            "embedder": OLLAMA_EMBED_MODEL,
            "embed_backend": DENSE_BGE_LABEL,
            "k": args.k,
            "per_page_max": args.per_page_max,
            "n_passages": len(picked),
            "retrieved_page_ids": retrieved_pages,
            "gold_page_ids": gold_pages,
            "context_tier": s.get("context_tier"),
            "bundle_id": bid,
            "answer_type": s.get("answer_type"),
            "regime": "rag_dense",
        })

    if not written:
        out.unlink(missing_ok=True)
        raise SystemExit(
            f"no dense-RAG prompts written — internal bundle text under {BUNDLES} is absent. "
            "Rebuild bundles locally; RAG needs the internal corpus."
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in written:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_split = collections.Counter(r["split"] for r in written)
    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    print(f"[ok] wrote {out} ({len(written)} RAG-dense[{DENSE_BGE_LABEL}] prompts; no_bundle skipped {no_bundle}, "
          f"no_passage {no_passage})")
    print(f"     by split: {dict(by_split)}  | k={args.k} per_page_max={args.per_page_max}")
    print(f"     gold-page recall@{args.k} (sanity, items with gold pages={len(recalls)}): {mean_recall:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
