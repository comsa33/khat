"""Retrieval helpers with an embedding cache for the open-region and oracle analyses.
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
import sys
import time
from pathlib import Path

from khat.paths import ROOT, LOCAL
DS = ROOT
REL = ROOT
OUT = LOCAL
BUNDLES = LOCAL / "bundles"
BASEL = LOCAL / "runs"
LEGAL = "cross_document_legal_reasoning"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import retrieval as R

# ---- embedding cache around R.ollama_embed ---------------------------------------------------------
CACHE_F = OUT / "cache" / "bge_m3_emb.pkl"
_orig_embed = R.ollama_embed
_cache: dict | None = None
_dirty = 0


def _load_cache() -> dict:
    global _cache
    if _cache is None:
        _cache = pickle.loads(CACHE_F.read_bytes()) if CACHE_F.exists() else {}
    return _cache


def save_cache() -> None:
    global _dirty
    if _cache is not None and _dirty:
        CACHE_F.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_F.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(_cache))
        tmp.replace(CACHE_F)
        _dirty = 0


def _key(t: str) -> str:
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


def cached_embed(texts: list, batch_size: int = 96) -> list:
    global _dirty
    c = _load_cache()
    keys = [_key(t) for t in texts]
    miss = [i for i, k in enumerate(keys) if k not in c]
    if miss:
        t0 = time.time()
        if len(miss) > 1:
            print(f"    [embed] {len(miss)} new texts (of {len(texts)}) ...", flush=True)
        vecs = _orig_embed([texts[i] for i in miss], batch_size)
        for i, v in zip(miss, vecs):
            c[keys[i]] = v
        _dirty += len(miss)
        if len(miss) > 1:
            print(f"    [embed] done in {time.time() - t0:.1f}s", flush=True)
            save_cache()
        elif _dirty >= 200:
            save_cache()
    return [c[k] for k in keys]


R.ollama_embed = cached_embed


# ---- data ------------------------------------------------------------------------------------------
def load_jsonl(p: Path) -> list:
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def test_qa() -> dict:
    return {r["qa_id"]: r for r in load_jsonl(REL / "data" / "qa_test_public.jsonl")}


def exclusions() -> set:
    txt = (REL / "data" / "exclusions" / "eval.txt").read_text(encoding="utf-8")
    return {l.strip() for l in txt.splitlines() if l.strip() and not l.startswith("#")}


def eval_scope(qa: dict, drop_excluded: bool = True) -> list:
    """test_public items with a bundle on disk, legal family and evaluation exclusions removed."""
    ex = exclusions() if drop_excluded else set()
    out = []
    for q, r in sorted(qa.items()):
        if r.get("task_type") == LEGAL or q in ex:
            continue
        bid = r.get("bundle_id")
        if not bid or not (BUNDLES / f"{bid}.txt").exists():
            continue
        out.append(r)
    return out


_chunks: dict = {}


def chunks_for(bid: str, size: int) -> list:
    key = (bid, size)
    if key not in _chunks:
        _chunks[key] = R.split_chunks((BUNDLES / f"{bid}.txt").read_text(encoding="utf-8"), size)
    return _chunks[key]


class Ranker:
    """Full rankings per (retriever, chunk size), using build_rag_smoke.rank_chunks / rrf / diversify."""

    def __init__(self):
        self.bm25: dict = {}
        self.dense: dict = {}

    def full(self, retriever: str, rec: dict, bid: str, size: int, query: str | None = None) -> list:
        chunks = chunks_for(bid, size)
        r = dict(rec)
        if query is not None:
            r["question"] = query
        bc = self.bm25.setdefault(size, {})
        dc = self.dense.setdefault(size, {})
        if retriever in ("bm25", "dense_bge"):
            full = R.rank_chunks(retriever, r, chunks, bid, bc, dc)
        elif retriever == "rrf":
            full = R.rrf([R.rank_chunks("bm25", r, chunks, bid, bc, dc),
                          R.rank_chunks("dense_bge", r, chunks, bid, bc, dc)])
        else:
            raise ValueError(retriever)
        return R.diversify(full, [pid for pid, _ in chunks], 0)


def wilson(x: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def fmt_ci(x: int, n: int) -> str:
    lo, hi = wilson(x, n)
    return f"{100 * x / n:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}] ({x}/{n})"
