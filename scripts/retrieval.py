#!/usr/bin/env python3
"""Passage retrieval over bundles (BM25, dense bge-m3, oracle) and RAG prompt building.

Usage: python3 scripts/retrieval.py --help
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path

from khat.paths import ROOT, LOCAL
QA = ROOT / "data" / "construction" / "qa_realistic_candidates.jsonl"
BUNDLES = LOCAL / "bundles"
PAGE_RE = re.compile(r"\[[^\]]*?([A-Za-z0-9][A-Za-z0-9-]*-p\d{3})[^\]]*?\]")
TOK_RE = re.compile(r"[0-9A-Za-z가-힣]+")


def tokenize(s: str) -> list:
    words = TOK_RE.findall(s.lower())
    grams = []
    for w in words:
        if len(w) >= 2 and any("가" <= c <= "힣" for c in w):
            grams += [w[i:i + 2] for i in range(len(w) - 1)]
    return words + grams


class BM25:
    def __init__(self, docs: list, k1: float = 1.5, b: float = 0.75):
        self.toks = [tokenize(d) for d in docs]
        self.N = len(self.toks)
        self.avgdl = (sum(len(d) for d in self.toks) / self.N) if self.N else 0.0
        self.df: dict = {}
        self.tf: list = []
        for d in self.toks:
            counts = collections.Counter(d)
            self.tf.append(counts)
            for t in counts:
                self.df[t] = self.df.get(t, 0) + 1
        self.k1, self.b = k1, b

    def _idf(self, t: str) -> float:
        n = self.df.get(t, 0)
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1)

    def top_k(self, query: str, k: int) -> list:
        q = set(tokenize(query))
        scores = []
        for i in range(self.N):
            tf, dl, s = self.tf[i], len(self.toks[i]), 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s += self._idf(t) * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
            scores.append((s, i))
        scores.sort(key=lambda x: (-x[0], x[1]))
        return [i for _, i in scores[:k]]


EMBED_MODEL = "text-embedding-3-small"
_OPENAI_CLIENT = None


def get_openai_client():
    """Return a cached OpenAI client (constructed once, reused across embed calls). Fails with a clear
    message if the optional baseline dependencies are missing (instead of a raw ImportError)."""
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        try:
            import openai
        except ImportError as exc:
            raise SystemExit(
                "dense/hybrid retrieval requires the optional baseline dependencies. "
                "Run: uv sync --extra baseline  (and set OPENAI_API_KEY)."
            ) from exc
        _OPENAI_CLIENT = openai.OpenAI()
    return _OPENAI_CLIENT


def embed_texts(texts: list) -> list:
    """OpenAI embeddings (needs OPENAI_API_KEY + baseline deps). Batched; reuses the cached client."""
    client = get_openai_client()
    out: list = []
    for i in range(0, len(texts), 256):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + 256])
        out.extend(d.embedding for d in resp.data)
    return out


def _embedding_similarity(a: list, b: list) -> float:
    """Dot-product similarity. text-embedding-3-small returns unit-normalized vectors, so for this
    embedding model the dot product is equivalent to cosine similarity (and is cheaper / deterministic)."""
    return sum(x * y for x, y in zip(a, b))


class DenseIndex:
    """Dense retrieval over chunk embeddings (dot-product similarity; see _embedding_similarity)."""

    def __init__(self, docs: list):
        self.vecs = embed_texts(docs)

    def top_k(self, query: str, k: int) -> list:
        qv = embed_texts([query])[0]
        sims = [(_embedding_similarity(qv, v), i) for i, v in enumerate(self.vecs)]
        sims.sort(key=lambda x: (-x[0], x[1]))
        return [i for _, i in sims[:k]]


BGE_MODEL_NAME = "BAAI/bge-m3"
_BGE_MODEL = None


def get_bge_model():
    """Return a cached bge-m3 SentenceTransformer (loaded once). Device: $BGE_DEVICE, else auto
    (mps/cuda/cpu). Fails with a clear message if sentence-transformers is missing."""
    global _BGE_MODEL
    if _BGE_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SystemExit(
                "dense_bge retrieval requires sentence-transformers + torch. Run with:\n"
                "  uv run --with sentence-transformers --with torch python scripts/build_baseline_rag_dense.py ..."
            ) from exc
        import os
        device = os.environ.get("BGE_DEVICE") or None
        _BGE_MODEL = SentenceTransformer(BGE_MODEL_NAME, device=device)
    return _BGE_MODEL


def bge_embed(texts: list):
    """Encode texts with bge-m3, unit-normalized (cosine == dot). Returns a numpy (n, d) array."""
    model = get_bge_model()
    return model.encode(
        texts, batch_size=32, normalize_embeddings=True,
        show_progress_bar=False, convert_to_numpy=True,
    )


class BGEDenseIndex:
    """Dense retrieval over bge-m3 chunk embeddings (cosine similarity via normalized dot product)."""

    def __init__(self, docs: list):
        self.vecs = bge_embed(docs)

    def top_k(self, query: str, k: int) -> list:
        qv = bge_embed([query])[0]
        sims = self.vecs @ qv
        order = sorted(range(len(sims)), key=lambda i: (-float(sims[i]), i))
        return order[:k]


OLLAMA_EMBED_MODEL = "bge-m3"
DENSE_BGE_LABEL = "bge-m3 (ollama /api/embed)"


def ollama_embed(texts: list, batch_size: int = 96) -> list:
    """Embed texts with bge-m3 via the local Ollama daemon's /api/embed (batched). Returns a list of
    float vectors. Raises RuntimeError on a short/failed response so a silent under-count can't corrupt
    the index."""
    import os
    import urllib.request
    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.environ.get("OLLAMA_EMBED_MODEL", OLLAMA_EMBED_MODEL)
    vecs: list = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        body = json.dumps({"model": model, "input": batch}).encode("utf-8")
        req = urllib.request.Request(
            f"{base}/api/embed", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=900) as r:
            data = json.loads(r.read().decode("utf-8"))
        emb = data.get("embeddings")
        if not emb or len(emb) != len(batch):
            raise RuntimeError(
                f"ollama /api/embed returned {len(emb) if emb else 0} embeddings for {len(batch)} inputs "
                f"(model={model}, error={data.get('error')}). Is the daemon up and `ollama pull {model}` done?"
            )
        vecs.extend(emb)
    return vecs


class OllamaBGEIndex:
    """Dense retrieval over bge-m3 embeddings from a local Ollama daemon. bge-m3 returns unit-normalized
    vectors (cosine == dot); we re-normalize defensively. numpy powers the top-k dot products."""

    def __init__(self, docs: list):
        import numpy as np
        v = np.asarray(ollama_embed(docs), dtype=np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vecs = v / norms

    def top_k(self, query: str, k: int) -> list:
        import numpy as np
        qv = np.asarray(ollama_embed([query])[0], dtype=np.float32)
        qv = qv / (float(np.linalg.norm(qv)) or 1.0)
        sims = self.vecs @ qv
        order = sorted(range(len(sims)), key=lambda i: (-float(sims[i]), i))
        return order[:k]


def rrf(rankings: list, k: int = 60) -> list:
    """Reciprocal-rank fusion of several full rankings (list of ranked index lists)."""
    scores: dict = collections.defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking, 1):
            scores[idx] += 1.0 / (k + rank)
    return sorted(scores, key=lambda i: (-scores[i], i))


def diversify(ranking: list, chunk_pages: list, per_page_max: int) -> list:
    """Cap how many chunks per page survive, preserving rank order. `per_page_max` <= 0 is a no-op.
    Pulls a lower-ranked gold-page chunk up when the top is dominated by repetitive sibling pages."""
    if not per_page_max or per_page_max <= 0:
        return ranking
    seen: dict = collections.defaultdict(int)
    out = []
    for idx in ranking:
        p = chunk_pages[idx]
        if seen[p] < per_page_max:
            seen[p] += 1
            out.append(idx)
    return out


def split_pages(text: str) -> list:
    """-> list of (page_id, page_text) split on the bundle's page markers."""
    out = []
    ms = list(PAGE_RE.finditer(text))
    if not ms:
        return [("unknown-p001", text.strip())] if text.strip() else []
    preamble = text[: ms[0].start()].strip()
    if preamble:
        out.append(("preamble-p000", preamble))
    for i, m in enumerate(ms):
        start = m.end()
        end = ms[i + 1].start() if i + 1 < len(ms) else len(text)
        out.append((m.group(1), text[start:end].strip()))
    return out


def split_chunks(text: str, max_chars: int = 1200) -> list:
    """Page-aware passage chunks: each page is windowed into ~max_chars blocks on line boundaries, so
    retrieval works at sub-page granularity (real RAG) while every chunk keeps its page_id."""
    chunks = []
    for pid, ptext in split_pages(text):
        cur = ""
        for ln in ptext.split("\n"):
            if cur and len(cur) + len(ln) + 1 > max_chars:
                if cur.strip():
                    chunks.append((pid, cur.strip()))
                cur = ln
            else:
                cur = f"{cur}\n{ln}" if cur else ln
        if cur.strip():
            chunks.append((pid, cur.strip()))
    return chunks


def rag_prompt(rec: dict, passages: list) -> str:
    instr = rec.get("instruction", "주어진 근거 자료로 질문에 답하라.")
    blocks = "\n\n".join(f"[{pid} (p.{pid.rsplit('-p', 1)[-1].lstrip('0') or '0'})]\n{txt}" for pid, txt in passages)
    return (
        f"{instr}\n\n"
        f"[검색된 문서 발췌] (질문과 관련된 상위 페이지만 제공)\n{blocks}\n\n"
        f"[질문]\n{rec['question']}\n\n"
        f"[지시] 위 발췌에 근거하여 답하세요. 발췌에서 찾을 수 없으면 "
        f"'제공된 자료만으로는 확정할 수 없음'이라고 답하세요. 군더더기 없이 최종 답만 간단히 출력하세요."
    )


def rank_chunks(retriever: str, rec: dict, chunks: list, bid: str,
                bm25_cache: dict, dense_cache: dict) -> list:
    """Full ranked chunk-index list for bm25/dense/hybrid (indexes cached per bundle; hybrid = RRF of
    BM25 + dense). dense/hybrid call embed_texts (paid)."""
    q = rec["question"]
    if retriever in ("bm25", "hybrid") and bid not in bm25_cache:
        bm25_cache[bid] = BM25([txt for _, txt in chunks])
    if retriever in ("dense", "hybrid") and bid not in dense_cache:
        dense_cache[bid] = DenseIndex([txt for _, txt in chunks])
    if retriever == "dense_bge" and bid not in dense_cache:
        dense_cache[bid] = OllamaBGEIndex([txt for _, txt in chunks])
    if retriever == "dense_bge_st" and bid not in dense_cache:
        dense_cache[bid] = BGEDenseIndex([txt for _, txt in chunks])
    n = len(chunks)
    if retriever == "bm25":
        return bm25_cache[bid].top_k(q, n)
    if retriever in ("dense", "dense_bge", "dense_bge_st"):
        return dense_cache[bid].top_k(q, n)
    if retriever == "hybrid":
        return rrf([bm25_cache[bid].top_k(q, n), dense_cache[bid].top_k(q, n)])
    raise ValueError(f"unsupported retriever for ranking: {retriever!r}")


def retrieve(retriever: str, rec: dict, chunks: list, k: int, bid: str,
             bm25_cache: dict, dense_cache: dict, per_page_max: int = 0) -> list:
    """Return the picked passages (page_id, text) in reading order. `per_page_max` > 0 caps chunks per
    page (page-diverse retrieval). oracle returns the gold-page chunks (unaffected by per_page_max)."""
    if retriever == "oracle":
        gold = set(rec.get("page_ids") or [])
        return [(pid, txt) for pid, txt in chunks if pid in gold]
    full = rank_chunks(retriever, rec, chunks, bid, bm25_cache, dense_cache)
    idx = diversify(full, [pid for pid, _ in chunks], per_page_max)[:k]
    return [chunks[i] for i in sorted(idx)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", choices=["bm25", "dense", "dense_bge", "dense_bge_st", "hybrid", "oracle"],
                    required=True,
                    help="bm25/oracle need no extra deps; dense/hybrid need OPENAI_API_KEY and the "
                         "optional baseline deps (uv sync --extra baseline); dense_bge = bge-m3 via a "
                         "local Ollama daemon (urllib only, fast); dense_bge_st = bge-m3 via "
                         "sentence-transformers+torch (offline fallback)")
    ap.add_argument("--k", type=int, default=5,
                    help="top-k passages for bm25/dense/hybrid (oracle uses gold-page chunks); "
                         "dense/hybrid embeddings need OPENAI_API_KEY + uv sync --extra baseline")
    ap.add_argument("--chunk-chars", type=int, default=1200, help="passage size for sub-page chunking")
    ap.add_argument("--per-page-max", type=int, default=0,
                    help="cap chunks per page in the top-k (page-diverse retrieval); 0 = off")
    ap.add_argument("--split", default="test_public", choices=["test_public", "dev"])
    ap.add_argument("--tiers", default="32k,64k")
    ap.add_argument("--max-items", type=int, default=40)
    ap.add_argument("--out", default=None, help="output JSONL (must be under local/)")
    args = ap.parse_args()

    variant_label = f"page_diverse_pp{args.per_page_max}" if args.per_page_max else "standard"
    suffix = f"pp{args.per_page_max}_" if args.per_page_max else ""
    out = Path(args.out).resolve() if args.out else \
        LOCAL / "runs" / f"rag_{args.retriever}_{suffix}prompts.jsonl"
    if not out.is_relative_to(LOCAL.resolve()):
        raise SystemExit(f"--out must be under local/ (embeds bundle text). Got: {out}")

    tiers = {t.strip() for t in args.tiers.split(",") if t.strip()}
    with QA.open(encoding="utf-8") as qf:
        rows = [json.loads(l) for l in qf if l.strip()]
    pool = [r for r in rows
            if r.get("split") == args.split and r.get("bundle_id") and r.get("context_tier") in tiers]
    pool.sort(key=lambda r: r["qa_id"])
    pool = pool[: args.max_items]
    if not pool:
        raise SystemExit("no QA matched the selection (split/tier/bundle)")

    out.parent.mkdir(parents=True, exist_ok=True)
    chunks_cache: dict = {}
    bm25_cache: dict = {}
    dense_cache: dict = {}
    written_recs: list = []
    no_passage = 0
    with out.open("w", encoding="utf-8") as f:
        for r in pool:
            bid = r["bundle_id"]
            if bid not in chunks_cache:
                bf = BUNDLES / f"{bid}.txt"
                chunks_cache[bid] = split_chunks(bf.read_text(encoding="utf-8"), args.chunk_chars) if bf.exists() else []
            chunks = chunks_cache[bid]
            if not chunks:
                print(f"  WARN: no bundle pages for {bid}; skipping {r['qa_id']}")
                continue
            picked = retrieve(args.retriever, r, chunks, args.k, bid, bm25_cache, dense_cache, args.per_page_max)
            if not picked:
                no_passage += 1
                print(f"  WARN: no passage retrieved ({args.retriever}) for {r['qa_id']}; skipping")
                continue
            rec = {"qa_id": r["qa_id"], "split": r["split"], "task_type": r["task_type"],
                   "context_tier": r["context_tier"], "bundle_id": bid, "answer_type": r.get("answer_type"),
                   "retriever": args.retriever, "per_page_max": args.per_page_max,
                   "retrieval_variant": variant_label, "n_passages": len(picked),
                   "prompt": rag_prompt(r, picked)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written_recs.append(rec)

    if not written_recs:
        out.unlink(missing_ok=True)
        try:
            disp = BUNDLES.relative_to(ROOT)
        except ValueError:
            disp = BUNDLES
        raise SystemExit(f"no RAG prompts written — internal bundle text under {disp} is absent or no "
                         "passages matched. Rebuild bundles locally; this smoke needs the internal corpus.")

    print(f"=== RAG smoke selection ({args.retriever}, {args.split}, tiers={sorted(tiers)}, k={args.k}) ===")
    print(f"wrote {len(written_recs)} items over {len({r['bundle_id'] for r in written_recs})} bundles "
          f"-> {out.relative_to(ROOT)} (INTERNAL); items with no passage: {no_passage}")
    print(f"avg passages/item: {sum(r['n_passages'] for r in written_recs) / len(written_recs):.1f}")
    print("qa_ids: " + ",".join(r["qa_id"] for r in written_recs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
