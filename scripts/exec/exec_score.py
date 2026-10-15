"""Deterministic scoring of executed values against the gold answers.
"""
import csv, json, re
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
from pathlib import Path

from khat.paths import ROOT as DS


_DEC = json.JSONDecoder()


def _objects(text):
    """top-level JSON objects: at each "{" try raw_decode; a success is kept and the scan resumes after its end (so an
    object nested inside a decoded one is not taken alone); a failure (e.g. an unpaired "{" in prose) moves on by one
    character"""
    out, i = [], text.find("{")
    while i != -1:
        try:
            obj, end = _DEC.raw_decode(text, i)
            out.append(obj); i = text.find("{", end)
        except json.JSONDecodeError:
            i = text.find("{", i + 1)
    return out


KEYS = {"table", "filters", "operator", "field"}


def parse(text):
    t = re.sub(r"```(?:json)?", "", text or "")
    for q in reversed(_objects(t)):
        if isinstance(q, dict) and KEYS <= set(q): return q, None
    return None, "invalid JSON"


NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def to_num(s):
    m = NUM_RE.search(str(s)); assert m, s
    x = m.group(0).replace(",", ""); return float(x), (len(x.split(".")[1]) if "." in x else 0)


@lru_cache(maxsize=None)
def _gold_src():
    G = {}
    for f in ("qa_test_public.jsonl", "qa_dev.jsonl"):
        for l in open(DS / "data" / f, encoding="utf-8"):
            r = json.loads(l); G[r["qa_id"]] = r
    fx = json.load(open(DS / "data/gold_fixes.json", encoding="utf-8"))
    dev_any = {r["qa_id"]: [x.strip() for x in r["new_answer_or_accepted"].split(" | ")]
               for r in csv.DictReader(open(DS / "data/errata/dev_review.csv", encoding="utf-8")) if r["status"] == "gold_accept_any"}
    return G, fx, dev_any


def gold_strings(q):
    G, fx, dev_any = _gold_src()
    if q in fx["replace"]: return [fx["replace"][q]]
    if q in fx["accept_any"]: return list(fx["accept_any"][q])
    if q in dev_any: return dev_any[q]
    return [str(G[q]["answer"])]


def gold_values(q):
    return [to_num(s) for s in gold_strings(q)]


def half_up(value, d):
    return float(Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def score(value, golds):
    if value is None: return False
    return any(half_up(value, d) == g for g, d in golds)
