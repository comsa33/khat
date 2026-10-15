#!/usr/bin/env python3
"""Review inputs of the high-risk families: question, gold and the cited pages and statute articles.

Usage: python3 scripts/construction/review_targets.py
"""
import json, re, os, bisect, glob, collections

from khat.paths import LOCAL
BUND = f"{LOCAL}/bundles"
PROC = f"{LOCAL}"
OUT = f"{LOCAL}/audit/triage_compare"
MARKER = re.compile(r"\[[^\]\n]*\(p\.\d+\)\]")
ART = re.compile(r"제\d+조(?:의\d+)?")
PC, TC = 9000, 30000
HIRISK = {"cross_document_legal_reasoning", "multi_document_comparison", "eligibility_reasoning"}

# ---- bundle page extraction ----
_bc = {}
def load_bundle(bid):
    if bid in _bc: return _bc[bid]
    p = f"{BUND}/{bid}.txt"
    if not os.path.exists(p): _bc[bid] = (None, None); return None, None
    t = open(p, encoding="utf-8", errors="replace").read()
    _bc[bid] = (t, [m.start() for m in MARKER.finditer(t)]); return _bc[bid]
def page(bid, pid):
    t, st = load_bundle(bid)
    if t is None: return None
    i = t.find(pid)
    if i < 0: return None
    k = bisect.bisect_right(st, i) - 1
    s = st[k] if k >= 0 else i; e = st[k + 1] if k + 1 < len(st) else len(t)
    return t[s:e][:PC]

# ---- statute article index: (source_id, 제N조) -> text ----
STAT = {}
for f in glob.glob(f"{PROC}/law-*/chunks.jsonl"):
    for l in open(f, encoding="utf-8"):
        if not l.strip(): continue
        try: r = json.loads(l)
        except Exception: continue
        cid = r.get("chunk_id", "")
        if "::" in cid:
            sid, art = cid.split("::", 1)
            m = ART.search(art)
            if m:
                STAT[(sid, m.group(0))] = r.get("text", "")
print(f"statute 조문 인덱스: {len(STAT)}개 (sources: {sorted(set(s for s,_ in STAT))})")

def statute_text(ev):
    out = []
    for e in (ev or []):
        sid = e.get("source_id", "")
        if not sid.startswith("law"): continue
        loc = e.get("locator", "")
        for art in dict.fromkeys(ART.findall(loc)):
            txt = STAT.get((sid, art))
            if txt:
                out.append(f"[{sid} {art}]\n{txt}")
    return out

def assemble(r):
    parts = [p for pid in (r.get("page_ids") or []) if (p := page(r.get("bundle_id"), pid))]
    parts += statute_text(r.get("evidence"))
    return ("\n\n----\n\n".join(parts))[:TC]

rows = []
for fn in ("construction/qa_dev.jsonl", "construction/qa_test_public.jsonl"):
    for l in open(f"{LOCAL}/{fn}", encoding="utf-8"):
        if not l.strip(): continue
        r = json.loads(l)
        if r["task_type"] not in HIRISK: continue
        ev = assemble(r)
        rows.append({"qa_id": r["qa_id"], "task_type": r["task_type"], "split": r["split"],
                     "question": r["question"], "gold_answer": str(r.get("answer", "")),
                     "evidence_text": ev, "has_evidence": bool(ev and len(ev) >= 120)})

json.dump(rows, open(f"{OUT}/review_targets_hirisk.json", "w"), ensure_ascii=False, indent=1)
byf = collections.Counter(x["task_type"] for x in rows)
noev = collections.Counter(x["task_type"] for x in rows if not x["has_evidence"])
withstat = sum(1 for fn in ("construction/qa_dev.jsonl","construction/qa_test_public.jsonl")
               for l in open(f"{LOCAL}/{fn}",encoding="utf-8") if l.strip()
               and json.loads(l)["task_type"] in HIRISK and statute_text(json.loads(l).get("evidence")))
print(f"총 {len(rows)}개 → review_targets_hirisk.json (statute 포함 항목 {withstat}개)")
for f in sorted(HIRISK): print(f"  {f:32} total {byf[f]:<4} 근거없음 {noev[f]}")
