"""Dense top-50 plus announcement page: for the items whose top-50 passages miss the gold page, add that page as one more passage.

Usage: python3 scripts/run/build_k50_page.py   (after build_k50.py and k50_coverage.py; writes local/runs/k50page/)
"""
import os, sys, json
from pathlib import Path
from khat.paths import LOCAL
HERE = LOCAL / "runs" / "k50page"
HERE.mkdir(parents=True, exist_ok=True)
K50 = LOCAL / "runs" / "k50"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import rag_lib as lib
from rag_lib import R
qa = lib.test_qa(); rk = lib.Ranker()
P = {json.loads(l)["qa_id"]: json.loads(l) for l in open(K50 / "prompts_dense_k50_agg81.jsonl")}
cov = json.load(open(K50 / "coverage_k50.json"))
unc = sorted(q for q, v in cov.items() if not v["page"])
assert len(unc) == 18
out, rep, ok63 = [], [], 0
for q, s in P.items():
    rec = qa[q]; bid = s["bundle_id"]; ch = lib.chunks_for(bid, 1200)
    gold = set(rec["page_ids"])
    if q not in unc:
        ok63 += any(f"[{g} (p." in s["prompt"] for g in gold)
        continue
    idx = sorted(rk.full("dense_bge", rec, bid, 1200)[:50])
    picked = [ch[i] for i in idx]
    assert R.rag_prompt(rec, picked) == s["prompt"], q
    assert not any(ch[i][0] in gold for i in idx), q
    pages = dict(R.split_pages((lib.BUNDLES / f"{bid}.txt").read_text(encoding="utf-8")))
    assert len(gold) == 1; g = next(iter(gold))
    first = min(i for i, (pid, _) in enumerate(ch) if pid == g)
    pos = sum(1 for i in idx if i < first)
    new = picked[:pos] + [(g, pages[g])] + picked[pos:]
    np_ = R.rag_prompt(rec, new)
    blk = f"[{g} (p.{g.rsplit('-p',1)[-1].lstrip('0') or '0'})]\n{pages[g]}\n\n"
    old = s["prompt"]
    assert len(np_) == len(old) + len(blk) or len(np_) == len(old) + len(blk.rstrip('\n')) + 2, q
    r = dict(s); r["prompt"] = np_; r["setting"] = "k50page"; r["added_page"] = g; r["insert_pos"] = pos
    out.append(r)
    rep.append({"qa_id": q, "page": g, "insert_pos": pos, "page_chars": len(pages[g]), "added_chars": len(np_) - len(old)})
print("other 63 contain a gold-page chunk:", ok63, "/ 63")
with open(HERE / "prompts_k50page_18.jsonl", "w", encoding="utf-8") as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump(rep, open(HERE / "build_report.json", "w"), indent=1)
for x in rep: print(x)
