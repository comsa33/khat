#!/usr/bin/env python3
"""Judge prompts for the dev runs, with each prediction layer checked against the initial gpt-4.1-mini verdicts.

Usage: python3 scripts/judges/build_judge_dev.py --out local/runs/judge_dev [--baselines DIR]
"""
import argparse, json, os, collections
from pathlib import Path

from judge_common import BASELINES, LEGAL, INITIAL_VERDICTS as REL, ROOT, judge_prompt

_ap = argparse.ArgumentParser()
_ap.add_argument("--out", required=True)
_ap.add_argument("--baselines", default=str(BASELINES))
_args = _ap.parse_args()
B = Path(_args.baselines)
OUT = Path(_args.out)
OUT.mkdir(parents=True, exist_ok=True)

def load(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def V(p): return {r["qa_id"]: bool(r["judge_correct"]) for r in load(p)}
def P(p):
    rs = load(p); d = {r["qa_id"]: r.get("prediction", "") for r in rs}
    assert len(d) == len(rs), ("dup qa_id", p); return d
def mt(n): return os.path.getmtime(B / n)

dev = {d["qa_id"]: d for d in load(ROOT / "data/qa_dev.jsonl")}
ins = {q for q, d in dev.items() if d.get("task_type") != LEGAL}

def layers(M, R, tag):
    if R == "rag":
        return [(f"jdev_{M}_bm25.judged.jsonl", f"ragbm25devfx_{M}_dev.jsonl", f"jdev_{M}_bm25", [])]
    if R == "cb":
        L = [(f"{tag}.judged.jsonl.layer1", f"cb_{M}_dev.jsonl", tag,
              [f"cb_{M}_dev.calls.jsonl"] + ([] if M == "gpt-5.5" else [f"_merged_cb_{M}.jsonl"]))]
        if M == "gpt-5.5":
            L.append((f"{tag}_retry.judged.jsonl", "cb_gpt-5.5_retry_merged.jsonl", f"{tag}_retry",
                      ["cb_gpt-5.5_dev_retry.jsonl", "cb_gpt-5.5_dev.jsonl"]))
        return L
    if M in ("gpt-5.4-mini", "gpt-5.4-nano"):
        return [(f"{tag}.judged.jsonl.layer1", f"fc_{M}_dev.jsonl", tag, [f"fc_{M}_dev.calls.jsonl", f"_merged_fc_{M}.jsonl"])]
    return [(f"{tag}.judged.jsonl.layer2", f"fc_{M}_dev.jsonl.layer2", tag, [f"fc_{M}_dev.calls.jsonl", f"_merged_fc_{M}.jsonl"]),
            (f"{tag}_layer3.judged.jsonl", f"fc_{M}_layer3_merged.jsonl", f"{tag}_layer3", [f"fc_{M}_dev_layer3.jsonl"]),
            (f"fc_{M}_dev_fixeddev512k.judged.jsonl", f"fc_{M}_dev_fixeddev512k.jsonl", f"fc_{M}_dev_fixeddev512k", [])]

TAGS = [(M, R, "g55fc" if (M, R) == ("gpt-5.5", "fc") else f"{M}_{R}")
        for M in ("gpt-4.1-mini", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.5") for R in ("cb", "rag", "fc")]

reqs, idx, mapping, problems = [], [], {}, []
for M, R, tag in TAGS:
    rel = V(REL / f"{tag}.jsonl")
    want = sorted(q for q in rel if q in ins)
    src = {}
    info = []
    for vf, pf, ttag, checks in layers(M, R, tag):
        v, p = V(B / vf), P(B / pf)
        tr = json.load(open(B / f"judge_track_{ttag}.json"))
        pre_ok = mt(pf) <= mt(f"judge_track_{ttag}.json")
        n_ok = {q for q in v if q in dev} <= set(p) and (len(v) == tr["n"])
        xc = {}
        for c in checks:
            cp = P(B / c); common = [q for q in p if q in cp and q in v]
            xc[c] = (len(common), sum(cp[q] == p[q] for q in common))
        for q in p:
            if q in v: src[q] = (pf, p[q], v[q])
        info.append({"layer": pf, "verdicts": vf, "rows": len(p), "judged": len(v), "track_n": tr["n"],
                     "file_predates_judge": pre_ok, "verdicts_subset_of_file": n_ok,
                     "crosscheck_equal/common": {k: f"{e}/{c}" for k, (c, e) in xc.items()}})
        if not pre_ok: problems.append((tag, pf, "file modified after judge submission (see crosschecks)"))
        for c, (cn, eq) in xc.items():
            if eq != cn: problems.append((tag, pf, f"crosscheck {c}: {cn-eq} differing preds"))
    miss = [q for q in want if q not in src]
    disagree = [q for q in want if q in src and src[q][2] != rel[q]]
    if R == "rag":
        old = V(B / f"{tag}.judged.jsonl.layer0")
        info.append({"release_differs_from_layer0_on": sum(1 for q in want if q in old and old[q] != rel[q]),
                     "release_differs_from_jdev_on": sum(1 for q in want if src.get(q, (0, 0, None))[2] != rel[q])})
    if miss: problems.append((tag, "UNRESOLVED", len(miss)))
    if disagree: problems.append((tag, "VERDICT-MISMATCH", disagree[:5], len(disagree)))
    per = collections.Counter(src[q][0] for q in want if q in src)
    mapping[tag] = {"n_inscope": len(want), "resolved_from": dict(per), "unresolved": len(miss),
                    "release_vs_layer_verdict_mismatch": len(disagree), "layers": info}
    for q in want:
        if q not in src or q in disagree: continue
        d = dev[q]; cid = f"d{len(reqs):05d}"
        reqs.append({"custom_id": cid, "prompt": judge_prompt(d.get("question", ""), str(d.get("answer", "")), src[q][1])})
        idx.append({"custom_id": cid, "set": "dev", "tag": tag, "qa_id": q})

with open(OUT / "prompts.jsonl", "w", encoding="utf-8") as f:
    for r in reqs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(OUT / "index.jsonl", "w", encoding="utf-8") as f:
    for r in idx: f.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump({"mapping": mapping, "problems": problems}, open(OUT / "dev_mapping.json", "w"), indent=1, ensure_ascii=False)
for t, m in mapping.items():
    print(t, m["n_inscope"], m["resolved_from"], "unres", m["unresolved"], "mism", m["release_vs_layer_verdict_mismatch"])
    for L in m["layers"]:
        if "layer" not in L: print("   ", L); continue
        print("   ", L["layer"], "rows", L["rows"], "judged", L["judged"], "track", L["track_n"], "predates", L["file_predates_judge"], "subset", L["verdicts_subset_of_file"], L["crosscheck_equal/common"])
print("PROBLEMS", problems)
print("requests", len(reqs), "unique", len({r['custom_id'] for r in reqs}), "pairs unique", len({(i['tag'], i['qa_id']) for i in idx}))
print("empty predictions", sum(1 for r in reqs if "[모델답변] (빈 답)" in r["prompt"]))
print("chars", sum(len(r["prompt"]) for r in reqs), "max", max(len(r["prompt"]) for r in reqs))
