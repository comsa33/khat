"""Dense top-50 on the 81 aggregation items against full context: accuracy, Wilson and pair-bootstrap intervals, coverage cross-check and panel judges (writes local/runs/k50/).

Usage: python3 scripts/score/score_k50.py   (after scripts/run/build_oracle.py and scripts/run/k50_coverage.py)
"""
import json, sys, glob, collections, random
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rerun_lib import *
from khat.paths import LOCAL
RUN = RESULTS / "runs/k50"
NEW = rerun_verdicts()
def new(judge, tag): return {k.split("|", 1)[1]: v for k, v in NEW[judge].get(tag, {}).items()}
JUD = {"claude": "claude-sonnet-5-5", "gpt": "gpt-4.1-mini"}
NEWM = {"claude-sonnet-5-5": "sonnet55", "claude-haiku-5-5": "haiku55"}
OLDFC = {"gpt-5.5": ["full_context_512k_rest_gpt-5.5", "full_context_512k_sample_gpt-5.5"],
         "gpt-4.1-mini": ["full_context_512k_rest_gpt-4.1-mini", "full_context_512k_sample_gpt-4.1-mini"],
         "minimax-m3": ["full_context_512k_rest_minimax-m3", "full_context_512k_sample_minimax-m3"],
         "glm-5.2": ["full_context_512k_rest_glm-5.2", "full_context_512k_sample_glm-5.2"]}
OLDDENSE = {"gpt-4.1-mini": "dense_gpt-4.1-mini", "gpt-5.5": "dense_gpt-5.5", "minimax-m3": "dense_minimax-m3"}
MODELS = ["gpt-5.5", "glm-5.2", "minimax-m3", "minimax-m3-reasoning", "claude-sonnet-5-5", "claude-haiku-5-5", "gpt-4.1-mini"]
BASE = lambda m: "minimax-m3" if m == "minimax-m3-reasoning" else m
def fc(m, j="claude"):
    if m in NEWM: return new(j, NEWM[m] + "_agg512k")
    v = {}
    for t in OLDFC[m]: v.update(rel(JUD[j], t) or {})
    return v
def dense5(m, j="claude"):
    if m in NEWM: return new(j, f"{NEWM[m]}_dense")
    if m == "glm-5.2": return new(j, "e2_glm-5.2_dense")
    return rel(JUD[j], OLDDENSE[m])
def e3(m, var, j="claude"): return new(j, f"{NEWM[m]}_{var}") if m in NEWM else new(j, f"{var}_{m}")
AGG = [json.loads(l)["qa_id"] for l in open(V / "data/item_sets/agg81.jsonl")]
SRC = {json.loads(l)["qa_id"]: json.loads(l) for l in open(LOCAL / "runs/oracle/check.jsonl")}
PAIR = {q: ((G[q].get("announcement_ids") or [""])[0], SRC[q]["source"]) for q in AGG}
# ---- k=50 verdicts (released verdict files; minimax-m3-reasoning is the reported minimax-m3 run)
K50N = {"minimax-m3-reasoning": "minimax-m3", "minimax-m3": "no_reasoning_minimax-m3"}
K50 = {"claude": collections.defaultdict(dict), "gpt": collections.defaultdict(dict)}; bad = collections.Counter()
for j in ("claude", "gpt"):
    for m in MODELS:
        K50[j][m] = rel(JUD[j], "k50_" + K50N.get(m, m))
B, SEED = 10000, 20261011
def boot(stat):
    groups = collections.defaultdict(list)
    for q in AGG: groups[PAIR[q]].append(q)
    keys = sorted(groups); rnd = random.Random(SEED); out = []
    for _ in range(B):
        samp = [q for k in (rnd.choice(keys) for _ in keys) for q in groups[k]]
        out.append(stat(samp))
    out.sort(); return out[int(0.025 * B)], out[int(0.975 * B) - 1]
acc = lambda v: (lambda s: sum(v[q] for q in s) / len(s))
cov = json.load(open(LOCAL / "runs/k50/coverage_k50.json"))
gen = json.load(open(RUN / "gen_report.json"))
TABAGG = {"dense5": {"gpt-4.1-mini": 14, "gpt-5.5": 33, "minimax-m3": 32, "glm-5.2": 32, "claude-sonnet-5-5": 28, "claude-haiku-5-5": 20},
          "fc": {"gpt-4.1-mini": 9, "gpt-5.5": 100, "minimax-m3": 68, "glm-5.2": 88, "claude-sonnet-5-5": 74, "claude-haiku-5-5": 23}}
res = {"models": {}, "bad": dict(bad)}
f = lambda t: f"[{t[0]*100:.0f}, {t[1]*100:.0f}]"
lines = ["## 1. Aggregation 81, dense bge-m3 k=50 vs full context 512k (judge claude-sonnet-5-5)",
         f"Pair bootstrap over {len(set(PAIR.values()))} announcement–table pairs, B={B}, seed {SEED}. Label per the pre-registered rule.", "",
         "| Model | k=50 acc | Wilson | pair bootstrap | FC 512k | d = k50 − FC (pp) | pair bootstrap | label | k=50 gpt-4.1-mini | dense k=5 (tab:agg) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for m in MODELS:
    if m not in K50["claude"] or len(K50["claude"][m]) < 81: lines.append(f"| {m} | (pending) |"); continue
    k = K50["claude"][m]; F = fc(BASE(m)); D5 = dense5(BASE(m))
    assert all(q in k and q in F and q in D5 for q in AGG), m
    h = sum(k[q] for q in AGG); hf = sum(F[q] for q in AGG); h5 = sum(D5[q] for q in AGG)
    assert round(100 * hf / 81) == TABAGG["fc"][BASE(m)] and round(100 * h5 / 81) == TABAGG["dense5"][BASE(m)], (m, hf, h5)
    pb = boot(acc(k)); st = lambda s: (sum(k[q] for q in s) - sum(F[q] for q in s)) / len(s)
    d = st(AGG); dpb = boot(st)
    label = "exceeds" if dpb[0] > 0 else ("below" if dpb[1] < 0 else "no detected difference")
    sec = K50["gpt"].get(m, {}); hs = sum(sec[q] for q in AGG) if len(sec) == 81 else None
    corr_not_full = sorted(q for q in AGG if k[q] and not cov[q]["full"])
    res["models"][m] = {"hits": h, "acc": h / 81, "wilson": wilson(h, 81), "pair": pb, "fc_hits": hf, "d": d, "d_pair": dpb,
                        "label": label, "secondary_hits": hs, "dense5_hits": h5, "correct_not_fully_covered": corr_not_full,
                        "correct_and_full": sum(1 for q in AGG if k[q] and cov[q]["full"]),
                        "wrong_but_full": sum(1 for q in AGG if not k[q] and cov[q]["full"]),
                        "median_prompt_tokens": gen[m]["median_in"]}
    lines.append(f"| {m} | {100*h/81:.0f}% ({h}) | {f(wilson(h, 81))} | {f(pb)} | {100*hf/81:.0f}% ({hf}) | {d*100:+.0f} | "
                 f"[{dpb[0]*100:+.0f}, {dpb[1]*100:+.0f}] | {label} | " + (f"{100*hs/81:.0f}% ({hs})" if hs is not None else "pending") +
                 f" | {100*h5/81:.0f}% ({h5}) |")
lines += ["", "## 2. Coverage cross-check (page + all matched rows in the k=50 context; 63/81 items fully covered)",
          "| Model | correct & fully covered | correct, not fully covered | wrong though fully covered | median prompt tokens |", "|---|---|---|---|---|"]
for m, r in res["models"].items():
    lines.append(f"| {m} | {r['correct_and_full']} | {len(r['correct_not_fully_covered'])} | {r['wrong_but_full']} | {r['median_prompt_tokens']:.0f} |")
pan = collections.defaultdict(dict)
for p in glob.glob(str(RESULTS / "verdicts/panel" / "panel_*_k50_claude-*.jsonl")):
    judge, who = Path(p).name[len("panel_"):-len(".jsonl")].split("_k50_")
    who = NEWM[who]
    rows_ = [json.loads(l) for l in open(p)]
    pan[who][judge] = rows_
lines += ["", "## 3. Panel judges on the Claude k=50 predictions (accuracy %, n=81 each)", "| Model | claude-sonnet-5-5 | gpt-4.1-mini | " + " | ".join(sorted({j for w in pan.values() for j in w})) + " |",
          "|---|---|---|" + "---|" * len({j for w in pan.values() for j in w})]
for m, who in (("claude-sonnet-5-5", "sonnet55"), ("claude-haiku-5-5", "haiku55")):
    if m not in res["models"]: continue
    cells = []
    for j in sorted({j for w in pan.values() for j in w}):
        rr = pan[who].get(j, [])
        key = next((k_ for k_ in ("judge_correct", "correct", "verdict") if rr and k_ in rr[0]), None)
        ok = [bool(r[key]) if key != "verdict" else str(r[key]).upper().startswith("Y") for r in rr] if key else []
        cells.append(f"{100*sum(ok)/len(ok):.0f}% (n={len(ok)})" if ok else "pending")
    r = res["models"][m]
    lines.append(f"| {m} | {100*r['acc']:.0f}% | " + (f"{100*r['secondary_hits']/81:.0f}%" if r["secondary_hits"] is not None else "pending") + " | " + " | ".join(cells) + " |")
OUT = LOCAL / "runs/k50"; OUT.mkdir(parents=True, exist_ok=True)
json.dump(res, open(OUT / "k50_results.json", "w"), indent=1, ensure_ascii=False)
open(OUT / "tables.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines)); print("bad", dict(bad))
