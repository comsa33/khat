"""Dense top-50 plus announcement page on 18 items: correct answers per model under the claude-sonnet-5-5 and gpt-4.1-mini judges, with the top-50 and open-region references and the panel judges (writes local/runs/k50page/).

Usage: python3 scripts/score/score_k50_page.py   (after scripts/run/build_k50_page.py)
"""
import json, glob, collections, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rerun_lib import *
from khat.paths import LOCAL
NEW = rerun_verdicts()
def new(j, tag): return {k.split("|", 1)[1]: v for k, v in NEW[j].get(tag, {}).items()}
ITEMS = sorted(json.loads(l)["qa_id"] for l in open(LOCAL / "runs/k50page/prompts_k50page_18.jsonl"))
MODELS = ["gpt-5.5", "minimax-m3", "glm-5.2", "claude-sonnet-5-5", "claude-haiku-5-5", "gpt-4.1-mini"]
def verdicts(run, names=None):
    names = names or {}
    out = {"claude": collections.defaultdict(dict), "gpt": collections.defaultdict(dict)}; bad = collections.Counter()
    for j, judge in (("claude", "claude-sonnet-5-5"), ("gpt", "gpt-4.1-mini")):
        for m in MODELS + [m for m in names if m not in MODELS]:
            out[j][m] = rel(judge, f"{run}_{names.get(m, m)}")
    return out, bad
PG, bad = verdicts("k50_page")
OLD, _ = verdicts("k50", {"minimax-m3-reasoning": "minimax-m3", "minimax-m3": "no_reasoning_minimax-m3"})
NEWM = {"claude-sonnet-5-5": "sonnet55", "claude-haiku-5-5": "haiku55"}
def openreg(m, j="claude"): return new(j, f"{NEWM[m]}_e3region") if m in NEWM else new(j, f"e3region_{m}")
res = {}
for m in MODELS:
    assert sorted(PG["claude"][m]) == ITEMS and sorted(PG["gpt"][m]) == ITEMS, m
    k50m = "minimax-m3-reasoning" if m == "minimax-m3" else m
    o = openreg(m)
    res[m] = {"k50page_primary": sum(PG["claude"][m][q] for q in ITEMS), "k50page_secondary": sum(PG["gpt"][m][q] for q in ITEMS),
              "k50_primary": sum(OLD["claude"][k50m][q] for q in ITEMS),
              "openregion_primary": sum(bool(o.get(q)) for q in ITEMS) if o else None}
panel = {}
for f in sorted(glob.glob(str(RESULTS / "verdicts/panel/panel_*_k50_page_claude-*.jsonl"))):
    judge, who = Path(f).name[len("panel_"):-len(".jsonl")].split("_k50_page_")
    name = judge + "_" + NEWM[who]
    rows = [json.loads(l) for l in open(f)]; panel[name] = {"n": len(rows), "correct": sum(r["judge_correct"] is True for r in rows),
                                                             "unparsed": sum(r["judge_correct"] is None for r in rows)}
OUT = LOCAL / "runs/k50page"; OUT.mkdir(parents=True, exist_ok=True)
json.dump({"results": res, "panel": panel, "judge_failures": dict(bad)}, open(OUT / "k50page_results.json", "w"), indent=1)
print("| model | k=50 | k=50+page (claude-sonnet-5-5) | k=50+page (gpt-4.1-mini) | open-region |")
for m, r in res.items(): print(f"| {m} | {r['k50_primary']}/18 | {r['k50page_primary']}/18 | {r['k50page_secondary']}/18 | {r['openregion_primary']}/18 |")
print("panel:", panel, "judge failures:", dict(bad))
