"""Accuracy of the claude-sonnet-5-5 and claude-haiku-5-5 predictions under the claude-sonnet-5-5, gpt-4.1-mini and four panel judges, and each panel judge's agreement with the claude-sonnet-5-5 judge. The kimi-k2.6 column uses the primary verdicts in results/verdicts/kimi-k2.6/; the other panel columns use results/verdicts/panel/.

Usage: python3 scripts/judges/panel_claude_compare.py   (writes local/runs/reruns/panel_compare.json)
"""
import json, glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "score"))
from rerun_lib import RERUNS, RESULTS, rerun_verdicts
from khat.paths import LOCAL
NEW = rerun_verdicts()
prim = {(t, k.split("|", 1)[1]): v for t, d in NEW["claude"].items() for k, v in d.items()}
gpt = {(t, k.split("|", 1)[1]): v for t, d in NEW["gpt"].items() for k, v in d.items()}
judges = ["gemma4_31b", "kimi-k2.6", "mistral-large-3", "deepseek-v4.1-flash"]
panel = {j: {} for j in judges}
KIMI = RESULTS / "verdicts/kimi-k2.6"
def kimi_files(name):
    if name.startswith("full_context_aggregation_"):
        return [KIMI / ("full_context_" + name[len("full_context_aggregation_"):] + ".jsonl"), KIMI / f"{name}.jsonl"]
    return [KIMI / f"{name}.jsonl"]
for j in judges:
    for t in [t for t in RERUNS if t.startswith(("sonnet55", "haiku55"))]:
        fs = kimi_files(RERUNS[t]) if j == "kimi-k2.6" else [RESULTS / "verdicts/panel" / f"panel_{j}_{RERUNS[t]}.jsonl"]
        for f in fs:
            for l in open(f):
                r = json.loads(l); panel[j][(t, r["qa_id"])] = bool(r["judge_correct"])
tags = sorted({k[0] for k in prim if k[0].startswith(("sonnet55", "haiku55"))})
def acc(d, keys): ks = [k for k in keys if k in d]; return (round(100 * sum(d[k] for k in ks) / len(ks)), len(ks)) if ks else ("-", 0)
print("| tag | n | claude-sonnet-5-5 | gpt-4.1-mini | " + " | ".join(judges) + " |")
out = {}
for t in tags:
    keys = [k for k in prim if k[0] == t]
    row = [acc(prim, keys), acc(gpt, keys)] + [acc(panel[j], keys) for j in judges]
    out[t] = row
    print(f"| {t} | {len(keys)} | " + " | ".join(f"{a}%" + ("" if n == len(keys) else f" (n={n})") for a, n in row) + " |")
for j in judges:
    ks = [k for k in panel[j] if k in prim]
    agree = sum(panel[j][k] == prim[k] for k in ks); py = sum(prim[k] and not panel[j][k] for k in ks); pn = sum(panel[j][k] and not prim[k] for k in ks)
    print(f"{j}: n={len(ks)} agree={100*agree/len(ks):.1f}% primaryYES_panelNO={py} primaryNO_panelYES={pn}")
OUT = LOCAL / "runs/reruns"; OUT.mkdir(parents=True, exist_ok=True)
json.dump(out, open(OUT / "panel_compare.json", "w"), indent=1)
