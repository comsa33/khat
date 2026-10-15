"""Same-day re-run of the eight dev items of one announcement under three generation rules.

Usage: python3 scripts/same_day/same_day_analysis.py <same-day run dir> <two-stage run dir>
"""
import collections, contextlib, io, json, random, runpy, sys
from pathlib import Path
from khat.paths import LOCAL
HERE = Path(__file__).resolve().parent; S = HERE.parent
W, TWO_STAGE = Path(sys.argv[1]), Path(sys.argv[2]); SELF = "--selftest" in sys.argv
sys.path.insert(0, str(S / "two_stage")); import judge_parse as JP
sys.argv = ["two_stage_analysis.py", str(TWO_STAGE)]
with contextlib.redirect_stdout(io.StringIO()):
    A = runpy.run_path(str(S / "two_stage/two_stage_analysis.py"))
cell, NONOV = A["cell"], A["NONOV"]
MODELS = ["gpt-4.1-mini", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.5", "minimax-m3", "glm-5.2", "claude-sonnet-5-5", "claude-haiku-5-5"]
FOUR = ["gpt-5.5", "minimax-m3", "glm-5.2", "claude-sonnet-5-5"]
RULES = ["bv1", "bv2", "bv3"]; JN = ["kimi", "sonnet", "gpt-4.1-mini"]
ITEMS = ["krhlrb_v05_1798"] + [f"krhlrb_v05_{i}" for i in range(1800, 1807)]
REST = [q for q in NONOV if q not in ITEMS]; assert len(REST) == 73 and set(ITEMS) <= set(NONOV)
WRONG = {"failed", "still_empty", "empty_retry_pending_or_failed"}
NEW = {j: collections.defaultdict(dict) for j in JN}
MISSING = {j: 0 for j in JN}
PB = {(x["rule"], x["qa_id"]): x["prompt"] for x in map(json.loads, open(LOCAL / "runs/same_day/prompts_B.jsonl", encoding="utf-8"))}
SAMEP = {(a, b): [q for q in ITEMS if PB[(a, q)] == PB[(b, q)]] for a, b in (("bv1", "bv2"), ("bv1", "bv3"), ("bv2", "bv3"))}
GENFAIL = collections.Counter()
if SELF:
    rnd = random.Random(20261014)
    for j in JN:
        for r in RULES:
            for m in MODELS:
                for q in ITEMS: NEW[j][(r, m)][q] = rnd.random() < 0.6
else:
    Jd = W / "judge"; idx = {i["custom_id"]: i for i in map(json.loads, open(Jd / "index.jsonl", encoding="utf-8"))}; V = JP.load(Jd)
    for j in JN:
        if V["missing"][j] and JP.rounds(Jd, j) < JP.MAX_ROUNDS: sys.exit(f"{j}: {len(V['missing'][j])} verdicts missing; run judge_parse.py missing first")
        print(f"verdicts missing for {j}: {len(V['missing'][j])}" + (" (final; counted incorrect)" if V["missing"][j] else ""))
        MISSING[j] = len(V["missing"][j])
    for cid, i in idx.items():
        if i["status"] in WRONG: GENFAIL[(i["condition"], i["model"])] += 1
        for j in JN: NEW[j][(i["condition"], i["model"])][i["qa_id"]] = (i["status"] not in WRONG) and V[j].get(cid, False)
    assert all(len(NEW[j][(r, m)]) == 8 for j in JN for r in RULES for m in MODELS)
out = ["SELFTEST (random verdicts; never reported)"] if SELF else []
for j in JN:
    old = {m: cell(j, "twostage", m, NONOV)[2] for m in MODELS}
    out.append(f"== judge {j}" + (" (primary)" if j == "kimi" else ""))
    out.append("  (1) the 8 items, correct / 8: " + "; ".join(f"{m} " + "/".join(str(sum(NEW[j][(r, m)].values())) for r in RULES) for m in MODELS) + "  (bv1/bv2/bv3)")
    acc, tv, dmax = {}, {}, {}
    for m in MODELS:
        base = sum(old[m][q] for q in REST)
        acc[m] = {r: 100 * (base + sum(NEW[j][(r, m)].values())) / 81 for r in RULES}
        rep = 100 * sum(old[m][q] for q in NONOV) / 81
        tv[m] = sum(NEW[j][("bv3", m)][q] != old[m][q] for q in ITEMS)
        dmax[m] = max(abs(sum(NEW[j][(r, m)].values()) - sum(NEW[j][("bv3", m)].values())) for r in RULES)
        out.append(f"  (2) {m:18s} dev' 81: bv1 {acc[m]['bv1']:.1f}  bv2 {acc[m]['bv2']:.1f}  bv3 {acc[m]['bv3']:.1f}  | reported (two-stage run) {rep:.1f}  "
                   f"| (3) bv3 vs two-stage-run verdicts differ on {tv[m]}/8 | max |rule - bv3| {dmax[m]} items")
    lo = min(acc[m][r] for m in FOUR for r in RULES); hi = max(acc[m][r] for m in FOUR for r in RULES)
    out.append(f"  (4) the four, min-max over three rules: {lo:.1f}-{hi:.1f}; per rule: " +
               "; ".join(f"{r} {min(acc[m][r] for m in FOUR):.1f}-{max(acc[m][r] for m in FOUR):.1f}" for r in RULES) +
               f"; max |rule - bv3| over the four {max(dmax[m] for m in FOUR)} items; time variation over the four {max(tv[m] for m in FOUR)} items")
# ---- values quoted in the text, kimi-k2.6
j = "kimi"; old = {m: cell(j, "twostage", m, NONOV)[2] for m in MODELS}
cnt = {m: {r: sum(NEW[j][(r, m)].values()) for r in RULES} for m in MODELS}
r1 = {m: 100 * (sum(old[m][q] for q in REST) + cnt[m]["bv3"]) / 81 for m in MODELS}
twelve = [100 * (sum(old[m][q] for q in REST) + cnt[m][r]) / 81 for m in FOUR for r in RULES]
lo, hi = round(min(twelve)), round(max(twelve))
k = {m: sum(NEW[j][("bv3", m)][q] != old[m][q] for q in ITEMS) for m in MODELS}
rep = {m: 100 * sum(old[m][q] for q in NONOV) / 81 for m in MODELS}
spread = {m: max(cnt[m].values()) - min(cnt[m].values()) for m in MODELS}
vs3 = {m: max(abs(cnt[m][r] - cnt[m]["bv3"]) for r in RULES) for m in MODELS}
D144, POOL = A["D144"], A["T81"] + A["D144"]
def comp_acc(m, qs):
    v = cell(j, "twostage", m, qs)[2]
    return 100 * sum((NEW[j][("bv3", m)][q] if q in ITEMS else v[q]) for q in qs) / len(qs)
others = [m for m in MODELS if not k[m]]
out += ["", "== values in the text (kimi-k2.6, primary)",
        "  Table 8 dev' (73 two-stage + 8 same-day bv3): " + ", ".join(f"{m} {r1[m]:.1f}" for m in MODELS),
        "  ladder Dev 144 / All 225 with the 8 replaced: " + ", ".join(f"{m} {comp_acc(m, D144):.1f} / {comp_acc(m, POOL):.1f}" for m in MODELS),
        f"  four models x three rules (12 values): {lo}-{hi}%" + (f" (single value {lo}%)" if lo == hi else "") + f"  [unrounded {min(twelve):.1f}-{max(twelve):.1f}]",
        "  bv3 same day vs 13 Oct, items differing: " + ", ".join(f"{m} {k[m]}" for m in MODELS)
        + "; with the 13 Oct run dev' = " + ", ".join(f"{m} {rep[m]:.1f}" for m in MODELS),
        "     sentence: " + ("The adopted rule's re-run gives the same verdicts as the 10/13 run on all 8 items for every model." if not any(k.values()) else
                             "The adopted rule's re-run differs from the 10/13 run on " + "; ".join(f"{k[m]} of the 8 items for {m}" for m in MODELS if k[m])
                             + (", and on none for the others" if others else "") + "; with the 10/13 run, dev' accuracy is "
                             + ", ".join(f"{rep[m]:.1f}% for {m}" for m in MODELS if k[m]) + "."),
        f"  max over the four of (max - min over the three rules) = {max(spread[m] for m in FOUR)} items (per model " + ", ".join(f"{m} {spread[m]}" for m in FOUR)
        + f"; max |rule - bv3| = {max(vs3[m] for m in FOUR)}); max over the four of the items differing from 13 Oct = {max(k[m] for m in FOUR)}",
        *[f"     {a} vs {b} on their {len(qs)} byte-identical prompts ({', '.join(q[-4:] for q in qs)}; not in the 13 Oct comparison), items differing: "
          + ", ".join(f"{m} {sum(NEW[j][(a, m)][q] != NEW[j][(b, m)][q] for q in qs)}" for m in MODELS) for (a, b), qs in SAMEP.items() if qs],
        f"  verdicts missing (kimi-k2.6) = {MISSING['kimi']}; other judges (SI column): " + ", ".join(f"{jj} {MISSING[jj]}" for jj in JN[1:]),
        "  generation failures (final failed / still empty, scored wrong): " + (", ".join(f"{r} {m} {c}" for (r, m), c in sorted(GENFAIL.items())) or "none")]
if not SELF:
    (W / "score").mkdir(exist_ok=True)
    ov = {jj: {m: {q: bool(NEW[jj][("bv3", m)][q]) for q in ITEMS} for m in MODELS} for jj in JN}
    json.dump(ov, open(W / "score" / "same_day_overrides.json", "w"), indent=0, sort_keys=True)
print("\n".join(out))
