"""Tables of the closed-book re-runs, reasoning-on, full-context, aggregation, open-region and oracle runs and judge agreement, from the released verdicts (writes local/runs/reruns/).

Usage: python3 scripts/score/score_reruns.py   (after scripts/run/build_oracle.py, which writes local/runs/oracle/check.jsonl)
"""
import json, collections, csv
from rerun_lib import *
from khat.paths import LOCAL
OUT = LOCAL / "runs/reruns"
OUT.mkdir(parents=True, exist_ok=True)
NEW = rerun_verdicts()
def new(judge, tag):
    return {k.split("|", 1)[1]: v for k, v in NEW[judge].get(tag, {}).items()}
TEST = {q for q, g in G.items() if g.get("split") == "test_public" and inscope(q)}
DEV = {q for q, g in G.items() if g.get("split") == "dev" and inscope(q)}
AGG = {json.loads(l)["qa_id"] for l in open(V / "data/item_sets/agg81.jsonl")}
SRC = {json.loads(l)["qa_id"]: json.loads(l)["source"] for l in open(LOCAL / "runs/oracle/check.jsonl")}
HUG = {q for q in AGG if SRC[q] == "hug"}; MOL = AGG - HUG
JUD = {"claude": "claude-sonnet-5-5", "gpt": "gpt-4.1-mini"}
R = {}
out = []
def P(*a): out.append(" ".join(str(x) for x in a))
def both(fn):
    return {j: fn(j) for j in ("claude", "gpt")}
def cell(sc, ci=False, plainci=False):
    c, g = sc["claude"], sc["gpt"]
    t = f"{c['plain']*100:.0f}% / {c['cw']*100:.0f}%"
    if ci: t += f" [{c['cw_ci'][0]*100:.0f}, {c['cw_ci'][1]*100:.0f}]"
    if plainci: t = f"{c['plain']*100:.0f}% [{c['plain_ci'][0]*100:.0f}, {c['plain_ci'][1]*100:.0f}] ({c['hits']})"
    diff = []
    if round(g["plain"]*100) != round(c["plain"]*100): diff.append(f"{g['plain']*100:.0f}%")
    if not plainci and round(g["cw"]*100) != round(c["cw"]*100): diff.append(f"cw {g['cw']*100:.0f}%")
    if diff: t += " (gpt " + " / ".join(diff) + ")"
    return t + ("" if plainci else f" n={c['n']}")

# ---- existing tag map (release verdicts)
OLD = {
 "gpt-4.1-mini": {"cb": "closed_book_gpt-4.1-mini", "bm25": "bm25_gpt-4.1-mini", "dense": "dense_gpt-4.1-mini", "fc": "full_context_gpt-4.1-mini"},
 "gpt-5.4-mini": {"cb": "closed_book_gpt-5.4-mini", "bm25": "bm25_gpt-5.4-mini", "dense": "dense_gpt-5.4-mini", "fc": "full_context_gpt-5.4-mini"},
 "gpt-5.4-nano": {"cb": "closed_book_gpt-5.4-nano", "bm25": "bm25_gpt-5.4-nano", "dense": "dense_gpt-5.4-nano", "fc": "full_context_gpt-5.4-nano"},
 "gpt-5.5": {"cb": "closed_book_gpt-5.5", "bm25": "bm25_gpt-5.5", "dense": "dense_gpt-5.5", "fc": "full_context_gpt-5.5"},
 "minimax-m3": {"cb": "closed_book_minimax-m3", "bm25": "bm25_minimax-m3", "dense": "dense_minimax-m3", "fc": "full_context_minimax-m3"},
 "glm-5.2": {"cb": "closed_book_no_reasoning_glm-5.2", "bm25": "bm25_no_reasoning_glm-5.2", "dense": "dense_no_reasoning_glm-5.2", "fc": "full_context_glm-5.2"},
}
NEWM = {"claude-sonnet-5-5": "sonnet55", "claude-haiku-5-5": "haiku55"}
def V_old(judge, model, reg): return rel(JUD[judge], OLD[model][reg])
def V_any(judge, model, reg):
    if model in NEWM: return new(judge, f"{NEWM[model]}_{reg}")
    return V_old(judge, model, reg)
CB_TEST = {m: f"e1_{m}_test" for m in ["gpt-4.1-mini", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.5", "glm-5.2", "minimax-m3"]}
def cb_corrected(judge, model):
    v = dict(V_any(judge, model, "cb"))
    if model in CB_TEST: v.update(new(judge, CB_TEST[model]))
    if model == "gpt-5.5": v.update(new(judge, "second_call_test"))
    return v

# ---- main test table (CB corrected), Claude rows added
P("## 1. test_public main table (CB = corrected; plain / cw)")
P("| Model | CB (n=342) | BM25 (n=339) | dense (n=339) | FC |"); P("|---|---|---|---|---|")
T1 = {}
for m in list(OLD) + list(NEWM):
    row = {"cb": both(lambda j: score(cb_corrected(j, m), TEST))}
    for reg in ("bm25", "dense", "fc"):
        row[reg] = both(lambda j: score(V_any(j, m, reg), TEST))
    T1[m] = row
    P(f"| {m} | {cell(row['cb'], ci=True)} | {cell(row['bm25'])} | {cell(row['dense'])} | {cell(row['fc'])} |")
R["t1"] = T1

# ---- CB before/after
P("\n## 2. Closed-book before / after removing the locator row identifiers and re-calling empty answers (test, Claude; gpt in parens if different)")
P("| Model | before | after | items changed |"); P("|---|---|---|---|")
for m in list(OLD):
    b = both(lambda j: score(V_old(j, m, "cb"), TEST)); a = T1[m]["cb"]
    ch = len(set(new("claude", CB_TEST[m])) | (set(new("claude", "second_call_test")) if m == "gpt-5.5" else set()))
    P(f"| {m} | {cell(b, ci=True)} | {cell(a, ci=True)} | {ch} |")
v = dict(V_old("claude", "gpt-5.5", "cb")); v.update(new("claude", "e1_gpt-5.5_test")); s1 = score(v, TEST)
v2 = dict(V_old("claude", "gpt-5.5", "cb")); v2.update(new("claude", "second_call_test")); s2 = score(v2, TEST)
P(f"- gpt-5.5 components (Claude): row identifiers removed only {s1['plain']*100:.1f}% / {s1['cw']*100:.1f}%; empty retry only {s2['plain']*100:.1f}% / {s2['cw']*100:.1f}%.")
P("| Model | test 18 with row identifiers: before | after |"); P("|---|---|---|")
for m in CB_TEST:
    ids = set(new("claude", CB_TEST[m])); b = score(V_old("claude", m, "cb"), ids); a = score(new("claude", CB_TEST[m]), ids)
    P(f"| {m} | {b['hits']}/{b['n']} | {a['hits']}/{a['n']} |")
g63 = set(new("claude", "second_call_test")); a = score(new("claude", "second_call_test"), g63)
P(f"- gpt-5.5 empty-retry 63: correct after retry {a['hits']}/{a['n']} (all were wrong before).")

# ---- glm-5.2 thinking OFF and ON
P("\n## 3. glm-5.2 thinking OFF (original) vs ON, test")
P("| Regime | OFF | ON |"); P("|---|---|---|")
T3 = {}
for reg, tag in (("cb", "e2_glm-5.2_cb"), ("bm25", "e2_glm-5.2_bm25"), ("dense", "e2_glm-5.2_dense")):
    off = T1["glm-5.2"][reg]; on = both(lambda j: score(new(j, tag), TEST)); T3[reg] = {"off": off, "on": on}
    P(f"| {reg} | {cell(off, ci=(reg=='cb'))} | {cell(on, ci=(reg=='cb'))} |")
P("- minimax-m3 CB/BM25/dense were already thinking-on in the original runs (thinking trace in 389/389, 386/386, 386/386 calls).")
R["t3"] = T3

# ---- paired on the full-context samples fc98 / fc77
FC98 = {q for q in V_old("claude", "gpt-5.5", "fc") if q in TEST}
FC77 = {q for q in V_old("claude", "gpt-5.4-mini", "fc") if q in TEST}
P(f"\n## 4. Paired (Claude, Wilson 95% on counts) — fc98 n={len(FC98)}, fc77 n={len(FC77)}")
P("| Model | n | CB | BM25 | dense | FC |"); P("|---|---:|---|---|---|---|")
T4 = {}
rows = [(m, FC98) for m in ("gpt-4.1-mini", "gpt-5.5", "minimax-m3", "glm-5.2", "claude-sonnet-5-5", "claude-haiku-5-5")] + [(m, FC77) for m in ("gpt-5.4-mini", "gpt-5.4-nano")]
for m, ids in rows:
    cbv = both(lambda j: score(cb_corrected(j, m), ids))
    r = {"cb": cbv}; r.update({reg: both(lambda j: score(V_any(j, m, reg), ids)) for reg in ("bm25", "dense", "fc")})
    T4[m] = r
    P(f"| {m} | {len(ids)} | {cell(r['cb'], plainci=True)} | {cell(r['bm25'], plainci=True)} | {cell(r['dense'], plainci=True)} | {cell(r['fc'], plainci=True)} |")
on = {reg: both(lambda j: score(new(j, t), FC98)) for reg, t in (("cb", "e2_glm-5.2_cb"), ("bm25", "e2_glm-5.2_bm25"), ("dense", "e2_glm-5.2_dense"))}
P(f"| glm-5.2 thinking ON | {len(FC98)} | {cell(on['cb'], plainci=True)} | {cell(on['bm25'], plainci=True)} | {cell(on['dense'], plainci=True)} | (FC as above) |")
T4["glm-5.2_on"] = on; R["t4"] = T4
P("\n### 4b. Decomposition on fc98 (Claude plain %, FC minus BM25 in items)")
sub = {"all": FC98, "no512k": {q for q in FC98 if G[q].get("context_tier") != "512k"}, "no_agg": {q for q in FC98 if q not in AGG}}
P("| Model | subset | n | BM25 | FC | FC−BM25 (items) |"); P("|---|---|---:|---|---|---|")
T4b = {}
for m in ("gpt-4.1-mini", "gpt-5.5", "minimax-m3", "glm-5.2", "claude-sonnet-5-5", "claude-haiku-5-5"):
    for k, ids in sub.items():
        b = score(V_any("claude", m, "bm25"), ids); f = score(V_any("claude", m, "fc"), ids)
        T4b[f"{m}|{k}"] = (b["n"], b["hits"], f["hits"])
        P(f"| {m} | {k} | {b['n']} | {b['hits']} | {f['hits']} | {f['hits']-b['hits']:+d} |")
R["t4b"] = T4b

# ---- FC by tier
TIERS = ["32k", "64k", "128k", "256k", "512k"]
P("\n## 5. FC by tier (Claude, plain)")
P("| Model | " + " | ".join(TIERS) + " |"); P("|---|" + "---|" * len(TIERS))
for m in list(OLD) + list(NEWM):
    v = V_any("claude", m, "fc"); cells = []
    for t in TIERS:
        ids = {q for q in TEST if G[q].get("context_tier") == t}
        s = score(v, ids); cells.append(f"{s['plain']*100:.0f}% ({s['hits']}/{s['n']})" if s["n"] else "—")
    P(f"| {m} | " + " | ".join(cells) + " |")

# ---- agg81 512k full context
def agg_old(judge, m):
    tags = {"gpt-5.5": ["full_context_512k_rest_gpt-5.5", "full_context_512k_sample_gpt-5.5"],
            "gpt-4.1-mini": ["full_context_512k_rest_gpt-4.1-mini", "full_context_512k_sample_gpt-4.1-mini"],
            "minimax-m3": ["full_context_512k_rest_minimax-m3", "full_context_512k_sample_minimax-m3"],
            "glm-5.2": ["full_context_512k_rest_glm-5.2", "full_context_512k_sample_glm-5.2"]}[m]
    v = {}
    for t in tags: v.update(rel(JUD[judge], t) or {})
    return v
def agg_any(judge, m): return new(judge, f"{NEWM[m]}_agg512k") if m in NEWM else agg_old(judge, m)
AGGM = ["gpt-5.5", "gpt-4.1-mini", "glm-5.2", "minimax-m3", "claude-sonnet-5-5", "claude-haiku-5-5"]
P("\n## 6. Aggregation set (n=81), full context on the 512k bundle (Claude plain / cw)")
P("| Model | all 81 | HUG 72 | MOLIT 9 |"); P("|---|---|---|---|")
T6 = {}
for m in AGGM:
    r = {k: both(lambda j: score(agg_any(j, m), ids)) for k, ids in (("all", AGG), ("hug", HUG), ("molit", MOL))}
    T6[m] = r
    P(f"| {m} | {cell(r['all'])} | {r['hug']['claude']['hits']}/{r['hug']['claude']['n']} | {r['molit']['claude']['hits']}/{r['molit']['claude']['n']} |")
R["t6"] = T6

# ---- open-region and oracle contexts
def e3(judge, m, var):
    k = {"claude-sonnet-5-5": "sonnet55", "claude-haiku-5-5": "haiku55"}
    return new(judge, f"{k[m]}_{var}") if m in k else new(judge, f"{var}_{m}")
P("\n## 7. Aggregation 81 — full context (512k) vs oracle rows vs open-region rows (Claude plain; HUG/MOLIT counts)")
P("| Model | FC 512k | open-region (all rows of the year) | oracle (matching rows only) | FC H/M | region H/M | oracle H/M |"); P("|---|---|---|---|---|---|---|")
T7 = {}
for m in AGGM:
    fc = both(lambda j: score(agg_any(j, m), AGG)); rg = both(lambda j: score(e3(j, m, "e3region"), AGG)); orc = both(lambda j: score(e3(j, m, "e3oracle"), AGG))
    hm = lambda vfun: f"{score(vfun('claude'), HUG)['hits']}/{score(vfun('claude'), MOL)['hits']}"
    T7[m] = {"fc": fc, "region": rg, "oracle": orc}
    P(f"| {m} | {cell(fc, plainci=True)} | {cell(rg, plainci=True)} | {cell(orc, plainci=True)} | {hm(lambda j: agg_any(j, m))} | {hm(lambda j: e3(j, m, 'e3region'))} | {hm(lambda j: e3(j, m, 'e3oracle'))} |")
R["t7"] = T7

# ---- dev CB
CB_DEV = {m: f"e1_{m}_dev" for m in ["gpt-4.1-mini", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.5"]}
P("\n## 8. dev closed-book (and pooled dev+test), before / after (Claude plain / cw; gpt in parens)")
P("| Model | dev before (n) | dev after | pooled before | pooled after |"); P("|---|---|---|---|---|")
T8 = {}
for m in CB_DEV:
    vb = V_old("claude", m, "cb")
    def after(judge):
        v = dict(V_old(judge, m, "cb")); v.update(new(judge, CB_DEV[m])); v.update(new(judge, CB_TEST[m]))
        if m == "gpt-5.5": v.update(new(judge, "second_call_dev")); v.update(new(judge, "second_call_test"))
        return v
    db = both(lambda j: score(V_old(j, m, "cb"), DEV)); da = both(lambda j: score(after(j), DEV))
    pb = both(lambda j: score(V_old(j, m, "cb"), DEV | TEST)); pa = both(lambda j: score(after(j), DEV | TEST))
    T8[m] = {"dev_before": db, "dev_after": da, "pooled_before": pb, "pooled_after": pa}
    P(f"| {m} | {cell(db)} | {cell(da)} | {cell(pb)} | {cell(pa)} |")
g = set(new("claude", "second_call_dev")); s = score(new("claude", "second_call_dev"), g)
P(f"- gpt-5.5 dev empty-retry 233: correct after retry {s['hits']}/{s['n']}.")
R["t8"] = T8

# ---- judge agreement / self-judging check
def kappa(a, b):
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n; pa = sum(a) / n; pb = sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb); return po, ((po - pe) / (1 - pe) if pe < 1 else 1.0)
def owner(tag):
    if tag.startswith("sonnet55"): return "claude-sonnet-5-5"
    if tag.startswith("haiku55"): return "claude-haiku-5-5"
    if tag.startswith("second_call"): return "gpt-5.5"
    return tag.split("_")[1]
pairs = collections.defaultdict(list); dis = []
for tag in NEW["claude"]:
    for k, cv in NEW["claude"][tag].items():
        gv = NEW["gpt"][tag][k]; pairs[owner(tag)].append((cv, gv))
        if cv != gv: dis.append({"tag": tag, "split": k.split("|")[0], "qa_id": k.split("|")[1], "claude": "YES" if cv else "NO", "gpt-4.1-mini": "YES" if gv else "NO"})
oldpairs = collections.defaultdict(list)
for m, tags in OLD.items():
    for reg, t in tags.items():
        c = rel(JUD["claude"], t); g2 = rel(JUD["gpt"], t)
        for q in c:
            if q in TEST and q in g2: oldpairs[m].append((c[q], g2[q]))
P("\n## 9. Judge agreement (Claude vs gpt-4.1-mini)")
P("| Predictions from | set | n | agree | κ | Claude YES / gpt NO | Claude NO / gpt YES |"); P("|---|---|---:|---|---|---:|---:|")
T9 = {}
for label, src in (("new runs", pairs), ("original runs (test)", oldpairs)):
    for m, ps in sorted(src.items()):
        a = [x for x, _ in ps]; b = [y for _, y in ps]; po, k = kappa(a, b)
        cy = sum(1 for x, y in ps if x and not y); gy = sum(1 for x, y in ps if y and not x)
        T9[f"{label}|{m}"] = {"n": len(ps), "agree": po, "kappa": k, "claude_yes_gpt_no": cy, "claude_no_gpt_yes": gy}
        P(f"| {m} | {label} | {len(ps)} | {po*100:.1f}% | {k:.3f} | {cy} | {gy} |")
with open(OUT / "disagreements.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["tag", "split", "qa_id", "claude", "gpt-4.1-mini"]); w.writeheader(); w.writerows(dis)
P(f"- disagreements.csv: {len(dis)} rows.")
R["t9"] = T9
hid = []
for l in open(RESULTS / "runs/claude/thinking_responses.jsonl"):
    r = json.loads(l)
    reg, q = r["custom_id"].split("__"); hid.append((reg, q, r["output_tokens"]))
P("\n### Sonnet hidden-reasoning responses and their verdicts")
P("| regime | qa_id | output tokens | Claude | gpt |"); P("|---|---|---:|---|---|")
for reg, q, o in hid:
    tag = f"sonnet55_{reg}"; c = new("claude", tag).get(q); g2 = new("gpt", tag).get(q)
    P(f"| {reg} | {q} | {o} | {'YES' if c else 'NO'} | {'YES' if g2 else 'NO'} |")
R["hidden"] = hid
json.dump(R, open(OUT / "results.json", "w"), ensure_ascii=False, indent=1, default=str)
open(OUT / "tables.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
