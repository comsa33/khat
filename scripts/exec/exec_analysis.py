"""Filter-execution results against the two-stage results (writes local/runs/exec/exec_analysis.txt).

Usage: python3 scripts/exec/exec_analysis.py <exec run dir>/main [--two-stage <two-stage run dir>]
"""
import collections, contextlib, io, json, runpy, statistics, sys, unicodedata
from pathlib import Path
from khat.paths import RESULTS, LOCAL
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "two_stage"))
import exec_run as runT, two_stage_run as runR, exec_build as B, exec_score as S

STRONG = ["gpt-5.5", "minimax-m3", "glm-5.2", "claude-sonnet-5-5"]
ORDER = ["gpt-4.1-mini", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.5", "minimax-m3", "glm-5.2", "claude-sonnet-5-5", "claude-haiku-5-5"]
nfc = lambda s: unicodedata.normalize("NFC", str(s)).strip()


def g_values(two_stage):
    sys.argv = ["two_stage_analysis.py", str(two_stage)]
    with contextlib.redirect_stdout(io.StringIO()):
        A = runpy.run_path(str(HERE.parent / "two_stage/two_stage_analysis.py"))
    out = {}
    for split, qs in (("test", A["T81"]), ("dev", A["NONOV"])):
        out[split] = {}
        for m in ORDER:
            k, n, v = A["cell"]("sonnet", "twostage", m, qs)
            assert n == 81, (m, split, n)
            out[split][m] = (100 * k / n, {q: v[q] for q in qs})
    assert sorted(A["T81"]) == B.T81 and sorted(A["NONOV"]) == B.DEVP
    return out


def one_filter(query, col):
    fs = [f for f in (query or {}).get("filters") or [] if isinstance(f, dict) and nfc(f.get("column", "")) == col]
    if len(fs) != 1: return None
    f = fs[0]; op = nfc(f.get("op", ""))
    if op == "=": return f.get("value")
    if op == "in" and isinstance(f.get("value"), list) and len(f["value"]) == 1: return f["value"][0]
    return None


def region_match(r, gold):
    if r["status"] != "executed": return False
    v = one_filter(r["query"], "지역")
    return v is not None and nfc(v) == nfc(gold)


def period_match(r, gp):
    if r["status"] != "executed": return False
    col, gv = ("연도", gp["filter"]["_query_year"]) if "_query_year" in gp["filter"] else ("거래연월", gp["filter"]["_deal_ymd"])
    v = one_filter(r["query"], col)
    try: return v is not None and float(str(v).replace(",", "")) == float(gv)
    except ValueError: return False


def median(xs):
    return statistics.median(xs)


def cat(d):
    return "above" if d >= 5 else "below" if d <= -5 else "close"


def direction(m):
    return "+" if m >= 5 else "-" if m <= -5 else "0"


def rng(xs):
    return f"{min(xs):.0f}–{max(xs):.0f}"


def run(Sdir, two_stage):
    Sdir = Path(Sdir); man = json.load(open(Sdir / "manifest.json"))
    assert man["set"] == "main", "only the main set is scored"
    G, _, _ = S._gold_src(); g = g_values(two_stage); out = []
    P = {m: {r["qa_id"]: r for r in runT.jl(Sdir / "preds" / f"tool_{m}.jsonl")} for m in ORDER}
    for m in ORDER: assert set(P[m]) == set(B.ITEMS), (m, "predictions missing")
    split_items = {"test": B.T81, "dev": B.DEVP}
    u = {s: {m: 100 * sum(runT.final_correct(P[m][q]) for q in qs) / len(qs) for m in ORDER} for s, qs in split_items.items()}
    delta = {s: {m: u[s][m] - g[s][m][0] for m in ORDER} for s in split_items}
    out.append("== u, g, delta (points, unrounded)")
    for s in split_items:
        for m in ORDER: out.append(f"  {s:4s} {m:18s} u {u[s][m]:5.1f}  g {g[s][m][0]:5.1f}  delta {delta[s][m]:+6.1f}  {cat(delta[s][m])}")
    cats = collections.Counter(cat(delta["test"][m]) for m in STRONG)
    maj = next((c for c, n in cats.items() if n >= 3), None)
    X = rng([u["test"][m] for m in STRONG]); D = rng([u["dev"][m] for m in STRONG]); A_ = rng([u["test"][m] for m in ORDER if m not in STRONG])
    lead = "Given the table schema and an executable filter format, executing the filter a model writes reaches"
    s52 = {"above": f"{lead} {X}% for these four models, above the two-stage baseline.", "close": f"{lead} {X}% for these four models, close to the two-stage baseline.",
           "below": f"{lead} {X}% for these four models, below the two-stage baseline.", None: f"{lead} {X}% for these four models (two-stage 84–86%)."}[maj]
    mt, md = median(list(delta["test"].values())), median(list(delta["dev"].values()))
    dt, dd = direction(mt), direction(md)
    out += ["", "== sentences in the text", f"  strong-four categories (test): {dict(cats)}; majority: {maj or 'none'}", f"  main: {s52}",
            f"  other four: For the other four models it reaches {A_}% (two-stage 0–25%).",
            f"  dev': On the 81 dev′ items it reaches {D}% for the four models" + (", unlike on the test items." if dt != dd else ".") ,
            f"  direction: test median delta {mt:+.2f} -> {dt}; dev' median delta {md:+.2f} -> {dd}",
            "  conclusion: " + (f"Executing a model-written filter over the table reaches {X}% for the same four models without placing rows in the prompt." if maj in ("above", "close") else "(none)")]
    out += ["", "== region filter match (HUG items), %"]
    rm = {}
    for s, qs in split_items.items():
        hug = [q for q in qs if G[q]["gold_predicate"]["source"] == "hug-sale-history"]
        rm[s] = {m: 100 * sum(region_match(P[m][q], G[q]["gold_predicate"]["filter"]["_query_area_name"]) for q in hug) / len(hug) for m in ORDER}
        med = median([rm[s][m] for m in STRONG])
        out.append(f"  {s} (n={len(hug)}): " + ", ".join(f"{m} {rm[s][m]:.1f}" for m in ORDER) + f"; strong-four median {med:.1f}")
    out += ["", "== final failure types per model (162 items)"]
    for m in ORDER:
        c = collections.Counter(P[m][q]["status"] if P[m][q]["status"] != "executed" else ("no_value" if P[m][q]["value"] is None else "executed") for q in B.ITEMS)
        pe = c["parse_fail"] + c["exec_error"]
        out.append(f"  {m:18s} parse_fail+exec_error {pe} ({100 * pe / 162:.1f}%); "
                   f"empty_output {c['empty_output']}; api_failed {c['api_failed']}; no_value {c['no_value']}; re-asked {sum(P[m][q]['reasked'] for q in B.ITEMS)}")
    out += ["", "== component match rates (all 162 items), %: table, region (HUG), period, operator, field (sum/avg/max)"]
    for m in ORDER:
        rs = [P[m][q] for q in B.ITEMS]; gp = {q: G[q]["gold_predicate"] for q in B.ITEMS}
        tab = sum(r["status"] == "executed" and nfc(r["query"].get("table")) == ("hug" if gp[r["qa_id"]]["source"] == "hug-sale-history" else "molit") for r in rs)
        per = sum(period_match(r, gp[r["qa_id"]]) for r in rs)
        op = sum(r["status"] == "executed" and nfc(r["query"].get("operator")) == gp[r["qa_id"]]["op"] for r in rs)
        fq = [r for r in rs if gp[r["qa_id"]]["op"] != "count"]
        fmap = {"TOT_HOCO": "총세대수(TOT_HOCO)", "dealAmount_manwon": "거래금액"}
        fld = sum(r["status"] == "executed" and nfc(r["query"].get("field") or "") == fmap[gp[r["qa_id"]]["field"]] for r in fq)
        hug = [q for q in B.ITEMS if gp[q]["source"] == "hug-sale-history"]
        reg = sum(region_match(P[m][q], gp[q]["filter"]["_query_area_name"]) for q in hug)
        out.append(f"  {m:18s} table {100 * tab / 162:.1f}; region {100 * reg / len(hug):.1f}; period {100 * per / 162:.1f}; operator {100 * op / 162:.1f}; field {100 * fld / len(fq):.1f}")
    out += ["", "== pairs: filter execution answers more / as many / fewer items than two-stage"]
    pair = lambda q: (G[q]["announcement_ids"][0], tuple(G[q]["source_ids"]))
    for s, qs in split_items.items():
        ps = sorted({pair(q) for q in qs}); line = []
        for m in ORDER:
            hsl = [0, 0, 0]
            for p in ps:
                pq = [q for q in qs if pair(q) == p]
                a = sum(runT.final_correct(P[m][q]) for q in pq); b = sum(g[s][m][1][q] for q in pq)
                hsl[0 if a > b else 1 if a == b else 2] += 1
            line.append(f"{m} {'/'.join(map(str, hsl))}")
        out.append(f"  {s} ({len(ps)} pairs): " + "; ".join(line))
    (LOCAL / "runs/exec").mkdir(parents=True, exist_ok=True)
    (LOCAL / "runs/exec/exec_analysis.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    pr = Path(sys.argv[sys.argv.index("--two-stage") + 1]) if "--two-stage" in sys.argv else RESULTS / "two_stage"
    print("\n".join(run(sys.argv[1], pr)))
