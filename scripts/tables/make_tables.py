"""Generate the LaTeX result tables and the numbers quoted in the text.

Usage: python3 scripts/tables/make_tables.py <two-stage run dir> <out dir> [--exec <exec run dir>/main] [--same-day <same-day run dir>]
"""
import contextlib, io, math, runpy, sys
from pathlib import Path
SCRIPTS = Path(__file__).resolve().parents[1]
W, OUT = sys.argv[1], Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
EXEC_DIR = Path(sys.argv[sys.argv.index("--exec") + 1]) if "--exec" in sys.argv else None
SAME_DAY = Path(sys.argv[sys.argv.index("--same-day") + 1]) if "--same-day" in sys.argv else None
sys.argv = ["two_stage_analysis.py", W]
with contextlib.redirect_stdout(io.StringIO()):
    A = runpy.run_path(str(SCRIPTS / "two_stage/two_stage_analysis.py"))
B_ITEMS = ["krhlrb_v05_1798"] + [f"krhlrb_v05_{i}" for i in range(1800, 1807)]
B_NOTE = " 8 dev items of one announcement use verdicts from a same-day re-run (Table~S19)." if SAME_DAY else ""
B_REF = " The 8 dev items of one announcement as in Table~8." if SAME_DAY else ""
if SAME_DAY:
    import json as _json
    _ov = _json.load(open(SAME_DAY / "score" / "same_day_overrides.json"))
    assert sorted(_ov) == sorted(["kimi", "sonnet", "gpt-4.1-mini"]), sorted(_ov)
    for _j, _mv in _ov.items():
        assert sorted(_mv) == sorted(A["MODELS"]), (_j, sorted(_mv))
        for _m, _qv in _mv.items():
            assert sorted(_qv) == B_ITEMS, (_j, _m)
            _d = A["NEW"][_j][("twostage", _m)]
            assert all(q in _d for q in B_ITEMS), (_j, _m)
            for q in B_ITEMS: _d[q] = bool(_qv[q])
E, cell, MODELS, FULLWIN, T81, D144, NONOV, PAIR = A["E"], A["cell"], A["MODELS"], A["FULLWIN"], A["T81"], A["D144"], A["NONOV"], A["PAIR"]
JN = ["kimi", "sonnet", "gpt-4.1-mini"]
PJ, SJ = "kimi", "sonnet"
NAME = {m: f"\\texttt{{{m}}}" for m in MODELS}
ORDER = ["gpt-4.1-mini", "gpt-5.5", "minimax-m3", "glm-5.2", "claude-sonnet-5-5", "claude-haiku-5-5", "gpt-5.4-mini", "gpt-5.4-nano"]
NUM = []
pc = lambda k, n: 100 * k / n
r0 = lambda x: f"{x:.0f}" if round(x) >= 10 else f"\\phantom{{0}}{x:.0f}"


def acc(j, cond, m, qs):
    k, n, _ = cell(j, cond, m, qs)
    return (pc(k, n), k, n) if n else None


def wil_half(k, n, z=1.959963984540054):
    p = k / n; d = 1 + z * z / n; return 100 * z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d

# ---------- Table 7: four settings on the full-context sample, primary (kimi)
V, _, _ = E.load("full")
rows7, half7 = [], []
for m in ORDER:
    qs = sorted(V["claude-sonnet-5-5"][("fc", m)])
    cells = []
    for s in ("cb", "bm25", "dense", "fc"):
        k7 = {jn: sum(V[jn][(s, m)][q] for q in qs) for jn in ("claude-sonnet-5-5", "kimi")}
        a = {jn: 100 * k7[jn] / len(qs) for jn in k7}
        cells.append(f"{r0(a['kimi'])} ({a['claude-sonnet-5-5']:.0f})"); half7.append(wil_half(k7["kimi"], len(qs)))
    rows7.append((m, len(qs), cells))
L = ["\\begin{table}[t]", "\\caption{Plain accuracy (\\%) on the full-context sample in the four settings, with accuracy under the \\texttt{claude-sonnet-5-5} judge in parentheses.}\\label{tab:paired}",
     "\\centering", "\\small", "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrcccc@{}}", "\\toprule",
     "Model & $n$ & CB & BM25 & Dense & FC \\\\", "\\midrule"]
for m, n, c in rows7:
    if m == "gpt-5.4-mini": L.append("\\midrule")
    ns = ("\\phantom{0}" + str(n)) if m.startswith("gpt-5.4") else str(n)
    L.append(NAME[m] + " & " + ns + " & " + " & ".join(c) + " \\\\")
L += ["\\bottomrule", "\\end{tabular*}",
      "\\footnotetext{\\tablefootnotefont\\raggedright CB = closed-book; Dense = dense \\texttt{bge-m3} RAG; FC = full context. Lower block: the 272k-window models, 77 items. Wilson 95\\% intervals have half-widths of " + f"{min(half7):.0f} to {max(half7):.0f} points.}}",
      "\\end{table}"]
(OUT / "tab_paired.tex").write_text("\n".join(L) + "\n")

# ---------- Table 8: aggregation items under each evidence condition, primary kimi-k2.6 (claude-sonnet-5-5)
V0, _, _ = E.load("none")
FC_SECOND = 100 * sum(E.agg81(V0, PJ, "minimax-m3", "fc").values()) / 81
C8 = [("top5", "BM25"), ("dense", "Dense"), ("k50", "$k{=}50$"), ("twostage", "Two-stage"), ("fc", "FC"), ("openregion", "Region"), ("oracle", "Oracle")]
EXEC = {}
if EXEC_DIR is not None:
    sys.path.insert(0, str(SCRIPTS / "exec"))
    import exec_run as _T
    for m in ORDER:
        f = EXEC_DIR / "preds" / f"tool_{m}.jsonl"
        d = {r["qa_id"]: r for r in _T.jl(f)}
        assert set(T81) <= set(d), (m, "Exec. predictions missing")
        EXEC[m] = {q: _T.final_correct(d[q]) for q in T81}
    C8.insert(C8.index(("twostage", "Two-stage")) + 1, ("exec", "Exec."))
SUP = {("k50", "minimax-m3"): "$^{a}$", ("k50", "glm-5.2"): "$^{b}$", ("fc", "minimax-m3"): "$^{c}$"}
L = ["\\begin{table}[t]", "\\caption{Plain accuracy (\\%) on the aggregation items under each evidence condition" + (" and with program execution" if EXEC else "") + ", with accuracy under the \\texttt{claude-sonnet-5-5} judge in parentheses.}\\label{tab:agg}",
     "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{" + ("1.5pt" if EXEC else "2pt") + "}", "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l" + "c" * (len(C8) + 1) + "@{}}", "\\toprule",
     f" & \\multicolumn{{{len(C8)}}}{{c}}{{Test ($n{{=}}81$)}} & Dev$'$ ($n{{=}}81$) \\\\", f"\\cmidrule(lr){{2-{len(C8) + 1}}}\\cmidrule(l){{{len(C8) + 2}-{len(C8) + 2}}}",
     "Model & " + " & ".join(h + ("$^{\\dagger}$" if c == "twostage" else "") for c, h in C8) + " & Two-stage \\\\", "\\midrule"]
half = []
for m in ORDER:
    if m == "gpt-5.4-mini": L.append("\\midrule")
    cs = []
    for c, _ in C8:
        if c == "exec":
            k_ = sum(EXEC[m].values()); cs.append(r0(pc(k_, len(T81)))); half.append(wil_half(k_, len(T81))); continue
        a = acc(PJ, c, m, T81)
        if a is None: cs.append("--"); continue
        k = acc(SJ, c, m, T81)
        cs.append(f"{r0(a[0])}{SUP.get((c, m), '')} ({k[0]:.0f})"); half.append(wil_half(a[1], a[2]))
    a = acc(PJ, "twostage", m, NONOV); k = acc(SJ, "twostage", m, NONOV); half.append(wil_half(a[1], a[2]))
    cs.append(f"{r0(a[0])} ({k[0]:.0f})")
    L.append(f"{NAME[m]} & " + " & ".join(cs) + " \\\\")
L += ["\\bottomrule", "\\end{tabular*}",
      "\\footnotetext{\\tablefootnotefont\\raggedright BM25 = top-5, re-run with the two-stage runs (Table~S21); Dense = \\texttt{bge-m3} top-5; $k{=}50$ = 50 \\texttt{bge-m3} passages; Region = open-region context (Section~5.2; month for MOLIT). "
      "Dev$'$: Table~6 (all 144 dev items in Table~S21). Table~S13 gives bootstrap intervals. "
      + ("Exec.: Section~5.2, scored without a judge. " if EXEC else "")
      + "MOLIT row identifiers in the FC, $k{=}50$, and one dense prompt reveal count answers and average denominators (Section~S3). "
      "$^{a}$~Reasoning on, chosen after both runs; without reasoning 23\\% (\\texttt{claude-sonnet-5-5} only; Table~S19). $^{b}$~Includes two empty answers, scored wrong (Section~S5). "
      f"$^{{c}}$~Scored on first calls; with the second calls for four items (Table~S19), {FC_SECOND:.0f}\\%. "
      "$^{\\dagger}$~Region rule adopted after three versions were compared on the test items' gold regions; Section~5.2." + B_NOTE + "}", "\\end{table}"]
(OUT / "tab_agg.tex").write_text("\n".join(L) + "\n")
NUM.append(f"Table 8 max Wilson half-width {max(half):.2f}")

# ---------- SI ladder: three judges, test conditions incl. reported top-5, dev, dev non-overlap, pooled
CL = [("top5rep", "BM25$^{m}$"), ("top5", "BM25"), ("dense", "Dense"), ("k50", "$k{=}50$"), ("twostage", "2-st."), ("fc", "FC"), ("openregion", "Reg."), ("oracle", "Orac.")]
ABB = {"gpt-4.1-mini": "4.1-m", "gpt-5.4-mini": "5.4-m", "gpt-5.4-nano": "5.4-n", "gpt-5.5": "5.5", "minimax-m3": "mmx", "glm-5.2": "glm", "claude-sonnet-5-5": "son", "claude-haiku-5-5": "hai"}
JL = {"kimi": "Primary (\\texttt{kimi-k2.6})", "sonnet": "\\texttt{claude-sonnet-5-5}", "gpt-4.1-mini": "\\texttt{gpt-4.1-mini}"}
L = ["\\begin{table}[t]", "\\caption{Aggregation items: accuracy (\\%) under each evidence condition and judge.}\\label{tab:ladder}", "\\centering", "\\footnotesize",
     "\\setlength{\\tabcolsep}{2pt}", "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l" + "c" * 11 + "@{}}", "\\toprule",
     " & \\multicolumn{8}{c}{Test ($n{=}81$)} & \\multicolumn{3}{c}{Two-stage} \\\\", "\\cmidrule(lr){2-9}\\cmidrule(l){10-12}",
     "Model & " + " & ".join(h for _, h in CL) + " & Dev & dev$'$ & All \\\\"]
for j in JN:
    L += ["\\midrule", f"\\multicolumn{{12}}{{@{{}}l}}{{\\emph{{{JL[j]}}}}}\\\\"]
    for m in ORDER:
        cs = []
        for c, _ in CL:
            a = acc(j, c, m, T81); cs.append("--" if a is None else f"{a[0]:.0f}")
        for qs in (D144, NONOV, T81 + D144):
            cs.append(f"{acc(j, 'twostage', m, qs)[0]:.0f}")
        L.append(f"\\texttt{{{ABB[m]}}} & " + " & ".join(cs) + " \\\\")
L += ["\\bottomrule", "\\end{tabular*}",
      "\\footnotetext{\\tablefootnotefont\\raggedright BM25$^{m}$ = the BM25 top-5 run of the main runs (Tables~7 and~S9); BM25 = its re-run with the two-stage settings, as in Table~8. dev$'$ = the 81 dev items whose table cell no test item asks for; All = test and dev (225). "
      "BM25 top-5 answers none of the 144 dev items for any model and judge. The two-stage and re-run BM25 columns come from the two-stage runs; BM25$^{m}$ and the other conditions are main runs." + B_NOTE + "}", "\\end{table}"]
(OUT / "tab_ladder.tex").write_text("\n".join(L) + "\n")

# ---------- SI pairs: two-stage / top-5 per announcement-table pair, primary
TNAME = {"cbdc-cbdc-2025-jecheon-yeongcheon-happy-notice": "Chungbuk Dev.\\ Corp., Jecheon", "ih-ih-2025-cheonwon-jeonse-notice": "Incheon Urban Corp., Cheonwon",
         "jpdc-jpdc-2025-04-29-ildo-notice": "JPDC, Ildo", "jpdc-jpdc-2025-08-25-beophwan-notice": "JPDC, Beophwan", "lh-announce-0000061016-public-sale-2025q4": "LH 0000061016",
         "lh-announce-0000061086-public-sale-2026q2": "LH 0000061086", "lh-daedong2-b1-public-sale-20251017": "LH 0000060998 (Daedong-2)",
         "lh-siheung-hajung-a1-newlywed-hope-20260430": "LH 0000061078 (Siheung Hajung)"}
def dname(a):
    p = a.split("-")[0].upper()
    if a.startswith("lh-announce-"): return "LH " + a.split("-")[2]
    if a.startswith("sh-sh-"): return "SH " + a.split("-")[2]
    return p + " " + "-".join(a.split("-")[2:5]) + (" (private)" if "private" in a else " (public)" if "public-notice" in a else "")
pairs = sorted(set(PAIR.values()), key=lambda p: (p[0] not in {PAIR[q][0] for q in T81}, p[1] != "hug-sale-history", p[0]))
L = ["\\begin{table}[t]", "\\caption{Correct answers per announcement--table pair (9 items each) as two-stage / BM25 top-5, primary judge (\\texttt{kimi-k2.6}).}\\label{tab:tspairs}", "\\centering", "\\footnotesize",
     "\\setlength{\\tabcolsep}{2pt}", "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l" + "c" * 8 + "@{}}", "\\toprule",
     "Announcement & " + " & ".join(f"\\texttt{{{ABB[m]}}}" for m in ORDER) + " \\\\", "\\midrule"]
V2 = {m: (cell(PJ, "twostage", m, T81 + D144)[2], cell(PJ, "top5", m, T81 + D144)[2]) for m in ORDER}
hl = {m: [0, 0, 0] for m in ORDER}; sect = None
for p in pairs:
    s = ("Test" if p[0] in TNAME else "Dev") + (", MOLIT table" if p[1] != "hug-sale-history" else ", HUG table")
    if s != sect: L.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{{s}}}}}\\\\"); sect = s
    qs = [q for q in T81 + D144 if PAIR[q] == p]
    cs = []
    for m in ORDER:
        a, b = sum(V2[m][0][q] for q in qs), sum(V2[m][1][q] for q in qs); cs.append(f"{a}/{b}")
        hl[m][0 if a > b else 1 if a == b else 2] += 1
    L.append(f"\\hspace*{{1em}}{TNAME.get(p[0], dname(p[0]))} & " + " & ".join(cs) + " \\\\")
L += ["\\midrule", "Higher / same / lower & " + " & ".join("/".join(map(str, hl[m])) for m in ORDER) + " \\\\", "\\bottomrule", "\\end{tabular*}",
      "\\footnotetext{\\tablefootnotefont\\raggedright Higher / same / lower: pairs in which the two-stage baseline answers more, as many, or fewer items than BM25 top-5. Dev announcements by release identifier. BMC = Busan Metropolitan Corp.; DTCO, GJUCO = the Daejeon and Gwangju urban development corporations; GH = Gyeonggi Housing and Urban Development Corp.; SH = Seoul Housing and Communities Corp.; JPDC = Jeju Province Development Corp." + B_REF + "}", "\\end{table}"]
(OUT / "tab_tspairs.tex").write_text("\n".join(L) + "\n")
NUM.append("pairs higher/same/lower " + str(hl))

# ---------- SI MOLIT six items: correct on count 3 / avg 3, and accuracy on the other 75, primary
CNT = ["krhlrb_v05_0425", "krhlrb_v05_0428", "krhlrb_v05_0431"]; AVG = ["krhlrb_v05_0426", "krhlrb_v05_0429", "krhlrb_v05_0432"]
assert set(CNT + AVG) <= set(T81)
R75 = [q for q in T81 if q not in CNT + AVG]
CM = [("top5", "BM25"), ("dense", "Dense"), ("k50", "$k{=}50$"), ("twostage", "Two-stage"), ("fc", "FC"), ("openregion", "Region"), ("oracle", "Oracle")]
L = ["\\begin{table}[t]", "\\caption{The six MOLIT count and average items: correct count / average items and accuracy (\\%) on the other 75 aggregation items, primary judge (\\texttt{kimi-k2.6}).}\\label{tab:molit6}", "\\centering", "\\footnotesize",
     "\\setlength{\\tabcolsep}{2pt}", "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l" + "c" * 7 + "@{}}", "\\toprule", "Model & " + " & ".join(h for _, h in CM) + " \\\\", "\\midrule"]
for m in ORDER:
    if m == "gpt-5.4-mini": L.append("\\midrule")
    cs = []
    for c, _ in CM:
        a = acc(PJ, c, m, T81)
        if a is None: cs.append("--"); continue
        v = cell(PJ, c, m, T81)[2]
        cs.append(f"{sum(v[q] for q in CNT)}/{sum(v[q] for q in AVG)}, {pc(sum(v[q] for q in R75), 75):.0f}")
    L.append(f"\\texttt{{{ABB[m]}}} & " + " & ".join(cs) + " \\\\")
L += ["\\bottomrule", "\\end{tabular*}", "\\footnotetext{\\tablefootnotefont\\raggedright Cells: correct count items (of 3) / correct average items (of 3), accuracy on the other 75.}", "\\end{table}"]
(OUT / "tab_molit6.tex").write_text("\n".join(L) + "\n")

# ---------- Table S13: Wilson, pair and cluster bootstrap intervals (method of analysis/si_cells.py and analysis/cluster_bootstrap.py)
import random as _rnd
from collections import defaultdict as _dd
G = A["G"]
UPAIR = lambda q: (G[q]["announcement_ids"][0], tuple(G[q]["source_ids"])); UCLU = lambda q: G[q]["cluster_id"]


def boot(qs, unit, v, B=10000, seed=20261011):
    groups = _dd(list)
    for q in qs: groups[unit(q)].append(q)
    keys = sorted(groups); rnd = _rnd.Random(seed)
    out = sorted((lambda s: sum(v[q] for q in s) / len(s))([q for k in (rnd.choice(keys) for _ in keys) for q in groups[k]]) for _ in range(B))
    return 100 * out[int(0.025 * B)], 100 * out[int(0.975 * B) - 1]


iv = lambda x: f"[{x[0]:.0f}, {x[1]:.0f}]"
CB = [("top5", "BM25"), ("dense", "Dense"), ("k50", "$k{=}50$"), ("twostage", "2-st.")] + ([("exec", "Exec.")] if EXEC else []) + \
     [("fc", "FC"), ("openregion", "Reg."), ("oracle", "Orac.")]
SORDER = ["gpt-4.1-mini", "gpt-5.5", "minimax-m3", "glm-5.2", "claude-sonnet-5-5", "claude-haiku-5-5", "gpt-5.4-mini", "gpt-5.4-nano"]
_cs = _dd(int)
for q in T81: _cs[UCLU(q)] += 1
_pd = {UPAIR(q) for q in NONOV}; _cd = {UCLU(q) for q in NONOV}


def vec(m, c, qs):
    if c == "exec": return EXEC[m]
    k, n, vv = cell(PJ, c, m, qs)
    if n == 0: return None
    assert n == len(qs), (m, c, n)
    return {q: vv[q] for q in qs}


L = ["\\begin{table}[t]", "\\caption{Aggregation items: accuracy (\\%) under the conditions of Table~8 (upper row) with 95\\% bootstrap intervals (lower row).}\\label{tab:aggboot}",
     "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{2pt}", "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l" + "c" * (len(CB) + 1) + "@{}}", "\\toprule",
     f" & \\multicolumn{{{len(CB)}}}{{c}}{{Test ($n{{=}}{len(T81)}$), cluster interval}} & Dev$'$ ($n{{=}}{len(NONOV)}$) \\\\",
     f"\\cmidrule(lr){{2-{len(CB) + 1}}}\\cmidrule(l){{{len(CB) + 2}-{len(CB) + 2}}}",
     "Model & " + " & ".join(h for _, h in CB) + " & 2-st., pair \\\\", "\\midrule"]
for m in SORDER:
    top, low = [], []
    for c, _ in CB:
        v = vec(m, c, T81)
        if v is None: top.append("--"); low.append(""); continue
        top.append(f"{100 * sum(v.values()) / len(T81):.0f}"); low.append(iv(boot(T81, UCLU, v)))
    v = vec(m, "twostage", NONOV); top.append(f"{100 * sum(v.values()) / len(NONOV):.0f}"); low.append(iv(boot(NONOV, UPAIR, v)))
    if m == "gpt-5.4-mini": L.append("\\midrule")
    L.append(f"\\texttt{{{ABB[m]}}} & " + " & ".join(top) + " \\\\")
    L.append(" & " + " & ".join(low) + " \\\\[2pt]")
L += ["\\bottomrule", "\\end{tabular*}",
      "\\footnotetext{\\tablefootnotefont\\raggedright Primary judge (\\texttt{kimi-k2.6})" + ("; Exec.: deterministic scoring" if EXEC else "") + f". Test: bootstrap over the {len(_cs)} near-duplicate clusters "
      f"(sizes {', '.join(map(str, sorted(_cs.values(), reverse=True)))}); dev$'$: bootstrap over its {len(_pd)} announcement--table pairs ({len(_cd)} clusters). "
      "10{,}000 resamples, seed 20261011. BM25 = BM25 top-5, re-run with the two-stage runs." + B_REF + "}", "\\end{table}"]
(OUT / "tab_aggboot.tex").write_text("\n".join(L) + "\n")

# ---------- numbers quoted in the prose
four = ["gpt-5.5", "minimax-m3", "glm-5.2", "claude-sonnet-5-5"]
rng = lambda c, qs, ms, j=PJ: (min(acc(j, c, m, qs)[0] for m in ms), max(acc(j, c, m, qs)[0] for m in ms))
for lab, c, qs, ms in [("two-stage test, 8", "twostage", T81, MODELS), ("two-stage test, four", "twostage", T81, four), ("two-stage dev, 8", "twostage", D144, MODELS),
                       ("two-stage dev, four", "twostage", D144, four), ("top5 test, 8", "top5", T81, MODELS), ("top5 dev, 8", "top5", D144, MODELS),
                       ("dense test, 8", "dense", T81, MODELS), ("k50 test, four", "k50", T81, four), ("fc test, 6", "fc", T81, FULLWIN), ("fc test, four", "fc", T81, four),
                       ("openregion test, four", "openregion", T81, four), ("oracle test, four", "oracle", T81, four)]:
    for j in JN:
        a, b = rng(c, qs, ms, j); NUM.append(f"{lab} [{j}]: {a:.1f}-{b:.1f}")
for m in ["claude-haiku-5-5", "gpt-4.1-mini", "gpt-5.4-mini", "gpt-5.4-nano"]:
    for c, qs in (("twostage", T81), ("twostage", D144), ("openregion", T81), ("oracle", T81)):
        a = acc(PJ, c, m, qs)
        if a: NUM.append(f"{m} {c} {'test' if qs is T81 else 'dev'}: {a[0]:.1f} ({a[1]}/{a[2]})")
(OUT / "numbers.txt").write_text("\n".join(NUM) + "\n")
print("written", sorted(p.name for p in OUT.iterdir()))
