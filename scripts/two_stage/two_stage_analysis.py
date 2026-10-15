"""Two-stage and BM25 top-5 results: accuracy per judge, model and split, pair counts and diagnostics.

Usage: python3 scripts/two_stage/two_stage_analysis.py <two-stage run dir>
"""
import csv, json, math, random, re, statistics, sys, collections
from pathlib import Path
W = Path(sys.argv[1]); SELF = "--selftest" in sys.argv; TESTONLY = "--selftest-g55-testonly" in sys.argv
_bad = [a for a in sys.argv[2:] if a not in ("--selftest", "--selftest-g55-testonly")]
if _bad or (TESTONLY and not SELF):
    sys.exit(f"unknown or invalid arguments {_bad or ['--selftest-g55-testonly without --selftest']}; usage: two_stage_analysis.py <two-stage run dir> [--selftest [--selftest-g55-testonly]]")
MAIN_JUDGE = "sonnet"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "judges")); sys.path.insert(0, str(HERE))
import panel_verdicts as E
import judge_parse as JP
DS = E.R
from khat.paths import DATA, LOCAL
sys.path.insert(0, str(HERE.parent))
import table_rows as q3
MODELS = ["gpt-4.1-mini", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.5", "minimax-m3", "glm-5.2", "claude-sonnet-5-5", "claude-haiku-5-5"]
FULLWIN = ["gpt-4.1-mini", "gpt-5.5", "minimax-m3", "glm-5.2", "claude-sonnet-5-5", "claude-haiku-5-5"]
JN = [MAIN_JUDGE] + [j for j in JP.JUDGES if j != MAIN_JUDGE]
LOADER = {"sonnet": "claude-sonnet-5-5", "gpt-4.1-mini": "gpt-4.1-mini", "kimi": "kimi"}
ids = lambda f: [l.strip() for l in open(DATA / "item_sets" / f) if l.strip()]
T81, D144 = ids("items_test81.txt"), ids("items_dev144.txt")
NONOV = [r["qa_id"] for r in csv.DictReader(open(DATA / "item_sets" / "dev_cells.tsv", encoding="utf-8"), delimiter="\t") if r["same_cell_as_test"] == "0"]
G = {}
for f in ("qa_test_public.jsonl", "qa_dev.jsonl"):
    for l in open(DS / "data" / f, encoding="utf-8"):
        r = json.loads(l); G[r["qa_id"]] = r
PAIR = {q: (G[q]["announcement_ids"][0], G[q]["gold_predicate"]["source"]) for q in T81 + D144}
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
WRONG = {"failed", "still_empty", "empty_retry_pending_or_failed"}


def wilson(k, n, z=1.959963984540054):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * (c - h) / d, 100 * (c + h) / d


# ---- new verdicts NEW[judge][(condition, model)][qa_id]; condition in twostage, devtop5, bm25test
NEW = {j: collections.defaultdict(dict) for j in JN}
COND = {"twostage": T81 + D144, "devtop5": D144, "bm25test": T81}
if SELF:
    rnd = random.Random(20261013)
    for c, qs in COND.items():
        for m in MODELS:
            for q in qs:
                if TESTONLY and m == "gpt-5.5" and q in D144: continue
                v = rnd.random() < {"twostage": 0.7, "devtop5": 0.15, "bm25test": 0.1}[c]
                for j in JN: NEW[j][(c, m)][q] = v if rnd.random() > 0.02 else not v
else:
    J = W / "main/judge"; idx = {i["custom_id"]: i for i in jl(J / "index.jsonl")}; V = JP.load(J)
    for j in JN:
        rd = JP.rounds(J, j)
        if V["missing"][j] and rd < JP.MAX_ROUNDS:
            sys.exit(f"{len(V['missing'][j])} {j} verdicts missing after {rd} resubmission rounds; run judge_parse.py missing first")
        print(f"verdicts missing for {j}: {len(V['missing'][j])}" + (" (final after 2 resubmissions; counted incorrect)" if V["missing"][j] else ""))
    for cid, i in idx.items():
        for j in JN:
            NEW[j][(i["condition"], i["model"])][i["qa_id"]] = False if i["status"] in WRONG else V[j].get(cid, False)
DEV_MODELS = [m for m in MODELS if set(NEW[MAIN_JUDGE].get(("twostage", m), {})) >= set(D144)]
for j in JN:
    for c, qs in COND.items():
        for m in MODELS:
            need = set(qs) if (c == "bm25test" or m in DEV_MODELS) else set(T81) & set(qs)
            assert set(NEW[j][(c, m)]) == need, (j, c, m)

# ---- reported test conditions REP[(judge, setting, model)] = {qa_id: bool}; FC must exist for every full-window model
V0, _, _ = E.load("full")
KX = {}
if (W / "judge_kimi/kimi_existing_R.jsonl").exists():
    for r in jl(W / "judge_kimi/kimi_existing_R.jsonl"):
        q, tag = r["qa_id"].split("__"); s, m = tag.split("_", 1)
        assert r["judge_correct"] is not None, r["qa_id"]; KX[(s, m, q)] = bool(r["judge_correct"])
REP = {}
for j in JN:
    for m in MODELS:
        for s, key in (("top5rep", "bm25"), ("dense", "dense"), ("fc", "fc"), ("k50", "k50"), ("openregion", "openregion"), ("oracle", "oracle")):
            if m not in FULLWIN and key not in ("bm25", "dense"): continue
            v = E.agg81(V0, LOADER[j], m, key)
            if j == "kimi" and KX:
                v = {q: KX[(key, m, q)] for q in v}
            REP[(j, s, m)] = v
for m in FULLWIN:
    assert all((j, "fc", m) in REP and len(REP[(j, "fc", m)]) == 81 for j in JN), m


def cell(j, cond, m, qs):
    if cond == "twostage": v = NEW[j][("twostage", m)]
    elif cond == "top5": v = {**NEW[j][("bm25test", m)], **NEW[j][("devtop5", m)]}
    else: v = REP.get((j, cond, m), {})
    qs = [q for q in qs if q in v]
    return sum(v[q] for q in qs), len(qs), v


tag = "SELFTEST " if SELF else ""
pc = lambda k, n: 100 * k / n
print(f"== {tag}judge: {MAIN_JUDGE} (columns: {', '.join(JN[1:])}); dev models: {len(DEV_MODELS)}"
      + ("" if "gpt-5.5" in DEV_MODELS else " (gpt-5.5: test only)"))
print(f"== {tag}(1) accuracy % [Wilson 95%] (correct/n); top5 = new BM25 top-5, top5rep = BM25 top-5 of the main runs")
SPL = (("test", T81), ("dev", D144), ("dev-nonoverlap", NONOV), ("pooled", T81 + D144))
for j in JN:
    print(f"-- judge {j}")
    for m in MODELS:
        parts = []
        for cond in ("top5", "top5rep", "twostage", "dense", "k50", "fc", "openregion", "oracle"):
            for split, qs in SPL:
                if cond not in ("top5", "twostage") and split != "test": continue
                if split != "test" and m not in DEV_MODELS: continue
                k, n, _ = cell(j, cond, m, qs)
                if n: lo, hi = wilson(k, n); parts.append(f"{cond}/{split} {pc(k, n):.1f} [{lo:.1f}, {hi:.1f}] ({k}/{n})")
        print(f"  {m}: " + " | ".join(parts))

print(f"\n== {tag}(2) pair level ({MAIN_JUDGE}): two-stage / top-5 correct per pair; higher / same / lower pairs")
pairs = sorted(set(PAIR.values()), key=lambda p: (p[0] not in {PAIR[q][0] for q in T81}, p))
for m in MODELS:
    _, _, vt = cell(MAIN_JUDGE, "top5", m, T81 + D144); _, _, vg = cell(MAIN_JUDGE, "twostage", m, T81 + D144)
    hi = same = lo = 0; row = []
    for p in pairs:
        qs = [q for q in T81 + D144 if PAIR[q] == p and q in vg]
        if not qs: continue
        a, b = sum(vg[q] for q in qs), sum(vt[q] for q in qs)
        hi += a > b; same += a == b; lo += a < b; row.append(f"{a}/{b}")
    print(f"  {m}: higher {hi} / same {same} / lower {lo} over {hi + same + lo} pairs | " + " ".join(row))

print(f"\n== {tag}(3) classification (test 81, {MAIN_JUDGE}; t = new BM25 top-5)")
cls, gvals, tvals, fvals = {}, {}, {}, {}
for m in MODELS:
    t = cell(MAIN_JUDGE, "top5", m, T81); g = cell(MAIN_JUDGE, "twostage", m, T81); tr = cell(MAIN_JUDGE, "top5rep", m, T81)
    gvals[m], tvals[m] = pc(g[0], g[1]), pc(t[0], t[1])
    if m not in FULLWIN:
        cls[m] = ("not classified (no full-context run)", None); print(f"  {m}: t {tvals[m]:.1f} (reported {pc(*tr[:2]):.1f}); g {gvals[m]:.1f}; {cls[m][0]}"); continue
    f = cell(MAIN_JUDGE, "fc", m, T81); fvals[m] = pc(f[0], f[1])
    if fvals[m] - tvals[m] < 10:
        cls[m] = (f"not classified (f - t = {fvals[m] - tvals[m]:.1f} < 10)", None)
    else:
        r = (gvals[m] - tvals[m]) / (fvals[m] - tvals[m])
        cls[m] = ("FC or above" if r >= 1 else "FC-near" if r >= 0.8 else "top5-near" if r <= 0.2 else "middle", r)
    c, r = cls[m]
    print(f"  {m}: t {tvals[m]:.1f} (reported {pc(*tr[:2]):.1f}); g {gvals[m]:.1f}; f {fvals[m]:.1f}; {c}" + (f"; r = {r:.2f}" if r is not None else ""))
lab = [c for c, r in cls.values() if r is not None]; M = len(lab); cnt = collections.Counter(lab)
fcside, mid, low = cnt["FC or above"] + cnt["FC-near"], cnt["middle"], cnt["top5-near"]
X, Y = min(gvals.values()), max(gvals.values())
if M == 0:
    s = (f"A two-stage baseline reaches {X:.0f}–{Y:.0f}% (top-5 retrieval {min(tvals.values()):.0f}–{max(tvals.values()):.0f}%, "
         f"full context {min(fvals.values()):.0f}–{max(fvals.values()):.0f}%).")
elif fcside * 2 > M:
    s = (f"A two-stage baseline that retrieves ten document passages and then twenty table passages with a query expanded by the "
         f"region and year it extracts reaches {X:.0f}–{Y:.0f}%, close to or above the full context for {fcside} of {M} models.")
elif low * 2 > M:
    s = f"A two-stage baseline reaches only {X:.0f}–{Y:.0f}%, so the aggregation set stays hard for retrieval baselines."
else:
    s = f"A two-stage baseline reaches {X:.0f}–{Y:.0f}%, between top-5 retrieval and the full context."
print(f"  classified M = {M}: {dict(cnt)}\n  abstract sentence: {s}")
sign = lambda x: "+" if x > 0 else "-" if x < 0 else "0"
dt_ = lambda m, qs, c: pc(*cell(MAIN_JUDGE, c, m, qs)[:2])
test_dir = sign(statistics.median(gvals[m] - tvals[m] for m in MODELS))
dg = {m: dt_(m, D144, "twostage") for m in DEV_MODELS}; dtp = {m: dt_(m, D144, "top5") for m in DEV_MODELS}
dev_dir = sign(statistics.median(dg[m] - dtp[m] for m in DEV_MODELS))
ds = (f"On 144 aggregation items from 16 other announcements, the two-stage baseline reaches {min(dg.values()):.0f}–{max(dg.values()):.0f}% "
      f"and top-5 retrieval {min(dtp.values()):.0f}–{max(dtp.values()):.0f}%" + (", unlike on the test items." if dev_dir != test_dir else "."))
print(f"  direction (sign of the median over models of g - t): test {test_dir}, dev {dev_dir}\n  dev sentence: {ds}"
      + ("" if "gpt-5.5" in DEV_MODELS else "  [gpt-5.5: test only]"))
keep = M > 0 and (fcside + mid) * 2 > M
print(f"  practical recommendation (retrieve document passages and the table separately): {'keep' if keep else 'delete'}")

if SELF:
    print("\n== SELFTEST: diagnostics skipped (they compare extraction with the gold filter and run only after the real runs)")
    sys.exit(0)
print("\n== (4) diagnostics (no model call)")
ex = {r["qa_id"]: r for r in jl(W / "extract.jsonl")}
loc = {r["qa_id"]: r["new_locator"] for r in csv.DictReader(open(DS / "data/errata/dev_review.csv", encoding="utf-8"))
       if r["status"] == "locator_fixed" and r.get("new_locator")}
import retrieval as RS
BT = (LOCAL / "bundles/mix_multiprovider_512k.txt").read_text(encoding="utf-8")
ch = RS.split_chunks(BT, 1200)
MK = list(re.finditer(r"\[[^\]\n]*?\s([^\s\]]+-p\d{3}) \(p\.\d+\)\]", BT))
SPANS = [(m.group(1), m.start(), MK[i + 1].start() if i + 1 < len(MK) else len(BT)) for i, m in enumerate(MK)]


def pages_of(i):
    off = BT.find(ch[i][1]); assert off >= 0, i
    return {p for p, a, b in SPANS if a < off + len(ch[i][1]) and off < b}


def gold_pages(q):
    if q in loc:
        return {x.split("#")[0].strip() for x in re.split(r"[;|,]", loc[q].split("->")[-1]) if x.strip()}
    return set(G[q].get("page_ids") or [])


def gold_region(q):
    f = G[q]["gold_predicate"]["filter"]
    return f.get("_query_area_name") or f.get("_lawd_name", "")


def gold_period(q):
    f = G[q]["gold_predicate"]["filter"]
    if "_deal_ymd" in f: return f"{f['_deal_ymd'][:4]}년 {int(f['_deal_ymd'][4:])}월"
    return f"{f['_query_year']}년"


def rendered(src, row):
    if src == "hug-sale-history":
        return " | ".join([row.get("_query_area_name", ""), row.get("_query_year", ""), row.get("BSU_NM", ""), str(row.get("TOT_HOCO", "")), str(row.get("GNRL_SILT_HOCO", ""))])
    return f"{row['_lawd_name']} {row['umdNm']} | {row['aptNm']} | {row['excluUseAr']}㎡ {row['dealAmount']}만원 | {row['_deal_ymd']}"


HUG_RE = re.compile(r"^[가-힣]+ \| 20\d\d \| .+ \| \d+ \| \d+$"); MOL_RE = re.compile(r"^MOLIT-APTTRADE-")
def table_of(i):
    ls = [l for l in ch[i][1].splitlines() if l.strip()]
    return "hug" if sum(bool(HUG_RE.match(l)) for l in ls) * 2 >= len(ls) else "molit" if sum(bool(MOL_RE.match(l)) for l in ls) * 2 >= len(ls) else None
TAB = collections.Counter(table_of(i) for i in range(len(ch)) if ch[i][0] == "preamble-p000")
HDR = {i for i in range(len(ch)) if "HUG 분양이력 자료" in ch[i][1] or "행ID | 시군구" in ch[i][1]}
for split, qs in (("test", T81), ("dev", D144)):
    hit = sum(gold_pages(q) <= set().union(*(pages_of(i) for i in ex[q]["stage1_ids"])) for q in qs)
    reg = sum(bool(ex[q]["region"]) and ex[q]["region"] == gold_region(q) for q in qs)
    per = sum(ex[q]["period"] == gold_period(q) for q in qs)
    cov, share, hdr = 0, [], 0
    for q in qs:
        lines = {rendered(G[q]["gold_predicate"]["source"], r) for r in q3.select(G[q]["gold_predicate"])}
        txt = "\n".join(re.sub(r"MOLIT-APTTRADE-\d{6}-\d{5}-\d{4} \| ", "", ch[i][1]) for i in ex[q]["stage2_ids"])
        cov += all(l in txt for l in lines)
        t = "hug" if G[q]["gold_predicate"]["source"] == "hug-sale-history" else "molit"
        share.append(f"{sum(table_of(i) == t for i in ex[q]['stage2_ids'])}/{TAB[t]}")
        hdr += bool(HDR & set(ex[q]["stage2_ids"]))
    print(f"  {split}: stage-1 page hit {hit}/{len(qs)} ({pc(hit, len(qs)):.1f}%); region match {reg}/{len(qs)} ({pc(reg, len(qs)):.1f}%); "
          f"period match {per}/{len(qs)}; stage-2 all matching rows {cov}/{len(qs)}"
          + ("  -> 80% rule: report in Results" if min(hit, reg) < 0.8 * len(qs) else ""))
    print(f"    table share (two-stage table passages of the asked table / that table's passages): {dict(sorted(collections.Counter(share).items()))}; header passage in {hdr}/{len(qs)}")
