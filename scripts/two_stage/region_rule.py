"""Region extraction rules for the two-stage query, compared on dev and test.

Usage: python3 scripts/two_stage/region_rule.py <two-stage run dir>
"""
import collections, csv, json, re, sys
from pathlib import Path
from khat.paths import ROOT, DATA, LOCAL
HERE = Path(__file__).resolve().parent; P = DATA / "item_sets"
sys.path.insert(0, str(HERE.parent))
import retrieval as RS, table_rows as C
DS = ROOT
W = Path(sys.argv[1]); EX = {json.loads(l)["qa_id"]: json.loads(l) for l in open(W / "extract.jsonl")}
G = {}
for f in ("qa_test_public.jsonl", "qa_dev.jsonl"):
    for l in open(DS / "data" / f): r = json.loads(l); G[r["qa_id"]] = r
ch = RS.split_chunks((LOCAL / "bundles/mix_multiprovider_512k.txt").read_text(encoding="utf-8"), 1200)
SIDO = sorted({r["_query_area_name"] for r in C.hug_rows()}, key=len, reverse=True)
SGG = sorted({r["_lawd_name"] for r in C.molit_rows()}, key=len, reverse=True)
ALIAS = {}
for s in SIDO:
    short = re.sub(r"(특별자치도|특별자치시|특별시|광역시|도)$", "", s); ALIAS[s] = s
    if len(short) >= 2: ALIAS[short] = s
ABBR = {"서울": "서울특별시", "부산": "부산광역시", "대구": "대구광역시", "인천": "인천광역시", "광주": "광주광역시", "대전": "대전광역시",
        "울산": "울산광역시", "세종": "세종특별자치시", "경기": "경기도", "강원": "강원특별자치도", "충북": "충청북도", "충남": "충청남도",
        "전북": "전북특별자치도", "전남": "전라남도", "경북": "경상북도", "경남": "경상남도", "제주": "제주특별자치도"}
for a, s_ in ABBR.items():
    m_ = [x for x in SIDO if x.startswith(s_[:2]) and (x == s_ or x[:2] == s_[:2])]
    if m_: ALIAS[a] = m_[0]
LOC = re.compile(r"(공급위치|위\s*치|소재지|사업지|대상지|건설위치)")
def region_from(texts, molit):
    loc = [l for t in texts for l in t.splitlines() if LOC.search(l)]
    for pool in (loc, [l for t in texts for l in t.splitlines()]):
        c = collections.Counter()
        for l in pool:
            if molit:
                for g in SGG:
                    if g in l or g.split()[-1] in l and g.split()[0][:2] in l: c[g] += 1
            else:
                for a, s in ALIAS.items():
                    if a in l: c[s] += 1
        if c: return c.most_common(1)[0][0]
    return None
def v(q, title_filter):
    rec = G[q]; top = [ch[i][1] for i in EX[q]["stage1_ids"]]
    molit = rec["gold_predicate"]["source"].startswith("molit")
    if title_filter:
        t = re.findall(r"「([^」]+)」", rec["question"]); toks = [x for x in re.split(r"[\s()·,]+", t[0]) if len(x) >= 2] if t else []
        own = [x for x in top if toks and sum(k in x for k in toks) >= max(1, len(toks) // 2)]
        top = own or top
    return region_from(top, molit)
gold = lambda q: G[q]["gold_predicate"]["filter"].get("_query_area_name") or G[q]["gold_predicate"]["filter"].get("_lawd_name")
T81 = [l.strip() for l in open(P / "items_test81.txt") if l.strip()]; D144 = [l.strip() for l in open(P / "items_dev144.txt") if l.strip()]
DP = [r["qa_id"] for r in csv.DictReader(open(P / "dev_cells.tsv"), delimiter="\t") if r["same_cell_as_test"] == "0"]
for name, qs in (("test 81", T81), ("dev 144", D144), ("dev' 81", DP)):
    r1 = sum(v(q, False) == gold(q) for q in qs); r2 = sum(v(q, True) == gold(q) for q in qs); r3 = sum(EX[q]["region"] == gold(q) for q in qs)
    print(f"{name}: v1 {r1}, v2 {r2}, v3 (adopted) {r3}")
