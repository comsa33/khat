"""Deterministic executor of a filter query over the HUG and MOLIT table rows of the 512k bundle.
"""
import hashlib, re, statistics, unicodedata
from functools import lru_cache
from pathlib import Path
from khat.paths import LOCAL

BUNDLE = LOCAL / "bundles/mix_multiprovider_512k.txt"
BUNDLE_MD5 = "927a291943e676bca3ebc6a6037b3e8e"
COLS = {"hug": {"지역": "text", "연도": "int", "사업장명": "text", "총세대수(TOT_HOCO)": "int", "일반분양세대(GNRL_SILT_HOCO)": "int"},
        "molit": {"시군구": "text", "읍면동": "text", "단지": "text", "전용면적": "num", "거래금액": "int", "거래연월": "int"}}
OPS = {"=", "!=", "<", "<=", ">", ">=", "in"}
OPERATORS = {"count", "sum", "avg", "max"}
HUG_RE = re.compile(r"^[가-힣]+ \| 20\d\d \| .+ \| \d+ \| \d+$")
MOL_RE = re.compile(r"^MOLIT-APTTRADE-\d{6}-\d{5}-\d{4} \| ([^|]+) \| (.+) \| ([\d.]+)㎡ ([\d,]+)만원 \| (20\d{4})$")
nfc = lambda s: unicodedata.normalize("NFC", str(s)).strip()


@lru_cache(maxsize=None)
def bundle_text():
    b = BUNDLE.read_bytes()
    assert hashlib.md5(b).hexdigest() == BUNDLE_MD5, "bundle is not the evaluated bundle B"
    return b.decode("utf-8")


def _block(title):
    t = bundle_text(); i = t.index(title); return t[i:t.index("===== [참고자료 끝] =====", i)]


@lru_cache(maxsize=None)
def rows(table):
    if table == "hug":
        out = []
        for l in _block("===== [참고자료] HUG").splitlines():
            if HUG_RE.match(l):
                a, y, n, tot, gen = l.split(" | ")
                out.append({"지역": a, "연도": y, "사업장명": n, "총세대수(TOT_HOCO)": tot, "일반분양세대(GNRL_SILT_HOCO)": gen})
        assert len(out) == 623, len(out)
        return tuple(out)
    out = []
    for l in _block("===== [참고자료] MOLIT").splitlines():
        m = MOL_RE.match(l)
        if l.startswith("MOLIT-APTTRADE-"):
            assert m, l
            loc = m.group(1).split(" ")
            out.append({"시군구": " ".join(loc[:2]), "읍면동": " ".join(loc[2:]), "단지": m.group(2), "전용면적": m.group(3),
                        "거래금액": m.group(4), "거래연월": m.group(5)})
    assert len(out) == 544, len(out)
    return tuple(out)


@lru_cache(maxsize=None)
def distinct(table, col):
    return tuple(sorted({r[col] for r in rows(table)}))


def num(x):
    if isinstance(x, bool) or x is None: return None
    if isinstance(x, (int, float)): return float(x)
    try: return float(nfc(x).replace(",", ""))
    except ValueError: return None


def execute(query):
    if not isinstance(query, dict): return {"ok": False, "error": "invalid JSON"}
    t = nfc(query.get("table", ""))
    if t not in COLS: return {"ok": False, "error": f"unknown table: {query.get('table')}"}
    op = nfc(query.get("operator", ""))
    if op not in OPERATORS: return {"ok": False, "error": f"invalid operator: {query.get('operator')}"}
    flt = query.get("filters")
    if flt is None: flt = []
    if not isinstance(flt, list) or not all(isinstance(f, dict) and {"column", "op", "value"} <= set(f) for f in flt):
        return {"ok": False, "error": "invalid filters"}
    conds = []
    for f in flt:
        c, o, v = nfc(f["column"]), nfc(f["op"]), f["value"]
        if c not in COLS[t]: return {"ok": False, "error": f"unknown column: {f['column']}"}
        if o not in OPS: return {"ok": False, "error": f"invalid op: {f['op']}"}
        kind = COLS[t][c]
        if kind == "text" and o not in ("=", "!=", "in"): return {"ok": False, "error": f"invalid op: {f['op']}"}
        vals = v if o == "in" else [v]
        if o == "in" and not isinstance(v, list): return {"ok": False, "error": f"invalid value: {f['column']}"}
        if o != "in" and isinstance(v, (list, dict)): return {"ok": False, "error": f"invalid value: {f['column']}"}
        if kind == "text": vals = [nfc(x) for x in vals]
        else:
            vals = [num(x) for x in vals]
            if any(x is None for x in vals): return {"ok": False, "error": f"non-numeric value: {f['column']}"}
        conds.append((c, o, kind, vals))
    fld = None
    if op != "count":
        if query.get("field") is None: return {"ok": False, "error": "missing field"}
        fld = nfc(query["field"])
        if fld not in COLS[t]: return {"ok": False, "error": f"unknown column: {query['field']}"}
        if COLS[t][fld] == "text": return {"ok": False, "error": f"non-numeric field: {query['field']}"}

    def ok(r):
        for c, o, kind, vals in conds:
            x = r[c] if kind == "text" else num(r[c])
            if o == "in":
                if x not in vals: return False
            elif o == "=" and x != vals[0]: return False
            elif o == "!=" and x == vals[0]: return False
            elif o == "<" and not x < vals[0]: return False
            elif o == "<=" and not x <= vals[0]: return False
            elif o == ">" and not x > vals[0]: return False
            elif o == ">=" and not x >= vals[0]: return False
        return True

    sel = [r for r in rows(t) if ok(r)]
    if op == "count": return {"ok": True, "value": len(sel), "n_rows": len(sel)}
    xs = [num(r[fld]) for r in sel]
    if op == "sum": return {"ok": True, "value": sum(xs), "n_rows": len(sel)}
    if not xs: return {"ok": True, "value": None, "n_rows": 0}
    return {"ok": True, "value": statistics.mean(xs) if op == "avg" else max(xs), "n_rows": len(sel)}


def from_gold(gp):
    """self-check (a) only: the dataset gold_predicate in the query format"""
    if gp["source"] == "hug-sale-history":
        q = {"table": "hug", "filters": [{"column": "지역", "op": "=", "value": gp["filter"]["_query_area_name"]},
                                          {"column": "연도", "op": "=", "value": gp["filter"]["_query_year"]}]}
        fmap = {"TOT_HOCO": "총세대수(TOT_HOCO)", "GNRL_SILT_HOCO": "일반분양세대(GNRL_SILT_HOCO)"}
    else:
        q = {"table": "molit", "filters": [{"column": "시군구", "op": "=", "value": gp["filter"]["_lawd_name"]},
                                            {"column": "거래연월", "op": "=", "value": gp["filter"]["_deal_ymd"]}]}
        fmap = {"dealAmount_manwon": "거래금액"}
    assert set(gp["filter"]) <= {"_query_area_name", "_query_year", "_lawd_name", "_deal_ymd"}, gp
    return {**q, "operator": gp["op"], "field": fmap[gp["field"]] if gp["op"] != "count" else None}
