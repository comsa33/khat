"""Unit tests of the table executor, answer parser, scorer and re-ask prompt (no model call).

Usage: uv run pytest tests/test_exec.py   or   python3 tests/test_exec.py
"""
import json, statistics, sys, unicodedata
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts" / "exec")); sys.path.insert(0, str(HERE.parent / "scripts" / "two_stage"))
import executor as X, exec_score as S, exec_build as B, two_stage_run as runR

FAILS, N = [], [0]


def t(name, cond):
    N[0] += 1
    if not cond: FAILS.append(name)


def q(table, filters, op, field=None):
    return {"table": table, "filters": [{"column": c, "op": o, "value": v} for c, o, v in filters], "operator": op, "field": field}


if __name__ != "__main__" and not X.BUNDLE.exists():
    import pytest
    pytest.skip("needs the rebuilt bundle under local/ (scripts/build)", allow_module_level=True)
H, M = X.rows("hug"), X.rows("molit")
t("bundle rows: HUG 623, MOLIT 544", len(H) == 623 and len(M) == 544)
t("MOLIT row identifiers dropped", all("MOLIT-APTTRADE" not in json.dumps(r, ensure_ascii=False) for r in M))
g23 = sum(r["지역"] == "경기도" and r["연도"] == "2023" for r in H)
dj06 = sum(r["시군구"] == "대전광역시 동구" and r["거래연월"] == "202506" for r in M)
cnt = lambda qq: X.execute(qq)["value"]

# ---- exact match, no aliases
t("region exact", cnt(q("hug", [("지역", "=", "경기도"), ("연도", "=", 2023)], "count")) == g23)
t("abbreviation 경기 -> 0 rows (no alias)", cnt(q("hug", [("지역", "=", "경기"), ("연도", "=", 2023)], "count")) == 0)
t("strip and NFC", cnt(q("hug", [("지역", "=", " " + unicodedata.normalize("NFD", "경기도") + " "), ("연도", "=", 2023)], "count")) == g23)
t("district without province -> 0 rows", cnt(q("molit", [("시군구", "=", "동구"), ("거래연월", "=", 202506)], "count")) == 0)
t("district exact", cnt(q("molit", [("시군구", "=", "대전광역시 동구"), ("거래연월", "=", 202506)], "count")) == dj06)
# ---- year and month forms (numeric columns)
for v in (2023, "2023", "2,023", 2023.0):
    t(f"year {v!r}", cnt(q("hug", [("지역", "=", "경기도"), ("연도", "=", v)], "count")) == g23)
for v in ("2023년", "23"):
    r = X.execute(q("hug", [("지역", "=", "경기도"), ("연도", "=", v)], "count"))
    t(f"year {v!r}", (not r["ok"] and r["error"] == "non-numeric value: 연도") if v == "2023년" else r["value"] == 0)
for v in (202506, "202506"):
    t(f"month {v!r}", cnt(q("molit", [("시군구", "=", "대전광역시 동구"), ("거래연월", "=", v)], "count")) == dj06)
for v in ("2025-06", "2025년 6월"):
    t(f"month {v!r} -> execution error", X.execute(q("molit", [("거래연월", "=", v)], "count"))["error"] == "non-numeric value: 거래연월")
t("month 2025.06 is the number 2025.06 -> 0 rows", cnt(q("molit", [("거래연월", "=", "2025.06")], "count")) == 0)
# ---- ops
yrs = lambda o, v: cnt(q("hug", [("지역", "=", "경기도"), ("연도", o, v)], "count"))
by = {y: sum(r["지역"] == "경기도" and r["연도"] == y for r in H) for y in ("2023", "2024", "2025")}
t("op !=", yrs("!=", 2023) == by["2024"] + by["2025"]); t("op <", yrs("<", 2024) == by["2023"]); t("op <=", yrs("<=", 2024) == by["2023"] + by["2024"])
t("op >", yrs(">", 2024) == by["2025"]); t("op >=", yrs(">=", 2024) == by["2024"] + by["2025"]); t("op in numeric", yrs("in", [2023, "2025"]) == by["2023"] + by["2025"])
t("op in text", cnt(q("hug", [("지역", "in", ["경기도", "서울특별시"]), ("연도", "=", 2023)], "count")) == g23 + sum(r["지역"] == "서울특별시" and r["연도"] == "2023" for r in H))
t("op != text", cnt(q("hug", [("지역", "!=", "경기도")], "count")) == 623 - sum(r["지역"] == "경기도" for r in H))
t("no filters = all rows", cnt(q("hug", [], "count")) == 623)
t("filters null = all rows", cnt({"table": "hug", "filters": None, "operator": "count", "field": None}) == 623)
# ---- values and zero rows
sel = [int(r["총세대수(TOT_HOCO)"]) for r in H if r["지역"] == "경기도" and r["연도"] == "2023"]
base = [("지역", "=", "경기도"), ("연도", "=", 2023)]
t("sum", cnt(q("hug", base, "sum", "총세대수(TOT_HOCO)")) == sum(sel))
t("avg exact mean", cnt(q("hug", base, "avg", "총세대수(TOT_HOCO)")) == statistics.mean(sel))
t("max", cnt(q("hug", base, "max", "총세대수(TOT_HOCO)")) == max(sel))
pr = [int(r["거래금액"].replace(",", "")) for r in M if r["시군구"] == "대전광역시 동구" and r["거래연월"] == "202506"]
t("molit max price (commas in the bundle)", cnt(q("molit", [("시군구", "=", "대전광역시 동구"), ("거래연월", "=", 202506)], "max", "거래금액")) == max(pr))
t("comma value on a numeric filter", cnt(q("hug", base + [("총세대수(TOT_HOCO)", ">=", "1,000")], "count")) == sum(x >= 1000 for x in sel))
zero = [("지역", "=", "경기"), ("연도", "=", 2023)]
t("0 rows count -> 0", X.execute(q("hug", zero, "count")) == {"ok": True, "value": 0, "n_rows": 0})
t("0 rows sum -> 0", X.execute(q("hug", zero, "sum", "총세대수(TOT_HOCO)"))["value"] == 0)
t("0 rows avg -> None", X.execute(q("hug", zero, "avg", "총세대수(TOT_HOCO)"))["value"] is None)
t("0 rows max -> None", X.execute(q("hug", zero, "max", "총세대수(TOT_HOCO)"))["value"] is None)
t("count ignores a field", cnt(q("hug", base, "count", "총세대수(TOT_HOCO)")) == g23)
# ---- execution errors: constant English text, kind and offending key only
E = lambda qq: X.execute(qq).get("error")
t("not an object", E([]) == "invalid JSON")
t("unknown table", E(q("HUG", base, "count")) == "unknown table: HUG")
t("unknown column", E(q("hug", [("region", "=", "경기도")], "count")) == "unknown column: region")
t("column alias not accepted", E(q("hug", [("총세대수", ">", 1)], "count")) == "unknown column: 총세대수")
t("invalid op", E(q("hug", [("연도", "==", 2023)], "count")) == "invalid op: ==")
t("op < on a text column", E(q("hug", [("지역", "<", "경기도")], "count")) == "invalid op: <")
t("invalid operator min", E(q("hug", base, "min", "총세대수(TOT_HOCO)")) == "invalid operator: min")
t("invalid operator median", E(q("hug", base, "median", "총세대수(TOT_HOCO)")) == "invalid operator: median")
t("missing field", E(q("hug", base, "avg")) == "missing field")
t("text field", E(q("hug", base, "sum", "지역")) == "non-numeric field: 지역")
t("unknown field", E(q("hug", base, "sum", "TOT_HOCO")) == "unknown column: TOT_HOCO")
t("non-numeric value", E(q("hug", [("총세대수(TOT_HOCO)", ">", "많음")], "count")) == "non-numeric value: 총세대수(TOT_HOCO)")
t("in without a list", E(q("hug", [("연도", "in", 2023)], "count")) == "invalid value: 연도")
t("list without in", E(q("hug", [("연도", "=", [2023])], "count")) == "invalid value: 연도")
t("filters as an object", E({"table": "hug", "filters": {"지역": "경기도"}, "operator": "count", "field": None}) == "invalid filters")
t("filter missing op", E({"table": "hug", "filters": [{"column": "지역", "value": "경기도"}], "operator": "count", "field": None}) == "invalid filters")
ERRS = [E(x) for x in (q("hug", base, "min", "총세대수(TOT_HOCO)"), q("hug", [("총세대수(TOT_HOCO)", ">", "많음")], "count"), q("molit", [("거래연월", "=", "2025-06")], "count"), q("hug", base, "avg"))]
t("error messages name no table value or row count", all(e and not any(ch.isdigit() for ch in e.replace("(TOT_HOCO)", "")) and "행" not in e for e in ERRS))
# ---- parser: the last object with the four keys
P_ = lambda s: S.parse(s)[0]
gq = q("hug", base, "count")
t("plain", P_(json.dumps(gq, ensure_ascii=False)) == gq)
t("fenced with prose", P_("조회:\n```json\n" + json.dumps(gq, ensure_ascii=False) + "\n```\n끝") == gq)
t("last of several objects", P_(json.dumps(q("molit", [], "count"), ensure_ascii=False) + " 수정: " + json.dumps(gq, ensure_ascii=False)) == gq)
t("object without the four keys is skipped", P_(json.dumps(gq, ensure_ascii=False) + ' {"note": 1}') == gq)
t("missing key field -> fail", S.parse('{"table": "hug", "filters": [], "operator": "count"}')[1] == "invalid JSON")
t("braces inside strings", P_('{"table":"hug","filters":[{"column":"사업장명","op":"=","value":"a}b{c"}],"operator":"count","field":null}')["filters"][0]["value"] == "a}b{c")
t("broken JSON", S.parse('{"table": "hug", "filters": [')[1] == "invalid JSON")
t("unpaired { in prose before the JSON", P_("조건 {지역은 경기도로 보면 됩니다.\n" + json.dumps(gq, ensure_ascii=False)) == gq)
t("unpaired { inside the prose after a draft", P_(json.dumps(q("molit", [], "count"), ensure_ascii=False) + " 수정 {초안 폐기} " + json.dumps(gq, ensure_ascii=False)) == gq)
t("no JSON", S.parse("답: 61건")[1] == "invalid JSON")
t("nested inner object is not taken alone", S.parse('{"a": {"table":"hug","filters":[],"operator":"count","field":null}}')[1] == "invalid JSON")
# ---- scorer: half-up at the gold's decimals, alternatives
t("gold parse", S.to_num("1,073세대") == (1073.0, 0) and S.to_num("57,000만원") == (57000.0, 0) and S.to_num("61건") == (61.0, 0) and S.to_num("12.5") == (12.5, 1))
t("exact", S.score(61, [(61.0, 0)]) and not S.score(60, [(61.0, 0)]))
t("half-up .5", S.score(1072.5, [(1073.0, 0)]) and not S.score(1072.5, [(1072.0, 0)]) and S.score(1073.5, [(1074.0, 0)]))
t("round down below .5", S.score(1072.4999, [(1072.0, 0)]))
t("one decimal half-up", S.score(12.45, [(12.5, 1)]) and not S.score(12.44, [(12.5, 1)]))
t("None is wrong", not S.score(None, [(0.0, 0)]))
t("count 0 can be right", S.score(0, [(0.0, 0)]))
t("MOLIT alternative (count without cancelled) accepted", S.score(171, S.gold_values("krhlrb_v05_0428")) and S.score(184, S.gold_values("krhlrb_v05_0428"))
  and not S.score(172, S.gold_values("krhlrb_v05_0428")))
t("MOLIT avg alternative accepted", len(S.gold_values("krhlrb_v05_0426")) == 2)
# ---- re-ask prompt
rp = B.retry_prompt("P", "first answer", "unknown column: region")
t("re-ask = first request + first answer + constant error", rp == "P\n\n[이전 답]\nfirst answer\n\n[실행 오류]\nunknown column: region\n\n" + B.RETRY_TAIL)


def test_exec_units():
    assert N[0] and not FAILS, FAILS


if __name__ == "__main__":
    print(f"unit tests {N[0]}; failures {len(FAILS)}")
    for f in FAILS: print("  FAIL", f)
    sys.exit(1 if FAILS else 0)
