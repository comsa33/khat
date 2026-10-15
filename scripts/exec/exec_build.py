"""Prompts and request files of the filter-execution run.

Usage: python3 scripts/exec/exec_build.py <run dir> main
"""
import csv, hashlib, json, sys
from pathlib import Path
from khat.paths import DATA, RESULTS
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "two_stage"))
import executor as X
import two_stage_run as runR
sys.path.insert(0, str(HERE.parent))
import retrieval as RS

TWO_STAGE = RESULTS / "two_stage"
MOLIT_EXAMPLES = "none"
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
T81 = sorted(l.strip() for l in open(DATA / "item_sets/items_test81.txt") if l.strip())
D144 = sorted(l.strip() for l in open(DATA / "item_sets/items_dev144.txt") if l.strip())
DEVP = sorted(r["qa_id"] for r in csv.DictReader(open(DATA / "item_sets/dev_cells.tsv", encoding="utf-8"), delimiter="\t")
              if r["same_cell_as_test"] == "0")
ITEMS = T81 + DEVP


def questions():
    Q = {}
    for f in ("qa_test_public.jsonl", "qa_dev.jsonl"):
        for r in jl(runR.DS / "data" / f): Q[r["qa_id"]] = (r["question"], r["bundle_id"])
    return Q


def asked_cells():
    """exempt from leak check (6): the (region, period) cells of the aggregation items of test 81 and dev 144 (cells
    only, no answers), used only to keep the example rows out of them"""
    cells = set()
    for f in ("qa_test_public.jsonl", "qa_dev.jsonl"):
        for r in jl(runR.DS / "data" / f):
            if r["qa_id"] in set(T81) | set(D144):
                c = r["gold_predicate"]["filter"]
                cells.add(("hug", c["_query_area_name"], c["_query_year"]) if "_query_year" in c else ("molit", c["_lawd_name"], c["_deal_ymd"]))
    return cells


def example_rows(cells):
    out = {}
    for t, (a, p) in (("hug", ("지역", "연도")), ("molit", ("시군구", "거래연월"))):
        out[t] = [r for r in X.rows(t) if (t, r[a], r[p]) not in cells][:3]
        if t == "molit" and MOLIT_EXAMPLES == "none": out[t] = []
        elif len(out[t]) < 3: sys.exit(f"STOP: fewer than 3 example rows outside the asked cells for {t}")
    return out


TYPE = {"text": "문자열", "int": "정수", "num": "실수"}
NOTE = {"연도": "정수(YYYY)", "거래연월": "정수(YYYYMM)", "거래금액": "정수(만원)", "전용면적": "실수(㎡)", "시군구": "문자열(시·도와 시·군·구)"}
DESC = {"hug": "HUG 분양이력 자료 (주택도시보증공사, 분양보증 발급 사업장). 각 행 = 분양 사업장 1건. ‘분양 사업장 건수’는 해당 지역·연도 행의 개수이고, ‘평균 총세대수(TOT_HOCO)’는 그 행들의 총세대수 평균입니다.",
        "molit": "MOLIT 아파트 매매 실거래 상세 (국토교통부). 각 행 = 매매 1건. '거래 건수'는 해당 시군구·거래연월 행의 개수이고, '평균 거래금액'은 그 행들의 거래금액(만원) 평균입니다."}
LISTED = {"hug": ("지역", "연도"), "molit": ("시군구", "거래연월")}


def schema(examples):
    out = ["[표 스키마]"]
    for t in ("hug", "molit"):
        out.append(f"표 {t}: {DESC[t]}")
        for c, k in X.COLS[t].items():
            line = f"- {c}: {NOTE.get(c, TYPE[k])}"
            if c in LISTED[t]: line += ". 값: " + ", ".join(X.distinct(t, c))
            out.append(line)
        if examples[t]:
            out.append("예시 행:")
            out += ["  " + ", ".join(f"{c}={str(r[c]).replace(',', '')}" for c in X.COLS[t]) for r in examples[t]]
    return "\n".join(out)


INSTR = ("주어진 문서 발췌와 표 스키마를 보고, 질문에 답하는 표 조회를 아래 형식의 JSON 객체 하나로 작성하라. "
         "지역과 기간은 문서 발췌와 질문에서 정하라. 고정된 실행기가 이 조회를 표에 실행하고, 그 결과를 답으로 쓴다.")
OUTFMT = ("[출력 형식]\n"
          "JSON 객체 하나만 출력하세요.\n"
          '{"table": "표 이름", "filters": [{"column": "열 이름", "op": "연산", "value": 값}], "operator": "count, sum, avg, max 중 하나", "field": "숫자 열 이름 또는 null"}\n'
          "- op: =, !=, <, <=, >, >=, in (in의 값은 목록). 문자열 열에는 =, !=, in만 씁니다.\n"
          "- operator: 행의 개수는 count, 합계는 sum, 평균은 avg, 최댓값은 max입니다. count이면 field는 null입니다.\n"
          "- 표 이름, 열 이름, 문자열 값은 스키마 표기와 정확히 같아야 합니다.")


def prompt(question, passages, sch):
    blocks = "\n\n".join(f"[{pid} (p.{pid.rsplit('-p', 1)[-1].lstrip('0') or '0'})]\n{txt}" for pid, txt in passages)
    return f"{INSTR}\n\n[검색된 문서 발췌] (질문과 관련된 상위 페이지만 제공)\n{blocks}\n\n{sch}\n\n[질문]\n{question}\n\n{OUTFMT}"


RETRY_TAIL = "위 오류를 고쳐 지정된 형식의 JSON 객체 하나만 다시 출력하세요."


def retry_prompt(first_prompt, prev_text, error):
    """first request + the model's first answer + the constant error text (error kind and offending key only)"""
    return f"{first_prompt}\n\n[이전 답]\n{(prev_text or '').strip()}\n\n[실행 오류]\n{error}\n\n{RETRY_TAIL}"


def build_prompts():
    Q = questions(); ex = {r["qa_id"]: r for r in jl(TWO_STAGE / "extract.jsonl")}
    ch = RS.split_chunks(X.bundle_text(), 1200)
    sch = schema(example_rows(asked_cells()))
    out = []
    for q in ITEMS:
        assert Q[q][1] == "mix_multiprovider_512k", q
        out.append({"qa_id": q, "split": "test" if q in T81 else "dev", "prompt": prompt(Q[q][0], [ch[i] for i in sorted(ex[q]["stage1_ids"])], sch)})
    return out, sch


def build(W, setname):
    W = Path(W); out = W / setname; (out / "requests").mkdir(parents=True, exist_ok=True)
    assert not any((out / "requests").glob("raw_*")), "raw outputs exist; refusing to rebuild"
    (DATA / "item_sets/items_devprime81.txt").write_text("\n".join(DEVP) + "\n", encoding="utf-8")
    P, sch = build_prompts(); PP = {r["qa_id"]: r for r in P}
    its = ITEMS
    for f in (out / "requests").glob("*_tool_*.jsonl"): f.unlink()
    man = {"set": setname, "items": its, "caps": runR.CAP, "openai_ids": runR.OAI, "anthropic_thinking": runR.ANT, "ollama_tags": runR.OLL,
           "num_ctx": runR.NUM_CTX, "files": {}, "schema_sha256": hashlib.sha256(sch.encode()).hexdigest(),
           "instr_sha256": hashlib.sha256((INSTR + "\n" + OUTFMT).encode()).hexdigest(), "molit_examples": MOLIT_EXAMPLES}
    runR.wjl(out / "prompts_tool.jsonl", [PP[q] for q in its])
    for m in runR.MODELS:
        parts = [("test", [q for q in its if q in T81]), ("dev", [q for q in its if q not in T81])]
        for part, qs in parts:
            rows = [runR.BUILD[runR.provider(m)](f"tool__{q}", m, PP[q]["prompt"], runR.CAP[m]) for q in qs]
            fn = out / "requests" / (f"{runR.provider(m)}_tool_{m}" + (f"__{part}" if part else "") + ".jsonl")
            runR.wjl(fn, rows); man["files"][fn.name] = len(rows)
    json.dump(man, open(out / "manifest.json", "w"), indent=1, ensure_ascii=False)
    print(setname, "items", len(its), "files", len(man["files"]), "requests", sum(man["files"].values()),
          "schema sha256", man["schema_sha256"])
    return man


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
