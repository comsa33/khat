#!/usr/bin/env python3
"""Oracle and open-region prompts for the 81 aggregation items (matching table rows plus the announcement page), checked against the gold answer, with OpenAI batch request files and Ollama run commands.

Usage: python3 scripts/run/build_oracle.py && python3 scripts/run/build_open_region_requests.py   (writes local/runs/oracle/)
"""
import json, re, statistics as st
from rerun_common import *
from run_baseline import _openai_chat_kwargs

E = OUT / "oracle"
E.mkdir(parents=True, exist_ok=True)
AGG = [r["qa_id"] for r in load(V / "data/item_sets/agg81.jsonl")]
SRCP = RO / "full_context/512k_all124_tp.jsonl"

def read_prompts(path, want):
    out = {}
    with open(path, encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            if r["qa_id"] in want:
                out[r["qa_id"]] = r["prompt"]
    return out

fcp = read_prompts(SRCP, set(AGG))
assert set(fcp) == set(AGG), len(fcp)

HEAD = "주어진 근거 자료로 질문에 답하라.\n\n[제공 문서] (아래 공고/번들 본문에서 근거를 찾으세요.)\n"
def split_prompt(p):
    assert p.startswith(HEAD)
    body, tail = p[len(HEAD):].rsplit("\n\n[질문]", 1)
    return body, "\n\n[질문]" + tail

BLOCK_RE = re.compile(r"===== \[참고자료\] (?P<title>[^\n]*?) =====\n(?P<body>.*?)\n===== \[참고자료 끝\] =====", re.S)
PAGE_RE = re.compile(r"^\[공고 [^\n]*? (?P<pid>[^\s]+) \(p\.\d+\)\]$", re.M)

def blocks(body):
    out = {}
    for m in BLOCK_RE.finditer(body):
        lines = m.group("body").split("\n")
        key = "hug" if m.group("title").startswith("HUG") else "molit"
        out[key] = {"title_line": f"===== [참고자료] {m.group('title')} =====", "desc": lines[0],
                    "header": lines[1], "rows": lines[2:]}
    return out

def page_text(body, pid):
    starts = [(m.start(), m.group("pid")) for m in PAGE_RE.finditer(body)]
    for i, (s, p) in enumerate(starts):
        if p == pid:
            end = starts[i + 1][0] if i + 1 < len(starts) else len(body)
            seg = body[s:end]
            cut = re.search(r"\n(\[법령 |===== \[참고자료\])", seg)
            return (seg[:cut.start()] if cut else seg).rstrip("\n")
    raise KeyError(pid)

def num(s):
    return float(s.replace(",", "").replace("만원", "").strip())

def match(src, cols, flt, open_region=False):
    if src == "hug":
        ok_year = cols[1].strip() == str(flt["_query_year"])
        ok_reg = cols[0].strip() == flt["_query_area_name"]
    else:
        ok_year = cols[4].strip() == str(flt["_deal_ymd"])
        ok_reg = (cols[1].strip() + " ").startswith(flt["_lawd_name"] + " ")
    return ok_year and (open_region or ok_reg)

def value(src, cols, field):
    if src == "hug":
        return num(cols[3]) if field == "TOT_HOCO" else num(cols[4])
    return num(cols[3].split("㎡", 1)[1])

def build(variant):
    rows_out, checks = [], []
    for q in AGG:
        g = GOLD[q]; gp = g["gold_predicate"]
        body, tail = split_prompt(fcp[q])
        bl = blocks(body)
        src = "hug" if gp["source"] == "hug-sale-history" else "molit"
        b = bl[src]
        sel = [r for r in b["rows"] if match(src, r.split(" | "), gp["filter"], variant == "open_region")]
        exact = [r for r in b["rows"] if match(src, r.split(" | "), gp["filter"])]
        vals = [value(src, r.split(" | "), gp.get("field")) for r in exact]
        calc = {"count": len(exact), "sum": sum(vals) if vals else None,
                "avg": (sum(vals) / len(vals)) if vals else None, "max": max(vals) if vals else None}[gp["op"]]
        gnums = [float(x) for x in (g.get("evaluation") or {}).get("gold_numbers") or []]
        ok = any(abs(calc - x) <= max(0.5, 0.005 * abs(x)) for x in gnums) if calc is not None else False
        desc = re.sub(r"^총 [\d,]+(개 사업장|건)\.\s*", "", b["desc"])
        table = "\n".join([b["title_line"], desc, b["header"], *sel, "===== [참고자료 끝] ====="])
        page = page_text(body, g["page_ids"][0])
        prompt = HEAD + table + "\n" + page + tail
        rows_out.append({"qa_id": q, "split": "test_public", "prompt": prompt, "variant": variant,
                         "source": src, "op": gp["op"], "n_rows": len(sel), "page_id": g["page_ids"][0]})
        checks.append({"qa_id": q, "op": gp["op"], "source": src, "n_rows_exact": len(exact), "n_rows_given": len(sel),
                       "recomputed": calc, "gold_numbers": gnums, "answer": g["answer"], "matches_gold": ok,
                       "fc_prompt_chars": len(fcp[q]), "oracle_prompt_chars": len(prompt)})
    return rows_out, checks

oracle, chk = build("oracle")
openr, chk2 = build("open_region")
write(E / "prompts_oracle.jsonl", oracle); write(E / "prompts_open_region.jsonl", openr)
write(E / "check.jsonl", chk)
(E / "items.txt").write_text("\n".join(AGG) + "\n")

# ---- settings: each model's FC settings for the 512k aggregation items
GPT_FC = {"gpt-5.5": 4096, "gpt-4.1-mini": 4096}
for M, cap in GPT_FC.items():
    write(E / f"batch_input_{M}.jsonl", [{"custom_id": f"oracle__test_public__{r['qa_id']}", "method": "POST",
          "url": "/v1/chat/completions", "body": _openai_chat_kwargs(M, r["prompt"], 0.0, cap, None)} for r in oracle])
OLL = {"glm-5.2:cloud": {"temperature": 0.0, "max_output_tokens": 8192, "num_ctx": 524288, "think": "on"},
       "minimax-m3:cloud": {"temperature": 0.0, "max_output_tokens": 16384, "num_ctx": 524288, "think": "on"}}
for M, s in OLL.items():
    s["cmd"] = (f"python3 scripts/run_baseline.py --provider ollama --model {M} --split test_public "
                f"--prompt-file local/runs/oracle/prompts_oracle.jsonl --out local/runs/predictions/oracle_{M.replace(':', '-')}_agg81.jsonl "
                f"--temperature 0 --max-output-tokens {s['max_output_tokens']} --num-ctx {s['num_ctx']} --think on --resume")
json.dump(OLL, open(E / "ollama_plan.json", "w"), ensure_ascii=False, indent=1)

lens = [len(r["prompt"]) for r in oracle]; lens2 = [len(r["prompt"]) for r in openr]
print("oracle checks matching gold:", sum(c["matches_gold"] for c in chk), "/", len(chk))
bad = [c for c in chk if not c["matches_gold"]]
for c in bad[:10]: print("  MISMATCH", c)
print("oracle prompt chars min/med/max", min(lens), int(st.median(lens)), max(lens), " open-region", min(lens2), int(st.median(lens2)), max(lens2))
print("rows given min/med/max", min(r['n_rows'] for r in oracle), int(st.median(r['n_rows'] for r in oracle)), max(r['n_rows'] for r in oracle))
