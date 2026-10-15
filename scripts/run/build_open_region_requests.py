"""Open-region OpenAI batch request files and Ollama commands, copied from the oracle ones with the open-region prompts.

Usage: python3 scripts/run/build_open_region_requests.py   (after build_oracle.py)
"""
import json
from khat.paths import LOCAL
E = LOCAL / "runs" / "oracle"
OR = {json.loads(l)["qa_id"]: json.loads(l) for l in open(E / "prompts_open_region.jsonl")}
for m in ["gpt-5.5", "gpt-4.1-mini"]:
    out = []
    for l in open(E / f"batch_input_{m}.jsonl"):
        r = json.loads(l)
        qid = r["custom_id"].split("__")[-1]
        msgs = r["body"]["messages"]
        assert len(msgs) == 1 and msgs[0]["role"] == "user", r["custom_id"]
        r["custom_id"] = r["custom_id"].replace("oracle__", "openregion__", 1)
        msgs[0]["content"] = OR[qid]["prompt"]
        out.append(r)
    assert len(out) == 81 and len({x["custom_id"] for x in out}) == 81
    with open(E / f"batch_input_openregion_{m}.jsonl", "w") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
p = json.load(open(E / "ollama_plan.json"))
for m, v in p.items():
    p[m]["cmd_openregion"] = v["cmd"].replace("prompts_oracle.jsonl", "prompts_open_region.jsonl").replace("oracle_", "openregion_")
json.dump(p, open(E / "ollama_plan.json", "w"), indent=1, ensure_ascii=False)
print("ok", list(p))
