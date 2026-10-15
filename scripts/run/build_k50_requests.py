"""OpenAI and Anthropic batch request files of the dense top-50 run.

Usage: python3 scripts/run/build_k50_requests.py   (after build_k50.py; writes local/runs/k50/gen/)
"""
import json, sys
from pathlib import Path
from khat.paths import LOCAL
HERE = LOCAL / "runs" / "k50"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_baseline import select_prompt, _openai_chat_kwargs
recs = [json.loads(l) for l in open(HERE / "prompts_dense_k50_agg81.jsonl")]
G = HERE / "gen"
G.mkdir(parents=True, exist_ok=True)
for m, cap in (("gpt-5.5", 16000), ("gpt-4.1-mini", 4000)):
    with open(G / f"batch_input_k50_{m}.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps({"custom_id": f"k50dense__test_public__{r['qa_id']}", "method": "POST",
                                "url": "/v1/chat/completions",
                                "body": _openai_chat_kwargs(m, select_prompt(r), 0.0, cap, None)}, ensure_ascii=False) + "\n")
for m, th, short in (("claude-sonnet-5-5", {"type": "between_tools"}, "sonnet55"), ("claude-haiku-5-5", {"type": "disabled"}, "haiku55")):
    with open(G / f"batch_k50_{short}.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps({"custom_id": f"k50dense__{r['qa_id']}", "params": {"model": m, "max_tokens": 16384, "thinking": th,
                                "messages": [{"role": "user", "content": select_prompt(r)}]}}, ensure_ascii=False) + "\n")
print("built")
