#!/usr/bin/env python3
"""Run a model over a prompt file (OpenAI, Azure OpenAI, Anthropic, Ollama) and write predictions.

Usage: python3 scripts/run_baseline.py --help
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from khat.paths import ROOT, LOCAL
DEFAULT_PROMPT_FILE = "local/runs/closed_book/locator_prompts.jsonl"
BASELINES_DIR = LOCAL / "runs"
PUBLIC_SPLITS = ("dev", "test_public")
PROVIDERS = ("openai", "azure_openai", "anthropic", "ollama")
MOCK_PREDICTION = "[MOCK] locator-only baseline placeholder — no model was called"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- prompt construction
def build_prompt(rec: dict) -> str:
    """Deterministic, simple closed-book prompt: instruction + question + locator hints (no doc text)."""
    instr = rec.get("instruction", "주어진 근거 자료로 질문에 답하라.")
    question = rec.get("question", "")
    spec = rec.get("context_spec", {}) or {}

    def _ids(key: str, limit: int = 20) -> str:
        vals = spec.get(key) or []
        if not isinstance(vals, list):
            vals = [vals]
        if not vals:
            return ""
        shown = vals[:limit]
        more = f" (+{len(vals) - limit} more)" if len(vals) > limit else ""
        return ", ".join(str(v) for v in shown) + more

    lines = []
    for label, key in (
        ("retrieval_mode", "retrieval_mode"),
        ("source_ids", "source_ids"),
        ("page_ids", "page_ids"),
        ("row_ids", "row_ids"),
        ("table_ids", "table_ids"),
        ("cell_ids", "cell_ids"),
        ("bundle_id", "bundle_id"),
        ("context_tier", "context_tier"),
        ("evidence_position", "evidence_position"),
        ("predicate_source", "predicate_source"),
    ):
        if key in ("retrieval_mode", "bundle_id", "context_tier", "evidence_position", "predicate_source"):
            v = spec.get(key)
            if v:
                lines.append(f"- {label}: {v}")
        else:
            s = _ids(key)
            if s:
                lines.append(f"- {label}: {s}")
    locator_block = "\n".join(lines) if lines else "- (근거 위치 메타데이터 없음)"

    return (
        f"{instr}\n\n"
        f"[질문]\n{question}\n\n"
        f"[근거 위치 정보] (원문 텍스트는 제공되지 않습니다. 아래 위치 식별자만 참고하세요.)\n"
        f"{locator_block}\n\n"
        f"[지시] 추측을 최소화하고, 답을 알 수 없으면 '제공된 자료만으로는 확정할 수 없음'이라고 답하세요. "
        f"군더더기 없이 최종 답만 간단히 출력하세요."
    )


def select_prompt(rec: dict) -> str:
    """Use a pre-built `prompt` if the record carries one (e.g. full-context prompts that embed
    bundle text, built by scripts/run/build_full_context.py); otherwise build the locator-only
    prompt. The runner never embeds bundle text itself — full-context prompts are produced offline and
    kept under local/."""
    p = rec.get("prompt")
    return p if isinstance(p, str) and p.strip() else build_prompt(rec)


# --------------------------------------------------------------------------- provider adapters
class BaseProvider:
    name = "base"

    def __init__(self, model: str, temperature: float, max_output_tokens: int, mock: bool = False,
                 reasoning: bool | None = None, num_ctx: int | None = None,
                 think: bool | None = None):
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.mock = mock
        self.reasoning = reasoning
        self.num_ctx = num_ctx
        self.think = think
        self.last_meta: dict = {}
        self._client = None

    def required_env(self) -> list[str]:
        return []

    def env_help(self) -> str:
        return ""

    def missing_env(self) -> list[str]:
        return [v for v in self.required_env() if not os.environ.get(v)]

    def ensure_ready(self) -> None:
        """Validate env + lazily build the client. Not called in --dry-run / --mock."""
        miss = self.missing_env()
        if miss:
            raise SystemExit(
                f"[{self.name}] missing required environment variable(s): {', '.join(miss)}\n"
                f"{self.env_help()}"
            )
        self._init_client()

    def _init_client(self) -> None:
        raise NotImplementedError

    def generate(self, prompt: str) -> str:
        self.last_meta = {}
        if self.mock:
            return MOCK_PREDICTION
        return self._generate(prompt)

    def _generate(self, prompt: str) -> str:
        raise NotImplementedError


def _openai_reasoning_model(model: str) -> bool:
    """Newer OpenAI 'reasoning' models (gpt-5*, o1/o3/o4*) require `max_completion_tokens` (and reject
    `max_tokens`); many also reject a non-default `temperature`, so we omit it for them."""
    m = model.lower()
    return m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4")


def _openai_chat_kwargs(model: str, prompt: str, temperature: float, max_output_tokens: int,
                        reasoning: bool | None = None) -> dict:
    """Shared OpenAI/Azure chat.completions kwargs, switching params for reasoning models.

    reasoning: None = auto-detect from the model name; True/False = force. Use the explicit form for
    Azure, where --model is an arbitrary deployment name the name heuristic cannot classify (e.g. a
    deployment named `prod-baseline` backed by a gpt-5/o-series model)."""
    is_reasoning = _openai_reasoning_model(model) if reasoning is None else reasoning
    kw = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if is_reasoning:
        kw["max_completion_tokens"] = max_output_tokens
    else:
        kw["max_tokens"] = max_output_tokens
        kw["temperature"] = temperature
    return kw


def _openai_usage(resp) -> dict:
    """Extract token usage from an OpenAI/Azure chat response (best-effort; empty if absent)."""
    u = getattr(resp, "usage", None)
    if u is None:
        return {}
    out = {"prompt_tokens": getattr(u, "prompt_tokens", None),
           "completion_tokens": getattr(u, "completion_tokens", None)}
    details = getattr(u, "completion_tokens_details", None)
    rt = getattr(details, "reasoning_tokens", None) if details is not None else None
    if rt is not None:
        out["reasoning_tokens"] = rt
    return out


class OpenAIProvider(BaseProvider):
    name = "openai"

    def required_env(self):
        return ["OPENAI_API_KEY"]

    def env_help(self):
        return "Set OPENAI_API_KEY. Install the SDK with: pip install 'khat[runs]' (or pip install openai)."

    def _init_client(self):
        import openai

        self._client = openai.OpenAI()

    def _generate(self, prompt):
        resp = self._client.chat.completions.create(
            **_openai_chat_kwargs(self.model, prompt, self.temperature, self.max_output_tokens, self.reasoning)
        )
        self.last_meta = _openai_usage(resp)
        return (resp.choices[0].message.content or "").strip()


class AzureOpenAIProvider(BaseProvider):
    name = "azure_openai"

    def required_env(self):
        return ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION"]

    def env_help(self):
        return (
            "Set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION. "
            "--model is the Azure DEPLOYMENT name. Install: pip install 'khat[runs]' (openai SDK)."
        )

    def _init_client(self):
        from openai import AzureOpenAI

        self._client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )

    def _generate(self, prompt):
        resp = self._client.chat.completions.create(
            **_openai_chat_kwargs(self.model, prompt, self.temperature, self.max_output_tokens, self.reasoning)
        )
        self.last_meta = _openai_usage(resp)
        return (resp.choices[0].message.content or "").strip()


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def required_env(self):
        return ["ANTHROPIC_API_KEY"]

    def env_help(self):
        return "Set ANTHROPIC_API_KEY. Install: pip install 'khat[runs]' (or pip install anthropic)."

    def _init_client(self):
        import anthropic

        self._client = anthropic.Anthropic()

    def _generate(self, prompt):
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_output_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return ("".join(parts)).strip()


class OllamaProvider(BaseProvider):
    """Local Ollama server via stdlib HTTP (no third-party SDK, no API key)."""

    name = "ollama"

    def base_url(self) -> str:
        return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

    def required_env(self):
        return []

    def env_help(self):
        return "Optionally set OLLAMA_BASE_URL (default http://localhost:11434). Start a local server: `ollama serve`."

    def _init_client(self):
        self._client = self.base_url()
        try:
            urllib.request.urlopen(self.base_url(), timeout=5)
        except urllib.error.HTTPError:
            pass
        except urllib.error.URLError as exc:
            raise SystemExit(
                f"[ollama] cannot reach server at {self.base_url()}: {exc}. Start it with `ollama serve`."
            )

    def _generate(self, prompt):
        options = {"temperature": self.temperature, "num_predict": self.max_output_tokens}
        if self.num_ctx:
            options["num_ctx"] = self.num_ctx
        payload = {"model": self.model, "prompt": prompt, "stream": False, "options": options}
        if self.think is not None:
            payload["think"] = self.think
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url()}/api/generate", data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"[ollama] request to {self.base_url()} failed: {exc}. Is `ollama serve` running and the model pulled?"
            )
        self.last_meta = {"thinking": data.get("thinking"),
                          "prompt_tokens": data.get("prompt_eval_count"),
                          "completion_tokens": data.get("eval_count")}
        return str(data.get("response", "")).strip()


PROVIDER_CLASSES = {
    "openai": OpenAIProvider,
    "azure_openai": AzureOpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
}


_REASONING_OVERRIDE = {"auto": None, "on": True, "off": False}
_THINK_OVERRIDE = {"auto": None, "on": True, "off": False}


def make_provider(args) -> BaseProvider:
    cls = PROVIDER_CLASSES[args.provider]
    return cls(args.model, args.temperature, args.max_output_tokens, mock=args.mock,
               reasoning=_REASONING_OVERRIDE[getattr(args, "reasoning", "auto")],
               num_ctx=getattr(args, "num_ctx", None),
               think=_THINK_OVERRIDE[getattr(args, "think", "auto")])


# --------------------------------------------------------------------------- io helpers
def safe_name(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "-._") else "-" for c in s)


def resolve_out_path(args) -> Path:
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out = out.resolve()
        wl = LOCAL.resolve()
        if not out.is_relative_to(wl):
            raise SystemExit(
                f"--out must be under local/. Got: {out}"
            )
        return out
    fname = f"{args.provider}_{safe_name(args.model)}_{args.split}.jsonl"
    return BASELINES_DIR / fname


def load_prompt_records(args) -> list[dict]:
    path = ROOT / args.prompt_file if not Path(args.prompt_file).is_absolute() else Path(args.prompt_file)
    if not path.exists():
        raise SystemExit(f"prompt file not found: {path} (run scripts/make_prompt.py first)")
    recs = []
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("split") == args.split:
            recs.append(r)
    if args.limit is not None:
        recs = recs[: args.limit]
    return recs


def already_done(out_path: Path) -> set:
    done = set()
    if out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            if line.strip():
                try:
                    done.add(json.loads(line)["qa_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


# --------------------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Provider-agnostic LLM baseline runner.")
    ap.add_argument("--provider", required=True, choices=PROVIDERS)
    ap.add_argument("--model", required=True, help="model name (Azure: deployment name)")
    ap.add_argument("--split", required=True, choices=list(PUBLIC_SPLITS))
    ap.add_argument("--prompt-file", default=DEFAULT_PROMPT_FILE)
    ap.add_argument("--out", default=None, help="output JSONL (must be under local/); default auto")
    ap.add_argument("--limit", type=int, default=None, help="only the first N prompts (smoke tests)")
    ap.add_argument("--dry-run", action="store_true", help="print planned requests; no SDK/key/API calls")
    ap.add_argument("--mock", action="store_true", help="run the loop with a deterministic fake response (offline)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-output-tokens", type=int, default=512)
    ap.add_argument("--num-ctx", type=int, default=None,
                    help="ollama only: context window (e.g. 65536). The ollama default (4096) silently "
                         "truncates long full-context bundles, so set this for long-context runs.")
    ap.add_argument("--think", choices=["auto", "on", "off"], default="auto",
                    help="ollama only: force chain-of-thought on/off via ollama's top-level `think`. "
                         "auto = model default. Use 'off' for cb/rag (reasoning models over-think short "
                         "prompts and can exhaust --max-output-tokens on the thinking trace → empty output); "
                         "keep 'auto'/'on' for full-context where the answer needs reasoning.")
    ap.add_argument("--reasoning", choices=["auto", "on", "off"], default="auto",
                    help="OpenAI/Azure token-param mode: auto = detect reasoning models (gpt-5*/o-series) "
                         "by name; on/off = force (use for Azure deployments whose name hides the model).")
    ap.add_argument("--resume", action="store_true", help="skip qa_ids already present in the output file")
    ap.add_argument("--sleep-seconds", type=float, default=0.0, help="pause between calls (rate limiting)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.dry_run and args.mock:
        print("choose at most one of --dry-run / --mock", file=sys.stderr)
        return 2

    records = load_prompt_records(args)
    out_path = resolve_out_path(args)
    provider = make_provider(args)

    if args.dry_run:
        print(f"== DRY RUN — provider={args.provider} model={args.model} split={args.split} ==")
        print(f"prompt_file: {args.prompt_file}")
        print(f"planned out: {out_path.relative_to(ROOT) if ROOT in out_path.parents else out_path}")
        print(f"temperature={args.temperature} max_output_tokens={args.max_output_tokens} "
              f"limit={args.limit} planned_requests={len(records)}")
        for i, r in enumerate(records, 1):
            prompt = select_prompt(r)
            if isinstance(r.get("prompt"), str) and r["prompt"].strip():
                preview = f"[internal prebuilt prompt: {len(prompt)} chars — INTERNAL, preview redacted]"
            else:
                preview = prompt.replace("\n", " ")[:160] + "…"
            print(f"  [{i}] qa_id={r.get('qa_id')} task={r.get('task_type')} prompt~= {preview}")
        print("== no SDK imported, no API called, no files written ==")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = already_done(out_path) if args.resume else set()
    queue = [r for r in records if r.get("qa_id") not in done]

    if not args.mock:
        provider.ensure_ready()

    log_path = out_path.with_suffix(".log")
    meta_path = out_path.with_suffix(".meta.json")
    calls_path = out_path.with_suffix(".calls.jsonl")
    started = now_iso()
    written, errors = 0, 0

    mode = "w" if not args.resume else "a"
    with out_path.open(mode, encoding="utf-8") as out_f, log_path.open(mode, encoding="utf-8") as log_f, \
            calls_path.open(mode, encoding="utf-8") as calls_f:
        log_f.write(f"# run start {started} provider={args.provider} model={args.model} "
                    f"split={args.split} mock={args.mock} pending={len(queue)} (skipped_resume={len(records) - len(queue)})\n")
        for i, r in enumerate(queue, 1):
            qid = r.get("qa_id")
            prompt = select_prompt(r)
            t0 = time.perf_counter()
            try:
                pred = provider.generate(prompt)
            except SystemExit:
                raise
            except Exception as exc:
                errors += 1
                log_f.write(f"{now_iso()} ERROR qa_id={qid}: {exc!r}\n")
                log_f.flush()
                continue
            latency_s = round(time.perf_counter() - t0, 3)
            out_f.write(json.dumps({"qa_id": qid, "prediction": pred}, ensure_ascii=False) + "\n")
            out_f.flush()
            calls_f.write(json.dumps(
                {"qa_id": qid, "prediction": pred, "latency_s": latency_s,
                 "prompt_chars": len(prompt), **provider.last_meta},
                ensure_ascii=False) + "\n")
            calls_f.flush()
            written += 1
            if args.sleep_seconds > 0 and i < len(queue):
                time.sleep(args.sleep_seconds)
    finished = now_iso()

    metadata = {
        "provider": args.provider,
        "model": args.model,
        "split": args.split,
        "prompt_file": args.prompt_file,
        "out_file": str(out_path.relative_to(ROOT)) if ROOT in out_path.parents else str(out_path),
        "started_at": started,
        "finished_at": finished,
        "limit": args.limit,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "mock": args.mock,
        "resume": args.resume,
        "split_prompt_count": len(records),
        "attempted": len(queue),
        "written": written,
        "skipped_resume": len(records) - len(queue),
        "errors": errors,
        "cost_usd": None,
        "args": vars(args),
        "argv": sys.argv,
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    tag = "MOCK" if args.mock else "LIVE"
    print(f"== baseline run ({tag}) — {written} predictions written, {errors} errors ==")
    print(f"predictions: {out_path.relative_to(ROOT) if ROOT in out_path.parents else out_path}")
    print(f"metadata:    {meta_path.relative_to(ROOT) if ROOT in meta_path.parents else meta_path}")
    print(f"score with:  python3 scripts/eval_harness.py --pred {out_path.relative_to(ROOT) if ROOT in out_path.parents else out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
