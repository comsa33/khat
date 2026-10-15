"""Two-stage run: build requests, submit and fetch batches, run Ollama models, parse outputs.

Usage: python3 scripts/two_stage/two_stage_run.py build|submit|status|fetch|ollama|parse ...
"""
import json, re, sys, time, collections, concurrent.futures as cf, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
from khat.paths import ROOT as DS, DATA, LOCAL
BL = LOCAL / "runs"
sys.path.insert(0, str(HERE.parent))
from run_baseline import _openai_chat_kwargs

OAI = {"gpt-5.5": "gpt-5.5-2026-04-23", "gpt-5.4-mini": "gpt-5.4-mini-2026-03-17",
       "gpt-5.4-nano": "gpt-5.4-nano-2026-03-17", "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14"}
ANT = {"claude-sonnet-5-5": {"type": "between_tools"}, "claude-haiku-5-5": {"type": "disabled"}}
OLL = {"glm-5.2": "glm-5.2:cloud", "minimax-m3": "minimax-m3:cloud"}
CAP = {"gpt-5.5": 16000, "gpt-5.4-mini": 16000, "gpt-5.4-nano": 16000, "gpt-4.1-mini": 4000,
       "claude-sonnet-5-5": 16384, "claude-haiku-5-5": 16384, "glm-5.2": 16384, "minimax-m3": 32768}
NUM_CTX, OLL_TEMP, MAX_RESUB = 131072, 0.0, 2
MODELS = list(OAI) + list(ANT) + list(OLL)
PRICE = {"gpt-5.5": (2.50, 15.0), "gpt-5.4-mini": (0.375, 2.25), "gpt-5.4-nano": (0.10, 0.625), "gpt-4.1-mini": (0.20, 0.80)}
ids = lambda f: [l.strip() for l in open(DATA / "item_sets" / f) if l.strip()]
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def wjl(p, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")


def base_model(name):
    for k, v in OAI.items():
        if name in (k, v): return k
    for k, v in OLL.items():
        if name in (k, v): return k
    return name


# ------------------------------------------------------------------ request builders
def openai_req(cid, m, prompt, cap):
    return {"custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
            "body": _openai_chat_kwargs(OAI[m], prompt, 0.0, cap, None)}


def anthropic_req(cid, m, prompt, cap):
    return {"custom_id": cid, "params": {"model": m, "max_tokens": cap, "thinking": ANT[m],
                                         "messages": [{"role": "user", "content": prompt}]}}


def ollama_req(cid, m, prompt, cap):
    return {"custom_id": cid, "model": OLL[m], "prompt": prompt, "think": True,
            "options": {"temperature": OLL_TEMP, "num_predict": cap, "num_ctx": NUM_CTX}}


def provider(m):
    return "openai" if m in OAI else "anthropic" if m in ANT else "ollama"


BUILD = {"openai": openai_req, "anthropic": anthropic_req, "ollama": ollama_req}


def set_cap(req, cap):
    r = json.loads(json.dumps(req))
    if "body" in r: r["body"]["max_completion_tokens" if "max_completion_tokens" in r["body"] else "max_tokens"] = cap
    elif "params" in r: r["params"]["max_tokens"] = cap
    else: r["options"]["num_predict"] = cap
    return r


def get_cap(req):
    if "body" in req: return req["body"].get("max_completion_tokens") or req["body"]["max_tokens"]
    if "params" in req: return req["params"]["max_tokens"]
    return req["options"]["num_predict"]


def sources(W):
    tw = {r["qa_id"]: r["prompt"] for r in jl(W / "prompts_twostage.jsonl")}
    dv = {r["qa_id"]: r["prompt"] for r in jl(BL / "rag_bm25_dev_fixed_prompts.jsonl")}
    bt = {r["qa_id"]: r["prompt"] for r in jl(BL / "rag_bm25_agg81_fixed_prompts.jsonl")}
    return {"twostage": tw, "devtop5": dv, "bm25test": bt}


def items(setname, S):
    T, D = ids("items_test81.txt"), ids("items_dev144.txt")
    assert setname == "main", setname
    return {"twostage": T + D, "devtop5": D, "bm25test": T}


def parts(setname, cond, qs):
    if setname == "main" and cond == "twostage":
        T = set(ids("items_test81.txt"))
        return [("test", [q for q in qs if q in T]), ("dev", [q for q in qs if q not in T])]
    return [(None, qs)]


FNAME = re.compile(r"(openai|anthropic|ollama)_([a-z0-9]+)_(.+?)(?:__(test|dev))?\.jsonl$")


def cmd_build(W, setname):
    S, out = sources(W), W / setname
    (out / "requests").mkdir(parents=True, exist_ok=True)
    IT = items(setname, S); man = {"set": setname, "items": IT, "caps": CAP, "openai_ids": OAI, "anthropic_thinking": ANT,
                                   "ollama_tags": OLL, "num_ctx": NUM_CTX, "files": {}}
    for cond, qs in IT.items():
        for m in MODELS:
            for part, pq in parts(setname, cond, qs):
                rows = [BUILD[provider(m)](f"{cond}__{q}", m, S[cond][q], CAP[m]) for q in pq]
                fn = out / "requests" / (f"{provider(m)}_{cond}_{m}" + (f"__{part}" if part else "") + ".jsonl")
                wjl(fn, rows); man["files"][fn.name] = len(rows)
    json.dump(man, open(out / "manifest.json", "w"), indent=1, ensure_ascii=False)
    print(setname, {c: len(q) for c, q in IT.items()}, "files", len(man["files"]), "requests", sum(man["files"].values()))


# ------------------------------------------------------------------ API steps
def cmd_batch(cmd, fn):
    fn = Path(fn).resolve(); prov = "openai" if "openai_" in fn.name else "anthropic"
    track = fn.with_name("track_" + fn.stem + ".json"); raw = fn.with_name("raw_" + fn.stem + ".jsonl")
    rows = jl(fn); assert len({r["custom_id"] for r in rows}) == len(rows)
    if prov == "openai":
        from openai import OpenAI; c = OpenAI()
        if cmd == "submit":
            assert not track.exists(), track
            f = c.files.create(file=open(fn, "rb"), purpose="batch")
            b = c.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions", completion_window="24h")
            json.dump({"id": b.id, "file": fn.name, "n": len(rows), "submitted": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, open(track, "w"), indent=1)
            print("submitted", b.id, len(rows)); return
        b = c.batches.retrieve(json.load(open(track))["id"]); print(fn.name, b.status, b.request_counts)
        if cmd == "fetch":
            assert b.status in ("completed", "expired", "cancelled") and not raw.exists(), (b.status, raw)
            raw.write_bytes(c.files.content(b.output_file_id).read() if b.output_file_id else b"")
            if b.error_file_id: fn.with_name("err_" + fn.stem + ".jsonl").write_bytes(c.files.content(b.error_file_id).read())
            print("saved", raw)
    else:
        import anthropic; c = anthropic.Anthropic()
        if cmd == "submit":
            assert not track.exists(), track
            b = c.messages.batches.create(requests=rows)
            json.dump({"id": b.id, "file": fn.name, "n": len(rows), "submitted": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, open(track, "w"), indent=1)
            print("submitted", b.id, len(rows)); return
        b = c.messages.batches.retrieve(json.load(open(track))["id"]); print(fn.name, b.processing_status, b.request_counts)
        if cmd == "fetch":
            assert b.processing_status == "ended" and not raw.exists(), (b.processing_status, raw)
            wjl(raw, [json.loads(r.model_dump_json()) for r in c.messages.batches.results(b.id)])
            print("saved", raw)


def ollama_show(base, tag):
    try:
        req = urllib.request.Request(f"{base}/api/show", data=json.dumps({"model": tag}).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read().decode())
        return {k: d.get(k) for k in ("digest", "modified_at", "remote_model", "remote_host")} | {"details": d.get("details")}
    except Exception as e:
        return {"error": repr(e)}


def cmd_ollama(fn, workers):
    import os
    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    fn = Path(fn).resolve(); raw = fn.with_name("raw_" + fn.stem + ".jsonl"); meta = fn.with_name("meta_" + fn.stem + ".json")
    rows = jl(fn); tags = sorted({r["model"] for r in rows})
    done = {r["custom_id"] for r in jl(raw) if r.get("ok")} if raw.exists() else set()
    queue = [r for r in rows if r["custom_id"] not in done]
    info = json.load(open(meta)) if meta.exists() else {"file": fn.name, "runs": []}
    info["runs"].append({"start": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "show_start": {t: ollama_show(base, t) for t in tags}, "queue": len(queue)})

    def one(r):
        payload = {"model": r["model"], "prompt": r["prompt"], "stream": False, "think": r["think"], "options": r["options"]}
        t0 = time.time()
        try:
            req = urllib.request.Request(f"{base}/api/generate", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=1800) as h:
                d = json.loads(h.read().decode())
            return {"custom_id": r["custom_id"], "ok": True, "response": d.get("response", ""), "thinking_chars": len(d.get("thinking") or ""),
                    "model": d.get("model"), "done_reason": d.get("done_reason"), "prompt_eval_count": d.get("prompt_eval_count"),
                    "eval_count": d.get("eval_count"), "latency_s": round(time.time() - t0, 2)}
        except Exception as e:
            return {"custom_id": r["custom_id"], "ok": False, "error": repr(e), "latency_s": round(time.time() - t0, 2)}

    with open(raw, "a", encoding="utf-8") as f, cf.ThreadPoolExecutor(workers) as ex:
        for res in ex.map(one, queue):
            f.write(json.dumps(res, ensure_ascii=False) + "\n"); f.flush()
    info["runs"][-1].update({"end": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "show_end": {t: ollama_show(base, t) for t in tags}})
    json.dump(info, open(meta, "w"), indent=1, ensure_ascii=False)
    print(fn.name, "done", len(queue), "->", raw.name)


# ------------------------------------------------------------------ parsing
def parse_raw(prov, r):
    """one raw row -> (custom_id, result) with result = {"ok", "text", "capped", "model", "in", "out", "think"}"""
    if prov == "openai":
        body = (r.get("response") or {}).get("body") or {}
        if r.get("error") or (r.get("response") or {}).get("status_code", 200) != 200 or not body.get("choices"):
            return r["custom_id"], {"ok": False}
        c, u = body["choices"][0], body.get("usage") or {}
        return r["custom_id"], {"ok": True, "text": (c["message"].get("content") or "").strip(), "capped": c.get("finish_reason") == "length",
                                "model": body.get("model"), "in": u.get("prompt_tokens"), "out": u.get("completion_tokens"),
                                "think": (u.get("completion_tokens_details") or {}).get("reasoning_tokens")}
    if prov == "anthropic":
        x = r["result"]
        if x["type"] != "succeeded":
            return r["custom_id"], {"ok": False}
        mm, u = x["message"], x["message"].get("usage") or {}
        txt = "".join(b.get("text", "") for b in mm["content"] if b.get("type") == "text").strip()
        return r["custom_id"], {"ok": True, "text": txt, "capped": mm.get("stop_reason") == "max_tokens", "model": mm.get("model"),
                                "in": u.get("input_tokens"), "out": u.get("output_tokens"), "think": (u.get("output_tokens_details") or {}).get("thinking_tokens")}
    if not r.get("ok"):
        return r["custom_id"], {"ok": False}
    return r["custom_id"], {"ok": True, "text": str(r.get("response") or "").strip(), "capped": r.get("done_reason") == "length",
                            "model": r.get("model"), "in": r.get("prompt_eval_count"), "out": r.get("eval_count"), "think": r.get("thinking_chars")}


RESP_MODEL = {**OAI, **{m: m for m in ANT}, **{m: t.removesuffix(":cloud") for m, t in OLL.items()}}


def pinned(m):
    return RESP_MODEL[m]


def merged(d, stem, prov, m):
    """results of request file <stem> merged over raw_<stem> and raw_resub<k>_<stem>; a success in a later round
    wins; a success whose model field differs from the pinned id counts as failed and is listed (every round)"""
    out, rounds, mism = {}, 0, []
    for k in range(0, MAX_RESUB + 1):
        f = d / (f"raw_{stem}.jsonl" if k == 0 else f"raw_resub{k}_{stem}.jsonl")
        if not f.exists(): continue
        rounds = k
        for r in jl(f):
            cid, res = parse_raw(prov, r)
            if res["ok"] and res["model"] != pinned(m):
                mism.append((f.name, cid, res["model"])); res = {"ok": False, "model_mismatch": res["model"]}
            if res["ok"] or cid not in out or not out[cid]["ok"]:
                out[cid] = res
    return out, rounds, mism


def digests(R, stem):
    """Ollama model identity seen at every run start / end of a request family (meta_*.json): the digest, or, when
    /api/show returns none (cloud tags), modified_at + parent_model; a change is a flag only"""
    seen = set()
    for f in R.glob(f"meta_*{stem}.json"):
        for run in json.load(open(f))["runs"]:
            for k in ("show_start", "show_end"):
                for d in (run.get(k) or {}).values():
                    ident = d.get("digest") or (f"{d.get('modified_at')}|{(d.get('details') or {}).get('parent_model')}" if d.get("modified_at") else None)
                    if ident: seen.add(ident)
    return sorted(seen)


def family(R, stem, prov, m, reqs, label, pending, fails, cap2=False):
    """one request family (original or one retry round) with its resubmission rounds"""
    res, rounds, mm = merged(R, stem, prov, m); fails += mm
    if not (R / f"raw_{stem}.jsonl").exists():
        pending.append(f"run {stem}.jsonl ({label})"); return res
    failed = [c for c in reqs if not res.get(c, {}).get("ok")]
    if failed and rounds < MAX_RESUB:
        nf = R / f"resub{rounds + 1}_{stem}.jsonl"
        if not nf.exists(): wjl(nf, [reqs[c] for c in failed])
        if not (R / f"raw_resub{rounds + 1}_{stem}.jsonl").exists(): pending.append(f"run {nf.name} ({len(failed)} failed)")
    return res


def cmd_parse(S):
    S = Path(S); R = S / "requests"; P = S / "preds"; P.mkdir(exist_ok=True)
    man = json.load(open(S / "manifest.json"))
    status, pending, fails, preds = {}, [], [], collections.defaultdict(list)
    for name in sorted(man["files"]):
        prov, cond, m, part = FNAME.match(name).groups()
        stem = name[:-6]; reqs = {r["custom_id"]: r for r in jl(R / name)}
        res = family(R, stem, prov, m, reqs, "first pass", pending, fails)
        if not (R / f"raw_{stem}.jsonl").exists(): continue
        empty = [c for c in reqs if res.get(c, {}).get("ok") and not res[c]["text"]]
        rfiles = sorted(R.glob(f"retry[0-9]*_{stem}.jsonl"), key=lambda f: int(re.match(r"retry(\d+)_", f.name).group(1)))
        asked = {r["custom_id"] for f in rfiles for r in jl(f)}
        new = [c for c in empty if c not in asked]
        if new:
            nf = R / f"retry{len(rfiles) + 1}_{stem}.jsonl"; wjl(nf, [set_cap(reqs[c], 2 * get_cap(reqs[c])) for c in new]); rfiles.append(nf)
        rres = {}
        for f in rfiles:
            rq = {r["custom_id"]: r for r in jl(f)}
            rres.update(family(R, f.stem, prov, m, rq, f"{len(rq)} empty", pending, fails))
        cnt = collections.Counter()
        for c in reqs:
            q = c.split("__")[1]; x = res.get(c, {})
            if not x.get("ok"):
                st, txt = "failed", ""
            elif x["text"]:
                st, txt = ("capped" if x["capped"] else "ok"), x["text"]
            else:
                y = rres.get(c, {})
                if y.get("ok") and y["text"]: st, txt = ("retry_capped" if y["capped"] else "retry_ok"), y["text"]
                elif y.get("ok"): st, txt = "still_empty", ""
                else: st, txt = "empty_retry_pending_or_failed", ""
            cnt[st] += 1
            preds[(cond, m)].append({"qa_id": q, "prediction": txt, "status": st})
        tok = [x for c, x in res.items() if x.get("ok")]
        dg = digests(R, stem) if prov == "ollama" else []
        status[name] = {"cond": cond, "model": m, "part": part, "n": len(reqs), "status": dict(cnt), "retry_files": len(rfiles),
                        "max_in": max((x["in"] or 0 for x in tok), default=0), "max_out": max((x["out"] or 0 for x in tok), default=0),
                        "mean_in": round(sum(x["in"] or 0 for x in tok) / max(len(tok), 1)), "mean_out": round(sum(x["out"] or 0 for x in tok) / max(len(tok), 1)),
                        "thinking_present": sum(bool(x.get("think")) for x in tok), "ollama_digests": dg, "digest_changed": len(dg) > 1}
    for (cond, m), rows in preds.items():
        wjl(P / f"{cond}_{m}.jsonl", rows)
    json.dump({"status": status, "pending": pending, "model_field_failures": fails},
              open(S / "status.json", "w"), indent=1, ensure_ascii=False)
    for n, s in status.items():
        print(f"  {n}: {s['status']} | in max {s['max_in']} mean {s['mean_in']} | out max {s['max_out']} mean {s['mean_out']} | "
              f"thinking {s['thinking_present']}" + (" | DIGEST CHANGED (flag)" if s["digest_changed"] else ""))
    print("pending:", pending or "none"); print("model field failures:", fails or "none")
    return status, pending, fails


# ------------------------------------------------------------------ cost
def openai_spend(W):
    tot, per = 0.0, collections.Counter()
    for f in Path(W).rglob("raw*openai*.jsonl"):
        for r in jl(f):
            body = (r.get("response") or {}).get("body") or {}
            u, m = body.get("usage"), base_model(body.get("model") or "")
            if not u or m not in PRICE: continue
            c = (u["prompt_tokens"] * PRICE[m][0] + u["completion_tokens"] * PRICE[m][1]) / 1e6
            tot += c; per[m] += c
    return tot, per


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "build": cmd_build(Path(sys.argv[2]), sys.argv[3])
    elif c in ("submit", "status", "fetch"): cmd_batch(c, sys.argv[2])
    elif c == "ollama": cmd_ollama(sys.argv[2], int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 3)
    elif c == "parse": sys.exit(1 if cmd_parse(sys.argv[2])[2] else 0)
    elif c == "cost": t, p = openai_spend(sys.argv[2]); print(f"OpenAI ${t:.2f}", {k: round(v, 2) for k, v in p.items()})
    else: sys.exit(__doc__)
