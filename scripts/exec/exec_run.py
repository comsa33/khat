"""Filter-execution run: build, submit, parse and score.

Usage: python3 scripts/exec/exec_run.py build|submit|status|fetch|ollama|parse|score ...
"""
import collections, hashlib, json, math, random, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "two_stage"))
import two_stage_run as runR
import exec_build as B
import executor as X
import exec_score as S

FN = re.compile(r"^(?:resub\d+_)?(openai|anthropic|ollama)_(toolretryempty|toolretry|toolempty|tool)_(.+?)(?:__(test|dev))?\.jsonl$")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
jl = runR.jl


def run_requests(cmd, fn, workers=3):
    if cmd == "ollama": runR.cmd_ollama(fn, workers)
    else: runR.cmd_batch(cmd, fn)


# ------------------------------------------------------------------ parsing
KIND_EMPTY = {"tool": "toolempty", "toolretry": "toolretryempty"}


def stage(R, prov, kind, m, part, reqs, pending, retry_ok):
    """one turn of one model file: first pass + resubmission rounds (API failures, at most 2), then one retry at twice
    the cap for empty answers (as in the two-stage run) with its own resubmission rounds. Returns {custom_id: result} or None while
    a file is still to be run."""
    stem = f"{prov}_{kind}_{m}" + (f"__{part}" if part else "")
    fr = family_results(R, stem, prov, m, reqs, pending)
    if fr is None: return None
    res = dict(fr["res"]); mism = list(fr["mismatch"])
    empties = sorted(c for c in reqs if res.get(c, {}).get("ok") and not (res[c].get("text") or "").strip())
    if empties:
        ename = f"{prov}_{KIND_EMPTY[kind]}_{m}" + (f"__{part}" if part else "") + ".jsonl"; ef = R / ename
        rows = [runR.set_cap(reqs[c], 2 * runR.get_cap(reqs[c])) for c in empties]
        if ef.exists():
            assert runR.jl(ef) == rows, f"{ename} differs from the empty answers"
        else:
            import check_requests as CR
            errs = CR.check_rows(rows, m, prov, set(empties), cap=2 * runR.CAP[m])
            if errs: sys.exit(f"{ename} failed validation: {errs[:3]}")
            runR.wjl(ef, rows); retry_ok[ename] = sha(ef)
        er = family_results(R, ename[:-6], prov, m, {r["custom_id"]: r for r in rows}, pending)
        if er is None: return None
        mism += er["mismatch"]
        for c in empties:
            if er["res"].get(c, {}).get("ok"): res[c] = er["res"][c]
    return {"res": res, "mismatch": mism}


def turn(res):
    """one final response -> dict(status, ...); status executed | parse_fail | exec_error | empty_output | api_failed"""
    if not res or not res.get("ok"): return {"status": "api_failed"}
    text = res.get("text") or ""
    if not text.strip(): return {"status": "empty_output", "capped": res.get("capped")}
    q, err = S.parse(text)
    if err: return {"status": "parse_fail", "error": err, "text": text, "capped": res.get("capped")}
    ex = X.execute(q)
    if not ex["ok"]: return {"status": "exec_error", "error": ex["error"], "query": q, "text": text, "capped": res.get("capped")}
    return {"status": "executed", "query": q, "exec": ex, "capped": res.get("capped")}


def family_results(R, stem, prov, m, reqs, pending):
    """first pass + resubmission rounds of one request file; writes the next resubmission file when needed"""
    if not (R / f"raw_{stem}.jsonl").exists():
        pending.append(f"run {stem}.jsonl"); return None
    res, rounds, mism = runR.merged(R, stem, prov, m)
    failed = [c for c in reqs if not res.get(c, {}).get("ok")]
    if failed and rounds < runR.MAX_RESUB:
        nf = R / f"resub{rounds + 1}_{stem}.jsonl"
        if not nf.exists(): runR.wjl(nf, [reqs[c] for c in failed])
        if not (R / f"raw_resub{rounds + 1}_{stem}.jsonl").exists():
            pending.append(f"run {nf.name} ({len(failed)} API failures)"); return None
        return family_results(R, stem, prov, m, reqs, pending)
    return {"res": res, "mismatch": mism}


def cmd_parse(Sdir, quiet=False):
    """no correctness here: statuses are mechanical; scoring is cmd_score (main only)"""
    Sdir = Path(Sdir); R = Sdir / "requests"; P = Sdir / "preds"; P.mkdir(exist_ok=True)
    man = json.load(open(Sdir / "manifest.json")); prompts = {r["qa_id"]: r["prompt"] for r in jl(Sdir / "prompts_tool.jsonl")}
    retry_ok = json.load(open(Sdir / "retry_ok.json")) if (Sdir / "retry_ok.json").exists() else {}
    pending, out, status = [], collections.defaultdict(dict), {}
    for name in sorted(man["files"]):
        prov, kind, m, part = FN.match(name).groups()
        reqs = {r["custom_id"]: r for r in jl(R / name)}
        st1 = stage(R, prov, "tool", m, part, reqs, pending, retry_ok)
        if st1 is None: continue
        first = {c.split("__", 1)[1]: turn(st1["res"].get(c)) for c in reqs}
        need = {q: t for q, t in first.items() if t["status"] in ("parse_fail", "exec_error")}
        second, mism = {}, list(st1["mismatch"])
        if need:
            rname = f"{prov}_toolretry_{m}" + (f"__{part}" if part else "") + ".jsonl"; rf = R / rname
            rows = [runR.BUILD[prov](f"toolretry__{q}", m, B.retry_prompt(prompts[q], t["text"], t["error"]), runR.CAP[m]) for q, t in sorted(need.items())]
            if rf.exists():
                assert jl(rf) == rows, f"{rname} differs from the items needing a re-ask"
            else:
                import check_requests as CR
                errs = CR.check_rows(rows, m, prov, {f"toolretry__{q}" for q in need})
                if errs: sys.exit(f"retry file {rname} failed validation: {errs[:3]}")
                runR.wjl(rf, rows); retry_ok[rname] = sha(rf)
            st2 = stage(R, prov, "toolretry", m, part, {r["custom_id"]: r for r in rows}, pending, retry_ok)
            if st2 is None: continue
            second = {c.split("__", 1)[1]: turn(st2["res"].get(c)) for c in st2["res"] if c.startswith("toolretry__")}
            for q in need: second.setdefault(q, {"status": "api_failed"})
            mism += st2["mismatch"]
        for q, t in first.items():
            fin = second[q] if q in need else t
            out[m][q] = {"qa_id": q, "split": "test" if q in B.T81 else "dev", "status": fin["status"], "first": t["status"], "reasked": q in need,
                         "query": fin.get("query"), "value": (fin.get("exec") or {}).get("value"), "n_rows": (fin.get("exec") or {}).get("n_rows"),
                         "error": fin.get("error"), "capped": bool(fin.get("capped"))}
        status[name] = {"model": m, "part": part, "n": len(reqs), "reasked": len(need), "model_mismatch": len(mism),
                        "final": dict(collections.Counter(out[m][q]["status"] for q in first))}
    json.dump(retry_ok, open(Sdir / "retry_ok.json", "w"), indent=1)
    for m, d in out.items():
        prev = {r["qa_id"]: r for r in jl(P / f"tool_{m}.jsonl")} if (P / f"tool_{m}.jsonl").exists() else {}
        runR.wjl(P / f"tool_{m}.jsonl", [d.get(q) or prev[q] for q in sorted(set(d) | set(prev))])
    json.dump({"status": status, "pending": pending}, open(Sdir / "status.json", "w"), indent=1, ensure_ascii=False)
    if not quiet:
        for n, s in status.items(): print(n, s)
        print("pending:", pending if pending else "none")
    return out, status, pending


# ------------------------------------------------------------------ scoring
def wilson(k, n, z=1.959963984540054):
    if n == 0: return float("nan"), float("nan")
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * (c - h) / d, 100 * (c + h) / d


def boot(items, unit, v, B_=10000, seed=20261011):
    groups = collections.defaultdict(list)
    for q in items: groups[unit(q)].append(q)
    keys = sorted(groups); rnd = random.Random(seed)
    out = sorted((lambda s: sum(v[q] for q in s) / len(s))([q for k in (rnd.choice(keys) for _ in keys) for q in groups[k]]) for _ in range(B_))
    return 100 * out[int(0.025 * B_)], 100 * out[int(0.975 * B_) - 1]


def final_correct(r):
    return r["status"] == "executed" and S.score(r["value"], S.gold_values(r["qa_id"]))


def cmd_score(Sdir, models=None):
    """main set only: deterministic scoring, Wilson, pair bootstrap (test and dev'), cluster bootstrap (test only;
    dev' has 3 clusters), failure types"""
    Sdir = Path(Sdir); G, _, _ = S._gold_src(); (Sdir / "score").mkdir(exist_ok=True)
    assert json.load(open(Sdir / "manifest.json"))["set"] == "main", "only the main set is scored"
    pair = lambda q: (G[q]["announcement_ids"][0], tuple(G[q]["source_ids"])); clu = lambda q: G[q]["cluster_id"]
    lines, tex = [], []
    for m in (models or runR.MODELS):
        f = Sdir / "preds" / f"tool_{m}.jsonl"
        if not f.exists(): lines.append(f"{m}: no predictions"); continue
        d = {r["qa_id"]: r for r in jl(f)}; cells = []
        for split in ("test", "dev"):
            qs = sorted(q for q, r in d.items() if r["split"] == split)
            if not qs: continue
            v = {q: final_correct(d[q]) for q in qs}; k = sum(v.values()); w = wilson(k, len(qs)); pb = boot(qs, pair, v)
            cb = f" cluster [{boot(qs, clu, v)[0]:.0f}, {boot(qs, clu, v)[1]:.0f}]" if split == "test" else ""
            types = collections.Counter("correct" if v[q] else ("no_value" if d[q]["status"] == "executed" and d[q]["value"] is None
                                                                   else "wrong_value" if d[q]["status"] == "executed" else d[q]["status"]) for q in qs)
            lines.append(f"{m:18s} {split:4s} {100 * k / len(qs):5.1f} ({k}/{len(qs)}) Wilson [{w[0]:.0f}, {w[1]:.0f}] pair [{pb[0]:.0f}, {pb[1]:.0f}]{cb} "
                         f"re-asked {sum(d[q]['reasked'] for q in qs)} types {dict(types)}")
            cells.append(f"{100 * k / len(qs):.0f}")
        tex.append(f"\\texttt{{{m}}} & " + " & ".join(cells) + " \\\\")
    (Sdir / "score" / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (Sdir / "score" / "tab_tool_rows.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return lines


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "build": B.build(sys.argv[2], sys.argv[3])
    elif c in ("submit", "status", "fetch"): run_requests(c, sys.argv[2])
    elif c == "ollama": run_requests("ollama", sys.argv[2], int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 3)
    elif c == "parse": sys.exit(1 if cmd_parse(sys.argv[2])[2] else 0)
    elif c == "score": cmd_score(sys.argv[2])
    else: sys.exit(__doc__)
