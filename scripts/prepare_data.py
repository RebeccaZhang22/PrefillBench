#!/usr/bin/env python3
"""Step 1: raw parquet -> message lists -> deterministic downsampling -> one jsonl per source.

For every source in the sampling config:
  1. convert each raw row to one prefill request: a complete message list (system / user / assistant /
     tool ...) cut right before the final assistant output; agent rows also carry `tools`
       wildchat     allenai/WildChat-1M                        one line per conversation
       swe_rebench  nebius/SWE-rebench-openhands-trajectories  one line per trajectory
  2. count its prefill length with the tokenizer's chat template (add_generation_prompt=True,
     enable_thinking=False). Lengths are cached in <out-dir>/.index/ and reused while the input files and
     tokenizer are unchanged, so re-sampling with another seed / rule takes seconds.
  3. downsample with the rule (numpy default_rng([seed, crc32(name)]), one uniform draw per row in file order):
       drop num_tokens < min_tokens or > max_input_tokens; keep rows shorter than short_len with prob short_keep;
       then draw num_samples rows uniformly without replacement
  4. write the selected rows, in file order, to <out-dir>/<name>.jsonl

Line format (keys in this order):
  {"id", "source", "num_tokens", "num_turns", "num_messages", "num_output_tokens", "meta", ["tools"], "messages"}
  num_turns = assistant messages in the context + 1 (the turn being prefilled)
  num_output_tokens = tokens of the assistant reply that was cut off (content + tool calls + <|im_end|>), i.e.
                      the real response length; used as the generation length in generation mode (0 = none)

Also written: <out-dir>/sample_index.csv (source,id,num_tokens,num_turns) and <out-dir>/manifest.json
(rule, per-source counts, sha256 fingerprint of the selected ids). The whole step is skipped when the
manifest matches the current rule and inputs.
"""
import argparse
import csv
import glob
import hashlib
import json
import multiprocessing as mp
import os
import time
import zlib

import numpy as np
import pyarrow.parquet as pq

TOK = None
INDEX_VERSION = 1
OUTPUT_VERSION = 2  # 2: lines carry num_output_tokens (length of the reply that was cut off)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="sampling json (written by load_config.py)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--workers", type=int, default=0, help="0 = all CPUs")
    return p.parse_args()


def log(msg):
    print(f"[prepare_data {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- raw row -> request

def strip_none(x):
    """parquet merges all tool schemas into one struct; drop the null fields it introduces."""
    if isinstance(x, dict):
        return {k: strip_none(v) for k, v in x.items() if v is not None}
    if isinstance(x, list):
        return [strip_none(v) for v in x if v is not None]
    return x


def drop_trailing_assistant(msgs):
    """Cut the conversation before its final assistant output; return (messages, that assistant message)."""
    answer = None
    while msgs and msgs[-1]["role"] == "assistant":
        answer = msgs.pop()
    return msgs, answer


def convert_wildchat(row):
    msgs = [{"role": m.get("role"), "content": m.get("content")} for m in (row.get("conversation") or [])]
    msgs = [m for m in msgs if m["role"] in ("user", "assistant") and isinstance(m["content"], str) and m["content"].strip()]
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    msgs, answer = drop_trailing_assistant(msgs)
    if not msgs:
        return None
    meta = {k: row.get(k) for k in ("conversation_hash", "model", "language") if row.get(k) is not None}
    return {"id": f"wildchat-{row.get('conversation_hash')}", "meta": meta, "tools": None, "messages": msgs,
            "answer": answer}


def convert_swe(row):
    msgs = []
    for m in row.get("trajectory") or []:
        m = strip_none(m)
        if m.get("role") not in ("system", "user", "assistant", "tool"):
            continue
        m["content"] = m.get("content") or ""
        msgs.append(m)
    msgs, answer = drop_trailing_assistant(msgs)
    if not any(m["role"] in ("user", "tool") for m in msgs):
        return None
    meta = {k: row.get(k) for k in ("trajectory_id", "instance_id", "repo", "resolved", "exit_status")
            if row.get(k) is not None}
    return {"id": f"swe_rebench-{row.get('trajectory_id')}", "meta": meta,
            "tools": strip_none(row.get("tools") or []) or None, "messages": msgs, "answer": answer}


FORMATS = {  # converter, parquet columns, read batch size
    "wildchat": (convert_wildchat, ["conversation_hash", "model", "language", "conversation"], 1024),
    "swe_rebench": (convert_swe, ["trajectory_id", "instance_id", "repo", "resolved", "exit_status",
                                  "trajectory", "tools"], 32),
}


def init_worker(tokenizer):
    global TOK
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    from transformers import AutoTokenizer
    TOK = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)


def measure(task):
    """(position, fmt, row) -> (position, id, num_tokens, num_turns) or None."""
    pos, fmt, row = task
    rec = FORMATS[fmt][0](row)
    if rec is None:
        return None
    text = TOK.apply_chat_template(rec["messages"], tools=rec["tools"], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    return pos, rec["id"], len(TOK(text, add_special_tokens=False)["input_ids"]), \
        sum(m["role"] == "assistant" for m in rec["messages"]) + 1


def answer_tokens(answer):
    """Tokens the model would generate for the cut-off reply: content + tool calls + <|im_end|>; 0 if none."""
    if not answer:
        return 0
    text = answer.get("content") or ""
    for tc in answer.get("tool_calls") or []:
        fn = tc.get("function") or {}
        text += '\n<tool_call>\n{"name": "' + (fn.get("name") or "") + '", "arguments": ' + \
            (fn.get("arguments") or "{}") + "}\n</tool_call>"
    return len(TOK(text, add_special_tokens=False)["input_ids"]) + 1


def render_line(fmt, row, ntok, nturn):
    """Returns (json line, num_output_tokens)."""
    rec = FORMATS[fmt][0](row)
    nout = answer_tokens(rec["answer"])
    out = {"id": rec["id"], "source": fmt, "num_tokens": int(ntok), "num_turns": int(nturn),
           "num_messages": len(rec["messages"]), "num_output_tokens": nout, "meta": rec["meta"]}
    if rec["tools"]:
        out["tools"] = rec["tools"]
    out["messages"] = rec["messages"]
    return json.dumps(out, ensure_ascii=False) + "\n", nout


# ---------------------------------------------------------------- helpers

def parquet_files(path):
    files = [path] if os.path.isfile(path) else sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"[prepare_data] no parquet files under {path} (see data/README.md for download)")
    return files


def input_signature(files):
    return [[os.path.abspath(f), os.path.getsize(f)] for f in files]


def iter_rows(files, columns, batch_size):
    pos = 0
    for f in files:
        pf = pq.ParquetFile(f)
        cols = [c for c in columns if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
            for row in batch.to_pylist():
                yield pos, row
                pos += 1


def build_index(src, fmt, files, tokenizer, workers, cache_path):
    """Per convertible row: position in the raw files, id, num_tokens, num_turns (cached)."""
    sig = {"version": INDEX_VERSION, "format": fmt, "tokenizer": tokenizer, "files": input_signature(files)}
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=False)
        if json.loads(str(z["sig"])) == sig:
            log(f"{src}: reusing token-length index {cache_path}")
            return z["pos"], list(z["ids"]), z["ntok"], z["nturn"]
    _, columns, batch_size = FORMATS[fmt]
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    log(f"{src}: counting prefill tokens of {total:,} rows with {workers} workers (cached afterwards) ...")
    pos, ids, ntok, nturn = [], [], [], []
    t0 = time.time()
    tasks = ((p, fmt, r) for p, r in iter_rows(files, columns, batch_size))
    with mp.Pool(workers, initializer=init_worker, initargs=(tokenizer,)) as pool:
        for i, res in enumerate(pool.imap(measure, tasks, chunksize=4 if fmt == "swe_rebench" else 256), 1):
            if res is not None:
                pos.append(res[0])
                ids.append(res[1])
                ntok.append(res[2])
                nturn.append(res[3])
            if i % (5000 if fmt == "swe_rebench" else 50000) == 0:
                el = time.time() - t0
                log(f"  {i:,}/{total:,} rows ({el:.0f}s, ETA {el / i * (total - i):.0f}s)")
    pos, ntok, nturn = np.array(pos, dtype=np.int64), np.array(ntok, dtype=np.int64), np.array(nturn, dtype=np.int64)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, sig=json.dumps(sig), pos=pos, ids=np.array(ids), ntok=ntok, nturn=nturn)
    log(f"{src}: {len(pos):,}/{total:,} rows usable [{time.time() - t0:.0f}s]")
    return pos, ids, ntok, nturn


def select(rule, ntok, max_input, seed):
    rng = np.random.default_rng([seed, zlib.crc32(rule["name"].encode())])
    u = rng.random(len(ntok))  # one draw per row, file order
    ok = (ntok >= rule.get("min_tokens", 0)) & (ntok <= max_input)
    ok &= (ntok >= rule.get("short_len", 0)) | (u < rule.get("short_keep", 1.0))
    cand = np.flatnonzero(ok)
    k = min(rule["num_samples"], len(cand))
    return np.sort(rng.choice(cand, size=k, replace=False)), int(ok.sum())


def main():
    args = parse_args()
    cfg = json.load(open(args.config))
    workers = args.workers or os.cpu_count() or 1
    os.makedirs(args.out_dir, exist_ok=True)
    man_path = os.path.join(args.out_dir, "manifest.json")

    sources = [(s, parquet_files(s["input"])) for s in cfg["sources"]]
    rule_key = {"sampling": {k: v for k, v in cfg.items() if k != "length_buckets"},
                "inputs": {s["name"]: input_signature(f) for s, f in sources}, "output_version": OUTPUT_VERSION}
    if os.path.exists(man_path):
        old = json.load(open(man_path))
        if old.get("rule_key") == rule_key and all(
                os.path.exists(os.path.join(args.out_dir, s["name"] + ".jsonl")) for s, _ in sources):
            log(f"outputs in {args.out_dir} already match this rule and these inputs, skipping")
            return

    init_worker(cfg["tokenizer"])  # main-process tokenizer, for the output lengths of the selected rows
    fp = hashlib.sha256()
    summary = []
    with open(os.path.join(args.out_dir, "sample_index.csv"), "w", newline="") as idx_f:
        w = csv.writer(idx_f)
        w.writerow(["source", "id", "num_tokens", "num_turns", "num_output_tokens"])
        for s, files in sources:
            name, fmt = s["name"], s["format"]
            cache = os.path.join(args.out_dir, ".index", f"{name}.npz")
            pos, ids, ntok, nturn = build_index(name, fmt, files, cfg["tokenizer"], workers, cache)
            pick, n_eligible = select(s, ntok, cfg["max_input_tokens"], cfg["seed"])
            want = dict(zip(pos[pick].tolist(), pick.tolist()))
            out_path = os.path.join(args.out_dir, name + ".jsonl")
            t0 = time.time()
            _, columns, batch_size = FORMATS[fmt]
            nout = {}
            with open(out_path + ".tmp", "w") as out:
                for p, row in iter_rows(files, columns, batch_size):
                    j = want.get(p)
                    if j is not None:
                        line, nout[j] = render_line(fmt, row, ntok[j], nturn[j])
                        out.write(line)
            os.replace(out_path + ".tmp", out_path)
            for j in pick:
                w.writerow([name, ids[j], ntok[j], nturn[j], nout[j]])
                fp.update(ids[j].encode() + b"\n")
            if len(pick) < s["num_samples"]:
                log(f"WARNING {name}: only {len(pick)} eligible rows, wanted {s['num_samples']}")
            summary.append({"name": name, "file": name + ".jsonl", "lines": len(ntok),
                            "over_max_input": int((ntok > cfg["max_input_tokens"]).sum()),
                            "eligible_after_rule": n_eligible, "sampled": int(len(pick)),
                            "sampled_tokens": int(ntok[pick].sum()),
                            "sampled_output_tokens": int(sum(nout.values())),
                            "size_bytes": os.path.getsize(out_path)})
            log(f"{name}: {len(ntok):,} rows -> {n_eligible:,} eligible -> {len(pick):,} sampled "
                f"({ntok[pick].sum() / 1e6:.1f}M tokens) -> {out_path} [{time.time() - t0:.0f}s]")
    manifest = {"config": cfg, "rule_key": rule_key, "sources": summary,
                "total_sampled": sum(x["sampled"] for x in summary), "fingerprint_sha256": fp.hexdigest()}
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log(f"fingerprint (sha256 of selected ids): {manifest['fingerprint_sha256']}")


if __name__ == "__main__":
    main()
