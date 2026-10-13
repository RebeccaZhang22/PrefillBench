#!/usr/bin/env python3
"""Convert a raw dataset into the full (not downsampled) benchmark jsonl.

One line = one prefill request = one complete message list (system / user / assistant / tool ...),
cut right before the final assistant output: everything in `messages` is context, and the model
would start generating the next assistant message. Agent lines also carry the `tools` schemas.

  wildchat     allenai/WildChat-1M parquet           -> one line per conversation
  swe_rebench  nebius/SWE-rebench-openhands-trajectories -> one line per trajectory

Line format (keys in this order):
  {"id", "source", "num_tokens", "num_turns", "num_messages", "meta", ["tools"], "messages"}
  num_tokens  prompt length with the tokenizer's chat template (add_generation_prompt=True,
              enable_thinking=False), i.e. the exact prefill length
  num_turns   assistant messages in the context + 1 (the turn being prefilled)

A <output>.manifest.json with row counts, length stats and the sha256 of the file is written next to it.
"""
import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import time

import numpy as np
import pyarrow.parquet as pq

TOK = None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, choices=["wildchat", "swe_rebench"])
    p.add_argument("--input", required=True, help="parquet file or directory (searched recursively)")
    p.add_argument("--output", required=True, help="output .jsonl")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-8B-FP8")
    p.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 1))
    p.add_argument("--limit", type=int, default=0, help="only convert the first N rows (for quick checks)")
    return p.parse_args()


def log(msg):
    print(f"[build_jsonl {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def strip_none(x):
    """parquet merges all tool schemas into one struct; drop the null fields it introduces."""
    if isinstance(x, dict):
        return {k: strip_none(v) for k, v in x.items() if v is not None}
    if isinstance(x, list):
        return [strip_none(v) for v in x if v is not None]
    return x


def drop_trailing_assistant(msgs):
    while msgs and msgs[-1]["role"] == "assistant":
        msgs.pop()
    return msgs


def convert_wildchat(row):
    msgs = [{"role": m.get("role"), "content": m.get("content")} for m in (row.get("conversation") or [])]
    msgs = [m for m in msgs if m["role"] in ("user", "assistant") and isinstance(m["content"], str) and m["content"].strip()]
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    msgs = drop_trailing_assistant(msgs)
    if not msgs:
        return None
    meta = {k: row.get(k) for k in ("conversation_hash", "model", "language") if row.get(k) is not None}
    return {"id": f"wildchat-{row.get('conversation_hash')}", "source": "wildchat", "meta": meta, "tools": None,
            "messages": msgs}


def convert_swe(row):
    msgs = []
    for m in row.get("trajectory") or []:
        m = strip_none(m)
        if m.get("role") not in ("system", "user", "assistant", "tool"):
            continue
        m["content"] = m.get("content") or ""
        msgs.append(m)
    msgs = drop_trailing_assistant(msgs)
    if not any(m["role"] in ("user", "tool") for m in msgs):
        return None
    meta = {k: row.get(k) for k in ("trajectory_id", "instance_id", "repo", "resolved", "exit_status")
            if row.get(k) is not None}
    return {"id": f"swe_rebench-{row.get('trajectory_id')}", "source": "swe_rebench", "meta": meta,
            "tools": strip_none(row.get("tools") or []) or None, "messages": msgs}


CONVERTERS = {"wildchat": (convert_wildchat, ["conversation_hash", "model", "language", "conversation"], 1024),
              "swe_rebench": (convert_swe, ["trajectory_id", "instance_id", "repo", "resolved", "exit_status",
                                            "trajectory", "tools"], 32)}


def init_worker(tokenizer):
    global TOK
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer
    TOK = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)


def process(args):
    source, row = args
    rec = CONVERTERS[source][0](row)
    if rec is None:
        return None
    text = TOK.apply_chat_template(rec["messages"], tools=rec["tools"], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    out = {"id": rec["id"], "source": source,
           "num_tokens": len(TOK(text, add_special_tokens=False)["input_ids"]),
           "num_turns": sum(m["role"] == "assistant" for m in rec["messages"]) + 1,
           "num_messages": len(rec["messages"]), "meta": rec["meta"]}
    if rec["tools"]:
        out["tools"] = rec["tools"]
    out["messages"] = rec["messages"]
    return json.dumps(out, ensure_ascii=False), out["num_tokens"], out["num_turns"]


def iter_rows(files, columns, batch_size, limit):
    n = 0
    for f in files:
        pf = pq.ParquetFile(f)
        cols = [c for c in columns if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
            for row in batch.to_pylist():
                yield row
                n += 1
                if limit and n >= limit:
                    return


def main():
    args = parse_args()
    files = [args.input] if os.path.isfile(args.input) else \
        sorted(glob.glob(os.path.join(args.input, "**", "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"no parquet files under {args.input}")
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    if args.limit:
        total = min(total, args.limit)
    _, columns, batch_size = CONVERTERS[args.source]
    log(f"{args.source}: {len(files)} parquet files, {total} rows -> {args.output} ({args.workers} workers)")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    tmp = args.output + ".tmp"
    sha = hashlib.sha256()
    n_in = n_out = 0
    ntok, nturn = [], []
    t0 = time.time()
    with mp.Pool(args.workers, initializer=init_worker, initargs=(args.tokenizer,)) as pool, open(tmp, "w") as f:
        rows = ((args.source, r) for r in iter_rows(files, columns, batch_size, args.limit))
        for res in pool.imap(process, rows, chunksize=4 if args.source == "swe_rebench" else 256):
            n_in += 1
            if res is not None:
                line = res[0] + "\n"
                f.write(line)
                sha.update(line.encode())
                ntok.append(res[1])
                nturn.append(res[2])
                n_out += 1
            if n_in % 20000 == 0 or (args.source == "swe_rebench" and n_in % 5000 == 0):
                el = time.time() - t0
                log(f"  {n_in}/{total} rows ({el:.0f}s, ETA {el / n_in * (total - n_in):.0f}s)")
    os.replace(tmp, args.output)

    ntok, nturn = np.array(ntok), np.array(nturn)

    def pct(a):
        q = np.percentile(a, [25, 50, 75, 90, 99])
        return {"mean": float(a.mean()), "p25": float(q[0]), "p50": float(q[1]), "p75": float(q[2]),
                "p90": float(q[3]), "p99": float(q[4]), "max": int(a.max()), "total": int(a.sum())}

    manifest = {"source": args.source, "file": os.path.basename(args.output), "rows_in": n_in, "rows_out": n_out,
                "tokenizer": args.tokenizer, "chat_template": {"add_generation_prompt": True, "enable_thinking": False},
                "num_tokens": pct(ntok), "num_turns": pct(nturn), "sha256": sha.hexdigest(),
                "size_bytes": os.path.getsize(args.output), "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(args.output + ".manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log(f"done: {n_out}/{n_in} lines, {manifest['size_bytes'] / 1e9:.2f} GB, tokens p50 {manifest['num_tokens']['p50']:.0f} "
        f"p90 {manifest['num_tokens']['p90']:.0f} max {manifest['num_tokens']['max']}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
