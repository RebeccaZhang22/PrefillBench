#!/usr/bin/env python3
"""Step 3a: render + tokenize the downsampled set into benchmark requests.

Each sampled line's `messages` (+ `tools`) is rendered with the chat template
(add_generation_prompt=True, enable_thinking=False) and tokenized, i.e. the request is
"all messages as context, prefill up to the start of the next assistant output".

Outputs (in --out-dir):
  prompts.jsonl     one record per request (orig_id, source, turns, input_len, tok_offset, length bucket)
  prompts_ids.bin   concatenated uint32 token ids
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

TOK = None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sampled", required=True, nargs="+", help="downsampled jsonl files from prepare_data.py")
    p.add_argument("--config", required=True)
    p.add_argument("--tokenizer", default="", help="default: tokenizer in the config")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    return p.parse_args()


def log(msg):
    print(f"[make_prompts {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def init_worker(tokenizer):
    global TOK
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer
    TOK = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)


def encode(line):
    r = json.loads(line)
    text = TOK.apply_chat_template(r["messages"], tools=r.get("tools"), tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    ids = np.asarray(TOK(text, add_special_tokens=False)["input_ids"], dtype=np.uint32)
    return r["id"], r["source"], r["num_tokens"], r["num_turns"], ids.tobytes()


def main():
    args = parse_args()
    cfg = json.load(open(args.config))
    tokenizer = args.tokenizer or cfg["tokenizer"]
    edges = cfg["length_buckets"]
    os.makedirs(args.out_dir, exist_ok=True)

    recs, offset, mismatch = [], 0, 0
    t0 = time.time()
    def lines():
        for path in args.sampled:
            with open(path) as f:
                yield from f

    with open(os.path.join(args.out_dir, "prompts_ids.bin"), "wb") as bin_f, \
            mp.Pool(args.workers, initializer=init_worker, initargs=(tokenizer,)) as pool:
        for i, (rid, src, ntok, nturn, raw) in enumerate(pool.imap(encode, lines(), chunksize=8)):
            n = len(raw) // 4
            mismatch += n != ntok
            lo, hi = next(((a, b) for a, b in zip(edges[:-1], edges[1:]) if a <= n < b), (edges[-2], edges[-1]))
            recs.append({"id": i, "orig_id": rid, "set": "mixed", "source": src, "turns": nturn, "bucket_lo": lo,
                         "bucket_hi": hi, "input_len": n, "tok_offset": offset})
            bin_f.write(raw)
            offset += n
            if (i + 1) % 10000 == 0:
                log(f"  {i + 1} requests ({time.time() - t0:.0f}s)")
    if mismatch:
        log(f"WARNING: {mismatch} requests differ from the precomputed num_tokens (different tokenizer/template?)")

    with open(os.path.join(args.out_dir, "prompts.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    log(f"{len(recs):,} requests ({offset / 1e6:.1f}M tokens) -> {args.out_dir} [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
