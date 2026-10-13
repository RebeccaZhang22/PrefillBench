#!/usr/bin/env python3
"""Pure prefill benchmark on our dataset, built on SGLang's official `sglang.benchmark.one_batch`.

No server, no HTTP, no scheduler, no chunked prefill: every batch is one `ModelRunner.extend()` call
(the same call one_batch times as "Prefill. latency"), bracketed by device synchronizes.

How it hooks in: `one_batch.load_model` is wrapped. The official `main()` / `latency_test()` run
unchanged (argument resolution, env setup, model loading, TP process spawning); right after the
model is loaded on each rank, our dataset loop runs, then the official flow continues with a tiny
1x16-token batch and exits. All SGLang server flags (--model-path, --tp-size, --mem-fraction-static,
--context-length, --json-model-override-args, --attention-backend, ...) are accepted as usual.

Pinned to SGLang 0.5.21 (checked against its source: load_model / prepare_synthetic_inputs_for_latency_test /
_TorchBenchRunner.{clear,extend,synchronize} / main / BenchArgs; TP workers are spawned, and this module
re-applies the hook when the spawned child re-imports it). run_pipeline.sh refuses other versions by default.

Inputs: prompts.jsonl + prompts_ids.bin from make_prompts.py.
Output (rank 0): <out-dir>/batches_shard<k>.csv, one row per extend() call; summary_shard<k>.json.

  python scripts/prefill_one_batch.py --model-path Qwen/Qwen3-8B-FP8 --context-length 131072 \
      --data-prompts results/x/dataset/prompts.jsonl --data-out-dir results/x/runs/tp1
"""
import argparse
import csv
import json
import logging
import os
import time

import numpy as np

try:
    from sglang.benchmark import one_batch as ob
except ImportError:  # older SGLang
    import sglang.bench_one_batch as ob
from sglang.srt.server_args import ServerArgs

ENV = "GPUBENCH_PREFILL_CFG"


def log(msg):
    print(f"[prefill {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def runner_ops(mr):
    """(clear, extend, synchronize, kv_capacity) for both the wrapped and the raw ModelRunner."""
    raw = getattr(mr, "torch_runner", mr)
    cap = getattr(raw, "max_total_num_tokens", None)
    if hasattr(mr, "extend") and hasattr(mr, "clear"):
        return mr.clear, mr.extend, mr.synchronize, cap

    def clear():
        raw.req_to_token_pool.clear()
        raw.token_to_kv_pool_allocator.clear()

    return clear, (lambda reqs: ob.extend(reqs, raw)), (lambda: ob.synchronize(raw.device)), cap


def make_batches(recs, batch_tokens, max_batch):
    if batch_tokens <= 0:
        return [[r] for r in recs]
    out, cur, n = [], [], 0
    for r in recs:
        if cur and (n + r["input_len"] > batch_tokens or len(cur) >= max_batch):
            out.append(cur)
            cur, n = [], 0
        cur.append(r)
        n += r["input_len"]
    if cur:
        out.append(cur)
    return out


def run_dataset(model_runner, tp_rank):
    cfg = json.loads(os.environ[ENV])
    is_main = tp_rank == 0
    say = log if is_main else (lambda *_: None)
    clear, extend, sync, cap = runner_ops(model_runner)

    recs = [json.loads(line) for line in open(cfg["prompts"])]
    recs = [r for r in recs if r.get("set", "mixed") == "mixed"][cfg["shard"]::cfg["num_shards"]]
    if cfg["max_requests"] > 0:
        recs = recs[:cfg["max_requests"]]
    ids = np.memmap(os.path.join(os.path.dirname(os.path.realpath(cfg["prompts"])), "prompts_ids.bin"),
                    dtype=np.uint32, mode="r")
    batches = make_batches(recs, cfg["batch_tokens"], cfg["max_batch"])
    say(f"shard {cfg['shard']}/{cfg['num_shards']}: {len(recs):,} requests, "
        f"{sum(r['input_len'] for r in recs) / 1e6:.1f}M tokens, {len(batches):,} batches, KV capacity {cap}")

    def run(batch):
        inputs = [ids[r["tok_offset"]:r["tok_offset"] + r["input_len"]].tolist() for r in batch]
        clear()
        reqs = ob.prepare_synthetic_inputs_for_latency_test(len(inputs), max(map(len, inputs)), inputs)
        sync()
        t0 = time.perf_counter()
        extend(reqs)
        sync()
        return t0, time.perf_counter()

    # warmup on a short, a median and the longest request (JIT / autotuning for the shapes in play)
    by_len = sorted(recs, key=lambda r: r["input_len"])
    for r in [by_len[0], by_len[len(by_len) // 2], by_len[-1]][:cfg["warmup"]]:
        run([r])
    say("warmup done")

    os.makedirs(cfg["out_dir"], exist_ok=True)
    f = open(os.path.join(cfg["out_dir"], f"batches_shard{cfg['shard']}.csv"), "w", newline="") if is_main else None
    w = csv.writer(f) if f else None
    if w:
        w.writerow(["batch", "n_reqs", "tokens", "max_len", "latency_s", "start_epoch", "end_epoch", "lens", "ids"])
    offset = time.time() - time.perf_counter()
    t_begin, done_tok, skipped, next_log = time.perf_counter(), 0, 0, time.perf_counter() + 60
    for bi, batch in enumerate(batches):
        tokens = sum(r["input_len"] for r in batch)
        if cap is not None and tokens > cap:
            skipped += len(batch)
            say(f"skip batch {bi}: {tokens} tokens > KV capacity {cap} (raise --mem-fraction-static)")
            continue
        t0, t1 = run(batch)
        done_tok += tokens
        if w:
            w.writerow([bi, len(batch), tokens, max(r["input_len"] for r in batch), f"{t1 - t0:.6f}",
                        f"{t0 + offset:.3f}", f"{t1 + offset:.3f}", ";".join(str(r["input_len"]) for r in batch),
                        ";".join(str(r["orig_id"]) for r in batch)])
        if t1 > next_log:
            next_log = t1 + 60
            el = t1 - t_begin
            say(f"  {bi + 1:,}/{len(batches):,} batches, {done_tok / el:,.0f} tok/s wall, elapsed {el / 60:.1f} min, "
                f"ETA {el / (bi + 1) * (len(batches) - bi - 1) / 60:.1f} min")
    if f:
        f.close()
        with open(os.path.join(cfg["out_dir"], f"summary_shard{cfg['shard']}.json"), "w") as sf:
            json.dump(dict(cfg, requests=len(recs), batches=len(batches), skipped_requests=skipped,
                           tokens=done_tok, wall_s=time.perf_counter() - t_begin, kv_capacity=cap), sf, indent=2)
    say(f"shard {cfg['shard']} done: {done_tok / 1e6:.1f}M tokens in {(time.perf_counter() - t_begin) / 60:.1f} min"
        + (f", {skipped} requests skipped" if skipped else ""))
    clear()


_official_load_model = ob.load_model


def _load_model_then_run_dataset(server_args, port_args, gpu_id, tp_rank):
    model_runner, tokenizer = _official_load_model(server_args, port_args, gpu_id, tp_rank)
    if os.environ.get(ENV):
        run_dataset(model_runner, tp_rank)
        os.environ.pop(ENV)  # the official warmup/sweep that follows must not re-run it
    return model_runner, tokenizer


# Applied at import time so that TP worker processes (spawned by one_batch.main) get it too.
ob.load_model = _load_model_then_run_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ServerArgs.add_cli_args(parser)
    ob.BenchArgs.add_cli_args(parser)
    g = parser.add_argument_group("dataset (gpubench)")
    g.add_argument("--data-prompts", required=True, help="prompts.jsonl from make_prompts.py")
    g.add_argument("--data-out-dir", required=True)
    g.add_argument("--data-shard", type=int, default=0, help="this process handles requests shard::num_shards")
    g.add_argument("--data-num-shards", type=int, default=1)
    g.add_argument("--data-batch-tokens", type=int, default=0,
                   help="0 = one request per extend() (pure per-request prefill); >0 = pack consecutive requests "
                        "into one extend() up to this many tokens")
    g.add_argument("--data-max-batch", type=int, default=256)
    g.add_argument("--data-max-requests", type=int, default=0, help="limit requests per shard (0 = all)")
    g.add_argument("--data-warmup", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.data_out_dir, exist_ok=True)
    os.environ[ENV] = json.dumps({
        "prompts": os.path.abspath(args.data_prompts), "out_dir": os.path.abspath(args.data_out_dir),
        "shard": args.data_shard, "num_shards": args.data_num_shards, "batch_tokens": args.data_batch_tokens,
        "max_batch": args.data_max_batch, "max_requests": args.data_max_requests, "warmup": args.data_warmup})
    # the official sweep that runs after our loop: one tiny batch, results to a side file
    args.batch_size, args.input_len, args.output_len = [1], [16], [1]
    args.result_filename = os.path.join(args.data_out_dir, f"one_batch_tail_shard{args.data_shard}.jsonl")
    server_args = ServerArgs.from_cli_args(args)
    bench_args = ob.BenchArgs.from_cli_args(args)
    logging.basicConfig(level=getattr(logging, server_args.log_level.upper()), format="%(message)s")
    try:
        ob.main(server_args, bench_args)
    finally:
        if server_args.tp_size != 1 and hasattr(ob, "kill_process_tree"):
            ob.kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
