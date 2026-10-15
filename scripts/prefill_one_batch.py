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

Latent mode (--data-latent-layer L, 0-based decoder block): the model is cut after block L before CUDA graphs are
captured, so every extend() runs embedding + blocks 0..L and returns the block-L residual stream
(hidden + residual, no final RMSNorm) of each request's last token. The final norm, lm_head and sampling are
skipped. See latent_exit.py; the installed SGLang files are untouched.

Inputs: prompts.jsonl + prompts_ids.bin from make_prompts.py.
With --data-generate 1 each request is then decoded to its real reply length (output_len from the data, capped
by --data-max-new-tokens, EOS ignored) with the official decode(), every step timed between device synchronizes:
TTFT = the extend() time, TPOT = decode step time, end-to-end = TTFT + all decode steps.

Output (rank 0): <out-dir>/batches_shard<k>.csv, one row per request / extend() call (latency_s = prefill time;
out_len, decode_s, tpot_p50_ms in generation mode); summary_shard<k>.json.

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

import latent_exit

ENV = "GPUBENCH_PREFILL_CFG"


def log(msg):
    print(f"[prefill {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def runner_ops(mr):
    """(clear, extend, decode, synchronize, kv_capacity) for both the wrapped and the raw ModelRunner."""
    raw = getattr(mr, "torch_runner", mr)
    cap = getattr(raw, "max_total_num_tokens", None)
    if hasattr(mr, "extend") and hasattr(mr, "clear"):
        return mr.clear, mr.extend, mr.decode, mr.synchronize, cap

    def clear():
        raw.req_to_token_pool.clear()
        raw.token_to_kv_pool_allocator.clear()

    return (clear, (lambda reqs: ob.extend(reqs, raw)), (lambda ids, batch: ob.decode(ids, batch, raw)),
            (lambda: ob.synchronize(raw.device)), cap)


def _latent_layer():
    cfg = os.environ.get(ENV)
    return json.loads(cfg).get("latent_layer", -1) if cfg else -1


def flops_fn(model_path, num_layers=None):
    """Prefill FLOPs of an L-token request (dense matmuls + causal attention over `num_layers` blocks, all by
    default); used to weight the ETA. Falls back to plain token counts if the model config cannot be read."""
    try:
        path = os.path.join(model_path, "config.json")
        if not os.path.exists(path):
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(model_path, "config.json")
        c = json.load(open(path))
        c = c.get("text_config", c)
        h, layers, nh = c["hidden_size"], num_layers or c["num_hidden_layers"], c["num_attention_heads"]
        nkv, hd = c.get("num_key_value_heads", nh), c.get("head_dim") or h // nh
        linear = layers * (h * nh * hd + 2 * h * nkv * hd + nh * hd * h + 3 * h * c["intermediate_size"])
        attn = layers * nh * hd
        return lambda L: 2.0 * linear * L + 2.0 * attn * L * L
    except Exception:
        return float


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
    clear, extend, decode, sync, cap = runner_ops(model_runner)
    gen = cfg["generate"]
    latent = cfg["latent_layer"]
    raw = getattr(model_runner, "torch_runner", model_runner)
    if latent >= 0 and getattr(raw, "_gpubench_latent", None) != latent:
        raise RuntimeError("latent exit was not applied before CUDA graph capture (ModelRunner.init_cuda_graphs hook)")
    hidden = raw.model_config.hidden_size

    recs = [json.loads(line) for line in open(cfg["prompts"])]
    recs = [r for r in recs if r.get("set", "mixed") == "mixed"][cfg["shard"]::cfg["num_shards"]]
    if 0 < cfg["max_requests"] < len(recs):
        # evenly spaced over the shard (sources are concatenated) + the longest request, so a trial run
        # covers short chat prompts as well as the longest agent context
        keep = set(np.linspace(0, len(recs) - 1, cfg["max_requests"]).round().astype(int).tolist())
        keep.add(max(range(len(recs)), key=lambda i: recs[i]["input_len"]))
        recs = [recs[i] for i in sorted(keep)]
    ids = np.memmap(os.path.join(os.path.dirname(os.path.realpath(cfg["prompts"])), "prompts_ids.bin"),
                    dtype=np.uint32, mode="r")
    for r in recs:  # generation length: the real reply length, capped, and fitting the context window
        n = r.get("output_len") or cfg["default_new_tokens"]
        r["gen_len"] = max(1, min(n, cfg["max_new_tokens"], cfg["context_length"] - r["input_len"] - 1)) if gen else 1
    batches = make_batches(recs, cfg["batch_tokens"], cfg["max_batch"])
    flops = flops_fn(cfg["model_path"], latent + 1 if latent >= 0 else None)
    # ETA weights: prefill FLOPs, plus decode steps counted as ~64 prefill tokens each (a memory-bound step at bs=1
    # costs roughly as long as prefilling a few dozen tokens)
    work = [sum(flops(r["input_len"]) + (r["gen_len"] - 1) * flops(64) for r in b) for b in batches]
    total_work = sum(work)
    if latent >= 0:
        say(f"latent mode: prefill runs decoder blocks 0..{latent} ({latent + 1} blocks), output = block-{latent} "
            f"residual stream of the last token; final norm / lm_head / sampling skipped")
    say(f"shard {cfg['shard']}/{cfg['num_shards']}: {len(recs):,} requests, "
        f"{sum(r['input_len'] for r in recs) / 1e6:.1f}M tokens, {len(batches):,} batches, KV capacity {cap}")

    def run(batch):
        """Prefill (one extend), then in generation mode decode gen_len-1 more tokens (EOS ignored), each step
        timed between device synchronizes like the official latency_test_run_once. Returns
        (prefill start, prefill end, per-step decode latencies)."""
        inputs = [ids[r["tok_offset"]:r["tok_offset"] + r["input_len"]].tolist() for r in batch]
        clear()
        reqs = ob.prepare_synthetic_inputs_for_latency_test(len(inputs), max(map(len, inputs)), inputs)
        sync()
        t0 = time.perf_counter()
        next_ids, out, fwd_batch = extend(reqs)
        sync()
        t1 = time.perf_counter()
        if latent >= 0 and tuple(out.shape) != (len(batch), hidden):
            raise RuntimeError(f"latent mode: expected output [{len(batch)}, {hidden}], got {tuple(out.shape)} "
                               "(the full model ran, e.g. a CUDA graph captured before the cut)")
        steps = []
        for _ in range(max(r["gen_len"] for r in batch) - 1):
            ts = time.perf_counter()
            next_ids, _ = decode(next_ids, fwd_batch)
            sync()
            steps.append(time.perf_counter() - ts)
        return t0, t1, steps

    # warmup on a short, a median and the longest request (JIT / autotuning for the shapes in play)
    by_len = sorted(recs, key=lambda r: r["input_len"])
    for r in [by_len[0], by_len[len(by_len) // 2], by_len[-1]][:cfg["warmup"]]:
        run([r])
    say("warmup done")

    os.makedirs(cfg["out_dir"], exist_ok=True)
    f = open(os.path.join(cfg["out_dir"], f"batches_shard{cfg['shard']}.csv"), "w", newline="") if is_main else None
    w = csv.writer(f) if f else None
    if w:
        w.writerow(["batch", "n_reqs", "tokens", "max_len", "latency_s", "start_epoch", "end_epoch", "lens", "ids",
                    "out_len", "decode_s", "tpot_p50_ms"])
    offset = time.time() - time.perf_counter()
    t_begin, done_tok, done_work, skipped, next_log = time.perf_counter(), 0, 0.0, 0, time.perf_counter() + 60
    for bi, batch in enumerate(batches):
        tokens = sum(r["input_len"] for r in batch)
        done_work += work[bi]
        if cap is not None and tokens + sum(r["gen_len"] for r in batch) > cap:
            skipped += len(batch)
            say(f"skip batch {bi}: {tokens} tokens > KV capacity {cap} (raise --mem-fraction-static)")
            continue
        t0, t1, steps = run(batch)
        done_tok += tokens
        if w:
            w.writerow([bi, len(batch), tokens, max(r["input_len"] for r in batch), f"{t1 - t0:.6f}",
                        f"{t0 + offset:.3f}", f"{t1 + offset:.3f}", ";".join(str(r["input_len"]) for r in batch),
                        ";".join(str(r["orig_id"]) for r in batch), len(steps) + 1, f"{sum(steps):.6f}",
                        f"{np.median(steps) * 1e3:.4f}" if steps else ""])
            f.flush()  # keep finished rows on disk even if the run is killed
        if t1 > next_log:
            next_log = t1 + 60
            el = t1 - t_begin
            say(f"  {bi + 1:,}/{len(batches):,} batches ({done_work / total_work * 100:.1f}% of compute), "
                f"{done_tok / el:,.0f} tok/s wall, elapsed {el / 60:.1f} min, "
                f"ETA {el / done_work * (total_work - done_work) / 60:.1f} min")
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
latent_exit.install(_latent_layer)


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
    g.add_argument("--data-generate", type=int, default=0,
                   help="1 = after prefill, decode each request's real reply length (output_len), EOS ignored")
    g.add_argument("--data-max-new-tokens", type=int, default=2048, help="cap on the generation length")
    g.add_argument("--data-default-new-tokens", type=int, default=256,
                   help="generation length for requests without a reference reply")
    g.add_argument("--data-latent-layer", type=int, default=-1,
                   help="-1 = full model; L >= 0 = stop prefill after decoder block L (0-based) and return its "
                        "residual stream (no final norm, lm_head or sampling)")
    args = parser.parse_args()

    os.makedirs(args.data_out_dir, exist_ok=True)
    os.environ[ENV] = json.dumps({
        "prompts": os.path.abspath(args.data_prompts), "out_dir": os.path.abspath(args.data_out_dir),
        "model_path": args.model_path,
        "shard": args.data_shard, "num_shards": args.data_num_shards, "batch_tokens": args.data_batch_tokens,
        "max_batch": args.data_max_batch, "max_requests": args.data_max_requests, "warmup": args.data_warmup,
        "generate": bool(args.data_generate), "max_new_tokens": args.data_max_new_tokens,
        "default_new_tokens": args.data_default_new_tokens, "context_length": args.context_length or 131072,
        "latent_layer": args.data_latent_layer})
    if args.data_generate and args.data_batch_tokens > 0:
        parser.error("--data-generate measures single-request generation; use --data-batch-tokens 0")
    if args.data_generate and args.data_latent_layer >= 0:
        parser.error("--data-latent-layer has no lm_head, so it cannot generate; use --data-generate 0")
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
