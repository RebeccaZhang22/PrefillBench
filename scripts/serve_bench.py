#!/usr/bin/env python3
"""Online prefill benchmark by request shape on a real SGLang scheduler (in-process Engine, one GPU).

Every request is sent as token ids with max_new_tokens=1: the scheduler prefills it (chunked prefill, continuous
batching, radix cache as configured), samples one token and finishes; there is no decode. Closed loop: C requests
(or C trajectories) are always in flight. With serve.latent_layer = L the model is cut after decoder block L
(latent_exit.py) and is built with only L+1 layers (json override num_hidden_layers), so weights and the KV pool
also cover only those layers.

Shapes (pools from make_prompts.py output: source wildchat = short, swe_rebench = long):
  short             one ~4k request per item                                  radix cache off
  long              one ~64k request per item                                 radix cache off
  long_seg_cache    a long trajectory cut at the assistant-turn boundaries nearest 1/4, 2/4, 3/4 of its length into
                    4 cumulative requests (prefixes of the same token ids), sent one after another;
                    radix cache on, so each request reuses the KV of the previous ones (while not evicted)
  long_seg_nocache  the same 4 requests with radix cache off: every request recomputes its whole prefix

Per concurrency level: issue until both level_min_seconds and level_min_requests_per_conc * C requests are done
(or level_max_seconds), then stop issuing and drain. Measurement window = [first issue, stop issuing]; a request
straddling the window edge contributes the overlapping fraction of its tokens.
  computed tok/s = sum(prompt_tokens - cached_tokens) / window     (GPU prefill work)
  prompt tok/s   = sum(prompt_tokens) / window                     (tokens served, incl. cache hits)
  latency        = submit -> finish of each request (queueing + prefill), from the client side

Outputs in <out-dir>: ps<prefill_size>/<shape>/c<C>.csv (one row per request), summary.jsonl (one row per level).
"""
import argparse
import asyncio
import csv
import json
import os
import time

import numpy as np

import latent_exit

LATENT_ENV = "GPUBENCH_LATENT_LAYER"
# runs in every process that imports this module, including SGLang's spawned scheduler (which re-imports __main__)
latent_exit.install(lambda: int(os.environ.get(LATENT_ENV, "-1")), sampler=True)

RADIX = {"short": False, "long": False, "long_seg_cache": True, "long_seg_nocache": False}
POOL = {"short": "wildchat", "long": "swe_rebench", "long_seg_cache": "swe_rebench", "long_seg_nocache": "swe_rebench"}
SAMPLING = {"max_new_tokens": 1, "temperature": 0.0, "ignore_eos": True}


def log(msg):
    print(f"[serve {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_cfg(path):
    import yaml
    cfg = yaml.safe_load(open(path))
    sv, rt = cfg["serve"], cfg["runtime"]
    lat = sv.get("latent_layer", "full")
    sv["latent_layer"] = -1 if str(lat).lower() in ("full", "none", "-1") else int(lat)
    unknown = set(sv["shapes"]) - set(RADIX)
    if unknown:
        raise SystemExit(f"[serve] unknown shapes {sorted(unknown)}; choose from {sorted(RADIX)}")
    return sv, rt


class Pools:
    """Token ids per request, shuffled per shape (seeded, identical across server configs); long items are cut
    into 4 segments on demand."""

    def __init__(self, dataset_dir, tokenizer, seed):
        self.ids = np.memmap(os.path.join(dataset_dir, "prompts_ids.bin"), dtype=np.uint32, mode="r")
        recs = [json.loads(line) for line in open(os.path.join(dataset_dir, "prompts.jsonl"))]
        self.by_src = {}
        for r in recs:
            self.by_src.setdefault(r["source"], []).append(r)
        t = tokenizer
        self.asst = [t.convert_tokens_to_ids("<|im_start|>"), t.convert_tokens_to_ids("assistant"),
                     t("\n", add_special_tokens=False)["input_ids"][0]]
        self.seed = seed
        self.cursor = {}

    def tokens(self, r):
        return np.asarray(self.ids[r["tok_offset"]:r["tok_offset"] + r["input_len"]], dtype=np.int64)

    def segments(self, ids):
        """Cuts right after '<|im_start|>assistant\\n' (where a reply would start) nearest 1/4, 2/4, 3/4."""
        a, b, c = self.asst
        p = np.flatnonzero((ids[:-2] == a) & (ids[1:-1] == b) & (ids[2:] == c)) + 3
        p = p[p < len(ids)]
        if len(p) < 3:
            return None
        cuts = []
        for k in (1, 2, 3):
            q = int(p[np.argmin(np.abs(p - len(ids) * k / 4))])
            if cuts and q <= cuts[-1]:
                return None
            cuts.append(q)
        return cuts + [len(ids)]

    def taker(self, shape):
        recs = self.by_src[POOL[shape]]
        # the two segmented shapes replay the same trajectories, so cache vs no-cache compares like with like
        key = "long_seg" if shape.startswith("long_seg") else shape
        order = np.random.default_rng([self.seed, sum(map(ord, key))]).permutation(len(recs))
        self.cursor[shape] = 0  # every server config replays the same request sequence

        def take():
            while True:
                i = self.cursor[shape]
                self.cursor[shape] += 1
                if i and i % len(recs) == 0:
                    log(f"WARNING {shape}: pool of {len(recs)} exhausted, reusing requests")
                r = recs[order[i % len(recs)]]
                ids = self.tokens(r)
                if not shape.startswith("long_seg"):
                    return r["orig_id"], [ids.tolist()]
                cuts = self.segments(ids)
                if cuts:
                    return r["orig_id"], [ids[:e].tolist() for e in cuts]
        return take


async def run_level(engine, take, conc, sv):
    rows, state = [], {"stop": False}
    t0 = time.perf_counter()
    need = sv["level_min_requests_per_conc"] * conc

    async def worker(w):
        while not state["stop"]:
            rid, segs = take()
            for k, ids in enumerate(segs):
                if state["stop"]:
                    break
                ts = time.perf_counter()
                out = await engine.async_generate(input_ids=ids, sampling_params=SAMPLING)
                te = time.perf_counter()
                m = out["meta_info"]
                rows.append({"item": rid, "seg": k, "worker": w, "prompt_tokens": m.get("prompt_tokens", len(ids)),
                             "cached_tokens": m.get("cached_tokens", 0) or 0, "start": ts - t0, "end": te - t0,
                             "latency_s": te - ts})

    async def monitor():
        next_log = 30
        while True:
            await asyncio.sleep(0.25)
            el = time.perf_counter() - t0
            if (el >= sv["level_min_seconds"] and len(rows) >= need) or el >= sv["level_max_seconds"]:
                state["stop"], state["t_stop"] = True, el
                return
            if el >= next_log:
                next_log += 30
                log(f"    {el:.0f}s: {len(rows)}/{need} requests done")

    await asyncio.gather(monitor(), *[worker(w) for w in range(conc)])
    return rows, state["t_stop"]


def summarize(rows, t_stop):
    st = np.array([r["start"] for r in rows])
    en = np.array([r["end"] for r in rows])
    frac = np.clip(np.minimum(en, t_stop) - np.maximum(st, 0), 0, None) / np.maximum(en - st, 1e-9)
    prompt = np.array([r["prompt_tokens"] for r in rows], dtype=np.float64)
    comp = prompt - np.array([r["cached_tokens"] for r in rows], dtype=np.float64)
    lat = np.array([r["latency_s"] for r in rows])
    return {"requests": len(rows), "window_s": t_stop, "drain_s": float(en.max() - t_stop),
            "computed_tok_s": float((comp * frac).sum() / t_stop), "prompt_tok_s": float((prompt * frac).sum() / t_stop),
            "req_s": float(frac.sum() / t_stop), "cache_hit_pct": float(100 * (prompt - comp).sum() / prompt.sum()),
            "prompt_len_mean": float(prompt.mean()), "computed_len_mean": float(comp.mean()),
            "lat_mean_s": float(lat.mean()), "lat_p50_s": float(np.percentile(lat, 50)),
            "lat_p90_s": float(np.percentile(lat, 90)), "lat_p99_s": float(np.percentile(lat, 99)),
            "lat_max_s": float(lat.max())}


def engine_kwargs(sv, rt, prefill_size, radix):
    override = {}
    if sv["latent_layer"] >= 0:
        override["num_hidden_layers"] = sv["latent_layer"] + 1
    ctx = int(rt.get("context_length", 81920))
    if ctx > 40960:  # Qwen3 YaRN, as in run_pipeline.sh
        override["rope_scaling"] = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
        os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    kw = dict(model_path=rt["model"], tp_size=1, mem_fraction_static=sv.get("mem_fraction_static", 0.75),
              context_length=ctx, disable_radix_cache=not radix, skip_tokenizer_init=True, log_level="error",
              json_model_override_args=json.dumps(override))
    if prefill_size != "default":  # default: SGLang picks both from the GPU memory (H20: 8192 / 16384)
        kw.update(chunked_prefill_size=int(prefill_size), max_prefill_tokens=int(prefill_size))
    kw.update(sv.get("extra_engine_args") or {})
    return kw


async def bench_server(engine, pools, shapes, sv, prefill_size, out_dir, summary_f):
    # warm up kernels / autotuning on both lengths
    for shape in ("short", "long"):
        _, segs = pools.taker(shape)()
        await engine.async_generate(input_ids=segs[-1], sampling_params=SAMPLING)
    await engine.tokenizer_manager.flush_cache()
    for shape in shapes:
        take = pools.taker(shape)
        for conc in sv["shapes"][shape]:
            log(f"  {shape} C={conc}")
            rows, t_stop = await run_level(engine, take, conc, sv)
            await engine.tokenizer_manager.flush_cache()
            d = os.path.join(out_dir, f"ps{prefill_size}", shape)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"c{conc}.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            s = dict({"prefill_size": prefill_size, "shape": shape, "radix_cache": RADIX[shape], "concurrency": conc,
                      "latent_layer": sv["latent_layer"]}, **summarize(rows, t_stop))
            summary_f.write(json.dumps(s) + "\n")
            summary_f.flush()
            log(f"    -> {s['computed_tok_s']:,.0f} computed tok/s, {s['prompt_tok_s']:,.0f} prompt tok/s, "
                f"latency p50 {s['lat_p50_s']:.2f}s p90 {s['lat_p90_s']:.2f}s, cache hit {s['cache_hit_pct']:.0f}%, "
                f"{s['requests']} requests")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="yaml with runtime + serve sections (e.g. shape.yaml)")
    ap.add_argument("--dataset", required=True, help="make_prompts.py output dir (prompts.jsonl + prompts_ids.bin)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shapes", nargs="*", help="subset of serve.shapes to run")
    ap.add_argument("--prefill-sizes", nargs="*", help="subset of serve.prefill_sizes (ints or default)")
    args = ap.parse_args()
    sv, rt = load_cfg(args.config)
    shapes = [s for s in sv["shapes"] if not args.shapes or s in args.shapes]
    os.makedirs(args.out_dir, exist_ok=True)
    os.environ[LATENT_ENV] = str(sv["latent_layer"])  # read by the scheduler process when it builds the model

    import sglang
    from transformers import AutoTokenizer
    pools = Pools(args.dataset, AutoTokenizer.from_pretrained(rt["model"]), seed=20261015)
    log(f"sglang {sglang.__version__}; pools: " + ", ".join(f"{k} {len(v):,}" for k, v in pools.by_src.items())
        + f"; latent_layer {sv['latent_layer']}")
    with open(os.path.join(args.out_dir, "summary.jsonl"), "a") as summary_f:
        for ps in [p if str(p) == "default" else int(p) for p in (args.prefill_sizes or sv["prefill_sizes"])]:
            for radix in (False, True):
                group = [s for s in shapes if RADIX[s] == radix]
                if not group:
                    continue
                kw = engine_kwargs(sv, rt, ps, radix)
                log(f"=== prefill size {ps}, radix cache {'on' if radix else 'off'}: {group}")
                t = time.time()
                engine = sglang.Engine(**kw)
                log(f"engine up in {time.time() - t:.0f}s")
                try:
                    engine.loop.run_until_complete(
                        bench_server(engine, pools, group, sv, ps, args.out_dir, summary_f))
                finally:
                    engine.shutdown()
    log(f"done -> {args.out_dir}/summary.jsonl")


if __name__ == "__main__":
    main()
