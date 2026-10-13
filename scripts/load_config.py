#!/usr/bin/env python3
"""Load and validate config.yaml for run_pipeline.sh.

  --emit shell          print `VAR='value'` lines for the bash driver (eval'd by run_pipeline.sh)
  --emit sampling-json  write the `sampling` section as JSON (consumed by downsample/visualize/make_prompts)
"""
import argparse
import json
import os
import shlex
import sys
import time

try:
    import yaml
except ImportError:
    sys.exit("[config] PyYAML is required to read the config: pip install pyyaml")

RUNTIME_DEFAULTS = {
    "python": "python3", "sglang_version": "0.5.21", "allow_version_mismatch": False, "model": "Qwen/Qwen3-8B-FP8",
    "gpus": [0], "tp": [1], "attention_backend": ["default"], "mem_fraction_static": 0.7, "context_length": 131072,
    "yarn": "auto", "batch_tokens": 0, "max_requests": 0, "extra_server_args": "", "peak_tflops": 0,
    "monitor_interval": 0.5, "base_port": 31000,
}
PIPELINE_DEFAULTS = {"confirm": True, "stats_only": False}
SOURCE_DEFAULTS = {"min_tokens": 0, "short_len": 0, "short_keep": 1.0}
FORMATS = ("wildchat", "swe_rebench")


def fail(msg):
    sys.exit(f"[config] {msg}")


def as_list(v):
    return v if isinstance(v, list) else [v]


def load(path, root):
    try:
        cfg = yaml.safe_load(open(path)) or {}
    except (OSError, yaml.YAMLError) as e:
        fail(f"cannot read {path}: {e}")
    samp = cfg.get("sampling") or fail("missing section: sampling")
    for k in ("seed", "tokenizer", "max_input_tokens", "length_buckets", "sources"):
        if k not in samp:
            fail(f"missing sampling.{k}")
    names = set()
    for i, s in enumerate(samp["sources"]):
        for k in ("name", "format", "input", "num_samples"):
            if k not in s:
                fail(f"missing sampling.sources[{i}].{k}")
        if s["format"] not in FORMATS:
            fail(f"sampling.sources[{i}].format must be one of {FORMATS}")
        if s["name"] in names:
            fail(f"duplicate source name: {s['name']}")
        names.add(s["name"])
        for k, v in SOURCE_DEFAULTS.items():
            s.setdefault(k, v)
        if not 0 <= float(s["short_keep"]) <= 1:
            fail(f"sampling.sources[{i}].short_keep must be in [0, 1]")
    if sorted(samp["length_buckets"]) != samp["length_buckets"]:
        fail("sampling.length_buckets must be increasing")

    rt = dict(RUNTIME_DEFAULTS, **(cfg.get("runtime") or {}))
    pl = dict(PIPELINE_DEFAULTS, **(cfg.get("pipeline") or {}))
    unknown = set(cfg.get("runtime") or {}) - set(RUNTIME_DEFAULTS)
    if unknown:
        fail(f"unknown runtime keys: {sorted(unknown)}")
    if int(samp["max_input_tokens"]) >= int(rt["context_length"]):
        fail(f"sampling.max_input_tokens ({samp['max_input_tokens']}) must be < runtime.context_length "
             f"({rt['context_length']})")

    paths = cfg.get("paths") or {}

    def resolve(p):
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(root, p))

    for s in samp["sources"]:
        s["input"] = resolve(str(s["input"]))

    out = str(paths.get("out_dir", "auto"))
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join("results", ts) if out == "auto" else out.replace("{timestamp}", ts)
    workers = int((cfg.get("processing") or {}).get("workers", 0))
    return cfg, samp, rt, pl, resolve(str(paths.get("processed_dir", "data/processed"))), resolve(out), workers


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--root", required=True, help="repo root, relative paths are resolved against it")
    ap.add_argument("--emit", choices=["shell", "sampling-json"], required=True)
    ap.add_argument("--out", help="output file for --emit sampling-json")
    args = ap.parse_args()
    cfg, samp, rt, pl, processed_dir, out_dir, workers = load(args.config, args.root)

    if args.emit == "sampling-json":
        with open(args.out, "w") as f:
            json.dump(samp, f, indent=2, ensure_ascii=False)
        return

    yarn = str(rt["yarn"]).lower()
    if yarn not in ("auto", "true", "false", "1", "0"):
        fail("runtime.yarn must be auto / true / false")
    shell = {
        "PROCESSED_DIR": processed_dir, "OUT_DIR": out_dir, "WORKERS": workers,
        "SOURCE_NAMES": ",".join(s["name"] for s in samp["sources"]), "PYTHON": rt["python"], "MODEL": rt["model"],
        "SGLANG_VERSION": str(rt["sglang_version"]), "ALLOW_VERSION_MISMATCH": int(bool(rt["allow_version_mismatch"])),
        "GPUS": ",".join(str(g) for g in as_list(rt["gpus"])), "TP_LIST": ",".join(str(t) for t in as_list(rt["tp"])),
        "ATTENTION_BACKEND_LIST": ",".join(str(a) for a in as_list(rt["attention_backend"])),
        "MEM_FRACTION": rt["mem_fraction_static"], "CONTEXT_LENGTH": rt["context_length"],
        "YARN": {"true": "1", "false": "0"}.get(yarn, yarn), "BATCH_TOKENS": rt["batch_tokens"],
        "MAX_REQUESTS": rt["max_requests"], "EXTRA_ARGS": rt["extra_server_args"] or "",
        "PEAK_TFLOPS": rt["peak_tflops"], "MONITOR_INTERVAL": rt["monitor_interval"], "BASE_PORT": rt["base_port"],
        "CONFIRM": int(bool(pl["confirm"])), "STATS_ONLY": int(bool(pl["stats_only"])),
        "MAX_INPUT": samp["max_input_tokens"],
    }
    for k, v in shell.items():
        print(f"{k}={shlex.quote(str(v))}")


if __name__ == "__main__":
    main()
