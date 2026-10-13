#!/usr/bin/env python3
"""Aggregate a benchmark result directory into report.md + figures (numpy/matplotlib only).

Expected layout:
  <result-dir>/dataset/{sample_manifest.json, sample_stats.json, sample_distribution.png}
  <result-dir>/runs/<tag>/{run.json, batches_shard*.csv, summary_shard*.json, gpu_monitor.csv}
"""
import argparse
import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", required=True)
    p.add_argument("--peak-tflops", type=float, default=0.0, help="per-GPU dense peak TFLOPS for MFU")
    return p.parse_args()


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def bucket_label(lo, hi):
    def k(x):
        return f"{x // 1024}k" if x >= 1024 and x % 1024 == 0 else str(x)
    return f"{k(lo)}-{k(hi)}"


def fmt(v, floatfmt):
    if isinstance(v, (float, np.floating)):
        return "-" if np.isnan(v) else floatfmt.format(v)
    return str(v)


def md_table(rows, floatfmt="{:,.1f}"):
    """rows: list of dicts with identical keys."""
    if not rows:
        return ""
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r[c], floatfmt) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def load_gpu(path):
    if not os.path.exists(path) or os.path.getsize(path) < 100:
        return None
    rows = read_csv(path)
    out = {}
    for k in ["epoch", "index", "util_gpu", "power_w"]:
        out[k] = np.array([float(r[k]) if r.get(k) not in (None, "", "[N/A]") else np.nan for r in rows])
    return out


def load_model_config(model):
    path = os.path.join(model, "config.json")
    try:
        if not os.path.exists(path):
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(model, "config.json")
        cfg = json.load(open(path))
        return cfg.get("text_config", cfg)
    except Exception as e:  # MFU is optional
        print(f"[report] cannot load model config ({e}); FLOPs/MFU disabled")
        return None


def prefill_flops(cfg, L):
    """Forward FLOPs of prefilling L tokens: dense matmuls + causal attention + last-token lm_head."""
    h, layers, nh = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["num_attention_heads"]
    nkv, hd = cfg.get("num_key_value_heads", nh), cfg.get("head_dim") or h // nh
    linear = layers * (h * nh * hd + 2 * h * nkv * hd + nh * hd * h + 3 * h * cfg["intermediate_size"])
    L = np.asarray(L, dtype=np.float64)
    return 2.0 * linear * L + 2.0 * layers * nh * hd * L ** 2 + 2.0 * h * cfg.get("vocab_size", 0)


def main():
    args = parse_args()
    rd = args.result_dir
    ds_dir = os.path.join(rd, "dataset")
    fig_dir = os.path.join(rd, "figs")
    os.makedirs(fig_dir, exist_ok=True)
    out = ["# Prefill efficiency report", ""]
    for name, title in [("config.yaml", "Config"), ("env.txt", "Environment")]:
        p = os.path.join(rd, name)
        if os.path.exists(p):
            out += [f"## {title}", "", "```", open(p).read().strip(), "```", ""]

    # ---------------- dataset (downsampled set) ----------------
    man_p, st_p = os.path.join(ds_dir, "sample_manifest.json"), os.path.join(ds_dir, "sample_stats.json")
    if os.path.exists(man_p) and os.path.exists(st_p):
        man, st = json.load(open(man_p)), json.load(open(st_p))
        out += ["## Benchmark set (downsampled)", "",
                f"- seed `{man['config']['seed']}`, max_input_tokens {man['config']['max_input_tokens']:,}, "
                f"fingerprint `{man['fingerprint_sha256'][:16]}`", ""]
        out += [md_table([{"source": x["name"], "lines (full)": f"{x['lines']:,}",
                           "> max_input": f"{x['over_max_input']:,}", "eligible": f"{x['eligible_after_rule']:,}",
                           "sampled": f"{x['sampled']:,}", "sampled tokens": f"{x['sampled_tokens']:,}"}
                          for x in man["sources"]]), ""]
        out += [md_table([{"source": r["source"], "requests": f"{r['requests']:,}", "tok mean": r["tok_mean"],
                           "tok p50": r["tok_p50"], "tok p90": r["tok_p90"], "tok p99": r["tok_p99"],
                           "tok max": f"{r['tok_max']:,}", "turn mean": r["turn_mean"], "turn p50": r["turn_p50"],
                           "turn p90": r["turn_p90"], "turn max": r["turn_max"]} for r in st["sources"]], "{:,.0f}"), ""]
        out += [md_table(st["length_buckets"], "{:.1f}"), "", "![](dataset/sample_distribution.png)", ""]

    # ---------------- runs ----------------
    run_dirs = sorted(d for d in glob.glob(os.path.join(rd, "runs", "*")) if glob.glob(os.path.join(d, "batches_shard*.csv")))
    if not run_dirs:
        out += ["_No prefill runs found._"]
        open(os.path.join(rd, "report.md"), "w").write("\n".join(out) + "\n")
        print(f"[report] wrote {os.path.join(rd, 'report.md')}")
        return
    cfg_path = os.path.join(rd, "sampling_config.json")
    edges = json.load(open(cfg_path))["length_buckets"] if os.path.exists(cfg_path) else [0, 2 ** 31]
    peak = args.peak_tflops * 1e12
    runs, model_cfg = {}, None
    for d in run_dirs:
        tag = os.path.basename(d)
        meta = json.load(open(os.path.join(d, "run.json"))) if os.path.exists(os.path.join(d, "run.json")) else {"tp": 1}
        if model_cfg is None and meta.get("model"):
            model_cfg = load_model_config(meta["model"])
        rows = [r for f in sorted(glob.glob(os.path.join(d, "batches_shard*.csv"))) for r in read_csv(f)]
        lens = [np.array([int(x) for x in r["lens"].split(";")]) for r in rows]
        lat = np.array([float(r["latency_s"]) for r in rows])
        runs[tag] = {"meta": meta, "lens": lens, "lat": lat, "tokens": np.array([int(r["tokens"]) for r in rows]),
                     "start": np.array([float(r["start_epoch"]) for r in rows]),
                     "end": np.array([float(r["end_epoch"]) for r in rows]),
                     "gpu": load_gpu(os.path.join(d, "gpu_monitor.csv"))}
    for r in runs.values():
        r["flops"] = np.array([prefill_flops(model_cfg, l).sum() for l in r["lens"]]) if model_cfg else None

    out += ["## Pure prefill results (SGLang one_batch, `ModelRunner.extend`)", "",
            "Each row of the batch log is one `extend()` call timed between two device synchronizes: no server, "
            "HTTP, scheduler or chunked prefill. `tok/s/GPU` = tokens / sum(extend time) / TP size.", ""]
    rows = []
    for tag, r in runs.items():
        tp, ns = r["meta"].get("tp", 1), r["meta"].get("num_shards", 1)
        busy = r["lat"].sum()
        row = {"config": tag, "GPUs": tp * ns, "requests": f"{sum(len(l) for l in r['lens']):,}",
               "extend calls": f"{len(r['lat']):,}", "tokens": f"{r['tokens'].sum():,}",
               "extend time (h)": busy / 3600, "tok/s/GPU": r["tokens"].sum() / busy / tp,
               "wall (h)": (r["end"].max() - r["start"].min()) / 3600}
        if peak and r["flops"] is not None:
            row["MFU %"] = r["flops"].sum() / busy / (peak * tp) * 100
        if r["gpu"] is not None:
            row["GPU util %"] = float(np.nanmean(r["gpu"]["util_gpu"]))
        rows.append(row)
    out += [md_table(rows, "{:,.2f}"), ""]

    # per length bucket
    for tag, r in runs.items():
        tp = r["meta"].get("tp", 1)
        single = all(len(l) == 1 for l in r["lens"])
        key = np.array([l.max() for l in r["lens"]])
        brow = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (key >= lo) & (key < hi)
            if not m.any():
                continue
            busy = r["lat"][m].sum()
            b = {"length": bucket_label(lo, hi), "extend calls": int(m.sum()),
                 "avg tokens": float(r["tokens"][m].mean()),
                 "latency p50 ms": float(np.median(r["lat"][m]) * 1e3), "latency p90 ms": float(np.percentile(r["lat"][m], 90) * 1e3),
                 "tok/s/GPU": r["tokens"][m].sum() / busy / tp}
            if peak and r["flops"] is not None:
                b["MFU %"] = r["flops"][m].sum() / busy / (peak * tp) * 100
            brow.append(b)
        note = "one request per extend() -> latency = pure prefill time (TTFT without serving overhead)" if single \
            else "packed batches, grouped by their longest request"
        out += [f"### `{tag}` by input length", "", f"_{note}_", "", md_table(brow), ""]

    # figures
    ncol = 3 if (peak and model_cfg) else 2
    fig, axes = plt.subplots(1, ncol, figsize=(6.2 * ncol, 4.6))
    for tag, r in runs.items():
        tp = r["meta"].get("tp", 1)
        x = r["tokens"]
        axes[0].scatter(x, r["lat"] * 1e3, s=3, alpha=0.3, label=tag)
        axes[1].scatter(x, x / r["lat"] / tp, s=3, alpha=0.3, label=tag)
        if ncol == 3:
            axes[2].scatter(x, r["flops"] / r["lat"] / (peak * tp) * 100, s=3, alpha=0.3, label=tag)
    titles = [("prefill latency (ms)", "latency vs input length"), ("prefill tok/s per GPU", "prefill throughput"),
              ("MFU (%)", f"MFU (peak {args.peak_tflops:g} TFLOPS/GPU)")]
    for ax, (yl, t) in zip(axes, titles):
        ax.set_xscale("log")
        ax.set_xlabel("tokens per extend() call")
        ax.set_ylabel(yl)
        ax.set_title(t)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, markerscale=4)
    axes[0].set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "prefill.png"), dpi=130)
    plt.close(fig)
    out += ["![](figs/prefill.png)", ""]

    gtags = [t for t, r in runs.items() if r["gpu"] is not None]
    if gtags:
        fig, axes = plt.subplots(len(gtags), 1, figsize=(12, 2.6 * len(gtags)), squeeze=False)
        for ax, t in zip(axes[:, 0], gtags):
            g = runs[t]["gpu"]
            t0 = np.nanmin(g["epoch"])
            for i in np.unique(g["index"]):
                m = g["index"] == i
                ax.plot(g["epoch"][m] - t0, g["util_gpu"][m], lw=0.6, label=f"GPU{int(i)}")
            ax.set_ylim(0, 105)
            ax.set_ylabel("util %")
            ax.set_title(t, fontsize=9)
            ax.legend(fontsize=7, ncol=8, loc="lower left")
        axes[-1, 0].set_xlabel("seconds")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "gpu_timeline.png"), dpi=110)
        plt.close(fig)
        out += ["## GPU utilization", "", "![](figs/gpu_timeline.png)", ""]

    with open(os.path.join(rd, "report.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"[report] wrote {os.path.join(rd, 'report.md')}")


if __name__ == "__main__":
    main()
