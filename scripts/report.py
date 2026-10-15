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


def prefill_flops(cfg, L, latent_layer=-1):
    """Forward FLOPs of prefilling L tokens: dense matmuls + causal attention + last-token lm_head.
    latent_layer >= 0: only decoder blocks 0..latent_layer, no lm_head."""
    h, nh = cfg["hidden_size"], cfg["num_attention_heads"]
    layers = latent_layer + 1 if latent_layer >= 0 else cfg["num_hidden_layers"]
    head = 0 if latent_layer >= 0 else 2.0 * h * cfg.get("vocab_size", 0)
    nkv, hd = cfg.get("num_key_value_heads", nh), cfg.get("head_dim") or h // nh
    linear = layers * (h * nh * hd + 2 * h * nkv * hd + nh * hd * h + 3 * h * cfg["intermediate_size"])
    L = np.asarray(L, dtype=np.float64)
    return 2.0 * linear * L + 2.0 * layers * nh * hd * L ** 2 + head


def main():
    args = parse_args()
    rd = args.result_dir
    ds_dir = os.path.join(rd, "dataset")
    fig_dir = os.path.join(rd, "figs")
    os.makedirs(fig_dir, exist_ok=True)
    out = ["# Prefill 效率测试报告", ""]
    for name, title in [("config.yaml", "配置"), ("env.txt", "运行环境")]:
        p = os.path.join(rd, name)
        if os.path.exists(p):
            out += [f"## {title}", "", "```", open(p).read().strip(), "```", ""]

    # ---------------- dataset (downsampled set) ----------------
    man_p, st_p = os.path.join(ds_dir, "sample_manifest.json"), os.path.join(ds_dir, "sample_stats.json")
    if os.path.exists(man_p) and os.path.exists(st_p):
        man, st = json.load(open(man_p)), json.load(open(st_p))
        out += ["## 测试数据（下采样后）", "",
                f"- seed：`{man['config']['seed']}`；长度上限 max_input_tokens：{man['config']['max_input_tokens']:,}；"
                f"样本指纹：`{man['fingerprint_sha256'][:16]}`", ""]
        out += [md_table([{"数据源": x["name"], "原始可用行数": f"{x['lines']:,}",
                           "超过长度上限": f"{x['over_max_input']:,}", "规则过滤后": f"{x['eligible_after_rule']:,}",
                           "抽样条数": f"{x['sampled']:,}", "抽样 token 数": f"{x['sampled_tokens']:,}"}
                          for x in man["sources"]]), ""]
        out += [md_table([{"数据源": "合计" if r["source"] == "ALL" else r["source"], "请求数": f"{r['requests']:,}", "长度均值": r["tok_mean"],
                           "长度 p50": r["tok_p50"], "长度 p90": r["tok_p90"], "长度 p99": r["tok_p99"],
                           "长度最大": f"{r['tok_max']:,}", "轮次均值": r["turn_mean"], "轮次 p50": r["turn_p50"],
                           "轮次 p90": r["turn_p90"], "轮次最大": r["turn_max"]} for r in st["sources"]], "{:,.0f}"), ""]
        names = [k for k in st["length_buckets"][0] if k not in ("range", "req_%", "token_%")]
        out += ["各长度区间的请求数，以及在全部请求数、全部 token 数中的占比：", "",
                md_table([dict({"长度区间": x["range"]}, **{("合计" if k == "ALL" else k): x[k] for k in names},
                               **{"请求占比 %": x["req_%"], "token 占比 %": x["token_%"]})
                          for x in st["length_buckets"]], "{:.1f}"), "",
                "![](dataset/sample_distribution.png)", ""]

    # ---------------- runs ----------------
    run_dirs = sorted(d for d in glob.glob(os.path.join(rd, "runs", "*")) if glob.glob(os.path.join(d, "batches_shard*.csv")))
    if not run_dirs:
        out += ["_没有找到 prefill 测试结果。_"]
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
        gen = bool(rows) and all(r.get("decode_s") not in (None, "") for r in rows)
        dec = np.array([float(r["decode_s"]) for r in rows]) if gen else np.zeros(len(rows))
        runs[tag] = {"meta": meta, "lens": lens, "lat": lat, "tokens": np.array([int(r["tokens"]) for r in rows]),
                     "start": np.array([float(r["start_epoch"]) for r in rows]),
                     "end": np.array([float(r["end_epoch"]) for r in rows]) + dec,  # end of prefill + decode
                     "gpu": load_gpu(os.path.join(d, "gpu_monitor.csv")), "gen": gen, "decode": dec,
                     "out": np.array([int(r["out_len"]) for r in rows]) if gen else None,
                     "tpot": np.array([float(r["tpot_p50_ms"]) if r["tpot_p50_ms"] else np.nan for r in rows])
                     if gen else None}
    for r in runs.values():
        lat_l = r["meta"].get("latent_layer", -1)
        r["flops"] = np.array([prefill_flops(model_cfg, l, lat_l).sum() for l in r["lens"]]) if model_cfg else None

    out += ["## 纯 prefill 测试结果（SGLang one_batch，`ModelRunner.extend`）", "",
            "每次 `extend()` 调用前后各做一次 device synchronize 计时，不经过 server、HTTP、调度器，也不做 chunked prefill。",
            "",
            "口径说明：多张卡按数据并行各跑一部分请求时，**数量类指标（请求数、token 数、耗时）是所有卡的合计**，"
            "**效率类指标（每卡 tok/s、延迟、MFU）是单卡口径**。",
            "",
            "- **每卡 tok/s（单卡）** = token 总数 ÷ extend 总耗时（各卡合计）÷ TP 大小；",
            "- **总吞吐 tok/s（全部卡）** = token 总数 ÷ 墙钟时间；",
            "- **MFU（单卡）** = 理论计算量（线性层 + causal attention）÷ extend 总耗时 ÷（单卡峰值 × TP 大小）；",
            "- **墙钟时间**：从第一条请求开始到最后一条结束的实际时间，多卡并行时取整体跨度"
            "（生成模式下包含 decode 时间）。",
            "- **计算到的层**：`全部` 为完整模型（含最终 norm、lm_head、采样）；`0..L` 为 latent 模式，配置名带 `_L<L>`，"
            "prefill 只跑 decoder block 0..L，输出第 L 层的残差流（不做最终 norm / lm_head / 采样），"
            "MFU 的理论计算量也只算这些层。", ""]
    rows = []
    for tag, r in runs.items():
        tp, ns = r["meta"].get("tp", 1), r["meta"].get("num_shards", 1)
        busy = r["lat"].sum()
        wall = r["end"].max() - r["start"].min()
        lat_l = r["meta"].get("latent_layer", -1)
        row = {"配置": tag, "计算到的层": f"0..{lat_l}" if lat_l >= 0 else "全部", "GPU 数": tp * ns,
               "请求数（合计）": f"{sum(len(l) for l in r['lens']):,}",
               "extend 次数（合计）": f"{len(r['lat']):,}", "token 总数（合计）": f"{r['tokens'].sum():,}",
               "extend 总耗时（合计，GPU·h）": busy / 3600, "每卡 tok/s（单卡）": r["tokens"].sum() / busy / tp,
               "墙钟时间 (h)": wall / 3600}
        if not r["gen"]:  # in generation mode the wall clock also contains decode, so it is not a prefill rate
            row["总吞吐 tok/s（全部卡）"] = r["tokens"].sum() / wall
        if peak and r["flops"] is not None:
            row["MFU %（单卡）"] = r["flops"].sum() / busy / (peak * tp) * 100
        if r["gpu"] is not None:
            row["GPU 利用率 %（平均）"] = float(np.nanmean(r["gpu"]["util_gpu"]))
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
            b = {"输入长度": bucket_label(lo, hi), "extend 次数（合计）": int(m.sum()),
                 "平均 token 数": float(r["tokens"][m].mean()),
                 "延迟 p50 (ms)": float(np.median(r["lat"][m]) * 1e3),
                 "延迟 p90 (ms)": float(np.percentile(r["lat"][m], 90) * 1e3),
                 "每卡 tok/s（单卡）": r["tokens"][m].sum() / busy / tp}
            if peak and r["flops"] is not None:
                b["MFU %（单卡）"] = r["flops"][m].sum() / busy / (peak * tp) * 100
            brow.append(b)
        note = "每条请求单独做一次 extend()，延迟即该请求的纯 prefill 时间（不含任何服务开销的 TTFT）。" if single \
            else "多条请求拼成一次 extend()，按其中最长的请求归入长度区间。"
        out += [f"### `{tag}`：按输入长度分段", "", note, "", md_table(brow), ""]

    # figures: linear axes everywhere; top row = full length range, bottom row = zoom on 0-16k
    ncol = 3 if (peak and model_cfg) else 2
    zoom = 16384
    one_per_call = all(len(l) == 1 for r in runs.values() for l in r["lens"])
    xlabel = "input tokens per request" if one_per_call else "tokens per extend() call (packed requests)"
    fig, axes = plt.subplots(2, ncol, figsize=(6.2 * ncol, 9.0), squeeze=False)
    thousands = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    for row, (xmax, lat_unit, lat_scale) in enumerate([(None, "s", 1.0), (zoom, "ms", 1e3)]):
        for tag, r in runs.items():
            tp = r["meta"].get("tp", 1)
            x = r["tokens"]
            m = np.ones(len(x), dtype=bool) if xmax is None else x <= xmax
            ys = [r["lat"][m] * lat_scale, x[m] / r["lat"][m] / tp]
            if ncol == 3:
                ys.append(r["flops"][m] / r["lat"][m] / (peak * tp) * 100)
            for ax, y in zip(axes[row], ys):
                ax.scatter(x[m], y, s=3, alpha=0.3, label=tag)
        rng = "full range" if xmax is None else f"0-{xmax:,} tokens"
        titles = [(f"prefill latency ({lat_unit})", f"latency vs input length ({rng})"),
                  ("prefill tok/s per GPU", f"prefill throughput ({rng})"),
                  ("MFU (%)", f"MFU, peak {args.peak_tflops:g} TFLOPS/GPU ({rng})")]
        for ax, (yl, t) in zip(axes[row], titles):
            ax.set_xlim(0, xmax if xmax else None)
            ax.set_ylim(bottom=0)
            ax.xaxis.set_major_formatter(thousands)
            ax.yaxis.set_major_formatter(thousands)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(yl)
            ax.set_title(t, fontsize=10)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, markerscale=4)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "prefill.png"), dpi=130)
    plt.close(fig)
    out += ["上排为完整长度范围，下排放大 0–16k 区间；两个坐标轴都是线性刻度。", "", "![](figs/prefill.png)", ""]

    # ---------------- generation (prefill + decode) ----------------
    gen_runs = {t: r for t, r in runs.items() if r["gen"]}
    if gen_runs:
        out += ["## 完整生成测试（prefill + decode）", "",
                "每条请求先做一次 prefill（`extend()`），再用官方 `decode()` 逐个 token 生成到**数据中真实回复的长度**"
                "（忽略 EOS，长度有上限），每一步前后都做 device synchronize。均为单请求（batch size 1）、单卡口径。", "",
                "- **TTFT**：prefill 时间（即首 token 延迟）；",
                "- **TPOT**：每个输出 token 的 decode 时间（取该请求各步的中位数）；",
                "- **端到端延迟** = TTFT + 全部 decode 步的时间，即生成完整回复所需的时间；",
                "- **decode 占比** = decode 时间 ÷ 端到端延迟；**每卡 decode tok/s** = 输出 token 数 ÷ decode 时间。", ""]
        rows = []
        for tag, r in gen_runs.items():
            e2e = r["lat"] + r["decode"]
            rows.append({"配置": tag, "请求数（合计）": f"{len(e2e):,}", "输出 token 总数（合计）": f"{r['out'].sum():,}",
                         "平均输出长度": float(r["out"].mean()), "TTFT p50 (ms)": float(np.median(r["lat"]) * 1e3),
                         "TPOT p50 (ms)": float(np.nanmedian(r["tpot"])),
                         "端到端 p50 (s)": float(np.median(e2e)), "端到端 p90 (s)": float(np.percentile(e2e, 90)),
                         "decode 占比 %": r["decode"].sum() / e2e.sum() * 100,
                         "每卡 decode tok/s（单卡）": (r["out"] - 1).sum() / max(r["decode"].sum(), 1e-9)})
        out += [md_table(rows, "{:,.2f}"), ""]
        for tag, r in gen_runs.items():
            e2e = r["lat"] + r["decode"]
            key = r["tokens"]
            brow = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (key >= lo) & (key < hi)
                if not m.any():
                    continue
                brow.append({"输入长度": bucket_label(lo, hi), "请求数（合计）": int(m.sum()),
                             "平均输入长度": float(key[m].mean()), "平均输出长度": float(r["out"][m].mean()),
                             "TTFT p50 (ms)": float(np.median(r["lat"][m]) * 1e3),
                             "TPOT p50 (ms)": float(np.nanmedian(r["tpot"][m])),
                             "端到端 p50 (s)": float(np.median(e2e[m])), "端到端 p90 (s)": float(np.percentile(e2e[m], 90)),
                             "decode 占比 %": r["decode"][m].sum() / e2e[m].sum() * 100})
            out += [f"### `{tag}`：按输入长度分段", "", md_table(brow), ""]

        fig, axes = plt.subplots(2, 3, figsize=(18.6, 9.0), squeeze=False)
        for row, (xmax, unit, scale) in enumerate([(None, "s", 1.0), (zoom, "ms", 1e3)]):
            for tag, r in gen_runs.items():
                x = r["tokens"]
                m = np.ones(len(x), dtype=bool) if xmax is None else x <= xmax
                e2e = (r["lat"] + r["decode"])[m]
                axes[row][0].scatter(x[m], e2e * scale, s=3, alpha=0.3, label=tag)
                axes[row][1].scatter(x[m], r["tpot"][m], s=3, alpha=0.3, label=tag)
                axes[row][2].scatter(x[m], r["decode"][m] / e2e * 100, s=3, alpha=0.3, label=tag)
            rng = "full range" if xmax is None else f"0-{xmax:,} tokens"
            titles = [(f"end-to-end latency ({unit})", f"end-to-end latency, prefill + decode ({rng})"),
                      ("TPOT (ms)", f"time per output token ({rng})"),
                      ("decode share of end-to-end (%)", f"decode share of latency ({rng})")]
            for ax, (yl, t) in zip(axes[row], titles):
                ax.set_xlim(0, xmax if xmax else None)
                ax.set_ylim(bottom=0)
                ax.xaxis.set_major_formatter(thousands)
                ax.yaxis.set_major_formatter(thousands)
                ax.set_xlabel("input tokens per request")
                ax.set_ylabel(yl)
                ax.set_title(t, fontsize=10)
                ax.grid(alpha=0.3)
                ax.legend(fontsize=8, markerscale=4)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "generation.png"), dpi=130)
        plt.close(fig)
        out += ["上排为完整长度范围，下排放大 0–16k 区间；两个坐标轴都是线性刻度。每个点是一条请求，"
                "输出长度为该请求在数据中的真实回复长度。", "", "![](figs/generation.png)", ""]

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
        out += ["## GPU 利用率", "",
                "nvidia-smi 采样的 `utilization.gpu`（有 kernel 在运行的时间占比，不等于 SM 占用率）。", "",
                "![](figs/gpu_timeline.png)", ""]

    with open(os.path.join(rd, "report.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"[report] wrote {os.path.join(rd, 'report.md')}")


if __name__ == "__main__":
    main()
