#!/usr/bin/env python3
"""summary.jsonl from serve_bench.py -> serve_report.md + figures.

Tables per shape (rows = concurrency, both prefill sizes side by side), the max concurrency whose p90 latency
meets each SLO, and throughput / latency curves.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SHAPE_TITLE = {"short": "短请求（~4k）", "long": "长请求（~64k，一次性 prefill）",
               "long_seg_cache": "长请求分 4 段，复用 KV cache（radix 开）",
               "long_seg_nocache": "长请求分 4 段，不复用 KV cache（radix 关，每段重算前缀）"}
SLO_KEY = {"short": "short", "long": "long", "long_seg_cache": "long", "long_seg_nocache": "long"}


def fmt_ps(ps):
    if ps == "default":
        return "默认"
    return f"{ps // 1024}k" if ps % 1024 == 0 else str(ps)


def fmt_ps_plot(ps):  # ASCII only: the default matplotlib font has no CJK glyphs
    return "default" if ps == "default" else fmt_ps(ps)


def md_table(rows):
    cols = list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(f"{v:,.2f}" if isinstance(v, float) else str(v) for v in r.values()) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--slo", default="{}", help="json {short: [s...], long: [s...]} from serve.slo_seconds")
    args = ap.parse_args()
    rd = args.result_dir
    slo = json.loads(args.slo)
    rows = [json.loads(line) for line in open(os.path.join(rd, "summary.jsonl"))]
    latest = {}  # a re-run of the same level replaces the earlier row
    for r in rows:
        latest[(r["shape"], r["prefill_size"], r["concurrency"])] = r
    rows = sorted(latest.values(), key=lambda r: (r["shape"], str(r["prefill_size"]), r["concurrency"]))
    shapes = [s for s in SHAPE_TITLE if any(r["shape"] == s for r in rows)]
    sizes = sorted({r["prefill_size"] for r in rows}, key=lambda p: 1 << 40 if p == "default" else p)
    lat = rows[0]["latent_layer"]

    out = ["# 请求形状 × 并发：prefill 吞吐与延迟", "",
           f"- 模型计算到：{'decoder block 0..%d（第 %d 层 latent，不做最终 norm / lm_head / 采样）' % (lat, lat) if lat >= 0 else '完整模型'}",
           "- 真实 SGLang 调度器（进程内 Engine，单卡），直接传 token ids，`max_new_tokens=1`：prefill 完即结束，没有 decode。",
           "- **prefill size** 同时设定 `chunked_prefill_size` 和 `max_prefill_tokens`：一次前向最多处理这么多 token，"
           "更长的请求被切块，跨多次前向完成。`默认` = 不设置，由 SGLang 按显存自动选择（H20：chunk 8192，"
           "max_prefill_tokens 16384）。",
           "- **并发 C**：闭环，始终有 C 个请求（分段形状为 C 条轨迹，每条轨迹的 4 段依次发送）在途。",
           "- **计算 tok/s** = 实际做 prefill 的 token（输入 − 命中缓存）÷ 时间，即 GPU 的工作量；"
           "**服务 tok/s** = 全部输入 token（含缓存命中）÷ 时间。",
           "- **延迟** = 客户端从提交到完成（排队 + prefill），按请求统计；分段形状里是每一段的延迟。", ""]

    for shape in shapes:
        out += [f"## {SHAPE_TITLE[shape]}", ""]
        sub = [r for r in rows if r["shape"] == shape]
        concs = sorted({r["concurrency"] for r in sub})
        table = []
        for c in concs:
            row = {"并发 C": c}
            for ps in sizes:
                r = next((x for x in sub if x["prefill_size"] == ps and x["concurrency"] == c), None)
                if r is None:
                    continue
                p = fmt_ps(ps)
                row[f"{p} 计算 tok/s"] = r["computed_tok_s"]
                if shape == "long_seg_cache":
                    row[f"{p} 服务 tok/s"] = r["prompt_tok_s"]
                    row[f"{p} 缓存命中 %"] = r["cache_hit_pct"]
                row[f"{p} 延迟 p50 (s)"] = r["lat_p50_s"]
                row[f"{p} 延迟 p90 (s)"] = r["lat_p90_s"]
            table.append(row)
        n_req = sum(r["requests"] for r in sub)
        out += [f"平均输入长度 {sub[0]['prompt_len_mean']:,.0f} token；共 {n_req:,} 个请求。", "", md_table(table), ""]

    # max concurrency under SLO
    if slo:
        out += ["## 满足延迟要求（p90）的最大并发", "",
                "在测过的并发档位里，p90 延迟不超过该值的最大 C；括号内是这个 C 下的计算 tok/s。"
                "`<1` 表示 C=1 也达不到。", ""]
        table = []
        for shape in shapes:
            for ps in sizes:
                sub = sorted((r for r in rows if r["shape"] == shape and r["prefill_size"] == ps),
                             key=lambda r: r["concurrency"])
                if not sub:
                    continue
                row = {"形状": shape, "prefill size": fmt_ps(ps)}
                for s in slo.get(SLO_KEY[shape], []):
                    ok = [r for r in sub if r["lat_p90_s"] <= s]
                    row[f"p90 ≤ {s:g}s"] = f"{ok[-1]['concurrency']} ({ok[-1]['computed_tok_s']:,.0f})" if ok else "<1"
                table.append(row)
        # shapes with different SLO lists have different columns: one table per SLO group
        for key in ("short", "long"):
            t = [r for r in table if SLO_KEY[r["形状"]] == key]
            if t:
                out += [md_table(t), ""]

    # figures
    fig, axes = plt.subplots(len(shapes), 2, figsize=(12, 3.6 * len(shapes)), squeeze=False)
    for ax_row, shape in zip(axes, shapes):
        for ps in sizes:
            sub = sorted((r for r in rows if r["shape"] == shape and r["prefill_size"] == ps),
                         key=lambda r: r["concurrency"])
            if not sub:
                continue
            c = [r["concurrency"] for r in sub]
            ax_row[0].plot(c, [r["computed_tok_s"] for r in sub], "o-", label=f"prefill size {fmt_ps_plot(ps)}")
            ax_row[1].plot(c, [r["lat_p90_s"] for r in sub], "o-", label=f"p90, {fmt_ps_plot(ps)}")
            ax_row[1].plot(c, [r["lat_p50_s"] for r in sub], "x--", alpha=0.6, label=f"p50, {fmt_ps_plot(ps)}")
        for ax, yl in zip(ax_row, ["computed prefill tok/s", "latency (s)"]):
            ax.set_xscale("log", base=2)
            ax.set_xlabel("concurrency C")
            ax.set_ylabel(yl)
            ax.set_ylim(bottom=0)
            ax.set_title(shape, fontsize=10)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.join(rd, "figs"), exist_ok=True)
    fig.savefig(os.path.join(rd, "figs", "serve.png"), dpi=120)
    plt.close(fig)
    out += ["## 曲线", "", "左：计算 tok/s 随并发变化；右：延迟随并发变化（横轴 log2）。", "", "![](figs/serve.png)", ""]

    with open(os.path.join(rd, "serve_report.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"[serve_report] wrote {os.path.join(rd, 'serve_report.md')}")


if __name__ == "__main__":
    main()
