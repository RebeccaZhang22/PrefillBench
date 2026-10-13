#!/usr/bin/env python3
"""Step 2: distribution of the downsampled set.

Figure (sample_distribution.png): left = prompt token distribution, right = turn distribution,
one color per source. Also prints a summary table and writes sample_stats.json.
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample-dir", required=True, help="output dir of downsample.py")
    p.add_argument("--config", required=True)
    return p.parse_args()


def fmt_k(x):
    return f"{x / 1024:g}k" if x >= 1024 else str(x)


def main():
    args = parse_args()
    cfg = json.load(open(args.config))
    tok, turns = defaultdict(list), defaultdict(list)
    with open(os.path.join(args.sample_dir, "sample_index.csv")) as f:
        for r in csv.DictReader(f):
            tok[r["source"]].append(int(r["num_tokens"]))
            turns[r["source"]].append(int(r["num_turns"]))
    names = [s["name"] for s in cfg["sources"] if s["name"] in tok]
    tok = {k: np.array(tok[k]) for k in names}
    turns = {k: np.array(turns[k]) for k in names}
    all_tok = np.concatenate([tok[k] for k in names])
    all_turn = np.concatenate([turns[k] for k in names])
    edges = cfg["length_buckets"]

    # ---------------- summary ----------------
    def row(name, a, t):
        q = np.percentile(a, [50, 90, 99])
        return {"source": name, "requests": int(len(a)), "total_tokens": int(a.sum()), "tok_mean": float(a.mean()),
                "tok_p50": float(q[0]), "tok_p90": float(q[1]), "tok_p99": float(q[2]), "tok_max": int(a.max()),
                "turn_mean": float(t.mean()), "turn_p50": float(np.median(t)), "turn_p90": float(np.percentile(t, 90)),
                "turn_max": int(t.max())}

    rows = [row(k, tok[k], turns[k]) for k in names] + [row("ALL", all_tok, all_turn)]
    buckets = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = {"range": f"{fmt_k(lo)}-{fmt_k(hi)}"}
        for k in names + ["ALL"]:
            a = all_tok if k == "ALL" else tok[k]
            m = (a >= lo) & (a < hi)
            b[k] = int(m.sum())
        m = (all_tok >= lo) & (all_tok < hi)
        b["req_%"] = m.mean() * 100
        b["token_%"] = all_tok[m].sum() / all_tok.sum() * 100
        buckets.append(b)

    w = 112
    print("=" * w)
    print(f" Downsampled benchmark set  (seed {cfg['seed']}, max_input_tokens {cfg['max_input_tokens']:,})")
    print("=" * w)
    print(f" {'source':<14}{'requests':>10}{'tokens':>14}{'mean':>9}{'p50':>9}{'p90':>9}{'p99':>9}{'max':>9}"
          f"   | turns {'mean':>6}{'p50':>6}{'p90':>6}{'max':>6}")
    for r in rows:
        print(f" {r['source']:<14}{r['requests']:>10,}{r['total_tokens']:>14,}{r['tok_mean']:>9,.0f}{r['tok_p50']:>9,.0f}"
              f"{r['tok_p90']:>9,.0f}{r['tok_p99']:>9,.0f}{r['tok_max']:>9,}   |       {r['turn_mean']:>6.1f}"
              f"{r['turn_p50']:>6.0f}{r['turn_p90']:>6.0f}{r['turn_max']:>6}")
    print("-" * w)
    print(f" {'tokens':<12}" + "".join(f"{k:>14}" for k in names) + f"{'ALL':>10}{'% requests':>13}{'% tokens':>11}")
    for b in buckets:
        print(f" {b['range']:<12}" + "".join(f"{b[k]:>14,}" for k in names) + f"{b['ALL']:>10,}"
              f"{b['req_%']:>12.1f}%{b['token_%']:>10.1f}%")
    print("=" * w)

    # ---------------- figure ----------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2))
    hi = max(edges[-1], int(all_tok.max()) + 1)
    width = max(256, hi // 128)
    bins = np.arange(0, hi + width, width)
    ax1.hist([tok[k] for k in names], bins=bins, stacked=True, color=COLORS[:len(names)],
             label=[f"{k} (n={len(tok[k]):,})" for k in names])
    ax1.set_yscale("log")
    for p, ls in [(50, "-"), (90, "--"), (99, ":")]:
        v = np.percentile(all_tok, p)
        ax1.axvline(v, color="k", lw=0.8, ls=ls, label=f"all p{p} = {v:,.0f}")
    ax1.set_xlim(0, hi)
    ax1.set_xlabel(f"prompt tokens (chat template applied), bin = {width}")
    ax1.set_ylabel("requests (log scale)")
    ax1.set_title(f"token distribution: {len(all_tok):,} requests, {all_tok.sum() / 1e6:,.0f}M tokens")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    tmax = int(all_turn.max())
    tb = np.arange(1, tmax + 2) - 0.5
    ax2.hist([turns[k] for k in names], bins=tb, stacked=True, color=COLORS[:len(names)],
             label=[f"{k} (p50 {np.median(turns[k]):.0f} turns)" for k in names])
    ax2.set_yscale("log")
    ax2.set_xlim(0.5, tmax + 0.5)
    ax2.set_xlabel("turns (assistant messages in context + 1)")
    ax2.set_ylabel("requests (log scale)")
    ax2.set_title("turn distribution")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = os.path.join(args.sample_dir, "sample_distribution.png")
    fig.savefig(fig_path, dpi=130)
    print(f" figure: {fig_path}")

    with open(os.path.join(args.sample_dir, "sample_stats.json"), "w") as f:
        json.dump({"sources": rows, "length_buckets": buckets}, f, indent=2)


if __name__ == "__main__":
    main()
