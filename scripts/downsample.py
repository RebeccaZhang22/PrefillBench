#!/usr/bin/env python3
"""Step 1: deterministic downsampling of the full per-source jsonl files (rule: sampling_config.json).

Only the header fields of each line (num_tokens / num_turns, written first by build_jsonl.py) are parsed,
so the scan is fast even for multi-GB files. Selected lines are copied verbatim.

Outputs (in --out-dir):
  sampled.jsonl         selected lines, grouped by source in config order, file order within a source
  sample_index.csv      source,id,num_tokens,num_turns,line_no for every selected line
  sample_manifest.json  config, per-source counts and a fingerprint (sha256 of the selected ids)
"""
import argparse
import csv
import hashlib
import json
import os
import re
import time
import zlib

import numpy as np

HEAD = re.compile(rb'^\{"id": "((?:[^"\\]|\\.)*)", "source": "[^"]*", "num_tokens": (\d+), "num_turns": (\d+)')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", required=True, help="directory holding the full jsonl files")
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def log(msg):
    print(f"[downsample {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scan(path):
    """Byte offsets, ids, num_tokens, num_turns of every line."""
    offs, ids, ntok, nturn = [], [], [], []
    with open(path, "rb") as f:
        pos = 0
        for line in f:
            m = HEAD.match(line)
            if m is None:
                raise SystemExit(f"{path}: line {len(offs) + 1} does not start with id/source/num_tokens/num_turns")
            offs.append(pos)
            ids.append(m.group(1).decode())
            ntok.append(int(m.group(2)))
            nturn.append(int(m.group(3)))
            pos += len(line)
    return np.array(offs, dtype=np.int64), ids, np.array(ntok), np.array(nturn)


def select(rule, ntok, max_input, seed):
    rng = np.random.default_rng([seed, zlib.crc32(rule["name"].encode())])
    u = rng.random(len(ntok))  # one draw per line, file order
    ok = (ntok >= rule.get("min_tokens", 0)) & (ntok <= max_input)
    ok &= (ntok >= rule.get("short_len", 0)) | (u < rule.get("short_keep", 1.0))
    cand = np.flatnonzero(ok)
    k = min(rule["num_samples"], len(cand))
    return np.sort(rng.choice(cand, size=k, replace=False)), int(ok.sum())


def main():
    args = parse_args()
    cfg = json.load(open(args.config))
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "sampled.jsonl")
    fp = hashlib.sha256()
    summary = []
    with open(out_path, "wb") as out, open(os.path.join(args.out_dir, "sample_index.csv"), "w", newline="") as idx_f:
        w = csv.writer(idx_f)
        w.writerow(["source", "id", "num_tokens", "num_turns", "line_no"])
        for rule in cfg["sources"]:
            path = os.path.join(args.data_dir, rule["file"])
            if not os.path.exists(path):
                raise SystemExit(f"missing {path}")
            t0 = time.time()
            offs, ids, ntok, nturn = scan(path)
            pick, n_eligible = select(rule, ntok, cfg["max_input_tokens"], cfg["seed"])
            with open(path, "rb") as f:
                for i in pick:
                    f.seek(offs[i])
                    out.write(f.readline())
                    w.writerow([rule["name"], ids[i], ntok[i], nturn[i], i + 1])
                    fp.update(ids[i].encode() + b"\n")
            if len(pick) < rule["num_samples"]:
                log(f"WARNING {rule['name']}: only {len(pick)} eligible lines, wanted {rule['num_samples']}")
            summary.append({"name": rule["name"], "file": rule["file"], "lines": len(ntok),
                            "over_max_input": int((ntok > cfg["max_input_tokens"]).sum()),
                            "eligible_after_rule": n_eligible, "sampled": int(len(pick)),
                            "sampled_tokens": int(ntok[pick].sum())})
            log(f"{rule['name']}: {len(ntok):,} lines -> {n_eligible:,} eligible -> {len(pick):,} sampled "
                f"({ntok[pick].sum() / 1e6:.1f}M tokens) [{time.time() - t0:.0f}s]")
    manifest = {"config": cfg, "sources": summary, "total_sampled": sum(s["sampled"] for s in summary),
                "fingerprint_sha256": fp.hexdigest()}
    with open(os.path.join(args.out_dir, "sample_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log(f"wrote {manifest['total_sampled']:,} lines to {out_path}")
    log(f"fingerprint (sha256 of selected ids): {manifest['fingerprint_sha256']}")


if __name__ == "__main__":
    main()
