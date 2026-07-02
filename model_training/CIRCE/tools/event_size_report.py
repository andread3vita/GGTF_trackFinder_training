#!/usr/bin/env python
"""Event-size + max_tokens report for the C-GATr training data.

Scans a parquet data dir (seed_*/{dc,vtx}_hits_train.parquet), prints the
per-event hit-count distribution, and recommends the largest OOM-safe
--max_tokens for 32 GB and 40 GB GPUs (fp32, no grad-checkpoint, the
xformers attention path).

    python tools/event_size_report.py <data_dir> [--seeds A-B]

Memory model (calibrated on Tesla V100-32GB, 4-GPU DDP, fp32, xformers
attention + memory-frugal geometric product, no grad-checkpoint):

    nvidia-smi peak per GPU  ~=  SLOPE * H/1000 + FIXED      [GB]

where H = hits in the heaviest batch on a rank = max(max_tokens, largest_event)
(events larger than the token budget are admitted as singleton batches, so the
largest single event sets a hard floor). FIXED folds in the CUDA context,
cuDNN workspaces, NCCL/gradient buffers and per-rank batch imbalance.

Calibration points (nvidia-smi rank-max, 4xV100):
    H=24000 -> 29.85 GB  =>  SLOPE=1.08 GB/1k, FIXED=4.3 GB  (model: 30.2 GB)
Re-measure and update these two constants if the model / GPU count changes.
"""
import argparse
import glob
import os
import sys

import numpy as np

try:
    import polars as pl
except ImportError:
    sys.exit("install polars")

SLOPE = 1.08    # GB per 1000 hits in a batch (per GPU)
FIXED = 4.3     # GB fixed overhead (context + cuDNN + NCCL + imbalance)
SAFETY = 2.0    # GB head-room left free for a long unattended run


def scan(data_dir, seeds=None):
    sizes = []
    dirs = sorted(glob.glob(os.path.join(data_dir, "seed_*")),
                  key=lambda p: int(p.split("_")[-1]))
    if seeds is not None:
        lo, hi = seeds
        dirs = [d for d in dirs if lo <= int(d.split("_")[-1]) <= hi]
    n_seed = 0
    for s in dirs:
        dc = os.path.join(s, "dc_hits_train.parquet")
        vt = os.path.join(s, "vtx_hits_train.parquet")
        if not (os.path.exists(dc) and os.path.exists(vt)):
            continue
        try:
            d = pl.scan_parquet(dc).group_by("event_id").agg(pl.len().alias("n")).collect()
            v = pl.scan_parquet(vt).group_by("event_id").agg(pl.len().alias("n")).collect()
            dm = dict(zip(d["event_id"], d["n"]))
            vm = dict(zip(v["event_id"], v["n"]))
            for e in set(dm) & set(vm):
                sizes.append(dm[e] + vm[e])
            n_seed += 1
        except Exception as ex:
            print(f"  skip {s}: {ex}")
    return np.array(sizes), n_seed


def max_tokens_for(gpu_gb, max_event):
    """Largest OOM-safe max_tokens, and whether the biggest event itself fits."""
    usable = gpu_gb - SAFETY
    max_H = (usable - FIXED) / SLOPE * 1000.0          # max batch hits that fit
    event_ok = max_event <= max_H
    # max_tokens budget capped by max_H; round down to a clean 1000.
    mt = int(max_H // 1000) * 1000
    # the budget should be >= the largest event (else needless singletons) but
    # capped by what fits; if the event doesn't fit, mt is irrelevant (cap hits).
    return mt, event_ok, max_H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", help="parquet dir containing seed_*/")
    ap.add_argument("--seeds", default=None, help="A-B inclusive seed range")
    args = ap.parse_args()
    seeds = None
    if args.seeds:
        a, b = args.seeds.split("-"); seeds = (int(a), int(b))

    a, n_seed = scan(args.data_dir, seeds)
    if len(a) == 0:
        sys.exit(f"no events found under {args.data_dir}")
    mx = int(a.max())
    print(f"\nevents: {len(a):,} from {n_seed} seeds  ({args.data_dir})")
    for p in [50, 90, 99, 99.9, 100]:
        print(f"  p{p:5.1f} = {np.percentile(a, p):8.0f} hits")
    print(f"  mean = {a.mean():.0f}   MAX = {mx}")

    print("\n--- recommended --max_tokens (fp32, no grad-ckpt, xformers attn) ---")
    print(f"  model: peak[GB] ~= {SLOPE}*H/1000 + {FIXED};  leave {SAFETY} GB free")
    for gpu in (32, 40):
        mt, ok, max_H = max_tokens_for(gpu, mx)
        peak = SLOPE * mt / 1000 + FIXED
        if ok:
            print(f"  {gpu} GB GPU: --max_tokens {mt:6d}   "
                  f"(predicted peak ~{peak:.1f} GB; largest event {mx} fits)")
        else:
            cap = int(max_H // 100) * 100
            print(f"  {gpu} GB GPU: LARGEST EVENT ({mx}) EXCEEDS capacity "
                  f"(~{max_H:.0f} hits). It becomes a singleton batch and WILL OOM. "
                  f"Either use a >GPU, or add --max_hits {cap} to clip the few "
                  f"oversized events.")


if __name__ == "__main__":
    main()
