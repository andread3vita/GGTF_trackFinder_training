"""Per-pT and per-eta efficiency breakdown for a CGATr checkpoint.

Runs beta-greedy clustering (default tbeta=0.10, td=0.20), joins the
per-cluster results back to the per-particle truth from the
`mc_particles_train.parquet` shards, and reports tracking efficiency
binned by transverse momentum and by pseudorapidity.

Usage:
    cd model_training
    PYTHONPATH=. python src/eval_cgatr_pt_eta.py \\
        --data_dir <path_to_parquet_data> \\
        --checkpoint checkpoints/cgatr/cgatr_best.pt \\
        --embed_dim 5 --eval_seeds 181-182 --max_events 500
"""

import os
import sys
import argparse
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from src.eval_cgatr import (
    CGATrParquetModel,
    get_clustering_greedy,
    parse_seed_range,
)
from src.dataset.parquet_dataset import IDEAParquetDataset, _cached_read


def compute_per_track_metrics(pred_labels, mc_index):
    """Compute per-track efficiency and match status.

    Returns list of dicts: [{mc_idx, efficiency, matched, n_hits}, ...]
    """
    unique_true = np.unique(mc_index)
    unique_true = unique_true[unique_true > 0]
    results = []
    for tid in unique_true:
        tmask = mc_index == tid
        n_true = tmask.sum()
        if n_true < 2:
            continue
        pred_for_track = pred_labels[tmask]
        pred_for_track = pred_for_track[pred_for_track >= 0]
        if len(pred_for_track) == 0:
            results.append({"mc_idx": int(tid), "efficiency": 0.0, "matched": False, "n_hits": int(n_true)})
            continue
        best_label, best_match = Counter(pred_for_track).most_common(1)[0]
        eff = best_match / n_true
        pur = best_match / (pred_labels == best_label).sum()
        results.append({
            "mc_idx": int(tid),
            "efficiency": float(eff),
            "matched": float(pur) > 0.75,
            "n_hits": int(n_true),
        })
    return results


@torch.no_grad()
def run_eval(model, dataset, device, max_events, embed_dim, tbeta, td):
    """Run inference + greedy clustering, return per-track metrics with event metadata."""
    model.eval()
    all_track_results = []
    n_events = min(max_events, len(dataset))

    for idx in range(n_events):
        event = dataset[idx]
        if event is None:
            continue

        features = event["features"].to(device)
        mc_index = event["mc_index"].numpy()
        is_secondary = event["is_secondary"].numpy().astype(bool)
        seq_lens = [event["n_hits"]]

        output = model(features, seq_lens)
        coords = output[:, :embed_dim].cpu().numpy()
        beta = torch.sigmoid(output[:, embed_dim]).cpu().numpy()

        sig_mask = (~is_secondary) & (mc_index != 0)
        if sig_mask.sum() < 4:
            continue

        sig_coords = coords[sig_mask]
        sig_beta = beta[sig_mask]
        sig_mc = mc_index[sig_mask]

        labels = get_clustering_greedy(sig_beta, sig_coords, tbeta=tbeta, td=td)
        track_metrics = compute_per_track_metrics(labels, sig_mc)

        dc_path, vtx_path, eid, _ = dataset._index[idx]
        seed = int(Path(dc_path).parent.name.replace("seed_", ""))

        for tm in track_metrics:
            tm["event_id"] = eid
            tm["seed"] = seed
        all_track_results.extend(track_metrics)

        if (idx + 1) % 50 == 0 or idx + 1 == n_events:
            print(f"  Events: {idx + 1}/{n_events} ({len(all_track_results)} tracks)", flush=True)

    return all_track_results


def load_mc_particles(data_dir, seed_start, seed_end):
    """Load mc_particles for given seed range, return DataFrame with pt, theta, eta."""
    dfs = []
    for seed in range(seed_start, seed_end):
        mc_path = Path(data_dir) / f"seed_{seed}" / "mc_particles_train.parquet"
        if not mc_path.exists():
            continue
        df = pl.read_parquet(str(mc_path), columns=["mc_index", "pt", "theta", "charge", "event_id", "seed"])
        dfs.append(df)
    if not dfs:
        return None
    combined = pl.concat(dfs)
    combined = combined.with_columns(
        (-pl.col("theta").truediv(2).tan().log()).alias("eta")
    )
    return combined


def bin_efficiency(track_df, col, bin_edges, label):
    """Compute efficiency and match rate per bin."""
    print(f"\n{'='*70}")
    print(f"Efficiency vs {label}")
    print(f"{'='*70}")
    print(f"{'Bin':>20}  {'Tracks':>7}  {'Efficiency':>10}  {'Match Rate':>10}")
    print(f"{'-'*55}")

    rows = []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (track_df[col] >= lo) & (track_df[col] < hi)
        subset = track_df.filter(mask)
        n = len(subset)
        if n == 0:
            continue
        avg_eff = subset["efficiency"].mean()
        match_rate = subset["matched"].cast(pl.Float64).mean()
        bin_label = f"[{lo:.2f}, {hi:.2f})"
        print(f"{bin_label:>20}  {n:>7}  {avg_eff:>10.3f}  {match_rate:>10.3f}")
        rows.append({"bin_lo": lo, "bin_hi": hi, "n_tracks": n,
                      "efficiency": float(avg_eff), "match_rate": float(match_rate)})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Per-pT and per-eta tracking efficiency for CGATr")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eval_seeds", type=str, default="181-182")
    parser.add_argument("--max_hits", type=int, default=3000)
    parser.add_argument("--max_events", type=int, default=500)
    parser.add_argument("--embed_dim", type=int, default=5)
    parser.add_argument("--tbeta", type=float, default=0.10)
    parser.add_argument("--td", type=float, default=0.20)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="eval_results/cgatr_pt_eta")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    model = CGATrParquetModel(embed_dim=args.embed_dim)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model = model.to(device)
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    seed_start, seed_end = parse_seed_range(args.eval_seeds)
    print(f"Loading eval data: seeds {seed_start}-{seed_end - 1}")
    dataset = IDEAParquetDataset(args.data_dir, seed_range=(seed_start, seed_end),
                                  max_hits_per_event=args.max_hits)
    print(f"Dataset: {len(dataset)} events")

    print(f"\n=== Running inference + clustering (tbeta={args.tbeta}, td={args.td}) ===")
    track_results = run_eval(model, dataset, device, args.max_events,
                              args.embed_dim, args.tbeta, args.td)
    print(f"Total tracks evaluated: {len(track_results)}")

    print(f"\nLoading mc_particles for seeds {seed_start}-{seed_end - 1}...")
    mc_df = load_mc_particles(args.data_dir, seed_start, seed_end)
    if mc_df is None:
        print("ERROR: No mc_particles found")
        return

    track_pl = pl.DataFrame(track_results)
    joined = track_pl.join(
        mc_df.select(["mc_index", "pt", "theta", "eta", "charge", "event_id", "seed"]),
        left_on=["mc_idx", "event_id", "seed"],
        right_on=["mc_index", "event_id", "seed"],
        how="left",
    )
    n_matched = joined.filter(pl.col("pt").is_not_null()).height
    n_total = joined.height
    print(f"Joined {n_matched}/{n_total} tracks with mc_particles")

    joined = joined.filter(pl.col("pt").is_not_null())
    if len(joined) == 0:
        print("ERROR: No tracks could be joined with mc_particles")
        return

    charged = joined.filter(pl.col("charge").abs() > 0.1)
    print(f"Charged tracks: {len(charged)} / {len(joined)} total")

    pt_bins = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    pt_results = bin_efficiency(charged, "pt", pt_bins, "pT [GeV]")

    eta_bins = [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    eta_results = bin_efficiency(charged, "eta", eta_bins, "eta")

    nhit_bins = [2, 5, 10, 20, 50, 100, 200, 500]
    nhit_results = bin_efficiency(
        charged.with_columns(pl.col("n_hits").cast(pl.Float64)),
        "n_hits", nhit_bins, "n_hits"
    )

    overall_eff = charged["efficiency"].mean()
    overall_match = charged["matched"].cast(pl.Float64).mean()
    print(f"\n{'='*70}")
    print(f"OVERALL (charged tracks): efficiency={overall_eff:.3f}, match_rate={overall_match:.3f}")
    print(f"{'='*70}")

    import json
    results = {
        "checkpoint": args.checkpoint,
        "eval_seeds": args.eval_seeds,
        "tbeta": args.tbeta,
        "td": args.td,
        "n_events_processed": min(args.max_events, len(dataset)),
        "n_tracks_evaluated": len(track_results),
        "n_charged_joined": len(charged),
        "overall_efficiency": float(overall_eff),
        "overall_match_rate": float(overall_match),
        "pt_bins": pt_results,
        "eta_bins": eta_results,
        "nhit_bins": nhit_results,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "pt_eta_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
