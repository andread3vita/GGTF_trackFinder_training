"""Comprehensive clustering analysis for a CGATr checkpoint.

Extends `eval_cgatr.py` (the coarse sweep) with:
- Fine-grained beta-greedy grid (td down to 0.05).
- Fine-grained DBSCAN grid (eps down to 0.05).
- Expanded HDBSCAN grid (min_cluster_size x min_samples x cluster_selection_epsilon).
- Full beta percentile histograms (0.0-1.0 in 0.1 steps) for signal + noise.
- Per-dimension embedding statistics (mean, median, std, min, max, skew).
- Composite ranking metrics:
    * rank_avg : average of ranks in purity/efficiency/match_rate (lower = better)
    * geo_mean : geometric mean of (purity, efficiency, match_rate)
    * harm_mean: harmonic mean  of (purity, efficiency, match_rate)
- Saves raw embeddings + per-event caches for the plotting script.

Designed to be run on a slice of the eval set per GPU; combine outputs across
GPUs with `merge_cgatr_analysis.py` and then plot with `plot_cgatr_analysis.py`.

Usage:
    cd model_training
    PYTHONPATH=. python src/eval_cgatr_analysis.py \\
        --data_dir <path_to_parquet_data> \\
        --checkpoint checkpoints/cgatr/cgatr_best.pt \\
        --embed_dim 5 --eval_seeds 181-182 --max_events 1500 \\
        --output_dir eval_results/cgatr_analysis
"""

import os
import sys
import json
import argparse
import pickle
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from src.eval_cgatr import (
    CGATrParquetModel,
    get_clustering_greedy,
    get_clustering_dbscan,
    compute_metrics,
    run_inference,
    parse_seed_range,
)
from src.dataset.parquet_dataset import IDEAParquetDataset


# ---------------------------------------------------------------------------
# Expanded clustering functions
# ---------------------------------------------------------------------------
def get_clustering_hdbscan_full(coords, min_cluster_size=5, min_samples=None,
                                 cluster_selection_epsilon=0.0):
    try:
        from hdbscan import HDBSCAN
        return HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
        ).fit_predict(coords)
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Composite ranking helpers
# ---------------------------------------------------------------------------
def add_composite_scores(results):
    """Add rank_avg, geo_mean, harm_mean to each result dict."""
    if not results:
        return results
    purs = np.array([r["purity"] for r in results])
    effs = np.array([r["efficiency"] for r in results])
    matches = np.array([r["match_rate"] for r in results])

    # ranks: 1 = best (highest value). Tie-break by order.
    def _ranks(arr):
        order = np.argsort(-arr)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(arr) + 1)
        return ranks

    r_pur = _ranks(purs)
    r_eff = _ranks(effs)
    r_match = _ranks(matches)

    for i, r in enumerate(results):
        r["rank_pur"] = int(r_pur[i])
        r["rank_eff"] = int(r_eff[i])
        r["rank_match"] = int(r_match[i])
        r["rank_avg"] = float((r_pur[i] + r_eff[i] + r_match[i]) / 3.0)
        clipped = max(min(purs[i], effs[i], matches[i]), 1e-6)
        r["geo_mean"] = float((purs[i] * effs[i] * matches[i]) ** (1 / 3))
        r["harm_mean"] = float(
            3.0 / (1.0 / max(purs[i], 1e-6)
                   + 1.0 / max(effs[i], 1e-6)
                   + 1.0 / max(matches[i], 1e-6))
        )
    return results


def print_top_n(results, key, n=10, reverse=True, label=""):
    sorted_r = sorted(results, key=lambda r: r[key], reverse=reverse)
    print(f"\n{'=' * 90}")
    print(f"TOP {n} by {label or key}")
    print(f"{'=' * 90}")
    print(f"{'Rank':>4}  {'Algorithm':<12} {'Params':<32} "
          f"{'Pur':>6} {'Eff':>6} {'Match':>6} {'GeoM':>6} {'HarM':>6} {'RankAvg':>7}")
    print(f"{'-' * 90}")
    for i, r in enumerate(sorted_r[:n]):
        print(f"{i + 1:>4}  {r['algorithm']:<12} {r['params']:<32} "
              f"{r['purity']:>6.3f} {r['efficiency']:>6.3f} {r['match_rate']:>6.3f} "
              f"{r['geo_mean']:>6.3f} {r['harm_mean']:>6.3f} {r['rank_avg']:>7.2f}")


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------
def sweep_greedy_fine(cached_events, tbeta_values, td_values):
    results = []
    total = len(tbeta_values) * len(td_values)
    done = 0
    for tbeta in tbeta_values:
        for td in td_values:
            all_m = []
            for evt in cached_events:
                labels = get_clustering_greedy(evt["beta"], evt["coords"], tbeta=tbeta, td=td)
                all_m.append(compute_metrics(labels, evt["mc_index"]))
            avg = {
                "purity": float(np.mean([m["purity"] for m in all_m])),
                "efficiency": float(np.mean([m["efficiency"] for m in all_m])),
                "match_rate": float(np.mean([m["match_rate"] for m in all_m])),
                "n_events": len(all_m),
                "algorithm": "greedy",
                "params": f"tbeta={tbeta:.2f}, td={td:.2f}",
                "tbeta": tbeta,
                "td": td,
            }
            results.append(avg)
            done += 1
            print(f"  greedy [{done}/{total}] tbeta={tbeta:.2f} td={td:.2f} | "
                  f"pur={avg['purity']:.3f} eff={avg['efficiency']:.3f} match={avg['match_rate']:.3f}",
                  flush=True)
    return results


def sweep_dbscan_fine(cached_events, eps_values, min_samples_values):
    results = []
    total = len(eps_values) * len(min_samples_values)
    done = 0
    for eps in eps_values:
        for ms in min_samples_values:
            all_m = []
            for evt in cached_events:
                labels = get_clustering_dbscan(evt["coords"], eps=eps, min_samples=ms)
                all_m.append(compute_metrics(labels, evt["mc_index"]))
            avg = {
                "purity": float(np.mean([m["purity"] for m in all_m])),
                "efficiency": float(np.mean([m["efficiency"] for m in all_m])),
                "match_rate": float(np.mean([m["match_rate"] for m in all_m])),
                "n_events": len(all_m),
                "algorithm": "DBSCAN",
                "params": f"eps={eps:.2f}, ms={ms}",
                "eps": eps,
                "min_samples": ms,
            }
            results.append(avg)
            done += 1
            print(f"  DBSCAN [{done}/{total}] eps={eps:.2f} ms={ms} | "
                  f"pur={avg['purity']:.3f} eff={avg['efficiency']:.3f} match={avg['match_rate']:.3f}",
                  flush=True)
    return results


def sweep_hdbscan_full(cached_events, mcs_values, ms_values, eps_values):
    results = []
    total = len(mcs_values) * len(ms_values) * len(eps_values)
    done = 0
    for mcs in mcs_values:
        for ms in ms_values:
            for eps in eps_values:
                all_m = []
                skip = False
                for evt in cached_events:
                    labels = get_clustering_hdbscan_full(
                        evt["coords"], min_cluster_size=mcs,
                        min_samples=ms, cluster_selection_epsilon=eps,
                    )
                    if labels is None:
                        print("  HDBSCAN not installed, skipping", flush=True)
                        skip = True
                        break
                    all_m.append(compute_metrics(labels, evt["mc_index"]))
                if skip:
                    return results
                avg = {
                    "purity": float(np.mean([m["purity"] for m in all_m])),
                    "efficiency": float(np.mean([m["efficiency"] for m in all_m])),
                    "match_rate": float(np.mean([m["match_rate"] for m in all_m])),
                    "n_events": len(all_m),
                    "algorithm": "HDBSCAN",
                    "params": f"mcs={mcs}, ms={ms}, eps={eps:.2f}",
                    "min_cluster_size": mcs,
                    "min_samples": ms,
                    "cluster_selection_epsilon": eps,
                }
                results.append(avg)
                done += 1
                print(f"  HDBSCAN [{done}/{total}] mcs={mcs} ms={ms} eps={eps:.2f} | "
                      f"pur={avg['purity']:.3f} eff={avg['efficiency']:.3f} match={avg['match_rate']:.3f}",
                      flush=True)
    return results


# ---------------------------------------------------------------------------
# Beta + embedding statistics
# ---------------------------------------------------------------------------
def full_beta_stats(all_betas, label):
    """Print fine-grained beta distribution from 0.0 to 1.0 in 0.1 steps."""
    print(f"\n{label} beta histogram ({len(all_betas):,} hits):")
    print(f"  {'Bin':>15}  {'Count':>10}  {'Fraction':>10}")
    print(f"  {'-' * 40}")
    edges = np.arange(0.0, 1.0001, 0.1)
    hist, _ = np.histogram(all_betas, bins=edges)
    for i, count in enumerate(hist):
        pct = 100 * count / max(len(all_betas), 1)
        print(f"  [{edges[i]:.1f}, {edges[i + 1]:.1f})  {count:>10}  {pct:>9.2f}%")
    print(f"  Stats: mean={all_betas.mean():.4f}  median={np.median(all_betas):.4f}  "
          f"std={all_betas.std():.4f}  min={all_betas.min():.4f}  max={all_betas.max():.4f}")
    print(f"  Percentiles (5, 25, 50, 75, 95): "
          f"{np.percentile(all_betas, [5, 25, 50, 75, 95])}")


def embedding_stats_full(coords, embed_dim):
    """Full per-dim stats: mean, median, std, min, max, skew."""
    print(f"\nEmbedding per-dim statistics ({len(coords):,} signal hits):")
    print(f"  {'Dim':>3}  {'Mean':>8} {'Median':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Skew':>8}")
    print(f"  {'-' * 60}")
    for d in range(embed_dim):
        v = coords[:, d]
        mean = v.mean()
        median = np.median(v)
        std = v.std()
        vmin = v.min()
        vmax = v.max()
        skew = float(((v - mean) ** 3).mean() / (std ** 3 + 1e-12))
        print(f"  {d:>3}  {mean:>8.3f} {median:>8.3f} {std:>8.3f} "
              f"{vmin:>8.3f} {vmax:>8.3f} {skew:>8.3f}")
    norms = np.linalg.norm(coords, axis=1)
    print(f"\n  Norm per hit: mean={norms.mean():.3f}  median={np.median(norms):.3f}  "
          f"std={norms.std():.3f}  min={norms.min():.3f}  max={norms.max():.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Comprehensive analysis sweep for a CGATr checkpoint")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eval_seeds", type=str, default="181-182")
    parser.add_argument("--max_hits", type=int, default=3000)
    parser.add_argument("--max_events", type=int, default=1500)
    parser.add_argument("--num_blocks", type=int, default=10)
    parser.add_argument("--hidden_mv_channels", type=int, default=16)
    parser.add_argument("--hidden_s_channels", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=5)
    parser.add_argument("--beta_mlp", action="store_true", default=False)
    parser.add_argument("--cosine_norm", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default="eval_results/cgatr_analysis")
    parser.add_argument("--save_samples", type=int, default=500000,
                        help="Save up to N (coords, mc_index, beta) rows for plotting")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load model ---
    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"embed_dim={args.embed_dim}")
    model = CGATrParquetModel(
        hidden_mv_channels=args.hidden_mv_channels,
        hidden_s_channels=args.hidden_s_channels,
        num_blocks=args.num_blocks,
        embed_dim=args.embed_dim,
        beta_mlp=args.beta_mlp,
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model = model.to(device)
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    # --- Load data ---
    start, end = parse_seed_range(args.eval_seeds)
    print(f"Loading eval data: seeds {start}-{end - 1}, max_hits={args.max_hits}")
    dataset = IDEAParquetDataset(args.data_dir, seed_range=(start, end),
                                  max_hits_per_event=args.max_hits)
    print(f"Dataset: {len(dataset)} events")

    # --- Inference ---
    print("\n=== Running inference ===")
    cached, noise_betas_list = run_inference(model, dataset, device, args.max_events,
                                              embed_dim=args.embed_dim,
                                              cosine_norm=args.cosine_norm)

    # --- Embedding stats ---
    all_coords = np.concatenate([e["coords"] for e in cached])
    all_mc = np.concatenate([e["mc_index"] for e in cached])
    all_sig_betas = np.concatenate([e["beta"] for e in cached])
    all_noise_betas = (np.concatenate(noise_betas_list) if noise_betas_list
                       else np.array([], dtype=np.float32))

    embedding_stats_full(all_coords, args.embed_dim)
    full_beta_stats(all_sig_betas, "SIGNAL")
    if len(all_noise_betas):
        full_beta_stats(all_noise_betas, "NOISE/SECONDARY")

    # --- Save raw samples for plotting (subsample if too many) ---
    n_to_save = min(args.save_samples, len(all_coords))
    rng = np.random.default_rng(42)
    idx_sig = rng.choice(len(all_coords), size=n_to_save, replace=False)
    n_noise_save = min(args.save_samples, len(all_noise_betas))
    idx_noise = (rng.choice(len(all_noise_betas), size=n_noise_save, replace=False)
                 if len(all_noise_betas) else np.array([], dtype=int))
    samples_path = os.path.join(args.output_dir, "samples.pkl")
    with open(samples_path, "wb") as f:
        pickle.dump({
            "coords": all_coords[idx_sig].astype(np.float32),
            "mc_index": all_mc[idx_sig].astype(np.int64),
            "beta_signal": all_sig_betas[idx_sig].astype(np.float32),
            "beta_noise": all_noise_betas[idx_noise].astype(np.float32) if len(all_noise_betas) else np.array([]),
            "embed_dim": args.embed_dim,
        }, f)
    print(f"\nSaved {n_to_save} signal + {n_noise_save} noise samples to {samples_path}")

    # --- Per-event data for t-SNE ---
    tsne_events_path = os.path.join(args.output_dir, "tsne_events.pkl")
    tsne_events = [cached[i] for i in range(min(8, len(cached)))]
    with open(tsne_events_path, "wb") as f:
        pickle.dump(tsne_events, f)
    print(f"Saved {len(tsne_events)} full events to {tsne_events_path} for t-SNE/UMAP")

    # --- Beta-greedy sweep (fine) ---
    all_results = []
    print("\n=== Beta-greedy fine sweep ===")
    tbeta_values = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    td_values = [0.05, 0.08, 0.10, 0.12, 0.15, 0.17, 0.20, 0.22, 0.25, 0.30, 0.40, 0.50]
    all_results.extend(sweep_greedy_fine(cached, tbeta_values, td_values))

    # --- DBSCAN sweep (fine) ---
    print("\n=== DBSCAN fine sweep ===")
    eps_values = [0.05, 0.08, 0.10, 0.12, 0.15, 0.17, 0.20, 0.22, 0.25, 0.30, 0.40]
    ms_values = [2, 3, 4, 5]
    all_results.extend(sweep_dbscan_fine(cached, eps_values, ms_values))

    # --- HDBSCAN sweep (expanded) ---
    print("\n=== HDBSCAN expanded sweep ===")
    mcs_values = [2, 3, 4, 5, 6, 8, 10]
    ms_values_h = [None, 1, 3]
    eps_values_h = [0.0, 0.1, 0.2]
    all_results.extend(sweep_hdbscan_full(cached, mcs_values, ms_values_h, eps_values_h))

    # --- Add composite scores ---
    all_results = add_composite_scores(all_results)

    # --- Print top-N tables ---
    print_top_n(all_results, "match_rate", n=15, label="Match Rate")
    print_top_n(all_results, "purity", n=15, label="Purity")
    print_top_n(all_results, "efficiency", n=15, label="Efficiency")
    print_top_n(all_results, "geo_mean", n=15, label="Geometric Mean (pur * eff * match)^(1/3)")
    print_top_n(all_results, "harm_mean", n=15, label="Harmonic Mean of (pur, eff, match)")
    print_top_n(all_results, "rank_avg", n=15, reverse=False,
                 label="Average Rank (lower = better across all 3)")

    # --- Best per algorithm by each metric ---
    print(f"\n{'=' * 90}")
    print("BEST PER ALGORITHM (per metric)")
    print(f"{'=' * 90}")
    for algo in ["greedy", "DBSCAN", "HDBSCAN"]:
        algo_results = [r for r in all_results if r["algorithm"] == algo]
        if not algo_results:
            continue
        print(f"\n  [{algo}] {len(algo_results)} configs")
        for metric, reverse in [("match_rate", True), ("purity", True),
                                 ("efficiency", True), ("geo_mean", True),
                                 ("rank_avg", False)]:
            best = sorted(algo_results, key=lambda r: r[metric], reverse=reverse)[0]
            print(f"    best {metric:<11}: {best['params']:<32}  "
                  f"pur={best['purity']:.3f} eff={best['efficiency']:.3f} "
                  f"match={best['match_rate']:.3f} geo={best['geo_mean']:.3f}")

    # --- Save JSON ---
    output_path = os.path.join(args.output_dir, "analysis_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "eval_seeds": args.eval_seeds,
            "embed_dim": args.embed_dim,
            "n_events": len(cached),
            "n_signal_hits": len(all_coords),
            "n_noise_hits": int(len(all_noise_betas)),
            "embedding_std_per_dim": [float(all_coords[:, d].std()) for d in range(args.embed_dim)],
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
