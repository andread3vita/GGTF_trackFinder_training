"""Merge per-GPU `eval_cgatr_analysis.py` outputs into a single set of results.

- Event-weighted averages of purity / efficiency / match rate per
  (algorithm, params) cell of the sweep.
- Concatenated signal+noise beta samples and embedding coordinate samples.
- Concatenated per-event caches (used by the t-SNE / UMAP plots).
- Re-computed composite scores (rank_avg, geo_mean, harm_mean).
- Produces a unified `analysis_results.json` and `samples.pkl` /
  `tsne_events.pkl` / `umap_events.pkl`.

Usage:
    cd model_training
    PYTHONPATH=. python src/merge_cgatr_analysis.py \\
        --dirs eval_results/cgatr_analysis_gpu0 eval_results/cgatr_analysis_gpu1 \\
               eval_results/cgatr_analysis_gpu2 eval_results/cgatr_analysis_gpu3 \\
        --out  eval_results/cgatr_analysis_merged
"""

import os
import sys
import json
import pickle
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def merge_sweep_results(all_json):
    """Event-weighted merge by (algorithm, params) key."""
    grouped = defaultdict(list)
    for payload in all_json:
        for r in payload["results"]:
            key = (r["algorithm"], r["params"])
            grouped[key].append(r)

    merged = []
    for (algo, params), rs in grouped.items():
        total_n = sum(r["n_events"] for r in rs)
        if total_n == 0:
            continue
        pur = sum(r["purity"] * r["n_events"] for r in rs) / total_n
        eff = sum(r["efficiency"] * r["n_events"] for r in rs) / total_n
        match = sum(r["match_rate"] * r["n_events"] for r in rs) / total_n
        new = {
            "algorithm": algo,
            "params": params,
            "purity": float(pur),
            "efficiency": float(eff),
            "match_rate": float(match),
            "n_events": int(total_n),
        }
        for k in ("tbeta", "td", "eps", "min_samples", "min_cluster_size",
                  "cluster_selection_epsilon"):
            if k in rs[0]:
                new[k] = rs[0][k]
        merged.append(new)
    return merged


def add_composite_scores(results):
    if not results:
        return results
    purs = np.array([r["purity"] for r in results])
    effs = np.array([r["efficiency"] for r in results])
    matches = np.array([r["match_rate"] for r in results])

    def _ranks(arr):
        order = np.argsort(-arr)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(arr) + 1)
        return ranks

    r_pur, r_eff, r_match = _ranks(purs), _ranks(effs), _ranks(matches)
    for i, r in enumerate(results):
        r["rank_pur"] = int(r_pur[i])
        r["rank_eff"] = int(r_eff[i])
        r["rank_match"] = int(r_match[i])
        r["rank_avg"] = float((r_pur[i] + r_eff[i] + r_match[i]) / 3.0)
        r["geo_mean"] = float((purs[i] * effs[i] * matches[i]) ** (1 / 3))
        r["harm_mean"] = float(
            3.0 / (1.0 / max(purs[i], 1e-6)
                   + 1.0 / max(effs[i], 1e-6)
                   + 1.0 / max(matches[i], 1e-6))
        )
    return results


def print_top_n(results, key, n=20, reverse=True, label=""):
    sorted_r = sorted(results, key=lambda r: r[key], reverse=reverse)
    print(f"\n{'=' * 96}")
    print(f"TOP {n} by {label or key}")
    print(f"{'=' * 96}")
    print(f"{'Rank':>4}  {'Algorithm':<10} {'Params':<34} "
          f"{'Pur':>6} {'Eff':>6} {'Match':>6} {'GeoM':>6} {'HarM':>6} {'RankAvg':>7}")
    print(f"{'-' * 96}")
    for i, r in enumerate(sorted_r[:n]):
        print(f"{i + 1:>4}  {r['algorithm']:<10} {r['params']:<34} "
              f"{r['purity']:>6.3f} {r['efficiency']:>6.3f} {r['match_rate']:>6.3f} "
              f"{r['geo_mean']:>6.3f} {r['harm_mean']:>6.3f} {r['rank_avg']:>7.2f}")


def merge_samples(dirs):
    """Concatenate sample pickle files (coords, mc_index, beta_signal, beta_noise)."""
    coords, mc_index, beta_signal, beta_noise = [], [], [], []
    embed_dim = None
    for d in dirs:
        pkl_path = os.path.join(d, "samples.pkl")
        if not os.path.exists(pkl_path):
            print(f"  [warn] {pkl_path} not found, skipping")
            continue
        with open(pkl_path, "rb") as f:
            s = pickle.load(f)
        coords.append(s["coords"])
        mc_index.append(s["mc_index"])
        beta_signal.append(s["beta_signal"])
        if len(s["beta_noise"]):
            beta_noise.append(s["beta_noise"])
        embed_dim = s["embed_dim"]
    return {
        "coords": np.concatenate(coords) if coords else np.zeros((0, embed_dim or 6), np.float32),
        "mc_index": np.concatenate(mc_index) if mc_index else np.zeros(0, np.int64),
        "beta_signal": np.concatenate(beta_signal) if beta_signal else np.zeros(0, np.float32),
        "beta_noise": np.concatenate(beta_noise) if beta_noise else np.zeros(0, np.float32),
        "embed_dim": embed_dim or 6,
    }


def merge_tsne_events(dirs):
    events = []
    for d in dirs:
        pkl_path = os.path.join(d, "tsne_events.pkl")
        if not os.path.exists(pkl_path):
            continue
        with open(pkl_path, "rb") as f:
            events.extend(pickle.load(f))
    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top_n", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # --- Load all JSONs ---
    all_json = []
    for d in args.dirs:
        p = os.path.join(d, "analysis_results.json")
        if not os.path.exists(p):
            print(f"[skip] {p} not found")
            continue
        with open(p) as f:
            all_json.append(json.load(f))
        print(f"Loaded {p} ({len(all_json[-1]['results'])} configs, "
              f"{all_json[-1]['n_events']} events)")

    # --- Merge sweeps ---
    merged = merge_sweep_results(all_json)
    merged = add_composite_scores(merged)
    print(f"\nMerged {len(merged)} configs across {sum(j['n_events'] for j in all_json)} events")

    # Per-dim std (averaged across runs)
    std_per_dim = np.mean(
        [np.array(j["embedding_std_per_dim"]) for j in all_json], axis=0
    ).tolist()
    total_sig = sum(j["n_signal_hits"] for j in all_json)
    total_noise = sum(j["n_noise_hits"] for j in all_json)

    # --- Print top-N tables ---
    print_top_n(merged, "match_rate", args.top_n, label="Match Rate")
    print_top_n(merged, "purity", args.top_n, label="Purity")
    print_top_n(merged, "efficiency", args.top_n, label="Efficiency")
    print_top_n(merged, "geo_mean", args.top_n, label="Geometric Mean")
    print_top_n(merged, "harm_mean", args.top_n, label="Harmonic Mean")
    print_top_n(merged, "rank_avg", args.top_n, reverse=False, label="Avg Rank (lower better)")

    # --- Best per algorithm ---
    print(f"\n{'=' * 96}")
    print("BEST PER ALGORITHM (per metric)")
    print(f"{'=' * 96}")
    for algo in ["greedy", "DBSCAN", "HDBSCAN"]:
        alg_r = [r for r in merged if r["algorithm"] == algo]
        if not alg_r:
            continue
        print(f"\n  [{algo}] {len(alg_r)} configs")
        for metric, rev in [("match_rate", True), ("purity", True), ("efficiency", True),
                             ("geo_mean", True), ("rank_avg", False)]:
            best = sorted(alg_r, key=lambda r: r[metric], reverse=rev)[0]
            print(f"    best {metric:<11}: {best['params']:<34}  "
                  f"pur={best['purity']:.3f} eff={best['efficiency']:.3f} "
                  f"match={best['match_rate']:.3f} geo={best['geo_mean']:.3f} "
                  f"rankAvg={best['rank_avg']:.2f}")

    # --- Save merged JSON ---
    out_json = os.path.join(args.out, "analysis_results.json")
    with open(out_json, "w") as f:
        json.dump({
            "source_dirs": args.dirs,
            "n_events_total": sum(j["n_events"] for j in all_json),
            "n_signal_hits_total": total_sig,
            "n_noise_hits_total": total_noise,
            "embedding_std_per_dim_avg": std_per_dim,
            "results": merged,
        }, f, indent=2)
    print(f"\nSaved merged JSON: {out_json}")

    # --- Merge & save sample pickles ---
    merged_samples = merge_samples(args.dirs)
    samples_path = os.path.join(args.out, "samples.pkl")
    with open(samples_path, "wb") as f:
        pickle.dump(merged_samples, f)
    print(f"Saved merged samples: {samples_path}  "
          f"({len(merged_samples['coords']):,} sig + {len(merged_samples['beta_noise']):,} noise)")

    tsne_events = merge_tsne_events(args.dirs)
    tsne_path = os.path.join(args.out, "tsne_events.pkl")
    with open(tsne_path, "wb") as f:
        pickle.dump(tsne_events, f)
    print(f"Saved {len(tsne_events)} events for t-SNE/UMAP: {tsne_path}")


if __name__ == "__main__":
    main()
