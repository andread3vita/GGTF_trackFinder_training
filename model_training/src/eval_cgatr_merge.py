"""Two-pass greedy-then-merge clustering post-processor for CGATr.

Step 1: beta-greedy with a tight `td` (default 0.05) — produces small, very
        pure proto-clusters (the head of each track).
Step 2: merge pairs of proto-clusters whose centroids are within `td_merge`,
        gated by an optional purity check on the combined cluster.
Step 3: compute purity / efficiency / match rate on the merged labels.

The script reports two variants:
- `--use_purity_gate` (default on): merges only when the union's MC-majority
  purity is above `purity_gate`. Useful as an **oracle upper bound** to see how
  much an ideal post-processor could recover from tight proto-clusters.
- `--no_use_purity_gate`: merges purely on centroid distance, mimicking what a
  practical post-processor without MC labels could do.

Usage:
    cd model_training
    PYTHONPATH=. python src/eval_cgatr_merge.py \\
        --data_dir <path_to_parquet_data> \\
        --checkpoint checkpoints/cgatr/cgatr_best.pt \\
        --embed_dim 5 --eval_seeds 181-182 --max_events 200 \\
        --output_dir eval_results/cgatr_merge --gpu 0
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from src.eval_cgatr import (
    CGATrParquetModel,
    get_clustering_greedy,
    compute_metrics,
    run_inference,
    parse_seed_range,
)
from src.dataset.parquet_dataset import IDEAParquetDataset


# ---------------------------------------------------------------------------
# Merge utilities
# ---------------------------------------------------------------------------
def _cluster_centroids(coords, labels):
    """Return (unique_labels, centroids[n_clusters, d])."""
    unique = np.unique(labels[labels >= 0])
    if len(unique) == 0:
        return unique, np.zeros((0, coords.shape[1]))
    centroids = np.stack([coords[labels == u].mean(axis=0) for u in unique])
    return unique, centroids


def _majority_purity(labels_slice):
    """Majority-label fraction within a subset of MC labels.

    labels_slice: 1D int array of MC ids (>=0) for the hits in the cluster.
    """
    if len(labels_slice) == 0:
        return 0.0
    counts = np.bincount(labels_slice)
    return counts.max() / len(labels_slice)


def merge_proto_clusters(coords, proto_labels, mc_index,
                          td_merge=0.20, purity_gate=0.75,
                          use_purity_gate=True):
    """Union-find merge of proto-clusters on centroid distance + optional purity gate.

    - Computes all pairwise centroid distances.
    - Sorts candidate pairs ascending by distance.
    - Merges greedily if pair distance < td_merge AND (if gate enabled) the
      combined majority-purity >= purity_gate.

    Returns merged labels array with same shape as proto_labels.
    """
    unique, centroids = _cluster_centroids(coords, proto_labels)
    n = len(unique)
    if n < 2:
        return proto_labels.copy()

    # Union-Find
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    # Snapshot of hit memberships per group (root -> indices)
    label_to_idx = {int(u): i for i, u in enumerate(unique)}
    group_hits = {i: np.where(proto_labels == u)[0] for i, u in enumerate(unique)}

    # Pairwise distances
    from scipy.spatial.distance import squareform, pdist
    dist_cond = pdist(centroids)
    D = squareform(dist_cond)

    # Candidate pairs sorted by distance
    iu, ju = np.triu_indices(n, k=1)
    d_flat = D[iu, ju]
    order = np.argsort(d_flat)

    for k in order:
        d = d_flat[k]
        if d >= td_merge:
            break
        i, j = int(iu[k]), int(ju[k])
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        # Gather hits currently in the two root groups
        hits = np.concatenate([group_hits[ri], group_hits[rj]])
        if use_purity_gate:
            pur = _majority_purity(mc_index[hits])
            if pur < purity_gate:
                continue
        union(i, j)
        # Move all hits to the new root
        new_root = find(i)
        other = rj if new_root == ri else ri
        group_hits[new_root] = np.concatenate([group_hits[ri], group_hits[rj]])
        # Blank out the merged-in side to avoid double-counting
        group_hits[other] = np.empty(0, dtype=int)

    # Build new labels: root_idx -> unique label id
    merged = proto_labels.copy()
    for i, u in enumerate(unique):
        root = find(i)
        if root != i:
            merged[proto_labels == u] = unique[root]
    return merged


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------
def sweep_merge(cached_events, tbeta, td_seed, td_merge_values,
                 purity_gate=0.75, use_purity_gate=True):
    results = []
    for tdm in td_merge_values:
        all_m = []
        for evt in cached_events:
            proto = get_clustering_greedy(evt["beta"], evt["coords"],
                                           tbeta=tbeta, td=td_seed)
            merged = merge_proto_clusters(
                evt["coords"], proto, evt["mc_index"],
                td_merge=tdm, purity_gate=purity_gate,
                use_purity_gate=use_purity_gate,
            )
            all_m.append(compute_metrics(merged, evt["mc_index"]))
        results.append({
            "variant": "gated" if use_purity_gate else "unsupervised",
            "tbeta": tbeta,
            "td_seed": td_seed,
            "td_merge": tdm,
            "purity_gate": purity_gate if use_purity_gate else None,
            "purity": float(np.mean([m["purity"] for m in all_m])),
            "efficiency": float(np.mean([m["efficiency"] for m in all_m])),
            "match_rate": float(np.mean([m["match_rate"] for m in all_m])),
            "n_events": len(all_m),
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Two-pass greedy-then-merge post-processor for CGATr")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eval_seeds", type=str, default="181-182")
    parser.add_argument("--max_hits", type=int, default=3000)
    parser.add_argument("--max_events", type=int, default=200)
    parser.add_argument("--embed_dim", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="eval_results/cgatr_merge")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--tbeta", type=float, default=0.10)
    parser.add_argument("--td_seed", type=float, default=0.05)
    parser.add_argument("--purity_gate", type=float, default=0.75)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    print(f"Loading checkpoint: {args.checkpoint}", flush=True)
    model = CGATrParquetModel(embed_dim=args.embed_dim)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model = model.to(device)

    start, end = parse_seed_range(args.eval_seeds)
    print(f"Loading eval data: seeds {start}-{end - 1}", flush=True)
    dataset = IDEAParquetDataset(args.data_dir, seed_range=(start, end),
                                  max_hits_per_event=args.max_hits)

    print("\n=== Running inference ===", flush=True)
    cached, _ = run_inference(model, dataset, device, args.max_events,
                              embed_dim=args.embed_dim, cosine_norm=False)

    # Baseline: plain greedy at (tbeta, td_seed) and (tbeta, 0.20) for reference
    print("\n=== Baselines (no merge) ===", flush=True)
    baselines = []
    for td_ref in [args.td_seed, 0.10, 0.15, 0.20]:
        all_m = []
        for evt in cached:
            labels = get_clustering_greedy(evt["beta"], evt["coords"],
                                            tbeta=args.tbeta, td=td_ref)
            all_m.append(compute_metrics(labels, evt["mc_index"]))
        avg = {
            "variant": "baseline",
            "tbeta": args.tbeta,
            "td": td_ref,
            "purity": float(np.mean([m["purity"] for m in all_m])),
            "efficiency": float(np.mean([m["efficiency"] for m in all_m])),
            "match_rate": float(np.mean([m["match_rate"] for m in all_m])),
            "n_events": len(all_m),
        }
        baselines.append(avg)
        print(f"  greedy tbeta={args.tbeta:.2f} td={td_ref:.2f} | "
              f"pur={avg['purity']:.3f} eff={avg['efficiency']:.3f} match={avg['match_rate']:.3f}",
              flush=True)

    td_merge_values = [0.10, 0.15, 0.20, 0.25, 0.30]

    print("\n=== Gated merge (oracle purity gate >= "
          f"{args.purity_gate}) ===", flush=True)
    gated = sweep_merge(cached, args.tbeta, args.td_seed, td_merge_values,
                        purity_gate=args.purity_gate, use_purity_gate=True)
    for r in gated:
        print(f"  td_seed={r['td_seed']:.2f} td_merge={r['td_merge']:.2f} | "
              f"pur={r['purity']:.3f} eff={r['efficiency']:.3f} match={r['match_rate']:.3f}",
              flush=True)

    print("\n=== Unsupervised merge (no gate) ===", flush=True)
    unsup = sweep_merge(cached, args.tbeta, args.td_seed, td_merge_values,
                        purity_gate=args.purity_gate, use_purity_gate=False)
    for r in unsup:
        print(f"  td_seed={r['td_seed']:.2f} td_merge={r['td_merge']:.2f} | "
              f"pur={r['purity']:.3f} eff={r['efficiency']:.3f} match={r['match_rate']:.3f}",
              flush=True)

    out_path = os.path.join(args.output_dir, "merge_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "eval_seeds": args.eval_seeds,
            "embed_dim": args.embed_dim,
            "n_events": len(cached),
            "tbeta": args.tbeta,
            "td_seed": args.td_seed,
            "purity_gate": args.purity_gate,
            "baselines": baselines,
            "gated_merge": gated,
            "unsupervised_merge": unsup,
        }, f, indent=2)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
