"""Fine-grained `td` probe of beta-greedy clustering on a CGATr checkpoint.

Sweeps the assignment radius `td` over a small low-end grid (default
{0.02, 0.03, 0.04, 0.05}) at two `tbeta` thresholds. Useful when the
embedding clusters are tight and the coarse sweep in `eval_cgatr.py`
plateaus at the lower edge of its `td` grid — this script tells you
whether match rate is still climbing below it (i.e. clusters are even
tighter than expected) or has truly saturated.

Usage:
    cd model_training
    PYTHONPATH=. python src/eval_cgatr_tdprobe.py \\
        --data_dir <path_to_parquet_data> \\
        --checkpoint checkpoints/cgatr/cgatr_best.pt \\
        --embed_dim 5 --eval_seeds 181-182 --max_events 200 \\
        --output_dir eval_results/cgatr_tdprobe --gpu 0
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


def main():
    parser = argparse.ArgumentParser(description="Fine-grained td probe for CGATr")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eval_seeds", type=str, default="181-182")
    parser.add_argument("--max_hits", type=int, default=3000)
    parser.add_argument("--max_events", type=int, default=200)
    parser.add_argument("--embed_dim", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="eval_results/cgatr_tdprobe")
    parser.add_argument("--gpu", type=int, default=0)
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
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)", flush=True)

    start, end = parse_seed_range(args.eval_seeds)
    print(f"Loading eval data: seeds {start}-{end - 1}", flush=True)
    dataset = IDEAParquetDataset(args.data_dir, seed_range=(start, end),
                                  max_hits_per_event=args.max_hits)
    print(f"Dataset: {len(dataset)} events", flush=True)

    print("\n=== Running inference ===", flush=True)
    cached, _ = run_inference(model, dataset, device, args.max_events,
                              embed_dim=args.embed_dim, cosine_norm=False)

    print("\n=== Tiny td probe ===", flush=True)
    td_values = [0.02, 0.03, 0.04, 0.05]
    tbeta_values = [0.05, 0.10]
    results = []
    total = len(td_values) * len(tbeta_values)
    done = 0
    for tbeta in tbeta_values:
        for td in td_values:
            all_m = []
            for evt in cached:
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

    out_path = os.path.join(args.output_dir, "tdprobe_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "eval_seeds": args.eval_seeds,
            "embed_dim": args.embed_dim,
            "n_events": len(cached),
            "results": results,
        }, f, indent=2)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
