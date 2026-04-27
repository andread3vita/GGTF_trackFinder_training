"""Zoom side-by-side: True MC | v34 predicted | v35 predicted.

Loads two checkpoints (with possibly different `--embed_dim`), runs inference
with each on the same events, then produces a 3-panel zoom view per event
showing the drift circles colored by track id. The same hits are used for
all three panels (only the predicted labels change), so any visual difference
is purely the model difference.

Usage:
    cd model_training
    PYTHONPATH=. python src/plot_events_3d_compare.py \
        --data_dir /home/marko.cechovic/cgatr/data_parquet_train \
        --ckpt_a checkpoints/cgatr_v34/cgatr_best.pt --embed_a 6 --label_a v34 \
        --ckpt_b checkpoints/cgatr_v35/cgatr_best.pt --embed_b 5 --label_b v35 \
        --algorithm greedy --tbeta 0.10 --td 0.20 \
        --eval_seeds 181-181 --n_events 6 --zoom_window 100 \
        --output_dir eval_results/v34_v35_zoom_compare
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from src.eval_sweep_v33 import (
    CGATrParquetModel,
    get_clustering_greedy,
    get_clustering_dbscan,
    parse_seed_range,
)
from src.dataset.parquet_dataset import IDEAParquetDataset
from src.plot_events_3d_v35_circles import (
    _build_drift_circles,
    _color_for,
    _remap_to_local,
    get_clustering_hdbscan_full,
)


@torch.no_grad()
def run_one(model, features, seq_lens, embed_dim):
    output = model(features, seq_lens)
    coords = output[:, :embed_dim].cpu().numpy()
    beta = torch.sigmoid(output[:, embed_dim]).cpu().numpy()
    return coords, beta


def cluster(coords, beta, args):
    if args.algorithm == "greedy":
        return get_clustering_greedy(beta, coords, tbeta=args.tbeta, td=args.td)
    if args.algorithm == "dbscan":
        return get_clustering_dbscan(coords, eps=args.eps, min_samples=args.min_samples)
    return get_clustering_hdbscan_full(
        coords, min_cluster_size=args.hdb_mcs, min_samples=args.hdb_ms,
        cluster_selection_epsilon=args.hdb_eps,
    )


def render_zoom_compare(features, mc_index, pred_a, pred_b, label_a, label_b,
                        out_path, title_suffix="", n_segments=40, window=100.0):
    is_dc = features[:, 3] > 0.5
    if not is_dc.any():
        return

    # Pick a target track that maximises (#distinct v34 clusters - #distinct
    # v35 clusters) over its DC hits, and is long enough (>=10 hits). I.e.
    # a track where v34 fragments and v35 consolidates — the regression-of-
    # interest case. Falls back to the densest track if nothing differs.
    tracks, counts = np.unique(mc_index[is_dc], return_counts=True)
    long_tracks = tracks[counts >= 10]
    best_score = -1
    target = tracks[np.argmax(counts)]  # default fallback
    for tid in long_tracks:
        m = is_dc & (mc_index == tid)
        n_a = len(np.unique(pred_a[m][pred_a[m] >= 0]))
        n_b = len(np.unique(pred_b[m][pred_b[m] >= 0]))
        score = (n_a - n_b)
        if score > best_score:
            best_score = score
            target = tid
    track_dc_mask = is_dc & (mc_index == target)
    track_wires = features[track_dc_mask][:, 4:7]
    z_order = np.argsort(track_wires[:, 2])
    center = track_wires[z_order[len(z_order) // 2]]
    # Annotate the target track's per-model fragmentation in the suptitle
    n_a_target = len(np.unique(pred_a[track_dc_mask][pred_a[track_dc_mask] >= 0]))
    n_b_target = len(np.unique(pred_b[track_dc_mask][pred_b[track_dc_mask] >= 0]))
    target_info = (
        f" | target true track has {track_dc_mask.sum()} DC hits, "
        f"split into {n_a_target} {label_a} cluster(s) vs {n_b_target} {label_b}"
    )
    title_suffix = title_suffix + target_info

    wire_or_pos = np.where(is_dc[:, None], features[:, 4:7], features[:, :3])
    diff = wire_or_pos - center
    in_window = (np.abs(diff[:, 0]) < window) & \
                (np.abs(diff[:, 1]) < window) & \
                (np.abs(diff[:, 2]) < window)
    feats_w = features[in_window]
    mc_w = mc_index[in_window]
    pa_w = pred_a[in_window]
    pb_w = pred_b[in_window]
    if len(feats_w) < 4:
        return

    is_vtx_w = feats_w[:, 3] < 0.5
    is_dc_w = ~is_vtx_w
    circles = _build_drift_circles(feats_w[is_dc_w], n_segments=n_segments)

    mc_local, n_true = _remap_to_local(mc_w)
    a_local, n_a = _remap_to_local(pa_w)
    b_local, n_b = _remap_to_local(pb_w)

    fig = plt.figure(figsize=(18, 7.2))
    panels = [
        (mc_local, n_true, "True MC tracks"),
        (a_local, n_a, f"{label_a} predicted"),
        (b_local, n_b, f"{label_b} predicted"),
    ]
    for k, (labels_full, n_uniq, name) in enumerate(panels):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        labels_dc = labels_full[is_dc_w]
        labels_vtx = labels_full[is_vtx_w]
        if len(circles) > 0:
            colors = np.array([_color_for(l) for l in labels_dc])
            ax.add_collection3d(Line3DCollection(circles, colors=colors,
                                                 linewidths=1.0))
            wire_xyz = feats_w[is_dc_w, 4:7]
            ax.scatter(wire_xyz[:, 0], wire_xyz[:, 1], wire_xyz[:, 2],
                       c=colors, s=6, depthshade=False, marker=".",
                       edgecolors="none", alpha=0.9)
        if is_vtx_w.any():
            vtx_xyz = feats_w[is_vtx_w, :3]
            vc = np.array([_color_for(l) for l in labels_vtx])
            ax.scatter(vtx_xyz[:, 0], vtx_xyz[:, 1], vtx_xyz[:, 2],
                       c=vc, s=40, edgecolors="black", linewidths=0.4,
                       depthshade=False)
        ax.set_xlim(center[0] - window, center[0] + window)
        ax.set_ylim(center[1] - window, center[1] + window)
        ax.set_zlim(center[2] - window, center[2] + window)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(f"{name}: {n_uniq} clusters in window\n"
                     f"({is_vtx_w.sum()} VTX, {is_dc_w.sum()} DC)")
        ax.view_init(elev=18, azim=35)
    # Wrap long suptitle into two lines
    main_title = f"{label_a} vs {label_b} zoom compare ({2*window:.0f}-unit window)"
    fig.suptitle(f"{main_title}\n{title_suffix}", fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--ckpt_a", type=str, required=True)
    parser.add_argument("--embed_a", type=int, required=True)
    parser.add_argument("--label_a", type=str, default="A")
    parser.add_argument("--ckpt_b", type=str, required=True)
    parser.add_argument("--embed_b", type=int, required=True)
    parser.add_argument("--label_b", type=str, default="B")
    parser.add_argument("--num_blocks", type=int, default=10)
    parser.add_argument("--eval_seeds", type=str, default="181-181")
    parser.add_argument("--max_hits", type=int, default=3000)
    parser.add_argument("--n_events", type=int, default=6)
    parser.add_argument("--algorithm", type=str, default="greedy",
                        choices=["greedy", "dbscan", "hdbscan"])
    parser.add_argument("--tbeta", type=float, default=0.10)
    parser.add_argument("--td", type=float, default=0.20)
    parser.add_argument("--eps", type=float, default=0.20)
    parser.add_argument("--min_samples", type=int, default=2)
    parser.add_argument("--hdb_mcs", type=int, default=2)
    parser.add_argument("--hdb_ms", type=int, default=1)
    parser.add_argument("--hdb_eps", type=float, default=0.10)
    parser.add_argument("--zoom_window", type=float, default=100.0)
    parser.add_argument("--n_segments", type=int, default=40)
    parser.add_argument("--output_dir", type=str,
                        default="eval_results/v34_v35_zoom_compare")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading {args.label_a}: {args.ckpt_a} (embed_dim={args.embed_a})")
    model_a = CGATrParquetModel(num_blocks=args.num_blocks, embed_dim=args.embed_a)
    sa = torch.load(args.ckpt_a, map_location="cpu", weights_only=False)
    if isinstance(sa, dict) and "model_state_dict" in sa:
        sa = sa["model_state_dict"]
    model_a.load_state_dict(sa)
    model_a = model_a.to(device).eval()

    print(f"Loading {args.label_b}: {args.ckpt_b} (embed_dim={args.embed_b})")
    model_b = CGATrParquetModel(num_blocks=args.num_blocks, embed_dim=args.embed_b)
    sb = torch.load(args.ckpt_b, map_location="cpu", weights_only=False)
    if isinstance(sb, dict) and "model_state_dict" in sb:
        sb = sb["model_state_dict"]
    model_b.load_state_dict(sb)
    model_b = model_b.to(device).eval()

    start, end = parse_seed_range(args.eval_seeds)
    dataset = IDEAParquetDataset(args.data_dir, seed_range=(start, end),
                                 max_hits_per_event=args.max_hits)
    print(f"Dataset: {len(dataset)} events")

    print(f"\n=== Comparing on {args.n_events} events ({args.algorithm}) ===")
    rendered = 0
    idx = 0
    while rendered < args.n_events and idx < len(dataset):
        event = dataset[idx]; idx += 1
        if event is None:
            continue
        features = event["features"].to(device)
        mc_index = event["mc_index"].numpy()
        is_secondary = event["is_secondary"].numpy().astype(bool)
        seq_lens = [event["n_hits"]]
        feats_np = features.cpu().numpy()

        coords_a, beta_a = run_one(model_a, features, seq_lens, args.embed_a)
        coords_b, beta_b = run_one(model_b, features, seq_lens, args.embed_b)

        sig_mask = (~is_secondary) & (mc_index != 0)
        if sig_mask.sum() < 200:
            continue

        feats_s = feats_np[sig_mask]
        mc_s = mc_index[sig_mask]
        pred_a = cluster(coords_a[sig_mask], beta_a[sig_mask], args)
        pred_b = cluster(coords_b[sig_mask], beta_b[sig_mask], args)

        out_path = os.path.join(args.output_dir,
                                f"event{idx-1:03d}_zoom_compare.png")
        render_zoom_compare(
            features=feats_s, mc_index=mc_s,
            pred_a=pred_a, pred_b=pred_b,
            label_a=args.label_a, label_b=args.label_b,
            out_path=out_path,
            title_suffix=f"event {idx-1} | {args.algorithm} tbeta={args.tbeta} td={args.td}",
            n_segments=args.n_segments, window=args.zoom_window,
        )
        rendered += 1

    print(f"\nRendered {rendered} comparison plots to {args.output_dir}/")


if __name__ == "__main__":
    main()
