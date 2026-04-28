"""3D event displays with DC drift circles (v35 / hot-startable).

Companion to plot_events_3d_v35.py. The point version draws each DC hit as a
single dot at the closest-approach coordinate. This version draws each DC hit
as the actual *drift circle* (centered on the wire, radius = drift_distance,
in the plane perpendicular to the wire), and VTX hits as solid dots.

The wire direction follows the same convention as the model's
`embed_circle_ipns` call:
    wire_dir = (sin(stereo)*cos(azim), sin(stereo)*sin(azim), cos(stereo))

Output PNGs go to <output_dir>/event<NNN>_3d_circles.png plus a multi-event
grid in <output_dir>/grid_circles_{true,predicted}.png.

Usage:
    cd model_training
    PYTHONPATH=. python src/plot_events_3d_v35_circles.py \
        --data_dir /home/marko.cechovic/cgatr/data_parquet_train \
        --checkpoint checkpoints/cgatr_v35/cgatr_best.pt \
        --embed_dim 5 --eval_seeds 181-181 \
        --algorithm greedy --tbeta 0.10 --td 0.20 \
        --n_events 6 --output_dir eval_results/v35_events_3d_circles

To reuse the same events the point-plot script picked, pass
`--n_events 6 --eval_seeds 181-181` (the collection logic is identical).
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


def get_clustering_hdbscan_full(coords, min_cluster_size=2, min_samples=1,
                                 cluster_selection_epsilon=0.0):
    from hdbscan import HDBSCAN
    return HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
    ).fit_predict(coords)


# ---- consistent qualitative colormap (shared with the points-only script) ----
def _build_palette():
    return np.concatenate([
        plt.cm.tab20.colors,
        plt.cm.tab20b.colors,
        plt.cm.tab20c.colors,
    ])


PALETTE = _build_palette()
NOISE_COLOR = (0.6, 0.6, 0.6, 0.55)


def _color_for(label):
    if label < 0:
        return NOISE_COLOR
    rgb = PALETTE[int(label) % len(PALETTE)]
    return (rgb[0], rgb[1], rgb[2], 0.85)


def _remap_to_local(labels):
    uniq = np.unique(labels[labels >= 0])
    mapping = {u: i for i, u in enumerate(uniq)}
    out = np.full_like(labels, -1)
    for i, lab in enumerate(labels):
        if lab >= 0:
            out[i] = mapping[lab]
    return out, len(uniq)


def _wire_basis(wire_dir):
    """Return (u, v) orthonormal basis spanning the plane perpendicular to wire_dir.

    wire_dir: (N, 3) array of unit-ish wire directions.
    Returns u, v: each (N, 3) and orthonormal w.r.t. wire_dir and each other.
    """
    w = wire_dir / np.linalg.norm(wire_dir, axis=1, keepdims=True).clip(min=1e-9)
    # Pick a helper vector that is least-parallel to w on a per-row basis.
    helper = np.tile(np.array([1.0, 0.0, 0.0]), (w.shape[0], 1))
    parallel_mask = np.abs(w[:, 0]) > 0.9  # if w is mostly along x, switch helper
    helper[parallel_mask] = np.array([0.0, 1.0, 0.0])
    u = np.cross(w, helper)
    u /= np.linalg.norm(u, axis=1, keepdims=True).clip(min=1e-9)
    v = np.cross(w, u)
    v /= np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-9)
    return u, v


def _build_drift_circles(features_dc, n_segments=24):
    """Build a list of (n_segments+1, 3) polylines, one per DC hit.

    features_dc: (N, 10) tensor rows where col 4-6 = wire xyz, col 7 = drift_dist,
                 col 8 = azim, col 9 = stereo.
    """
    if len(features_dc) == 0:
        return np.zeros((0, n_segments + 1, 3), dtype=np.float32)
    wire_xyz = features_dc[:, 4:7]
    drift = features_dc[:, 7:8]
    azim = features_dc[:, 8]
    stereo = features_dc[:, 9]

    cos_s = np.cos(stereo); sin_s = np.sin(stereo)
    cos_a = np.cos(azim);   sin_a = np.sin(azim)
    w_dir = np.stack([sin_s * cos_a, sin_s * sin_a, cos_s], axis=-1)

    u, v = _wire_basis(w_dir)
    t = np.linspace(0.0, 2.0 * np.pi, n_segments + 1)  # closed circle (last == first)
    cos_t = np.cos(t)[None, :]   # (1, S)
    sin_t = np.sin(t)[None, :]
    # Per-row outer: u (N,3) * cos_t (1,S) -> (N, S, 3)
    circles = (
        wire_xyz[:, None, :]
        + drift[:, None, :] * (u[:, None, :] * cos_t[..., None]
                               + v[:, None, :] * sin_t[..., None])
    )
    return circles  # (N, S+1, 3)


def render_3d_circles(features, mc_index, pred_labels,
                      out_path, title_suffix="", n_segments=20):
    """Side-by-side 3D plot with VTX as dots and DC as drift circles.

    features: (N_signal, 10) numpy array. mc_index/pred_labels: (N_signal,).
    """
    is_vtx = features[:, 3] < 0.5
    is_dc = ~is_vtx

    vtx_xyz = features[is_vtx, :3]
    vtx_mc = mc_index[is_vtx]
    vtx_pred = pred_labels[is_vtx]

    dc_features = features[is_dc]
    dc_mc = mc_index[is_dc]
    dc_pred = pred_labels[is_dc]
    dc_circles = _build_drift_circles(dc_features, n_segments=n_segments)

    mc_local, n_true = _remap_to_local(mc_index)
    pred_local, n_pred = _remap_to_local(pred_labels)

    fig = plt.figure(figsize=(16, 7))
    for k, (labels_full, labels_name, n_uniq) in enumerate([
        (mc_local, "True MC tracks", n_true),
        (pred_local, "Predicted tracks", n_pred),
    ]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")

        # Map colors using the local labels for THIS panel
        labels_vtx = labels_full[is_vtx]
        labels_dc = labels_full[is_dc]

        # DC drift circles via Line3DCollection (fast batch rendering)
        if len(dc_circles) > 0:
            colors = np.array([_color_for(l) for l in labels_dc])
            segs = dc_circles  # (N, S+1, 3) — Line3DCollection wants (N, S+1, 3)
            lc = Line3DCollection(segs, colors=colors, linewidths=0.6)
            ax.add_collection3d(lc)

        # VTX hits as solid dots, slightly larger so they pop against the circles
        if len(vtx_xyz) > 0:
            vtx_colors = np.array([_color_for(l) for l in labels_vtx])
            ax.scatter(
                vtx_xyz[:, 0], vtx_xyz[:, 1], vtx_xyz[:, 2],
                c=vtx_colors, s=22, depthshade=False, edgecolors="black",
                linewidths=0.3, marker="o", label=f"VTX ({len(vtx_xyz)})",
            )

        # axis bounds: union of all circle and VTX points so collection3d shows up
        all_pts = []
        if len(dc_circles) > 0:
            all_pts.append(dc_circles.reshape(-1, 3))
        if len(vtx_xyz) > 0:
            all_pts.append(vtx_xyz)
        if all_pts:
            pts = np.concatenate(all_pts, axis=0)
            mins = pts.min(axis=0); maxs = pts.max(axis=0)
            pad = 0.02 * (maxs - mins).max()
            ax.set_xlim(mins[0] - pad, maxs[0] + pad)
            ax.set_ylim(mins[1] - pad, maxs[1] + pad)
            ax.set_zlim(mins[2] - pad, maxs[2] + pad)
        ax.set_xlabel("x [a.u.]")
        ax.set_ylabel("y [a.u.]")
        ax.set_zlabel("z [a.u.]")
        ax.set_title(
            f"{labels_name}: {n_uniq} clusters\n"
            f"({len(vtx_xyz)} VTX dots, {len(dc_features)} DC circles)"
        )
        ax.view_init(elev=18, azim=35)

    fig.suptitle(f"v35 event display (drift circles) | {title_suffix}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def render_xy_circles(features, mc_index, pred_labels, out_path,
                      title_suffix=""):
    """xy projection with DC drift circles drawn as 2D circles via patches.

    Stereo wires project to ellipses in xy, but for clarity we draw the
    in-plane component only (i.e. ignore the z-component of the circle).
    """
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection

    is_vtx = features[:, 3] < 0.5
    is_dc = ~is_vtx
    dc_features = features[is_dc]
    vtx_xyz = features[is_vtx, :3]
    mc_local, n_true = _remap_to_local(mc_index)
    pred_local, n_pred = _remap_to_local(pred_labels)
    labels_vtx_t = mc_local[is_vtx]; labels_dc_t = mc_local[is_dc]
    labels_vtx_p = pred_local[is_vtx]; labels_dc_p = pred_local[is_dc]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, lvtx, ldc, n_uniq, name in [
        (axes[0], labels_vtx_t, labels_dc_t, n_true, "True MC tracks"),
        (axes[1], labels_vtx_p, labels_dc_p, n_pred, "Predicted tracks"),
    ]:
        # DC: 2D circles centered on (wire_x, wire_y) with radius = drift_distance.
        # NOTE: this is the projection of a 3D circle onto xy. For stereo wires
        # the true projection is an ellipse, but the deviation is small.
        if len(dc_features) > 0:
            patches = [
                Circle((row[4], row[5]), row[7])
                for row in dc_features
            ]
            colors = np.array([_color_for(l) for l in ldc])
            pc = PatchCollection(patches, edgecolor=colors, facecolor="none",
                                 linewidths=0.4)
            ax.add_collection(pc)
        if len(vtx_xyz) > 0:
            vc = np.array([_color_for(l) for l in lvtx])
            ax.scatter(vtx_xyz[:, 0], vtx_xyz[:, 1], c=vc, s=14,
                       edgecolors="black", linewidths=0.3, marker="o")
        # bounds
        if len(dc_features) > 0:
            xs = dc_features[:, 4]; ys = dc_features[:, 5]
            r = dc_features[:, 7]
            xmin, xmax = (xs - r).min(), (xs + r).max()
            ymin, ymax = (ys - r).min(), (ys + r).max()
        else:
            xmin, xmax = vtx_xyz[:, 0].min(), vtx_xyz[:, 0].max()
            ymin, ymax = vtx_xyz[:, 1].min(), vtx_xyz[:, 1].max()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{name}: {n_uniq} clusters")
        ax.grid(alpha=0.3)
    fig.suptitle(f"v35 xy with drift circles | {title_suffix}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def render_3d_circles_zoom(features, mc_index, pred_labels,
                            out_path, title_suffix="", n_segments=40,
                            window=80.0):
    """3D plot zoomed to a small window around a high-DC-hit track so the
    drift circles are visible at their true scale."""
    is_dc = features[:, 3] > 0.5
    dc_idx_global = np.where(is_dc)[0]
    if len(dc_idx_global) == 0:
        return
    # Pick the true track with the most DC hits, fall back to any track.
    tracks, counts = np.unique(mc_index[is_dc], return_counts=True)
    if len(tracks) == 0:
        return
    target_track = tracks[np.argmax(counts)]
    track_dc_mask = is_dc & (mc_index == target_track)
    if not track_dc_mask.any():
        return
    # Anchor on a specific DC hit (the median by z of the densest track) rather
    # than the mean position, which for curved tracks lands far from any hit.
    track_wires = features[track_dc_mask][:, 4:7]
    z_order = np.argsort(track_wires[:, 2])
    center = track_wires[z_order[len(z_order) // 2]]

    # Restrict to hits whose wire (or vtx pos) falls inside a window around center.
    wire_or_pos = np.where(is_dc[:, None], features[:, 4:7], features[:, :3])
    diff = wire_or_pos - center
    in_window = (np.abs(diff[:, 0]) < window) & \
                (np.abs(diff[:, 1]) < window) & \
                (np.abs(diff[:, 2]) < window)
    feats_w = features[in_window]
    mc_w = mc_index[in_window]
    pred_w = pred_labels[in_window]
    if len(feats_w) < 4:
        return

    is_vtx_w = feats_w[:, 3] < 0.5
    is_dc_w = ~is_vtx_w
    circles = _build_drift_circles(feats_w[is_dc_w], n_segments=n_segments)

    mc_local, n_true = _remap_to_local(mc_w)
    pred_local, n_pred = _remap_to_local(pred_w)

    fig = plt.figure(figsize=(15, 6.5))
    for k, (labels_full, n_uniq, name) in enumerate([
        (mc_local, n_true, "True MC"),
        (pred_local, n_pred, "Predicted"),
    ]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
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
                       depthshade=False, marker="o")
        ax.set_xlim(center[0] - window, center[0] + window)
        ax.set_ylim(center[1] - window, center[1] + window)
        ax.set_zlim(center[2] - window, center[2] + window)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(f"{name}: {n_uniq} clusters in window\n"
                     f"({is_vtx_w.sum()} VTX dots, {is_dc_w.sum()} DC drift circles)")
        ax.view_init(elev=18, azim=35)
    fig.suptitle(
        f"v35 zoom view ({2*window:.0f}-unit window around densest track) | {title_suffix}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def render_grid_circles(events, out_path, view_kind="pred", n_segments=18):
    """Compact multi-event grid with drift circles."""
    n = len(events)
    cols = 3
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(5.8 * cols, 5.2 * rows))
    for k, evt in enumerate(events):
        ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
        feats = evt["features_full"]
        if view_kind == "pred":
            labels = evt["pred_labels"]; label_name = "predicted"
        else:
            labels = evt["mc_index"]; label_name = "true MC"
        local, n_uniq = _remap_to_local(labels)

        is_vtx = feats[:, 3] < 0.5
        is_dc = ~is_vtx
        circles = _build_drift_circles(feats[is_dc], n_segments=n_segments)
        if len(circles) > 0:
            colors = np.array([_color_for(l) for l in local[is_dc]])
            ax.add_collection3d(Line3DCollection(circles, colors=colors,
                                                 linewidths=0.45))
        if is_vtx.any():
            vc = np.array([_color_for(l) for l in local[is_vtx]])
            ax.scatter(feats[is_vtx, 0], feats[is_vtx, 1], feats[is_vtx, 2],
                       c=vc, s=12, edgecolors="black", linewidths=0.2,
                       depthshade=False)

        all_pts = [feats[is_vtx, :3]]
        if len(circles) > 0:
            all_pts.append(circles.reshape(-1, 3))
        pts = np.concatenate([p for p in all_pts if len(p)], axis=0)
        mins = pts.min(axis=0); maxs = pts.max(axis=0)
        pad = 0.02 * (maxs - mins).max()
        ax.set_xlim(mins[0] - pad, maxs[0] + pad)
        ax.set_ylim(mins[1] - pad, maxs[1] + pad)
        ax.set_zlim(mins[2] - pad, maxs[2] + pad)
        ax.set_title(
            f"Event {evt['event_idx']}: {is_vtx.sum()} VTX + {is_dc.sum()} DC, "
            f"{n_uniq} {label_name} tracks", fontsize=9
        )
        ax.view_init(elev=18, azim=35)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    fig.suptitle(f"v35 3D event grid (drift circles) — {label_name} clusters",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


@torch.no_grad()
def collect_events(model, dataset, device, n_events, embed_dim,
                   algorithm, tbeta, td, eps, min_samples,
                   hdb_mcs, hdb_ms, hdb_eps,
                   min_signal_hits=200):
    """Same selection logic as plot_events_3d_v35.collect_events, but also keep
    the FULL feature row (cols 0..9) so we can draw circles, and keep the
    signal-mask filtering of mc/pred labels."""
    model.eval()
    out = []
    idx = 0
    tries = 0
    while len(out) < n_events and idx < len(dataset) and tries < 600:
        tries += 1
        event = dataset[idx]
        idx += 1
        if event is None:
            continue

        features = event["features"].to(device)
        mc_index = event["mc_index"].numpy()
        is_secondary = event["is_secondary"].numpy().astype(bool)
        seq_lens = [event["n_hits"]]
        feats_np = features.cpu().numpy()

        output = model(features, seq_lens)
        coords = output[:, :embed_dim].cpu().numpy()
        beta = torch.sigmoid(output[:, embed_dim]).cpu().numpy()

        sig_mask = (~is_secondary) & (mc_index != 0)
        if sig_mask.sum() < min_signal_hits:
            continue

        coords_s = coords[sig_mask]
        beta_s = beta[sig_mask]
        mc_s = mc_index[sig_mask]
        feats_s = feats_np[sig_mask]

        if algorithm == "greedy":
            pred = get_clustering_greedy(beta_s, coords_s, tbeta=tbeta, td=td)
            algo_label = f"greedy tbeta={tbeta}, td={td}"
        elif algorithm == "dbscan":
            pred = get_clustering_dbscan(coords_s, eps=eps, min_samples=min_samples)
            algo_label = f"DBSCAN eps={eps}, min_samples={min_samples}"
        elif algorithm == "hdbscan":
            pred = get_clustering_hdbscan_full(coords_s, min_cluster_size=hdb_mcs,
                                               min_samples=hdb_ms,
                                               cluster_selection_epsilon=hdb_eps)
            algo_label = f"HDBSCAN mcs={hdb_mcs}, ms={hdb_ms}, eps={hdb_eps}"
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        out.append({
            "event_idx": idx - 1,
            "features_full": feats_s,    # (N_sig, 10)
            "mc_index": mc_s,
            "pred_labels": pred,
            "n_signal": int(sig_mask.sum()),
            "algo_label": algo_label,
        })
        n_vtx = int((feats_s[:, 3] < 0.5).sum())
        n_dc = int(len(feats_s) - n_vtx)
        print(f"  collected event {idx-1}: {n_vtx} VTX + {n_dc} DC signal hits, "
              f"{len(np.unique(mc_s))} true tracks, "
              f"{len(np.unique(pred[pred>=0]))} predicted clusters", flush=True)

    return out


def main():
    parser = argparse.ArgumentParser(description="v35 3D event displays with drift circles")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eval_seeds", type=str, default="181-181")
    parser.add_argument("--max_hits", type=int, default=3000)
    parser.add_argument("--embed_dim", type=int, default=5)
    parser.add_argument("--num_blocks", type=int, default=10)
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
    parser.add_argument("--n_segments", type=int, default=22,
                        help="Polyline segments per drift circle in the 3D view")
    parser.add_argument("--zoom_window", type=float, default=80.0,
                        help="Half-side of zoom box (a.u.) around the densest track")
    parser.add_argument("--output_dir", type=str,
                        default="eval_results/v35_events_3d_circles")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    model = CGATrParquetModel(num_blocks=args.num_blocks, embed_dim=args.embed_dim)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model = model.to(device)
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    start, end = parse_seed_range(args.eval_seeds)
    dataset = IDEAParquetDataset(args.data_dir, seed_range=(start, end),
                                 max_hits_per_event=args.max_hits)
    print(f"Dataset: {len(dataset)} events")

    print(f"\n=== Collecting {args.n_events} events with algorithm={args.algorithm} ===")
    events = collect_events(
        model, dataset, device,
        n_events=args.n_events, embed_dim=args.embed_dim,
        algorithm=args.algorithm, tbeta=args.tbeta, td=args.td,
        eps=args.eps, min_samples=args.min_samples,
        hdb_mcs=args.hdb_mcs, hdb_ms=args.hdb_ms, hdb_eps=args.hdb_eps,
    )
    if not events:
        print("No events collected (signal threshold too high?)")
        return

    print("\n=== Rendering circle plots ===")
    for e in events:
        base = f"event{e['event_idx']:03d}"
        render_3d_circles(
            features=e["features_full"], mc_index=e["mc_index"],
            pred_labels=e["pred_labels"],
            out_path=os.path.join(args.output_dir, f"{base}_3d_circles.png"),
            title_suffix=f"{e['algo_label']} | {e['n_signal']} signal hits",
            n_segments=args.n_segments,
        )
        render_xy_circles(
            features=e["features_full"], mc_index=e["mc_index"],
            pred_labels=e["pred_labels"],
            out_path=os.path.join(args.output_dir, f"{base}_xy_circles.png"),
            title_suffix=f"event {e['event_idx']} | {e['algo_label']}",
        )
        render_3d_circles_zoom(
            features=e["features_full"], mc_index=e["mc_index"],
            pred_labels=e["pred_labels"],
            out_path=os.path.join(args.output_dir, f"{base}_3d_circles_zoom.png"),
            title_suffix=f"{e['algo_label']} | {e['n_signal']} signal hits",
            n_segments=40,
            window=args.zoom_window,
        )

    render_grid_circles(events, os.path.join(args.output_dir, "grid_circles_predicted.png"),
                        view_kind="pred", n_segments=max(12, args.n_segments - 6))
    render_grid_circles(events, os.path.join(args.output_dir, "grid_circles_true.png"),
                        view_kind="true", n_segments=max(12, args.n_segments - 6))

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
