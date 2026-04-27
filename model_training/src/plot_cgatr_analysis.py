"""Generate figures from `eval_cgatr_analysis.py` (or merged) outputs.

Produces:
- beta_histograms.png      : signal vs noise beta distribution (linear + log scale)
- embedding_hist_per_dim.png: per-dim histograms of embedding values
- embedding_pairs.png      : scatter of embeddings for a few dim pairs, colored by track id
- tsne_events.png          : t-SNE of signal hits per event (colored by track id)
- umap_events.png          : UMAP of signal hits per event
- sweep_pareto.png         : purity vs efficiency scatter colored by match rate
- greedy_scan_per_td.png   : best match rate per `td` for the beta-greedy sweep

Usage:
    cd model_training
    PYTHONPATH=. python src/plot_cgatr_analysis.py \\
        --analysis_dir eval_results/cgatr_analysis_merged \\
        --output_dir   eval_results/cgatr_analysis_merged/plots
"""

import os
import sys
import json
import pickle
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_beta_histograms(samples, out_path):
    sig = samples["beta_signal"]
    noise = samples["beta_noise"]
    bins = np.linspace(0, 1, 41)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, yscale in zip(axes, ["linear", "log"]):
        ax.hist(sig, bins=bins, alpha=0.6, label=f"Signal (n={len(sig):,})",
                color="tab:blue", density=True)
        if len(noise):
            ax.hist(noise, bins=bins, alpha=0.6, label=f"Noise/secondary (n={len(noise):,})",
                    color="tab:red", density=True)
        ax.set_xlabel("beta (sigmoid of head output)")
        ax.set_ylabel("Density")
        ax.set_yscale(yscale)
        ax.set_title(f"Beta distribution ({yscale} scale)")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("CGATr beta distribution: signal vs noise", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_embedding_per_dim(samples, out_path):
    coords = samples["coords"]
    d = coords.shape[1]
    fig, axes = plt.subplots(2, (d + 1) // 2, figsize=(3.5 * ((d + 1) // 2), 6))
    axes = np.array(axes).flatten()
    for i in range(d):
        ax = axes[i]
        ax.hist(coords[:, i], bins=80, color="tab:blue", alpha=0.8)
        ax.set_title(f"Dim {i} | std={coords[:, i].std():.3f}")
        ax.set_xlabel("value")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    for i in range(d, len(axes)):
        axes[i].axis("off")
    fig.suptitle(f"CGATr embedding distribution per dimension ({len(coords):,} hits)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_embedding_pairs(samples, out_path, max_pairs=6):
    coords = samples["coords"]
    mc = samples["mc_index"]
    d = coords.shape[1]
    n = min(50000, len(coords))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(coords), size=n, replace=False)
    coords = coords[idx]
    mc = mc[idx]
    colors = mc % 20

    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)][:max_pairs]
    cols = 3
    rows = (len(pairs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows))
    axes = np.array(axes).flatten()
    for k, (i, j) in enumerate(pairs):
        ax = axes[k]
        sc = ax.scatter(coords[:, i], coords[:, j], c=colors, cmap="tab20",
                         s=1.2, alpha=0.5)
        ax.set_xlabel(f"dim {i}")
        ax.set_ylabel(f"dim {j}")
        ax.grid(alpha=0.3)
    for k in range(len(pairs), len(axes)):
        axes[k].axis("off")
    fig.suptitle(f"CGATr embedding 2D projections (colored by track id mod 20)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_tsne_events(events, out_path, n_events=4):
    from sklearn.manifold import TSNE

    events = events[:n_events]
    cols = 2
    rows = (len(events) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.array(axes).flatten()
    for k, evt in enumerate(events):
        coords = evt["coords"]
        mc = evt["mc_index"]
        if len(coords) < 20:
            axes[k].axis("off")
            continue
        if len(coords) > 3000:
            rng = np.random.default_rng(k)
            sel = rng.choice(len(coords), size=3000, replace=False)
            coords = coords[sel]
            mc = mc[sel]
        perp = min(40, max(5, len(coords) // 20))
        try:
            emb = TSNE(n_components=2, perplexity=perp, init="pca",
                        random_state=k, n_jobs=-1).fit_transform(coords)
        except Exception as e:
            print(f"  t-SNE failed for event {k}: {e}")
            axes[k].axis("off")
            continue
        n_unique = len(np.unique(mc))
        sc = axes[k].scatter(emb[:, 0], emb[:, 1], c=mc % 20, cmap="tab20",
                              s=2.5, alpha=0.7)
        axes[k].set_title(f"Event {k}: {len(coords)} hits, {n_unique} unique tracks")
        axes[k].grid(alpha=0.3)
    for k in range(len(events), len(axes)):
        axes[k].axis("off")
    fig.suptitle("t-SNE of signal embeddings per event (color = track id mod 20)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_umap_events(events, out_path, n_events=4):
    try:
        import umap
    except ImportError:
        print("  umap not installed, skipping UMAP")
        return

    events = events[:n_events]
    cols = 2
    rows = (len(events) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.array(axes).flatten()
    for k, evt in enumerate(events):
        coords = evt["coords"]
        mc = evt["mc_index"]
        if len(coords) < 30:
            axes[k].axis("off")
            continue
        if len(coords) > 3000:
            rng = np.random.default_rng(k)
            sel = rng.choice(len(coords), size=3000, replace=False)
            coords = coords[sel]
            mc = mc[sel]
        try:
            reducer = umap.UMAP(n_components=2, n_neighbors=15,
                                 min_dist=0.05, random_state=k)
            emb = reducer.fit_transform(coords)
        except Exception as e:
            print(f"  UMAP failed for event {k}: {e}")
            axes[k].axis("off")
            continue
        n_unique = len(np.unique(mc))
        axes[k].scatter(emb[:, 0], emb[:, 1], c=mc % 20, cmap="tab20",
                         s=2.5, alpha=0.7)
        axes[k].set_title(f"Event {k}: {len(coords)} hits, {n_unique} unique tracks")
        axes[k].grid(alpha=0.3)
    for k in range(len(events), len(axes)):
        axes[k].axis("off")
    fig.suptitle("UMAP of signal embeddings per event (color = track id mod 20)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_sweep_pareto(results, out_path):
    algos = {}
    for r in results:
        algos.setdefault(r["algorithm"], []).append(r)

    fig, ax = plt.subplots(figsize=(9, 7))
    markers = {"greedy": "o", "DBSCAN": "s", "HDBSCAN": "^"}
    for algo, rs in algos.items():
        pur = np.array([r["purity"] for r in rs])
        eff = np.array([r["efficiency"] for r in rs])
        match = np.array([r["match_rate"] for r in rs])
        sc = ax.scatter(eff, pur, c=match, cmap="viridis", marker=markers.get(algo, "o"),
                         s=60, edgecolor="black", linewidth=0.3, alpha=0.8,
                         label=algo, vmin=0.0, vmax=0.8)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Match rate (>75% purity)")
    ax.set_xlabel("Efficiency (mean over true tracks)")
    ax.set_ylabel("Purity (mean over predicted clusters)")
    ax.set_title("CGATr sweep: purity vs efficiency (color = match rate)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_beta_scan_per_td(results, out_path):
    greedy = [r for r in results if r["algorithm"] == "greedy"]
    tds = sorted({r["td"] for r in greedy})
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.viridis
    for i, td in enumerate(tds):
        sub = sorted([r for r in greedy if abs(r["td"] - td) < 1e-6], key=lambda r: r["tbeta"])
        tbetas = [r["tbeta"] for r in sub]
        matches = [r["match_rate"] for r in sub]
        ax.plot(tbetas, matches, marker="o",
                 color=cmap(i / max(len(tds) - 1, 1)),
                 label=f"td={td:.2f}")
    ax.set_xlabel("tbeta (beta threshold for condensation seed)")
    ax.set_ylabel("Match rate")
    ax.set_title("CGATr beta-greedy: match rate vs tbeta, per td")
    ax.legend(loc="best", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    out = args.output_dir or os.path.join(args.analysis_dir, "plots")
    os.makedirs(out, exist_ok=True)

    with open(os.path.join(args.analysis_dir, "samples.pkl"), "rb") as f:
        samples = pickle.load(f)
    with open(os.path.join(args.analysis_dir, "tsne_events.pkl"), "rb") as f:
        events = pickle.load(f)
    with open(os.path.join(args.analysis_dir, "analysis_results.json"), "r") as f:
        results_json = json.load(f)

    print(f"Loaded {len(samples['coords'])} signal samples, {len(samples['beta_noise'])} noise samples")
    print(f"Embed dim: {samples['embed_dim']}")

    plot_beta_histograms(samples, os.path.join(out, "beta_histograms.png"))
    plot_embedding_per_dim(samples, os.path.join(out, "embedding_hist_per_dim.png"))
    plot_embedding_pairs(samples, os.path.join(out, "embedding_pairs.png"))
    plot_sweep_pareto(results_json["results"], os.path.join(out, "sweep_pareto.png"))
    plot_beta_scan_per_td(results_json["results"], os.path.join(out, "greedy_scan_per_td.png"))
    plot_tsne_events(events, os.path.join(out, "tsne_events.png"))
    plot_umap_events(events, os.path.join(out, "umap_events.png"))

    print(f"\nAll plots saved to {out}/")


if __name__ == "__main__":
    main()
