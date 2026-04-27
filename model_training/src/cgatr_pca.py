"""PCA of CGATr clustering embeddings.

Loads `samples.pkl` produced by `merge_cgatr_analysis.py`, fits PCA on the
N-dimensional signal-hit embeddings, and prints cumulative explained
variance per component. Useful for deciding whether the trained embedding
is "really" using all of its dimensions or whether `embed_dim` can be
reduced (a 99 %-cumulative-variance test is the rule of thumb we used to
pick the default `embed_dim=5`).

Usage:
    PYTHONPATH=. python src/cgatr_pca.py \\
        --samples eval_results/cgatr_analysis_merged/samples.pkl \\
        --output  eval_results/cgatr_analysis_merged/pca_results.json
"""

import os
import sys
import pickle
import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="PCA of CGATr clustering embeddings")
    parser.add_argument("--samples", type=str,
        default="eval_results/cgatr_analysis_merged/samples.pkl",
        help="Path to samples.pkl produced by merge_cgatr_analysis.py.")
    parser.add_argument("--output", type=str,
        default="eval_results/cgatr_analysis_merged/pca_results.json")
    args = parser.parse_args()

    print(f"Loading {args.samples}", flush=True)
    with open(args.samples, "rb") as f:
        samples = pickle.load(f)

    # samples expected to be a dict with 'coords' array or a list of arrays
    if isinstance(samples, dict) and "coords" in samples:
        X = samples["coords"]
    elif isinstance(samples, list):
        X = np.concatenate([s["coords"] for s in samples], axis=0)
    else:
        raise ValueError(f"Unknown samples structure: {type(samples)} keys="
                          f"{list(samples.keys()) if isinstance(samples, dict) else None}")

    X = np.asarray(X)
    print(f"Embedding matrix: {X.shape} ({X.dtype})", flush=True)
    print(f"Per-dim mean  : {np.round(X.mean(axis=0), 4).tolist()}", flush=True)
    print(f"Per-dim std   : {np.round(X.std(axis=0), 4).tolist()}", flush=True)

    from sklearn.decomposition import PCA
    d = X.shape[1]
    pca = PCA(n_components=d)
    pca.fit(X)

    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)

    print("\n=== PCA explained variance ===", flush=True)
    for i, (v, c) in enumerate(zip(evr, cum), 1):
        print(f"  Component {i}: evr={v:.4f}  cum={c:.4f}", flush=True)

    print("\n=== Decision signal ===", flush=True)
    for k in range(1, d + 1):
        mark = ""
        if cum[k - 1] >= 0.99:
            mark = "  <- >=99%"
        elif cum[k - 1] >= 0.95:
            mark = "  <- >=95%"
        print(f"  Top {k} components cover {cum[k - 1] * 100:.2f}%{mark}", flush=True)

    if d >= 5:
        five_cov = float(cum[4])
        if five_cov >= 0.99:
            verdict = "Top-5 components cover >=99%; reducing to embed_dim=5 is safe."
        elif five_cov >= 0.95:
            verdict = ("Top-5 components cover 95-99%. Borderline; reducing to "
                       "embed_dim=5 is plausible but warrants a small probe run.")
        else:
            verdict = "Top-5 components cover <95%; keep the current embed_dim."
        print(f"\nVerdict: {verdict}", flush=True)

    # Cosine of principal axes against standard basis to see which dims are dominant
    comps = pca.components_  # (n_comp, d)
    print("\n=== Principal components (|loadings| per dim, %) ===", flush=True)
    header = "    " + "   ".join(f"d{j}" for j in range(d))
    print(header, flush=True)
    for i, c in enumerate(comps, 1):
        frac = np.abs(c) * 100
        print(f"  PC{i}: " + " ".join(f"{v:5.1f}" for v in frac), flush=True)

    import json
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "n_samples": int(X.shape[0]),
            "d": int(d),
            "explained_variance_ratio": evr.tolist(),
            "cumulative_explained_variance": cum.tolist(),
            "components_abs": np.abs(comps).tolist(),
        }, f, indent=2)
    print(f"\nSaved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
