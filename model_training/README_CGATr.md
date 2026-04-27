# CGATr — Conformal Geometric Algebra Transformer

CGATr is a **Conformal Geometric Algebra (Cl(4,1))** variant of the GGTF
track-finder in this repository. It keeps the high-level recipe of the
upstream model (an equivariant transformer that maps detector hits to a
clustering embedding + a per-hit β score, trained with the Object Condensation
loss) but swaps the underlying algebra and adds a small loss term that we
found necessary for tight track clusters.

This document explains:

1. What CGATr changes vs. upstream and **why**.
2. How the CGA hit encoding works.
3. How the loss differs.
4. Where the new code lives.
5. How to train and evaluate the new model.

> Empirical results (training curves, sweeps, plots) will be added separately
> once the run finishes. This document only covers the code that has been
> ported to this branch.

---

## 1. What changed at a glance

| Aspect | Upstream (GGTF in `gatr_v111` / `train_lightning.py`) | CGATr (this branch) |
|---|---|---|
| Geometric algebra | **PGA Cl(3,0,1)**, 16-component multivectors | **CGA Cl(4,1)**, 32-component multivectors |
| Equivariant package | `src/gatr_v111/` (unchanged in this PR) | new `src/cgatr/` (parallel package) |
| Cayley tables on disk | `model_training/gatr_utils/*.pt` | new `model_training/cga_utils/*.pt` |
| DC hit encoding | "left + right" point pair on the wire | a single **IPNS drift circle** = drift sphere ∧ wire-perpendicular plane (grade-2 bivector) |
| VTX hit encoding | grade-1 null point | grade-1 null point (same idea, in 32-dim CGA) |
| MV channels | as in upstream | single MV channel — both VTX points and DC circles live in the same channel |
| OC loss | attractive + repulsive + β-sig + β-noise | same + **within-cluster variance regularizer** `L_var` (default weight 0.3, 2-epoch warmup) |
| Default clustering dim | upstream default | **5** (chosen by PCA on trained embeddings) |
| Training entry point | `src/train_lightning.py` (Weaver / Lightning / ROOT) | new `src/train_cgatr_parquet.py` (PyTorch DDP / Parquet) |
| Eval entry point | `notebooks/1_evaluation_IDEA_tracking.ipynb` | new `src/eval_cgatr.py` (sweeps β-greedy, DBSCAN, HDBSCAN) |

The upstream GATr code (`src/gatr_v111`, `src/gatr_v111_onnx`,
`src/train_lightning.py`, `src/models/Gatr*.py`, etc.) is **not modified** in
this branch. The CGA variant is a parallel addition you can review side by
side with the original.

---

## 2. Why CGA?

A drift-chamber hit is fundamentally not a point. The detector measures the
identity of the wire that fired and the drift radius around it; the actual
particle hit lies somewhere on the **circle** described by that wire and
drift radius. Upstream encodes this by either taking the wire midpoint or
splitting the hit into two "left/right" point candidates on the wire.

In **Conformal Geometric Algebra Cl(4,1)** every classical geometric
primitive — points, spheres, planes, circles, lines, motors — lives in the
*same* graded vector space (32-dim multivectors split into grades 0–5). In
particular:

- A 3-D point P(x,y,z) is a null grade-1 vector (`embed_point`).
- A sphere of radius r centred at c is a non-null grade-1 vector `S = P(c) − (r²/2)·e∞` (`embed_sphere`).
- A circle is the intersection of a sphere with a plane and is the outer
  product `C = S ∧ π`, a grade-2 bivector (`embed_circle_ipns`).

This means a drift-chamber hit can be encoded **losslessly** as the IPNS
circle defined by its sense wire and drift radius — no left/right
disambiguation, no information loss. Equivariance under rotations,
translations, scalings and Lorentz transformations is automatic for any
network built from the geometric product, grade projection and norms.

The CGA backbone (`src/cgatr/`) is structured exactly like upstream's PGA
package (`src/gatr_v111/`): same module names, same primitives, same block
recipe. Only the underlying algebra and the input encoding differ.

---

## 3. CGA hit encoding

```python
# src/train_cgatr_parquet.py — `CGATrParquetModel.forward`
mv = torch.zeros(N, 32)              # one CGA multivector per hit

# Vertex-detector hits → null point (grade 1)
mv[is_vtx] = embed_point(vtx_xyz_normed)

# Drift-chamber hits → IPNS circle (grade 2)
mv[is_dc]  = embed_circle_ipns(wire_xyz_normed, wire_dir, drift_radius_normed,
                               op_table=basis_outer)

# Per-hit type as a scalar grade-0 component
mv = mv + embed_scalar(hit_type)
```

The single MV channel keeps the model small (≈0.92 M parameters with the
default `num_blocks=10`, `hidden_mv_channels=16`, `hidden_s_channels=64`).

The Cayley tables for the geometric and outer products in Cl(4,1) are
pre-computed (`model_training/cga_utils/*.pt`); they can be regenerated with
`python -m src.cgatr.generate_cga_tables` (requires the `clifford` package).

---

## 4. Loss: OC + within-cluster variance regularizer

The base Object-Condensation loss is unchanged in spirit:

```
L_OC = L_attr + L_rep + L_β,sig + L_β,noise + λ_β · L_β,suppress
```

CGATr adds one more term:

```
L_var = mean over tracks k of  mean over hits i in track k of  ||x_i − μ_k||²
```

where `μ_k` is the mean clustering embedding of the hits assigned to track
k. `L_var` is

- **unweighted by β** (every hit contributes, including the low-β tail of a
  track that the standard `q · log(d²+1)` attractive term effectively
  ignores),
- **quadratic in distance** (gradients grow at the cluster edge instead of
  saturating), and
- **anchored at the centroid**, not the alpha point (translation-invariant
  within the cluster, more stable target).

It is added with weight `var_weight` (default **0.3**), warmed up linearly
over `var_warmup_epochs` epochs (default **2**) so clusters can form before
they are tightened. With `--var_weight 0` the loss reduces exactly to the
standard OC formulation.

We added this term after observing in t-SNE / UMAP / PCA of trained
embeddings that tracks formed **elongated 1-D blobs** rather than tight
spheres, forcing the post-hoc clustering to use very small radii (`td≈0.05`)
and hurting efficiency. `L_var` directly attacks that failure mode.

The implementation is in `object_condensation_loss(...)` in
`src/train_cgatr_parquet.py`, with `var_weight` and `return_components`
flags.

---

## 5. New code map

```
model_training/
├── cga_utils/                              ← NEW: Cl(4,1) Cayley tables
│   ├── cga_geometric_product.pt
│   ├── cga_outer_product.pt
│   └── cga_metadata.pt
│
├── src/
│   ├── cgatr/                              ← NEW: equivariant Cl(4,1) backbone
│   │   ├── interface/                      ←   embed/extract for points,
│   │   │   ├── point.py                    ←     spheres, planes, circles,
│   │   │   ├── sphere.py                   ←     scalars (mirrors gatr_v111
│   │   │   ├── plane.py                    ←     interface package)
│   │   │   ├── circle.py
│   │   │   └── scalar.py
│   │   ├── primitives/                     ←   GP, outer, dual, attention,
│   │   │   ├── linear.py                   ←     linear, normalization,
│   │   │   ├── bilinear.py                 ←     invariants — all Cl(4,1)
│   │   │   ├── attention.py
│   │   │   ├── invariants.py
│   │   │   ├── nonlinearities.py
│   │   │   ├── normalization.py
│   │   │   ├── dropout.py
│   │   │   └── dual.py
│   │   ├── layers/                         ←   EquiLinear, LayerNorm,
│   │   │   ├── linear.py                   ←     attention, MLP, blocks
│   │   │   ├── layer_norm.py
│   │   │   ├── attention/
│   │   │   ├── mlp/
│   │   │   ├── gatr_block.py
│   │   │   └── dropout.py
│   │   ├── nets/cgatr.py                   ←   top-level CGATr network
│   │   ├── tests/test_cga.py               ←   sanity tests for the algebra
│   │   └── generate_cga_tables.py          ←   regenerates cga_utils/*.pt
│   │
│   ├── dataset/parquet_dataset.py          ← NEW: lightweight Polars-based
│   │                                              IDEA Parquet IterableDataset
│   ├── train_cgatr_parquet.py              ← NEW: DDP training entry point
│   └── eval_cgatr.py                       ← NEW: sweep eval (greedy/DBSCAN/HDBSCAN)
│
├── README.md                               ← unchanged
├── README_CGATr.md                         ← THIS FILE
└── (everything else unchanged: gatr_v111/, gatr_v111_onnx/, train_lightning.py, ...)
```

---

## 6. How to use

### 6.1 Environment

CGATr reuses the upstream environment plus three lightweight packages:

```bash
# in the existing GATr container
pip install torch_scatter polars hdbscan
```

`torch_scatter` is required by the OC loss (`scatter_max`, `scatter_add`,
`scatter_mean`); `polars` reads the Parquet shards; `hdbscan` is only used
in the eval sweep.

### 6.2 Data layout

`train_cgatr_parquet.py` expects per-seed Parquet shards:

```
<data_dir>/
  vtx/event_seed_<N>.parquet      # vertex-detector hits
  dc/event_seed_<N>.parquet       # drift-chamber hits
```

with the columns used in `parquet_dataset.py`
(`hit_x/y/z`, `wire_x/y/z`, `drift_distance`, `wire_azimuthal_angle`,
`wire_stereo_angle`, `mc_index`, `produced_by_secondary`, `event_id`,
`hit_id`, …). The conversion from the upstream ROOT/edm4hep format to this
Parquet layout is done outside of this PR.

### 6.3 Training (single GPU)

```bash
cd model_training
PYTHONPATH=. python src/train_cgatr_parquet.py \
    --data_dir <path_to_parquet_data> \
    --train_seeds 1-160 \
    --val_seeds   101-110 \
    --batch_size 4 \
    --num_epochs 40 \
    --gpus 0 \
    --output_dir checkpoints/cgatr
```

### 6.4 Training (4× DDP via torchrun)

```bash
cd model_training
PYTHONPATH=. torchrun --nproc_per_node=4 src/train_cgatr_parquet.py \
    --data_dir <path_to_parquet_data> \
    --train_seeds 1-160 \
    --val_seeds   101-110 \
    --batch_size 4 \
    --num_epochs 40 \
    --output_dir checkpoints/cgatr
```

Important flags:

| Flag | Default | Notes |
|---|---|---|
| `--embed_dim` | 5 | Clustering embedding dimension. |
| `--var_weight` | 0.3 | Within-cluster variance regularizer; set to 0 for plain OC. |
| `--var_warmup_epochs` | 2 | Linear warmup of `var_weight` from 0. |
| `--qmin` | 0.1 | OC charge floor. |
| `--beta_suppress_weight` | 0.1 | β-suppression on non-alpha signal hits. |
| `--ema_decay` | 0.999 | Set to 0 to disable EMA. |
| `--num_blocks` | 10 | Equivariant transformer depth. |
| `--hidden_mv_channels` | 16 | MV hidden channels per block. |
| `--hidden_s_channels` | 64 | Scalar hidden channels per block. |
| `--cosine_norm` | off | L2-normalize clustering coords before loss. |

### 6.5 Evaluation sweep

```bash
cd model_training
PYTHONPATH=. python src/eval_cgatr.py \
    --data_dir <path_to_parquet_data> \
    --checkpoint checkpoints/cgatr/cgatr_best.pt \
    --embed_dim 5 \
    --eval_seeds 181-190 --max_events 5000 \
    --output_dir eval_results/cgatr
```

This sweeps β-greedy `(tbeta, td)`, DBSCAN `(eps, min_samples)` and HDBSCAN
`min_cluster_size`, and writes `sweep_results.json` plus a top-30 summary
table to stdout. `--embed_dim` must match the dimension used during
training.

---

## 7. Backward compatibility

This branch only **adds** files. No upstream training, eval, ONNX
conversion, dataset-creation or notebook code is modified. To run the
original GGTF model, follow the existing instructions in `README.md` —
nothing has changed.
