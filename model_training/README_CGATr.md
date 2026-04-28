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

## 2. CGA hit encoding

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

## 3. Loss: OC + within-cluster variance regularizer

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

## 4. New code map

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
│   ├── eval_cgatr.py                       ← NEW: sweep eval (greedy/DBSCAN/HDBSCAN)
│   ├── eval_cgatr_analysis.py              ← NEW: fine-grained sweep + embedding stats,
│   │                                              composite ranking, raw caches for plots
│   ├── eval_cgatr_tdprobe.py               ← NEW: low-`td` greedy probe
│   ├── eval_cgatr_merge.py                 ← NEW: two-pass greedy-then-merge post-processor
│   ├── eval_cgatr_pt_eta.py                ← NEW: per-pT and per-eta efficiency breakdown
│   ├── merge_cgatr_analysis.py             ← NEW: merge per-GPU eval_cgatr_analysis outputs
│   ├── plot_cgatr_analysis.py              ← NEW: figure generator for the analysis sweep
│   ├── cgatr_pca.py                        ← NEW: PCA of trained embeddings (sizes embed_dim)
│   └── analyze_cgatr_log.py                ← NEW: training-log parser + preview report
│
├── README.md                               ← unchanged
├── README_CGATr.md                         ← THIS FILE
└── (everything else unchanged: gatr_v111/, gatr_v111_onnx/, train_lightning.py, ...)

data_creation/
├── edm4hep_to_parquet.py                   ← NEW: digitized edm4hep ROOT → Parquet shards
└── (everything else unchanged)
```

---

## 5. Dataset

The CGATr training run consumed Z-pole `e+e-` events simulated for the
**IDEA o1_v03** detector concept at FCC-ee. The whole pipeline is in the
upstream `data_creation/` folder; CGATr only adds an
edm4hep → Parquet converter at the end.

### 6.1 Generation and simulation pipeline

| Stage | Tool | Notes |
|---|---|---|
| Event generator | **Pythia 8** | `e+e- → f f̄` via `WeakSingleBoson:ffbar2ffbar(s:gmZ)`; ISR + FSR on. |
| Beam | √s = **90 GeV** (Z-pole) | `Beams:idA = 11`, `Beams:idB = -11`, no momentum spread. |
| Vertex smearing | Gaussian | σ_x = 5.96 µm, σ_y = 23.8 nm, σ_z = 0.397 mm (FCC-ee nominal). |
| Decay cylinder | r ≤ 2250 mm, z ≤ 2500 mm | matches the IDEA tracking volume. |
| Detector | **IDEA o1_v03** | from `$FCCCONFIG/FullSim/IDEA/IDEA_o1_v03/`; calo disabled (`simulateCalo = False`) — we only use the tracker. |
| Geometry / simulation | **key4hep + DD4hep + Geant4** | nightly stack `/cvmfs/sw-nightlies.hsf.org/key4hep/`. |
| Digitization | `runIDEAv3o1_trackerDigitizer.py` | upstream digitizer; no pile-up / background. |
| Output | edm4hep ROOT | one file per Pythia seed. |
| Parquet conversion | `data_creation/edm4hep_to_parquet.py` (this PR) | per-seed `dc_hits_*.parquet`, `vtx_hits_*.parquet`, `mc_particles_*.parquet`. |

The Pythia card and detector pipeline scripts live under
`data_creation/condor_pipeline/IDEA/noBackground/` and
`data_creation/utils/Pythia_generation/` — unchanged from upstream.

### 6.2 Sub-detectors fed to the model

| Detector | edm4hep collection | CGA primitive |
|---|---|---|
| Vertex barrel + endcaps (VTXB / VTXD) | `TrackerHitPlane` | grade-1 null point. |
| Silicon wrapper barrel + endcaps (SiWrB / SiWrD) | `TrackerHitPlane` | grade-1 null point. |
| Drift chamber (DCH) | `SenseWire` (`DCH_DigiCollection`) | **grade-2 IPNS circle** = wire-perpendicular plane ∧ drift sphere. |

The drift chamber dominates the hit count by an order of magnitude, so the
gain from encoding it as a true geometric circle is the dominant effect.

### 6.3 Volume of data and split

We generated **599 Pythia seeds** with ≈500 events per seed (≈300 k events
total) and use a contiguous split:

| Split | Seeds | Events | Total hits (after the 3 000-cap, see 6.4) |
|---|---|---|---|
| Train | 1 – 160       | 79 996 | 154.7 M |
| Val   | 161 – 180     | 10 000 |  19.3 M |
| Eval  | 181 – 200     | 10 000 |  ~19 M  |

Numbers are taken from an actual training run and are reproduced verbatim
by `train_cgatr_parquet.py` printing `Hits/event: min=62, median=2766,
max=3000, total=154,695,134` at startup. The remaining ≈400 seeds are
held out for future studies.

### 6.4 The 3 000-hit cap (VRAM budget)

In raw form the IDEA event sizes have a long tail:

```
DC + VTX hits per event:
  mean ≈ 2 890
  p95  ≈ 6 880
  p99  ≈ 10 000
  max  ≈ 17 000
  ~50 % of events exceed 3 000 hits
```

CGATr is an O(N²) attention model on 32-component multivectors. We train
on **4× V100 32 GB** with `batch_size=4`, `num_blocks=10`,
`hidden_mv_channels=16`, `hidden_s_channels=64`; per GPU, even
FlashAttention runs out of memory above ≈3 000 hits per event. Increasing
`--max_hits` beyond that fails on our hardware during the second backward
pass.

We therefore truncate every event to **≤ 3 000 hits** in
`IDEAParquetDataset.__getitem__`:

- All vertex / silicon-wrapper hits are kept (typically a few hundred).
- Drift-chamber hits are uniformly subsampled with a per-event seed
  derived from the event index, so the truncation is **deterministic and
  reproducible** across runs.

The cap is a CLI flag: `--max_hits 3000` (default). Larger GPUs can lift
it.

In practice the median raw event has ≈ 2 770 hits which fits the cap; we
lose hits only on the upper-tail half of events, and only from the
drift-chamber multiplicity excess (where MC truth links remain consistent
because we sample without replacement).

### 6.5 Polars + LRU caching

The dataset class (`src/dataset/parquet_dataset.py`) keeps only file
metadata at init time and lazily reads each seed-shard with Polars on
first access. A module-level `lru_cache(maxsize=32)` on the per-seed
grouped DataFrames means every event in a seed is served from a single
parquet read, and a worker switching to a new seed evicts the oldest one.
This is what makes `num_workers=0` viable even with 80 k training events:
disk I/O is amortised over the ≈500 events per seed.

---

## 7. How to use

### 7.1 Environment

CGATr reuses the upstream environment plus three lightweight packages:

```bash
# in the existing GATr container
pip install torch_scatter polars hdbscan
```

`torch_scatter` is required by the OC loss (`scatter_max`, `scatter_add`,
`scatter_mean`); `polars` reads the Parquet shards; `hdbscan` is only used
in the eval sweep.

### 7.2 Data layout

`train_cgatr_parquet.py` expects per-seed Parquet shards:

```
<data_dir>/
  seed_<N>/
    dc_hits_<split>.parquet         # drift-chamber hits
    vtx_hits_<split>.parquet        # vertex / silicon-wrapper hits
    mc_particles_<split>.parquet    # MC particle properties (used by pt/eta eval)
```

with the columns used in `src/dataset/parquet_dataset.py`
(`hit_x/y/z`, `wire_x/y/z`, `drift_distance`, `wire_azimuthal_angle`,
`wire_stereo_angle`, `mc_index`, `produced_by_secondary`, `event_id`,
`hit_id`, …).

The conversion from the upstream digitized edm4hep ROOT files to this
Parquet layout is provided by `data_creation/edm4hep_to_parquet.py`:

```bash
# typically run on lxplus where podio/edm4hep are available
source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh
pip install --user pyarrow
python data_creation/edm4hep_to_parquet.py \
    --input_dir  <raw_root_dir>     \
    --output_dir <parquet_out_dir>  \
    --split      train
```

The script extracts wire geometry + drift radius for the drift chamber
(everything CGA needs to reconstruct the IPNS circle), point positions
for the vertex/silicon hits, and the MC particle properties used for
truth-matching and efficiency breakdowns.

### 7.3 Training (single GPU)

```bash
cd model_training
PYTHONPATH=. python src/train_cgatr_parquet.py \
    --data_dir <path_to_parquet_data> \
    --train_seeds 1-160 \
    --val_seeds   161-180 \
    --max_hits    3000 \
    --batch_size  4 \
    --num_epochs  40 \
    --gpus 0 \
    --output_dir  checkpoints/cgatr
```

### 7.4 Training (4× DDP via torchrun)

```bash
cd model_training
PYTHONPATH=. torchrun --nproc_per_node=4 src/train_cgatr_parquet.py \
    --data_dir <path_to_parquet_data> \
    --train_seeds 1-160 \
    --val_seeds   161-180 \
    --max_hits    3000 \
    --batch_size  4 \
    --num_epochs  40 \
    --output_dir  checkpoints/cgatr
```

This is the configuration of the training run reported with this branch:
160 train seeds (≈80 k events) / 20 val seeds (10 k events), `--max_hits
3000`, `--batch_size 4` per GPU, AdamW with cosine schedule, EMA 0.999,
40 epochs.

Important flags:

| Flag | Default | Notes |
|---|---|---|
| `--max_hits` | 3000 | Per-event hit cap (VRAM budget — see § 6.4). VTX kept, DC subsampled deterministically. |
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

### 7.5 Evaluation sweep (coarse)

```bash
cd model_training
PYTHONPATH=. python src/eval_cgatr.py \
    --data_dir <path_to_parquet_data> \
    --checkpoint checkpoints/cgatr/cgatr_best.pt \
    --embed_dim 5 \
    --eval_seeds 181-200 --max_events 5000 \
    --max_hits 3000 \
    --output_dir eval_results/cgatr
```

Eval seeds are deliberately disjoint from the train (1-160) and val
(161-180) ranges.

This sweeps β-greedy `(tbeta, td)`, DBSCAN `(eps, min_samples)` and HDBSCAN
`min_cluster_size`, and writes `sweep_results.json` plus a top-30 summary
table to stdout. `--embed_dim` must match the dimension used during
training.

### 7.6 Fine analysis + figures (multi-GPU friendly)

A second pass that scans much finer grids, dumps raw embedding samples and
per-event caches, and produces figures (β-distributions, per-dim embedding
histograms, t-SNE / UMAP per event, sweep Pareto plot, greedy scan per
`td`):

```bash
cd model_training
# split eval seeds across GPUs and run in parallel
for G in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$G PYTHONPATH=. python src/eval_cgatr_analysis.py \
      --data_dir   <path_to_parquet_data> \
      --checkpoint checkpoints/cgatr/cgatr_best.pt \
      --embed_dim  5 \
      --eval_seeds $((181 + 2*G))-$((181 + 2*(G+1))) \
      --output_dir eval_results/cgatr_analysis_gpu$G &
done; wait

PYTHONPATH=. python src/merge_cgatr_analysis.py \
    --dirs eval_results/cgatr_analysis_gpu0 eval_results/cgatr_analysis_gpu1 \
           eval_results/cgatr_analysis_gpu2 eval_results/cgatr_analysis_gpu3 \
    --out  eval_results/cgatr_analysis_merged

PYTHONPATH=. python src/plot_cgatr_analysis.py \
    --analysis_dir eval_results/cgatr_analysis_merged \
    --output_dir   eval_results/cgatr_analysis_merged/plots
```

Optional follow-ups on the merged outputs:

- `src/cgatr_pca.py --samples eval_results/cgatr_analysis_merged/samples.pkl` —
  reports cumulative explained variance per embedding dimension. We used
  this to confirm that the trained 6-D embedding sat in a 5-D subspace,
  motivating `embed_dim=5`.
- `src/eval_cgatr_tdprobe.py` — fine `td` probe (e.g. 0.02–0.05) when the
  coarse sweep saturates at the lower edge.
- `src/eval_cgatr_merge.py` — two-pass greedy-then-merge clustering with
  an optional MC-purity gate (oracle upper bound for what a smarter
  post-processor could buy).
- `src/eval_cgatr_pt_eta.py` — per-pT and per-eta efficiency breakdown by
  joining clustering output with the MC-particle Parquet.

### 7.7 Live training-log analysis

While a run is in progress (or after) you can produce a quick plot/report
pack from its stdout log without disturbing the run:

```bash
python model_training/src/analyze_cgatr_log.py \
    --log run.log \
    --total_epochs 40 \
    --out_dir model_training/eval_results/cgatr_preview
# optional head-to-head against a previous run:
python model_training/src/analyze_cgatr_log.py \
    --log run.log --baseline_log baseline-run.log \
    --baseline_label baseline --current_label cgatr \
    --out_dir model_training/eval_results/cgatr_preview
```

---

## 8. Backward compatibility

This branch only **adds** files. No upstream training, eval, ONNX
conversion, dataset-creation or notebook code is modified. To run the
original GGTF model, follow the existing instructions in `README.md` —
nothing has changed.
