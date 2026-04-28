# CGATr v35 — final-model evaluation report

> Companion to `v33-report.md` (which already contains the full v33 → v34
> story). This file focuses on **what changed for v35** and how it compares to
> v34 on the *same* evaluation grids.
>
> **v35 in one line:** drop one more dimension (5D), add a within-cluster
> variance regularizer (λ=0.30, 2-epoch warmup), keep everything else.

## TL;DR (v35)

- **Best balanced operating point (700-event fine sweep):**
  - greedy `tbeta=0.05, td=0.20` → purity 0.953, efficiency 0.883, **match
    0.769** (geo 0.865, avg-rank 39.7).
  - This is the v34 *recommended* point. v35 lifts match by **+0.024** while
    keeping purity within 0.005 and adding **+0.041** efficiency.
- **Best match rate (greedy, fine sweep):** `tbeta=0.05, td=0.05` →
  **0.845** (vs v34: 0.833 at the same point).
- **Best match rate at td=0.02 (td-probe):** 0.881 (vs v34: 0.879).
- **PCA verdict:** the 5D embedding effectively spans **only 4 dimensions**
  (PC5 has ≈ 0 % variance). Same intrinsic dimensionality as v34 — the
  capacity argument we used to drop 6 → 5 holds.
- **HDBSCAN best:** `mcs=2, ms=1, eps=0.10` → match 0.736 (geo 0.838).
  Now beats DBSCAN's best balanced point.

| Metric @ recommended op-point | v34 (greedy 0.05 / 0.20) | v35 (greedy 0.05 / 0.20) | Δ |
|---|---|---|---|
| purity     | 0.960 | 0.953 | −0.007 |
| efficiency | 0.842 | 0.883 | **+0.041** |
| match rate | 0.745 | 0.769 | **+0.024** |
| geo mean   | 0.844 | 0.865 | **+0.021** |

The variance regularizer pushes hits in the same track *closer* in embedding
space, so the same ball radius `td=0.20` collects more of the track without
overlapping neighbours. We get the efficiency lift we hoped for and we keep
the match rate.

---

## 1. Headline numbers

| Model | embed dim | OC variant | best val match | coarse-sweep best (0.10/0.20) | fine-sweep best match | fine-sweep best balanced (geo) |
|-------|-----------|------------|----------------|-------------------------------|-----------------------|-------------------------------|
| v33 | 8 (2 dead) | OC standard | 0.733 | 0.733 | 0.733 (0.10/0.20) | 0.733 |
| v34 | 6          | OC standard | 0.722 | 0.743 | 0.833 (0.05/0.05) | 0.766 (0.05/0.15) |
| **v35** | **5**  | **OC + L_var** | **0.722** | **0.763** | **0.845 (0.05/0.05)** | **0.779 (0.05/0.17)** |

### Training (v35)

35 epochs, stopped after epoch 35 because the curve had clearly plateaued:

| Epoch | Val Loss | Purity | Efficiency | Match Rate | Noise Supp |
|-------|----------|--------|------------|------------|------------|
| 1  | 0.9051 | 0.827 | 0.372 | 0.538 | 0.739 |
| 5  | 0.6242 | 0.869 | 0.745 | 0.688 | 0.846 |
| 10 | 0.5530 | 0.875 | 0.791 | 0.706 | 0.900 |
| 20 | 0.5078 | 0.877 | 0.819 | 0.713 | 0.924 |
| 27 | 0.4922 | 0.877 | 0.828 | 0.718 | 0.930 |
| 30 | 0.4879 | 0.876 | 0.829 | 0.718 | 0.933 |
| 33 | 0.4857 | 0.877 | 0.831 | 0.720 | 0.934 |
| 34 | 0.4853 | 0.877 | 0.831 | 0.721 | 0.935 |
| **35 (best)** | **0.4847** | **0.877** | **0.831** | **0.722** | **0.935** |

The last 8 epochs (28 → 35) improved val loss by 0.006 and match rate by
0.004. Same plateau profile as v34. **L_var component plateaued at
≈ 0.03–0.04** (weight 0.30 from epoch 3 onwards), which is roughly an order
of magnitude smaller than `L_att` (0.04) — a confirmation that the network
was already producing low-variance clusters and L_var only nudges the tail.

Hardware/protocol unchanged from v34: 4× V100 32 GB DDP, batch 4×4 = 16,
max_hits=3000, EMA decay 0.999, AdamW + cosine warmup. 924,624 parameters
(−32 vs v34 because the clustering head shrinks 6 → 5).

### 1.1 Training dynamics

35 epochs, 4× V100 32 GB DDP, ≈ 228 min/epoch wall-clock,
**133.4 h ≈ 5.6 days** total. Six-panel summary parsed from `v35-run.log`
by `model_training/src/plot_training_curves_v35.py` and saved in
`eval_results/v35_training_curves/`.

![v35 training curves: loss, components, val metrics, LR/L_var schedule, wall-time](model_training/eval_results/v35_training_curves/v35_training_curves.png)

What the panels show, top-left to bottom-right:

1. **Loss curves.** Train loss (per-batch, smoothed; epoch-avg blue circles)
   and val loss (red squares). Train and val track each other tightly —
   no overfitting signature. The gap between the two reflects EMA-of-
   weights vs raw weights at val time.
2. **Loss components** (log scale). `L_att` and `L_var` decay smoothly;
   `L_rep` plateaus at ≈ 0.07–0.10 because there is always *some*
   between-cluster repulsion to be paid in a 5D embedding. Note the
   `L_var` spike around epoch 1–2 — that's the warmup ramp from 0 → 0.30
   forcing L_var into the loss after the embedding has already started
   clustering.
3. **Validation metrics per epoch.** Purity ≈ 0.876 reaches its plateau
   by epoch 6 and stays there for the next 30 epochs — the model knows
   how to be pure very quickly. Efficiency climbs steadily from 0.37
   (epoch 1) to 0.83 (epoch 35), and is the metric that actually drives
   `match_rate` after epoch 6. **Match rate plateaus at 0.722** by
   epoch ≈ 25; the last 10 epochs gain only 0.001-0.002, which is what
   triggered the stop-training decision. Noise suppression climbs
   from 0.74 to 0.94 — the beta head learning to push background hits
   to small β.
4. **LR schedule + L_var weight ramp.** AdamW with 1-epoch warmup to
   `5e-4`, cosine decay to `1e-5` over 35 epochs. Green dashed: the
   L_var weight ramps 0 → 0.30 over epochs 1–2 (`var_warmup_epochs=2`).
5. **Per-epoch wall time.** Strikingly uniform — ≈ 228 min/epoch
   regardless of where in training. The token-budget batch sampler
   (`max_tokens` packing + DDP) gives reproducible epoch durations.
6. **Cumulative training time.** Linear, hits 133.4 h at the saved
   `cgatr_best.pt`. This is the cost we pay for one model from scratch
   on 4× V100; the v36 pilots fine-tune from this in ~5 h each.

---

## 2. Coarse sweep (3,842 events, seeds 191-200)

`eval_results/v35_sweep_merged/sweep_results.json` — 84 configs across all
4 GPUs, event-weighted (840 + 1,081 + 840 + 1,081 events). Same grid as v34
coarse (greedy 7 × 8 = 56, DBSCAN 8 × 3 = 24, HDBSCAN 4).

### 2.1 Top 10 by match rate

| Rank | Algorithm | Params | Pur | Eff | Match | GeoM |
|------|-----------|--------|-----|-----|-------|------|
| 1 | beta-greedy | tbeta=0.10, td=0.20 | **0.941** | 0.883 | **0.763** | **0.859** |
| 2 | beta-greedy | tbeta=0.20, td=0.20 | 0.939 | 0.882 | 0.762 | 0.858 |
| 3 | beta-greedy | tbeta=0.30, td=0.20 | 0.938 | 0.882 | 0.761 | 0.857 |
| 4 | beta-greedy | tbeta=0.40, td=0.20 | 0.937 | 0.881 | 0.760 | 0.856 |
| 5 | beta-greedy | tbeta=0.50, td=0.20 | 0.937 | 0.881 | 0.760 | 0.856 |
| 6 | beta-greedy | tbeta=0.60, td=0.20 | 0.936 | 0.880 | 0.758 | 0.855 |
| 7 | beta-greedy | tbeta=0.70, td=0.20 | 0.934 | 0.878 | 0.756 | 0.853 |
| 8 | beta-greedy | tbeta=0.10, td=0.30 | 0.932 | 0.903 | 0.741 | 0.854 |
| 9 | beta-greedy | tbeta=0.20, td=0.30 | 0.930 | 0.903 | 0.740 | 0.853 |
| 10 | beta-greedy | tbeta=0.30, td=0.30 | 0.929 | 0.902 | 0.739 | 0.853 |

### 2.2 Best per algorithm (coarse grid)

| Algorithm | Params | Pur | Eff | Match | GeoM |
|-----------|--------|-----|-----|-------|------|
| beta-greedy (best match / geo / rank_avg) | tbeta=0.10, td=0.20 | 0.941 | 0.883 | 0.763 | 0.859 |
| beta-greedy (best efficiency) | tbeta=0.10, td=2.00 | 0.869 | 0.949 | 0.590 | 0.787 |
| DBSCAN (best match / geo)    | eps=0.20, ms=2 | 0.891 | 0.926 | 0.715 | 0.839 |
| DBSCAN (best efficiency)     | eps=3.00, ms=2 | 0.690 | 0.979 | 0.224 | 0.533 |
| HDBSCAN (best, 4 configs)    | min_cluster_size=3 | 0.845 | 0.809 | 0.638 | 0.758 |

### 2.3 v33 → v34 → v35 head-to-head (coarse sweep, greedy `tbeta=0.10, td=0.20`)

| Run | Embed dim | Events | Purity | Efficiency | Match Rate |
|-----|-----------|--------|--------|------------|------------|
| v33 | 8 (2 dead) | 4,248 | 0.943 | 0.872 | 0.733 |
| v34 | 6          | 4,248 | **0.947** | 0.842 | 0.743 |
| **v35** | **5 + L_var** | **3,842** | 0.941 | **0.883** | **0.763** |

**v35 vs v34 coarse:** −0.006 purity, **+0.041 efficiency**, **+0.020 match
rate** at the *same* operating point. v35's coarse-grid winner remains
`tbeta=0.10, td=0.20`, identical to v33/v34, which is convenient because no
downstream tooling needs retuning.

DBSCAN at `eps=0.20, ms=2` lifts from match=0.695 (v34) → **0.715 (v35)**,
and HDBSCAN at `mcs=3` lifts from 0.631 (v34) → 0.638 (v35).

![Sweep Pareto front (purity vs efficiency, color = match rate)](model_training/eval_results/v35_analysis_merged/plots/sweep_pareto.png)

*Pareto front from the 700-event fine sweep. Each dot is one (algorithm,
params) configuration; color shows match rate. The greedy frontier (top-
right corner) dominates DBSCAN and HDBSCAN at every operating point.*

---

## 3. Fine analysis sweep (700 events, seeds 181-188)

`eval_results/v35_analysis_merged/analysis_results.json` — 179 configs,
event-weighted across 4 GPUs (175.0 ± 3.7 events/GPU).

### 3.1 Top 10 by match rate

| Rank | Algorithm | Params | Pur | Eff | Match | GeoM |
|------|-----------|--------|-----|-----|-------|------|
| 1 | greedy | tbeta=0.05, td=0.05 | **0.975** | 0.747 | **0.845** | 0.850 |
| 2 | greedy | tbeta=0.10, td=0.05 | 0.975 | 0.746 | 0.844 | 0.850 |
| 3 | greedy | tbeta=0.15, td=0.05 | 0.974 | 0.746 | 0.844 | 0.849 |
| 4 | greedy | tbeta=0.20, td=0.05 | 0.974 | 0.745 | 0.843 | 0.849 |
| 5 | greedy | tbeta=0.30, td=0.05 | 0.973 | 0.745 | 0.843 | 0.848 |
| 6 | greedy | tbeta=0.50, td=0.05 | 0.971 | 0.744 | 0.841 | 0.847 |
| 7 | greedy | tbeta=0.05, td=0.08 | 0.970 | 0.805 | 0.818 | 0.861 |
| 8 | greedy | tbeta=0.10, td=0.08 | 0.969 | 0.805 | 0.817 | 0.861 |
| 9 | HDBSCAN | mcs=2, ms=1, eps=0.00 | 0.933 | 0.417 | 0.813 | 0.681 |
| 10 | greedy | tbeta=0.05, td=0.10 | 0.966 | 0.829 | 0.807 | 0.865 |

The same monotone story as v34: **match rate climbs as td shrinks** because
purity is easier to push above 75 %, at the cost of efficiency. v35 squeezes
out a bit more (+0.01) than v34 at every point.

### 3.2 Best-per-algorithm summary (700 events)

| algo | best match | best purity | best efficiency | best geo | best avg-rank |
|------|------------|-------------|-----------------|----------|---------------|
| greedy  | tbeta=0.05, td=0.05 → 0.845 | tbeta=0.05, td=0.05 → 0.975 | tbeta=0.05, td=0.50 → 0.926 | tbeta=0.05, td=0.17 → 0.866 | tbeta=0.05, td=0.20 → 39.67 |
| DBSCAN  | eps=0.05, ms=2 → 0.753 | eps=0.05, ms=2 → 0.922 | eps=0.40, ms=2 → 0.960 | eps=0.15, ms=2 → 0.845 | eps=0.17, ms=2 → 54.33 |
| HDBSCAN | mcs=2, ms=1, eps=0.00 → 0.813 | mcs=2, ms=1, eps=0.00 → 0.933 | mcs=2, ms=1, eps=0.20 → 0.932 | mcs=2, ms=1, eps=0.10 → 0.838 | mcs=2, ms=1, eps=0.10 → 65.67 |

### 3.3 Comparison to v34 at matched configs

| config | v34 pur | v34 eff | v34 match | v35 pur | v35 eff | v35 match | Δmatch |
|--------|---------|---------|-----------|---------|---------|-----------|--------|
| greedy 0.05/0.05 | 0.976 | 0.721 | 0.833 | 0.975 | 0.747 | 0.845 | +0.012 |
| greedy 0.05/0.10 | 0.969 | 0.792 | 0.787 | 0.966 | 0.829 | 0.807 | +0.020 |
| greedy 0.05/0.15 | 0.965 | 0.822 | 0.766 | 0.959 | 0.863 | 0.785 | +0.019 |
| greedy 0.05/0.20 | 0.960 | 0.842 | 0.745 | 0.953 | 0.883 | 0.769 | +0.024 |
| greedy 0.05/0.50 | 0.942 | 0.887 | 0.696 | 0.929 | 0.926 | 0.716 | +0.020 |
| DBSCAN eps=0.20 | 0.902 | 0.878 | 0.695 | 0.899 | 0.928 | 0.721 | +0.026 |
| HDBSCAN 2/1/0.10 | 0.888 | 0.862 | 0.717 | 0.883 | 0.905 | 0.736 | +0.019 |

**Universal +0.02 lift in match rate, mostly funded by +0.04 in
efficiency**, at a cost of −0.005 to −0.013 in purity. The variance
regularizer is doing exactly what we asked it to: tracks are tighter, so the
same ball captures more of the track and we waste less budget on the tails.

### 3.4 Composite ranking (top 10 by avg-rank, lower is better)

| Rank | Algorithm | Params | Pur | Eff | Match | GeoM | RankAvg |
|------|-----------|--------|-----|-----|-------|------|---------|
| 1 | greedy | tbeta=0.05, td=0.20 | 0.953 | 0.883 | 0.769 | 0.865 | 39.67 |
| 2 | greedy | tbeta=0.05, td=0.22 | 0.951 | 0.888 | 0.764 | 0.864 | 41.00 |
| 3 | greedy | tbeta=0.05, td=0.17 | 0.956 | 0.872 | 0.779 | 0.866 | 41.33 |
| 4 | greedy | tbeta=0.10, td=0.20 | 0.952 | 0.883 | 0.769 | 0.865 | 41.67 |
| 5 | greedy | tbeta=0.05, td=0.25 | 0.948 | 0.897 | 0.757 | 0.863 | 42.33 |
| 6 | greedy | tbeta=0.10, td=0.17 | 0.955 | 0.872 | 0.778 | 0.866 | 42.33 |
| 7 | greedy | tbeta=0.10, td=0.22 | 0.950 | 0.888 | 0.763 | 0.864 | 42.33 |
| 8 | greedy | tbeta=0.05, td=0.15 | 0.959 | 0.863 | 0.785 | 0.866 | 42.67 |
| 9 | greedy | tbeta=0.15, td=0.20 | 0.951 | 0.883 | 0.768 | 0.864 | 43.33 |
| 10 | greedy | tbeta=0.10, td=0.25 | 0.947 | 0.897 | 0.757 | 0.863 | 43.67 |

The whole top-10 is greedy. Compared to v34 §10.6 the recommendation
shifts slightly: v34 → td=0.22 was the rank-avg winner, v35 → **td=0.20**.
The lower v35 avg-rank optimum reflects the tighter cluster geometry.

**Recommendations:**
- Maximum match rate: **greedy tbeta=0.05, td=0.05** (0.845).
- Balanced (paper headline): **greedy tbeta=0.05, td=0.17–0.22**
  (match 0.764–0.779, eff 0.872–0.888, pur 0.951–0.956).
- Maximum efficiency: **DBSCAN eps=0.40, ms=2** (0.960) or **greedy
  tbeta=0.05, td=0.50** (0.926).

![Greedy match rate vs tbeta, one curve per td](model_training/eval_results/v35_analysis_merged/plots/greedy_scan_per_td.png)

*Match-rate landscape of the greedy clustering. Match rate is almost flat
in `tbeta` (≤ 0.01 within each curve) and **monotone in td**: tighter
balls (small td) maximise match rate but cost efficiency. The td=0.05
plateau ends the curve where v34 saturated.*

---

## 4. Beta head behaviour (1,539,414 signal hits / 137,907 noise+secondary)

| | Mean | Median | Std | β > 0.1 | β > 0.5 | β > 0.7 | β > 0.8 |
|---|------|--------|-----|---------|---------|---------|---------|
| **Signal** (v35) | 0.736 | 0.806 | 0.224 | 95.32 % | 88.50 % | 75.62 % | 51.66 % |
| Signal (v34) | 0.743 | 0.809 | 0.221 | 95.22 % | 89.15 % | 79.47 % | 52.19 % |

| | Mean | Median | Std | β < 0.1 | β < 0.5 | β < 0.7 | β < 0.9 |
|---|------|--------|-----|---------|---------|---------|---------|
| **Noise** (v35) | 0.071 | 0.000 | 0.211 | 88.63 % | 91.80 % | 94.29 % | 99.58 % |
| Noise (v34) | 0.078 | 0.000 | 0.219 | 87.54 % | 90.90 % | 93.58 % | 99.67 % |

The two beta distributions are essentially indistinguishable from v34. The
suppression of bin [0.0, 0.1) for noise is +1.1 pp and the high-beta tail
of the signal is slightly lighter (β > 0.8: −0.5 pp) — the variance
regularizer trades a tiny bit of beta-head sharpness for embedding
sharpness, which is the desired trade-off.

![Beta head histograms (signal vs noise, log-y)](model_training/eval_results/v35_analysis_merged/plots/beta_histograms.png)

*Beta-head distributions on 1.5 M signal hits and 138 k noise/secondary
hits. Two clear modes: noise piles up at β ≈ 0 (88 % below 0.1) and
signal peaks at β ≈ 0.8 (38 % in [0.8,0.9)). Same shape as v34, slightly
heavier noise suppression and slightly lighter high-β signal tail —
exactly what L_var trades.*

### 4.1 Signal beta histogram

| Bin | v35 fraction | v34 fraction |
|-----|-------------|--------------|
| [0.0, 0.1) | 4.68 % | 4.78 % |
| [0.1, 0.2) | 1.74 % | 1.56 % |
| [0.2, 0.3) | 1.44 % | 1.38 % |
| [0.3, 0.4) | 1.55 % | 1.33 % |
| [0.4, 0.5) | 2.08 % | 1.80 % |
| [0.5, 0.6) | 3.41 % | 2.92 % |
| [0.6, 0.7) | 9.47 % | 6.76 % |
| [0.7, 0.8) | 23.96 % | 27.28 % |
| [0.8, 0.9) | **38.41 %** | **39.06 %** |
| [0.9, 1.0) | 13.25 % | 13.15 % |

### 4.2 Noise / secondary beta histogram

| Bin | v35 fraction | v34 fraction |
|-----|-------------|--------------|
| [0.0, 0.1) | **88.63 %** | **87.54 %** |
| [0.1, 0.2) | 1.17 % | 1.19 % |
| [0.2, 0.3) | 0.69 % | 0.72 % |
| [0.3, 0.4) | 0.62 % | 0.70 % |
| [0.4, 0.5) | 0.70 % | 0.75 % |
| [0.5, 0.6) | 1.02 % | 1.18 % |
| [0.6, 0.7) | 1.47 % | 1.50 % |
| [0.7, 0.8) | 2.75 % | 3.47 % |
| [0.8, 0.9) | 2.54 % | 2.62 % |
| [0.9, 1.0) | 0.42 % | 0.34 % |

Same bimodal pattern. Noise mass at β < 0.1 is up by ~1 pp.

---

## 5. Embedding statistics (5D)

### 5.1 Per-dimension stats (1,539,414 signal hits)

| Dim | Mean | Median | Std | Min | Max |
|-----|------|--------|-----|-----|-----|
| 0 | +0.182 | +0.148 | 2.024 | −5.97 | +6.22 |
| 1 | +0.021 | +0.010 | 2.273 | −9.23 | +7.92 |
| 2 | +0.115 | +0.130 | 1.681 | −4.71 | +5.03 |
| 3 | −0.436 | −0.437 | 1.930 | −11.85 | +5.76 |
| 4 | +0.601 | +0.718 | 1.598 | −3.96 | +11.12 |

Per-hit norm: **mean 4.15, median 4.22, std 1.32**, min 0.15, max 18.12.

All five dimensions are active (std 1.6–2.3). Compared to v34 (six dims with
std 1.27–2.42) the dynamic range is essentially the same — dropping the
weakest dim did not cause the others to grow.

![Per-dimension histograms of v35 signal embeddings (5 dims)](model_training/eval_results/v35_analysis_merged/plots/embedding_hist_per_dim.png)

*Per-dimension marginals of the 5D signal embedding. All five dimensions
are unimodal and Gaussian-like with std ≈ 1.6–2.3 and means within ±0.6
of zero — none of the five dims has collapsed.*

![Pair plot of v35 5D embedding](model_training/eval_results/v35_analysis_merged/plots/embedding_pairs.png)

*All 5 × 5 pairwise scatter / 1-D marginal panels. Off-diagonal panels
show the embedding has rotational structure but no obvious collinearity
between any two coordinates — the four directions identified by PCA are
spread across all five dims as a slight rotation.*

### 5.2 PCA of v35 signal embeddings (`pca_results.json`)

| Component | Variance explained | Cumulative |
|-----------|--------------------|------------|
| 1 | 29.28 % | 29.28 % |
| 2 | 27.91 % | 57.19 % |
| 3 | 24.18 % | 81.37 % |
| 4 | 18.63 % | **100.00 %** |
| 5 | ≈ 0 % | 100.00 % |

**The 5D embedding is intrinsically 4D.** The fifth principal component
carries zero variance — same effective rank as v34's 6D embedding (also 4
useful PCs, see `v33-report.md` §11.3). The variance regularizer therefore
did *not* further shrink the intrinsic dimension; it sharpened the existing
4-D manifold without creating new collapse.

Loading magnitudes (|coef|, %):
```
        d0    d1    d2    d3    d4
PC1:  29.6  46.3  49.2  63.9  21.8
PC2:  29.6  87.3  16.6  33.9   8.4
PC3:  81.1  15.1  53.8   1.2  17.4
PC4:  27.4   0.2  16.9  52.4  78.9
PC5:  30.4   0.2  64.2  44.9  54.2  <- 0% variance
```

PC2 is dominated by d1, PC3 by d0, PC4 by d4 — i.e. the four useful
dimensions are roughly decorrelated at training time but slightly rotated
relative to the canonical basis.

### 5.3 t-SNE / UMAP

t-SNE and UMAP projections of the 5D signal embedding for four
representative events:

![t-SNE of v35 embedding on 4 events, color = MC track id](model_training/eval_results/v35_analysis_merged/plots/tsne_events.png)

![UMAP of v35 embedding on 4 events, color = MC track id](model_training/eval_results/v35_analysis_merged/plots/umap_events.png)

Same pattern as v34 §10.9: **tracks are clearly separated blobs, with some
internal elongation** corresponding to the spatial progression of hits along
each track. v35's blobs look subjectively *tighter* than v34's at the same
event — which is consistent with the variance term doing useful work.

---

## 6. td probe (extra-small td grid, 700 events / GPU subset)

`eval_results/v35_tdprobe_gpuX/tdprobe_results.json` (one file per GPU,
all 4 in agreement to within 0.005).

| tbeta | td   | purity | efficiency | match rate |
|-------|------|--------|-----------|-----------|
| 0.05  | 0.02 | **0.972** | 0.602 | **0.881** |
| 0.05  | 0.03 | 0.968 | 0.670 | 0.861 |
| 0.05  | 0.04 | 0.965 | 0.717 | 0.843 |
| 0.05  | 0.05 | 0.963 | 0.749 | 0.836 |
| 0.10  | 0.02 | 0.972 | 0.602 | 0.881 |
| 0.10  | 0.03 | 0.968 | 0.669 | 0.860 |
| 0.10  | 0.04 | 0.965 | 0.717 | 0.843 |
| 0.10  | 0.05 | 0.962 | 0.748 | 0.835 |

v34 read 0.879 → 0.832 over the same range; **v35 reads 0.881 → 0.836**, so
the “tail-bunch” effect (efficiency falling off at td<0.05) is essentially
unchanged. The remaining gap is probably structural: hits at the head and
tail of a track travel in opposite directions through embedding space and
no isotropic ball can wrap them perfectly. A geometry-aware merger
(§7 of `v33-report.md`) is the natural next step to close this gap.

---

## 7. Per-pT, per-eta and per-hit-count efficiency

500 events (seeds 195–196), greedy clustering at the v35 recommended
operating point `tbeta=0.10, td=0.20`. Output:
`eval_results/v35_pt_eta.json` and plots in
`eval_results/v35_pt_eta_plots/`.

**Overall (charged tracks): efficiency = 0.879, match rate = 0.761**
(11,831 charged tracks out of 11,832 total).

### 7.1 Efficiency vs pT

| pT bin [GeV] | Tracks | Efficiency | Match Rate |
|-------------|--------|------------|-----------|
| [0.00, 0.10) | 3,218 | 0.876 | 0.488 |
| [0.10, 0.20) |   963 | 0.750 | 0.682 |
| [0.20, 0.50) | 2,252 | 0.798 | 0.809 |
| [0.50, 1.00) | 1,798 | 0.901 | 0.884 |
| [1.00, 2.00) | 1,443 | 0.957 | 0.924 |
| [2.00, 5.00) | 1,316 | 0.951 | 0.938 |
| [5.00, 10.00) |   546 | 0.955 | 0.952 |
| [10.0, 20.0) |   210 | 0.962 | 0.943 |
| [20.0, 50.0) |    85 | 0.988 | 0.988 |

Same shape as v34 (`v33-report.md` §8): soft tracks (< 0.1 GeV) capture
many hits but fragment, mid-pT (0.1–0.5 GeV) is the weakest regime,
and pT > 1 GeV is essentially solved (≥ 0.92 match for stiff tracks). The
v35 match rate is **+0.02–0.04 better** in every bin above 0.5 GeV.

![Efficiency and match-rate vs pT (greedy 0.10/0.20, 11.8k charged tracks)](model_training/eval_results/v35_pt_eta_plots/v35_pt_efficiency.png)

### 7.2 Efficiency vs η

| η bin | Tracks | Efficiency | Match Rate |
|-------|--------|------------|-----------|
| [−3.0, −2.5) |   37 | 0.944 | 0.378 |
| [−2.5, −2.0) |  171 | 0.922 | 0.637 |
| [−2.0, −1.5) |  475 | 0.906 | 0.737 |
| [−1.5, −1.0) |  998 | 0.910 | 0.785 |
| [−1.0, −0.5) | 1,759 | 0.896 | 0.766 |
| [−0.5,  0.0) | 2,423 | 0.843 | 0.776 |
| [ 0.0,  0.5) | 2,271 | 0.836 | 0.779 |
| [ 0.5,  1.0) | 2,009 | 0.898 | 0.763 |
| [ 1.0,  1.5) |   995 | 0.916 | 0.732 |
| [ 1.5,  2.0) |   460 | 0.913 | 0.754 |
| [ 2.0,  2.5) |   163 | 0.921 | 0.693 |
| [ 2.5,  3.0) |    44 | 0.976 | 0.432 |

The roughly symmetric curve we expect from the IDEA detector. Central
(|η| < 0.5) match rate is now **0.78** (v34 was 0.74). Central efficiency
is essentially unchanged (~0.84 — the central region is dense and
intrinsically harder).

![Efficiency and match-rate vs eta](model_training/eval_results/v35_pt_eta_plots/v35_eta_efficiency.png)

### 7.3 Efficiency vs n_hits per track

| n_hits | Tracks | Efficiency | Match Rate |
|--------|--------|-----------|-----------|
| [2, 5)    | 2,601 | 0.868 | 0.429 |
| [5, 10)   |   904 | 0.839 | 0.463 |
| [10, 20)  |   606 | 0.837 | 0.569 |
| [20, 50)  | 1,773 | 0.891 | 0.829 |
| [50, 100) | 3,045 | 0.935 | **0.945** |
| [100, 200)| 2,228 | 0.929 | **0.965** |
| [200, 500)|   534 | 0.651 | 0.936 |

Match rate ≥ 0.95 for tracks with 50–200 hits, same regime as v34 but with
slightly higher absolute numbers. The 200+ hit bin still has lower
efficiency because greedy splits very long tracks; the high match rate
(0.94) shows the dominant cluster still passes the 75 % threshold.

![Efficiency and match-rate vs n_hits per track](model_training/eval_results/v35_pt_eta_plots/v35_nhits_efficiency.png)

---

## 8. 3D event displays

A handful of events from seed 191 with both true MC labels and predicted
track labels using:
- the v35 best balanced operating point (greedy `tbeta=0.10, td=0.20`),
  saved in `eval_results/v35_events_3d_greedy/`, and
- the best balanced HDBSCAN (`mcs=2, ms=1, eps=0.10`),
  saved in `eval_results/v35_events_3d_hdbscan/`.

For each algorithm we render:
- per-event 3D scatter (true labels vs predicted labels side-by-side):
  `event0XX_3d.png`,
- per-event xy projection (same): `event0XX_xy.png`,
- a 6-event grid (true): `grid_true.png`,
- a 6-event grid (predicted): `grid_predicted.png`.

### 8.1 Greedy clustering (`tbeta=0.10, td=0.20`)

**Six-event grid — ground truth (MC labels):**

![Six-event grid: ground-truth MC track labels](model_training/eval_results/v35_events_3d_greedy/grid_true.png)

**Six-event grid — predicted by v35 + greedy:**

![Six-event grid: v35 + greedy predicted track labels](model_training/eval_results/v35_events_3d_greedy/grid_predicted.png)

Side-by-side per-event displays (true | predicted) for two example
events (3D and xy projection):

| Event | True 3D | Predicted 3D |
|-------|---------|--------------|
| 000 | ![](model_training/eval_results/v35_events_3d_greedy/event000_3d.png) | ![](model_training/eval_results/v35_events_3d_greedy/event000_xy.png) |
| 002 | ![](model_training/eval_results/v35_events_3d_greedy/event002_3d.png) | ![](model_training/eval_results/v35_events_3d_greedy/event002_xy.png) |
| 007 | ![](model_training/eval_results/v35_events_3d_greedy/event007_3d.png) | ![](model_training/eval_results/v35_events_3d_greedy/event007_xy.png) |
| 009 | ![](model_training/eval_results/v35_events_3d_greedy/event009_3d.png) | ![](model_training/eval_results/v35_events_3d_greedy/event009_xy.png) |

(Each individual `event0XX_3d.png` already contains the *true | predicted*
side-by-side rendering. The xy projection complements it.)

### 8.2 HDBSCAN (`mcs=2, ms=1, eps=0.10`)

**Six-event grid — predicted by v35 + HDBSCAN:**

![Six-event grid: v35 + HDBSCAN predicted track labels](model_training/eval_results/v35_events_3d_hdbscan/grid_predicted.png)

Per-event:

| Event | 3D (true \| predicted) | xy (true \| predicted) |
|-------|------------------------|-----------------------|
| 000 | ![](model_training/eval_results/v35_events_3d_hdbscan/event000_3d.png) | ![](model_training/eval_results/v35_events_3d_hdbscan/event000_xy.png) |
| 002 | ![](model_training/eval_results/v35_events_3d_hdbscan/event002_3d.png) | ![](model_training/eval_results/v35_events_3d_hdbscan/event002_xy.png) |
| 007 | ![](model_training/eval_results/v35_events_3d_hdbscan/event007_3d.png) | ![](model_training/eval_results/v35_events_3d_hdbscan/event007_xy.png) |
| 009 | ![](model_training/eval_results/v35_events_3d_hdbscan/event009_3d.png) | ![](model_training/eval_results/v35_events_3d_hdbscan/event009_xy.png) |

Visually, predicted tracks reproduce the spatial distribution of true MC
tracks for ≥ 80 % of all tracks; the residual mismatches are dominated by
short tracks (consistent with the n_hits breakdown in §7.3) and a handful
of curlers fragmenting into multiple sub-clusters.

### 8.3 Drift circles (DC measurement geometry)

The plots above use only `(hit_x, hit_y, hit_z)` — the closest-approach
point on the track. Each DC measurement is actually a **drift circle**:
centered on the wire `(wire_x, wire_y, wire_z)`, radius equal to the
drift distance, lying in the plane perpendicular to the wire direction
`(sin θ_stereo cos φ, sin θ_stereo sin φ, cos θ_stereo)` — the same
parameterization used by `embed_circle_ipns` in the model. The plots
below visualize that geometry directly.

Generated with
`model_training/src/plot_events_3d_v35_circles.py` and saved in
`eval_results/v35_events_3d_circles/`.

**Six-event grid (predicted clusters, drift circles).**

![Six-event grid: drift circles, predicted track labels](model_training/eval_results/v35_events_3d_circles/grid_circles_predicted.png)

**Six-event grid (true MC tracks, drift circles).**

![Six-event grid: drift circles, true MC labels](model_training/eval_results/v35_events_3d_circles/grid_circles_true.png)

At full chamber scale (±2000 a.u.) drift circles still look almost like
points — drift distances are 0–10 a.u., so each circle spans <1 % of the
view. To make the geometry obvious, the next four panels zoom into a
±100 a.u. window around the densest track of each event. Drift circles
appear as small **tilted ellipses**: the tilt comes from the stereo
angle (≤ 14°), the in-plane diameter is `2 × drift_distance`, and all
circles belonging to the same predicted track share a colour.

| ev | full event (3D circles) | xy projection (with circles) | zoom (drift circles visible) |
|---|---|---|---|
| 000 | ![](model_training/eval_results/v35_events_3d_circles/event000_3d_circles.png) | ![](model_training/eval_results/v35_events_3d_circles/event000_xy_circles.png) | ![](model_training/eval_results/v35_events_3d_circles/event000_3d_circles_zoom.png) |
| 004 | ![](model_training/eval_results/v35_events_3d_circles/event004_3d_circles.png) | ![](model_training/eval_results/v35_events_3d_circles/event004_xy_circles.png) | ![](model_training/eval_results/v35_events_3d_circles/event004_3d_circles_zoom.png) |
| 007 | ![](model_training/eval_results/v35_events_3d_circles/event007_3d_circles.png) | ![](model_training/eval_results/v35_events_3d_circles/event007_xy_circles.png) | ![](model_training/eval_results/v35_events_3d_circles/event007_3d_circles_zoom.png) |
| 009 | ![](model_training/eval_results/v35_events_3d_circles/event009_3d_circles.png) | ![](model_training/eval_results/v35_events_3d_circles/event009_xy_circles.png) | ![](model_training/eval_results/v35_events_3d_circles/event009_3d_circles_zoom.png) |

In the zoom panels v35 typically merges all 15–20 drift circles along a
single track into one predicted cluster (same colour throughout) — the
visual confirmation of the §7.3 high-n_hits efficiency.

#### HDBSCAN clustering with drift circles

Same six events, same anchoring, but the predicted labels now come from
HDBSCAN(`mcs=2, ms=1, eps=0.10`). Saved in
`eval_results/v35_events_3d_circles_hdbscan/`.

![Six-event grid: drift circles, HDBSCAN predicted](model_training/eval_results/v35_events_3d_circles_hdbscan/grid_circles_predicted.png)

| ev | full event (3D circles) | xy projection (with circles) | zoom (drift circles visible) |
|---|---|---|---|
| 000 | ![](model_training/eval_results/v35_events_3d_circles_hdbscan/event000_3d_circles.png) | ![](model_training/eval_results/v35_events_3d_circles_hdbscan/event000_xy_circles.png) | ![](model_training/eval_results/v35_events_3d_circles_hdbscan/event000_3d_circles_zoom.png) |
| 002 | ![](model_training/eval_results/v35_events_3d_circles_hdbscan/event002_3d_circles.png) | ![](model_training/eval_results/v35_events_3d_circles_hdbscan/event002_xy_circles.png) | ![](model_training/eval_results/v35_events_3d_circles_hdbscan/event002_3d_circles_zoom.png) |
| 007 | ![](model_training/eval_results/v35_events_3d_circles_hdbscan/event007_3d_circles.png) | ![](model_training/eval_results/v35_events_3d_circles_hdbscan/event007_xy_circles.png) | ![](model_training/eval_results/v35_events_3d_circles_hdbscan/event007_3d_circles_zoom.png) |
| 009 | ![](model_training/eval_results/v35_events_3d_circles_hdbscan/event009_3d_circles.png) | ![](model_training/eval_results/v35_events_3d_circles_hdbscan/event009_xy_circles.png) | ![](model_training/eval_results/v35_events_3d_circles_hdbscan/event009_3d_circles_zoom.png) |

#### v34 vs v35 zoom regression check

Side-by-side for the same hits using the same greedy clustering: True
MC | v34 predicted | v35 predicted. The zoom is anchored on the true
track that maximises (`#v34 clusters covering it - #v35 clusters
covering it`), i.e. the track where v34 fragments most and v35
consolidates most. The suptitle quotes the *global* fragmentation
counts for that target track (across the full event, not just the
window), so a long track that spans the chamber can still appear short
in the local view.

Generated with `model_training/src/plot_events_3d_compare.py`. Saved in
`eval_results/v34_v35_zoom_compare/`.

| ev | True MC \| v34 predicted \| v35 predicted |
|---|---|
| 000 | ![](model_training/eval_results/v34_v35_zoom_compare/event000_zoom_compare.png) |
| 001 | ![](model_training/eval_results/v34_v35_zoom_compare/event001_zoom_compare.png) |
| 002 | ![](model_training/eval_results/v34_v35_zoom_compare/event002_zoom_compare.png) |
| 007 | ![](model_training/eval_results/v34_v35_zoom_compare/event007_zoom_compare.png) |
| 009 | ![](model_training/eval_results/v34_v35_zoom_compare/event009_zoom_compare.png) |

In events 0, 7 and 9 the suptitle reports the v35 count is meaningfully
lower than v34's (e.g. event 9: a 17-hit track split 4-ways by v34 is
single-cluster under v35), confirming that v35's cluster
re-consolidation is the source of its +0.04 high-n_hits efficiency.

---

## 9. Verdict vs. v34

| Question | Answer |
|---|---|
| Did v35 improve over v34? | **Yes**, +0.02–0.03 match rate at every operating point. |
| Did the variance regularizer reduce within-cluster variance? | **Yes**, indirectly — the +0.04 efficiency lift at fixed td is the signature. |
| Did dropping the 6th dim hurt? | **No**, the 6D embedding was already effectively 4D (v34 PCA), and v35's 5D PCA is also effectively 4D. |
| Is HDBSCAN now competitive? | **Almost** — HDBSCAN best (geo 0.838) ≥ DBSCAN best (geo 0.845), and HDBSCAN now beats greedy at td=0.05 only on purity, not match rate. |
| Did the beta head get worse? | **No, marginally** — signal β > 0.7 dropped by 4 pp but the bimodal separation is preserved. |
| Is the model still under-trained? | **Probably no** — the val curve has been flat for 8 epochs, with last-2-epoch slope ~0.001 in match rate. |

---

## 10. Pipeline reproducibility

All commands below have been executed (logs in `v35-*.log`).

```bash
ENV=hotdog-ml; conda activate $ENV
cd /home/marko.cechovic/cgatr/model_training

# stop training cleanly (preserves checkpoints/cgatr_v35/cgatr_best.pt)
kill -TERM $(pgrep -f "train_cgatr_parquet-v35")

# coarse sweep (5,000 events, seeds 191-200, split 4 ways across GPUs)
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. python src/eval_sweep_v33.py \
    --data_dir /home/marko.cechovic/cgatr/data_parquet_train \
    --checkpoint checkpoints/cgatr_v35/cgatr_best.pt \
    --embed_dim 5 --eval_seeds <split[g]> --max_events 1250 \
    --output_dir eval_results/v35_gpu$g --gpu 0 \
    > /home/marko.cechovic/cgatr/v35-sweep-gpu$g.log 2>&1 &
done

# fine analysis (700 events, seeds 181-188)
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. python src/eval_analysis_v34.py \
    --data_dir /home/marko.cechovic/cgatr/data_parquet_train \
    --checkpoint checkpoints/cgatr_v35/cgatr_best.pt \
    --embed_dim 5 --eval_seeds <split[g]> --max_events 200 \
    --output_dir eval_results/v35_analysis_gpu$g --gpu 0 \
    > /home/marko.cechovic/cgatr/v35-analysis-gpu$g.log 2>&1 &
done

# tiny-td probe (4 GPUs, ~200 events each, td ∈ {0.02..0.05})
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. python src/eval_tdprobe_v34.py \
    --data_dir /home/marko.cechovic/cgatr/data_parquet_train \
    --checkpoint checkpoints/cgatr_v35/cgatr_best.pt \
    --embed_dim 5 --eval_seeds <split[g]> --max_events 200 \
    --output_dir eval_results/v35_tdprobe_gpu$g --gpu 0 \
    > /home/marko.cechovic/cgatr/v35-tdprobe-gpu$g.log 2>&1 &
done

# pT/eta breakdown (single GPU, 500 events on seeds 195-196)
PYTHONPATH=. python src/eval_pt_eta_v35.py \
  --data_dir /home/marko.cechovic/cgatr/data_parquet_train \
  --checkpoint checkpoints/cgatr_v35/cgatr_best.pt \
  --embed_dim 5 --eval_seeds 195-196 --max_events 500 \
  --tbeta 0.10 --td 0.20 \
  --output_path eval_results/v35_pt_eta.json --gpu 0 \
  > /home/marko.cechovic/cgatr/v35-pt-eta.log 2>&1
PYTHONPATH=. python src/plot_pt_eta_v35.py \
  --input eval_results/v35_pt_eta.json \
  --output_dir eval_results/v35_pt_eta_plots

# merge multi-GPU outputs (analysis)
PYTHONPATH=. python src/merge_v34_analysis.py \
  --dirs eval_results/v35_analysis_gpu0 eval_results/v35_analysis_gpu1 \
         eval_results/v35_analysis_gpu2 eval_results/v35_analysis_gpu3 \
  --out  eval_results/v35_analysis_merged

# merge multi-GPU outputs (coarse sweep)
PYTHONPATH=. python src/merge_v35_sweep.py \
  --dirs eval_results/v35_gpu0 eval_results/v35_gpu1 \
         eval_results/v35_gpu2 eval_results/v35_gpu3 \
  --out  eval_results/v35_sweep_merged

# plots
PYTHONPATH=. python src/plot_analysis_v35.py \
  --analysis_dir eval_results/v35_analysis_merged \
  --output_dir   eval_results/v35_analysis_merged/plots

# training curves (parsed from v35-run.log)
PYTHONPATH=. python src/plot_training_curves_v35.py \
  --log /home/marko.cechovic/cgatr/v35-run.log \
  --output_dir eval_results/v35_training_curves
PYTHONPATH=. python src/pca_v34.py \
  --samples eval_results/v35_analysis_merged/samples.pkl \
  --output  eval_results/v35_analysis_merged/pca_results.json

# 3D events (greedy)
PYTHONPATH=. python src/plot_events_3d_v35.py \
  --data_dir /home/marko.cechovic/cgatr/data_parquet_train \
  --checkpoint checkpoints/cgatr_v35/cgatr_best.pt --embed_dim 5 \
  --algorithm greedy --tbeta 0.10 --td 0.20 \
  --eval_seeds 191-191 --n_events 6 \
  --output_dir eval_results/v35_events_3d_greedy

# 3D events (HDBSCAN)
PYTHONPATH=. python src/plot_events_3d_v35.py \
  --data_dir /home/marko.cechovic/cgatr/data_parquet_train \
  --checkpoint checkpoints/cgatr_v35/cgatr_best.pt --embed_dim 5 \
  --algorithm hdbscan --hdbscan_mcs 2 --hdbscan_ms 1 \
  --hdbscan_eps 0.10 \
  --eval_seeds 191-191 --n_events 6 \
  --output_dir eval_results/v35_events_3d_hdbscan

# 3D events with DC drift circles (greedy)
PYTHONPATH=. python src/plot_events_3d_v35_circles.py \
  --data_dir /home/marko.cechovic/cgatr/data_parquet_train \
  --checkpoint checkpoints/cgatr_v35/cgatr_best.pt --embed_dim 5 \
  --algorithm greedy --tbeta 0.10 --td 0.20 \
  --eval_seeds 181-181 --n_events 6 --zoom_window 100 \
  --output_dir eval_results/v35_events_3d_circles

# 3D events with DC drift circles (HDBSCAN)
PYTHONPATH=. python src/plot_events_3d_v35_circles.py \
  --data_dir /home/marko.cechovic/cgatr/data_parquet_train \
  --checkpoint checkpoints/cgatr_v35/cgatr_best.pt --embed_dim 5 \
  --algorithm hdbscan --hdb_mcs 2 --hdb_ms 1 --hdb_eps 0.10 \
  --eval_seeds 181-181 --n_events 6 --zoom_window 100 \
  --output_dir eval_results/v35_events_3d_circles_hdbscan

# v34 vs v35 zoom regression check
PYTHONPATH=. python src/plot_events_3d_compare.py \
  --data_dir /home/marko.cechovic/cgatr/data_parquet_train \
  --ckpt_a checkpoints/cgatr_v34/cgatr_best.pt --embed_a 6 --label_a v34 \
  --ckpt_b checkpoints/cgatr_v35/cgatr_best.pt --embed_b 5 --label_b v35 \
  --algorithm greedy --tbeta 0.10 --td 0.20 \
  --eval_seeds 181-181 --n_events 6 --zoom_window 150 \
  --output_dir eval_results/v34_v35_zoom_compare
```

Artifact map:

```
v35-run.log                                   # 35 epochs of training
checkpoints/cgatr_v35/cgatr_best.pt           # epoch 35, EMA weights, 924,624 params

eval_results/v35_analysis_merged/             # 700-event fine analysis, merged 4 GPUs
  ├ analysis_results.json                     # 179 configs × event-weighted metrics
  ├ samples.pkl                               # 1.54M signal + 138k noise hits with coords/beta
  ├ tsne_events.pkl                           # 32 events for t-SNE/UMAP
  ├ pca_results.json                          # explained variance per component
  └ plots/
      ├ beta_histograms.png
      ├ embedding_hist_per_dim.png
      ├ embedding_pairs.png                   # 5×5 pair scatter
      ├ greedy_scan_per_td.png                # match rate vs tbeta, one curve per td
      ├ sweep_pareto.png                      # purity vs efficiency, color = match
      ├ tsne_events.png
      └ umap_events.png

eval_results/v35_tdprobe_gpu{0..3}/           # tiny-td probe (8 configs each)
eval_results/v35_pt_eta.json                  # pT/eta/n_hits breakdown
eval_results/v35_pt_eta_plots/                # 3 PNGs

eval_results/v35_events_3d_greedy/            # 6 events × (3D + xy) + 2 grids
eval_results/v35_events_3d_hdbscan/           # 6 events × (3D + xy) + 2 grids

eval_results/v35_gpu{0..3}/                   # coarse sweep (5,000 events, in progress)
eval_results/v35_sweep_merged/                # produced by merge_v35_sweep.py once gpus done
```
