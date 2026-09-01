# CGA (conformal) variant of the GGTF track finder

This adds a Conformal Geometric Algebra (Cl(4,1)) version of the transformer
that runs inside the existing GGTF pipeline. Nothing about the pipeline
changes: same data, same loss, same trainer - you just point the network
config at the conformal wrapper.

## Run it

Training, exactly like GATR but with one flag changed:

```bash
torchrun --nproc_per_node=4 -m src.train_lightning \
  --network-config src/models/wrapper/model_tracking_cgatr.py \
  ... everything else as in your usual GATR command ...
```

Works in the same docker image as GATR (`dologarcia/gatr:v9`). The CGA
product tables and equivariant bases are generated at first use, no extra
files needed.

## Why conformal

Drift-chamber hits are circles (wire position, wire direction, drift radius),
and CGA represents circles, spheres and planes as native objects. Each drift
hit enters the network as its actual measured circle instead of a point or a
point pair, so the left/right ambiguity never has to be resolved upstream.
The encoding provably keeps the wire direction, which the two-point (left,
right) encoding discards.

## What is in this PR

- `src/cgatr/` - the CGA library (layers, primitives, interfaces, tests).
  Equivariance and geometry are covered by `src/cgatr/tests/test_cga.py`.
- `src/models/Cgatr_withModifications.py` - the conformal LightningModule,
  same structure as `Gatr_withModifications.py`, capacity-matched to it
  within 1.4% so A/B comparisons are about the algebra and nothing else.
- `src/models/wrapper/model_tracking_cgatr.py` - the network config to pass
  via `--network-config`.
- `src/dataset/parquet_ggtf_adapter.py` - feeds parquet datasets (see
  converters below) into the standard DGL graph contract, byte-identical
  inputs for both algebras.
- `src/train_algebra_ab.py` + `src/models/smoke_algebra_ab.py` - a paired
  A/B trainer and a five-minute integration check. The smoke test runs both
  algebras forward and backward through the real loss on real events and
  verifies they see identical inputs:

  ```bash
  python -m src.models.smoke_algebra_ab --data_dir <parquet_dir>
  ```

- `data_creation/edm4hep_to_parquet.py` - converts digitised edm4hep ROOT to
  the parquet layout the adapter reads (per-seed directories with drift,
  vertex/silicon and MC particle tables, full circle geometry per drift hit).
- `data_creation/edm4hep_to_parquet_lowmem.py` - same output, streaming
  writer. Use this one for keepAllParticles samples: their MC tables are
  ~100k particles per event and the in-memory version needs tens of GB per
  worker (we found out the hard way). Also writes through a .tmp rename so
  interrupted conversions never leave a truncated file.

  ```bash
  python data_creation/edm4hep_to_parquet_lowmem.py \
    --input_dir <dir with seed_N/digi_edm4hep/*.root> \
    --output_dir <parquet_out> --split train
  ```

## Checkpoint

A trained checkpoint (IDEA v4 o1, 91 GeV Zqq) is available - it is too large
for the repo, ask us for the link and drop it wherever you like; the wrapper
loads it through the normal Lightning mechanisms.
