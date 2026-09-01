#!/bin/bash
# CIRCE reference configuration, runnable as-is inside the gatr:v9 docker
# image. This is exactly the setup our reported numbers come from (up to the
# 3-d output contract of the capacity-matched variant and EMA, which our own
# trainer adds on top).
#
#   data      Zqq_uds_minKineticEnergy0 seeds 1-180 + Loopers 201-1000
#             (point --data_dir at your parquet; converters in data_creation/)
#   targets   >= 3 hits, stored secondaries kept (the adapter applies the
#             create_garbage_label-style relabel)
#   loss      circe backend: attr 1.0, repul 1.0, qmin 0.1,
#             beta_suppress 0.1, var 0.3
#   optim     AdamW 4e-4, weight decay 1e-4, 2 warm-up epochs, flat, then a
#             half-cosine anneal to 1e-5 over the last 6 epochs
#   batching  token budget 16k hits/batch
set -euo pipefail

DATA=${1:?usage: train_circe.sh <parquet_dir> <output_dir>}
OUT=${2:?usage: train_circe.sh <parquet_dir> <output_dir>}

python -u -m src.train_algebra_ab \
  --algebra conformal \
  --loss_backend circe \
  --recipe circe \
  --reference_width \
  --data_dir "$DATA" \
  --train_seeds 1-180 --val_seeds 181-190 \
  --epochs 30 \
  --start_lr 4e-4 \
  --max_tokens 16000 \
  --num_devices 4 \
  --output_dir "$OUT"
