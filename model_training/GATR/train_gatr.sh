#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <output_dir> [wandb_project] [wandb_entity]"
    exit 1
fi

OUTPUT_DIR="$1"
WANDB_PROJECT="${2:-IDEA_v3_o1_tracking_andrea}"
WANDB_ENTITY="${3:-ml4hep}"

TRAINING_NAME=$(basename "$OUTPUT_DIR")

mkdir -p "$OUTPUT_DIR"

torchrun \
    --nproc_per_node=4 \
    -m src.train_lightning \
    --data-train /eos/experiment/fcc/ee/simulation/key4hep_2026_06_16/IDEA_v4_o1/91GeV/Zqq_uds_minKineticEnergy0/graph/Graphs_* \
    --data-config ../../config_files/config_tracking.yaml \
    -clust -clust_dim 3 \
    --network-config src/models/wrapper/model_tracking_gatr.py \
    --model-prefix "${OUTPUT_DIR}/" \
    --num-workers 0 \
    --gpus 0,1,2,3 \
    --batch-size 4 \
    --start-lr 3e-4 \
    --num-epochs 100 \
    --optimizer ranger \
    --fetch-step 0.04 \
    --condensation \
    --log-wandb \
    --wandb-displayname "$TRAINING_NAME" \
    --wandb-projectname "$WANDB_PROJECT" \
    --wandb-entity "$WANDB_ENTITY" \
    --frac_cluster_loss 0 \
    --qmin 3