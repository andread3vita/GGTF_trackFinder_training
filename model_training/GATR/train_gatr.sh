#!/bin/bash

torchrun    --nproc_per_node=4 \
            -m src.train_lightning \
            --data-train /eos/experiment/fcc/ee/simulation/key4hep_2026_06_16/IDEA_v4_o1/91GeV/Zqq_uds_minKineticEnergy0/graph/Graphs_* \
            --data-config ../../config_files/config_tracking.yaml \
            -clust -clust_dim 3 \
            --network-config src/models/wrapper/model_tracking_gatr.py \
            --model-prefix /afs/cern.ch/work/a/adevita/public/trainingFolder/GATr_smallDataset_IDEAv4o1_standardLoss/ \
            --num-workers 0 \
            --gpus 0,1,2,3 \
            --batch-size 4 \
            --start-lr 3e-4 \
            --num-epochs 100 \
            --optimizer ranger \
            --fetch-step 0.04 \
            --condensation \
            --log-wandb \
            --wandb-displayname GATr_smallDataset_IDEAv4o1_standardLoss \
            --wandb-projectname IDEA_v3_o1_tracking_andrea \
            --wandb-entity ml4hep \
            --frac_cluster_loss 0 \
            --qmin 3  