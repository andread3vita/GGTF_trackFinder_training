#!/usr/bin/env python

import os
import ast
import sys
import shutil
import glob
import argparse
import functools
import numpy as np
import math
import torch
import wandb
import warnings
import random

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from src.utils.parser_args import parser


import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    TQDMProgressBar,
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.profilers import AdvancedProfiler

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from src.utils.train_utils import (
    train_load,
    test_load,
)
from src.utils.import_tools import import_module
from src.utils.train_utils import get_samples_steps_per_epoch, model_setup, get_gpu_dev

print("Using PyTorch version:", torch.__version__)

def main():

    args = parser.parse_args()
    
    missing_files = 0
    for i in args.data_train:
        if not os.path.isfile(i):
            print("Missing File!", i)
            missing_files += 1

    if missing_files > 0:
        print(f"{missing_files} missing files. Quit.")
        sys.exit(1)   
        
    args = get_samples_steps_per_epoch(args)
    args.local_rank = 0
    training_mode = not args.predict
    
    
    if training_mode:
        train_loader, val_loader, data_config, train_input_names = train_load(args)
    else:
        test_loaders, data_config = test_load(args)

    model = model_setup(args, data_config)
    if args.gpus:
        gpus = [int(i) for i in args.gpus.split(",")]
    else:
        print("No GPUs flag provided - Setting GPUs to [0]")
        gpus = [0]
        
    wandb_logger = WandbLogger(
        project=args.wandb_projectname,
        entity=args.wandb_entity,
        name=args.wandb_displayname,
        offline="True"
    )
       

    if training_mode:
        print("USING TRAINING MODE")

        checkpoint_callback = ModelCheckpoint(
            dirpath=args.model_prefix,  # checkpoints_path, # <--- specify this on the trainer itself for version control
            filename="_{epoch}_{step}",
            every_n_train_steps=1000,
            save_top_k=-1,  # <--- this is important!
            save_weights_only=True,
        )
        
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks = [
            TQDMProgressBar(refresh_rate=10),
            checkpoint_callback,
            lr_monitor,
        ]
        # only needed for the GNN baseline
        gpus = [int(i) for i in args.gpus.split(",")]
        print("Using GPUs:", gpus)
        trainer = L.Trainer(
            callbacks=callbacks,
            accelerator="gpu",
            devices=gpus,
            default_root_dir=args.model_prefix,
            logger=wandb_logger,
            # max_epochs=5,
            strategy="ddp",
            # limit_train_batches=20,
            # limit_train_batches=890,
            # gradient_clip_val=1.0,
            accumulate_grad_batches=2,
            limit_val_batches=5,
        )
        args.local_rank = trainer.global_rank
        train_loader, val_loader, data_config, train_input_names = train_load(args)

        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            # ckpt_path=args.load_model_weights,
        )

    elif args.data_test:
        trainer = L.Trainer(
            callbacks=[TQDMProgressBar(refresh_rate=1)],
            accelerator="gpu",
            devices=[0],
            default_root_dir=args.model_prefix,
            logger=wandb_logger,
            # limit_val_batches=5,
        )


        for name, get_test_loader in test_loaders.items():
            test_loader = get_test_loader()

            trainer.validate(
                model=model,
                ckpt_path=args.load_model_weights,
                dataloaders=test_loader,
            )


if __name__ == "__main__":
        
    L.seed_everything(42, workers=True)
    
    main()
