#!/bin/bash
# Example training commands for WM

# Quick test run (1 epoch, 50 steps per epoch)
python model.py code_train \
    --dataset PrimeIntellect/deepcoder-gold-standard-solutions \
    --split train \
    --epochs 1 \
    --steps_per_epoch 50 \
    --save_dir runs/test_run

# Full training run (3 epochs, all steps)
python model.py code_train \
    --dataset PrimeIntellect/deepcoder-gold-standard-solutions \
    --split train \
    --epochs 3 \
    --sft_steps 100 \
    --save_dir runs/full_training

# With Weights & Biases logging
python model.py code_train \
    --dataset PrimeIntellect/deepcoder-gold-standard-solutions \
    --split train \
    --epochs 3 \
    --wandb \
    --wandb_project wm-gfn-code \
    --save_dir runs/wandb_training

