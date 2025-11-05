#!/bin/bash
# Simple script to run GNN training with the three HuggingFace datasets

mkdir -p gnn_outputs

UV_CACHE_DIR=.uv_cache uv run python GNN.py \
  --path "/mnt/c/Users/Admin/.cache/huggingface/datasets/PrimeIntellect___deepcoder-gold-standard-solutions,/mnt/c/Users/Admin/.cache/huggingface/datasets/codeparrot___codecomplex,/mnt/c/Users/Admin/.cache/huggingface/datasets/deepmind___code_contests" \
  --arch gtr \
  --k 15 \
  --mutual \
  --lappe \
  --rwse \
  --use_neighbor_loader \
  --fanout "15,10,5" \
  --epochs 10 \
  --save_model "gnn_outputs/model_best.pt" \
  --save_results "gnn_outputs/training_results.json"

