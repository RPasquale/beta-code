#!/bin/bash
# Production ML Training Script for GNN with HuggingFace Datasets
# This script implements proper ML practices: train/val/test splits, regularization, early stopping, etc.

# Using 3 datasets that load successfully
DATASETS="C:\Users\Admin\.cache\huggingface\datasets\PrimeIntellect___deepcoder-gold-standard-solutions,C:\Users\Admin\.cache\huggingface\datasets\codeparrot___codecomplex,C:\Users\Admin\.cache\huggingface\datasets\deepmind___code_contests"
# Commented out datasets that have loading issues:
# DATASETS_FULL="C:\Users\Admin\.cache\huggingface\datasets\codeparrot___codecomplex,C:\Users\Admin\.cache\huggingface\datasets\deepmind___code_contests,C:\Users\Admin\.cache\huggingface\datasets\justus27___deepcoder-train,C:\Users\Admin\.cache\huggingface\datasets\mbpp,C:\Users\Admin\.cache\huggingface\datasets\PrimeIntellect___deepcoder-gold-standard-solutions,C:\Users\Admin\.cache\huggingface\datasets\princeton-nlp___swe-bench_lite"

OUTPUT_DIR="gnn_outputs"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_PATH="$OUTPUT_DIR/gnn_model_${TIMESTAMP}.pt"
RESULTS_PATH="$OUTPUT_DIR/gnn_results_${TIMESTAMP}.json"

echo "=========================================="
echo "GNN Production Training"
echo "=========================================="
echo "Datasets: $DATASETS"
echo "Model will be saved to: $MODEL_PATH"
echo "Results will be saved to: $RESULTS_PATH"
echo ""

uv run python GNN.py \
    --path "$DATASETS" \
    --arch gtr \
    --lappe \
    --rwse \
    --k 15 \
    --mutual \
    --epochs 1000 \
    --use_neighbor_loader \
    --fanout "15,10,5" \
    --save_model "$MODEL_PATH" \
    --save_results "$RESULTS_PATH"

echo ""
echo "Training complete!"
echo "Model: $MODEL_PATH"
echo "Results: $RESULTS_PATH"

