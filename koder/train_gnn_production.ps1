# Production ML Training Script for GNN with HuggingFace Datasets (PowerShell)
# This script implements proper ML practices: train/val/test splits, regularization, early stopping, etc.

# Using 3 datasets that load successfully
$DATASETS = "C:\Users\Admin\.cache\huggingface\datasets\PrimeIntellect___deepcoder-gold-standard-solutions,C:\Users\Admin\.cache\huggingface\datasets\codeparrot___codecomplex,C:\Users\Admin\.cache\huggingface\datasets\deepmind___code_contests"
# Commented out datasets that have loading issues:
# $DATASETS_FULL = "C:\Users\Admin\.cache\huggingface\datasets\codeparrot___codecomplex,C:\Users\Admin\.cache\huggingface\datasets\deepmind___code_contests,C:\Users\Admin\.cache\huggingface\datasets\justus27___deepcoder-train,C:\Users\Admin\.cache\huggingface\datasets\mbpp,C:\Users\Admin\.cache\huggingface\datasets\PrimeIntellect___deepcoder-gold-standard-solutions,C:\Users\Admin\.cache\huggingface\datasets\princeton-nlp___swe-bench_lite"

$OUTPUT_DIR = "gnn_outputs"
if (-not (Test-Path $OUTPUT_DIR)) {
    New-Item -ItemType Directory -Path $OUTPUT_DIR | Out-Null
}

$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$MODEL_PATH = "$OUTPUT_DIR\gnn_model_${TIMESTAMP}.pt"
$RESULTS_PATH = "$OUTPUT_DIR\gnn_results_${TIMESTAMP}.json"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "GNN Production Training" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Datasets: $DATASETS" -ForegroundColor Yellow
Write-Host "Model will be saved to: $MODEL_PATH" -ForegroundColor Yellow
Write-Host "Results will be saved to: $RESULTS_PATH" -ForegroundColor Yellow
Write-Host ""

uv run python GNN.py `
    --path "$DATASETS" `
    --arch gtr `
    --lappe `
    --rwse `
    --k 15 `
    --mutual `
    --epochs 1000 `
    --use_neighbor_loader `
    --fanout "15,10,5" `
    --save_model "$MODEL_PATH" `
    --save_results "$RESULTS_PATH"

Write-Host ""
Write-Host "Training complete!" -ForegroundColor Green
Write-Host "Model: $MODEL_PATH" -ForegroundColor Green
Write-Host "Results: $RESULTS_PATH" -ForegroundColor Green

