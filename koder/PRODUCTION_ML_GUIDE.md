# Production ML Training Guide for GNN.py

This guide explains the proper ML practices built into GNN.py and how to use them.

## Built-in ML Best Practices

The GNN.py script includes:

### ✅ Train/Validation/Test Splits
- **Automatic stratified splitting** (70% train, 10% validation, 20% test)
- **Stratified sampling** for classification tasks to preserve class distribution
- **Group-aware splitting** for graph-level tasks

### ✅ Regularization
- **Weight Decay (L2)**: Default `1e-4` (AdamW optimizer)
- **Dropout**: Default `0.2` in encoder layers
- **Gradient Clipping**: Max norm `1.0` to prevent exploding gradients
- **Layer Normalization**: Applied after each layer for stability

### ✅ Training Best Practices
- **Early Stopping**: Best model selection based on validation performance
- **Automatic Mixed Precision (AMP)**: Faster training with CUDA
- **Learning Rate Scheduling**: Optional cosine annealing with warm restarts
- **Model Checkpointing**: Save best model state

### ✅ Hyperparameter Optimization
- **AutoML Search**: Automatically tests combinations of:
  - Heads: [4, 8]
  - Layers: [2, 3]
  - Optimizers: [Adam, AdamW]
  - Schedulers: [None, Cosine]
  - Learning Rates: [3e-4, 1e-3]

## Recommended Training Command

### Full Production Training (All Datasets)

**PowerShell:**
```powershell
uv run python GNN.py `
    --path "C:\Users\Admin\.cache\huggingface\datasets\codeparrot___codecomplex,C:\Users\Admin\.cache\huggingface\datasets\deepmind___code_contests,C:\Users\Admin\.cache\huggingface\datasets\justus27___deepcoder-train,C:\Users\Admin\.cache\huggingface\datasets\mbpp,C:\Users\Admin\.cache\huggingface\datasets\PrimeIntellect___deepcoder-gold-standard-solutions,C:\Users\Admin\.cache\huggingface\datasets\princeton-nlp___swe-bench_lite" `
    --arch gtr `
    --lappe `
    --rwse `
    --k 15 `
    --mutual `
    --epochs 1000 `
    --use_neighbor_loader `
    --fanout "15,10,5" `
    --save_model "gnn_outputs/model_best.pt" `
    --save_results "gnn_outputs/training_results.json"
```

**Bash/WSL:**
```bash
uv run python GNN.py \
    --path "C:\Users\Admin\.cache\huggingface\datasets\codeparrot___codecomplex,C:\Users\Admin\.cache\huggingface\datasets\deepmind___code_contests,C:\Users\Admin\.cache\huggingface\datasets\justus27___deepcoder-train,C:\Users\Admin\.cache\huggingface\datasets\mbpp,C:\Users\Admin\.cache\huggingface\datasets\PrimeIntellect___deepcoder-gold-standard-solutions,C:\Users\Admin\.cache\huggingface\datasets\princeton-nlp___swe-bench_lite" \
    --arch gtr \
    --lappe \
    --rwse \
    --k 15 \
    --mutual \
    --epochs 1000 \
    --use_neighbor_loader \
    --fanout "15,10,5" \
    --save_model "gnn_outputs/model_best.pt" \
    --save_results "gnn_outputs/training_results.json"
```

### Using the Production Script

**PowerShell:**
```powershell
cd koder
.\train_gnn_production.ps1
```

**Bash/WSL:**
```bash
cd koder
bash train_gnn_production.sh
```

## Command Arguments Explained

| Argument | Value | Purpose |
|----------|-------|---------|
| `--path` | Comma-separated dataset paths | Input datasets |
| `--arch` | `gtr` or `gatv2` | GraphTransformer (recommended) or GAT |
| `--lappe` | Flag | Laplacian Positional Encoding |
| `--rwse` | Flag | Random Walk Structural Encoding |
| `--k` | 15 | kNN neighbors for graph construction |
| `--mutual` | Flag | Mutual kNN (more stable) |
| `--epochs` | 1000 | Training steps |
| `--use_neighbor_loader` | Flag | Mini-batching for large graphs |
| `--fanout` | "15,10,5" | Neighbor sampling per layer |
| `--save_model` | Path | Save best model checkpoint |
| `--save_results` | Path | Save training metrics JSON |

## What Gets Saved

### Model Checkpoint (`--save_model`)
Contains:
- Encoder state dict
- Head state dict
- Architecture config
- Hyperparameters
- Task information

### Results JSON (`--save_results`)
Contains:
- Best validation metric
- Test metric
- Full hyperparameter config
- Dataset statistics
- Training configuration

## Training Process

1. **Data Loading**: Loads and combines all HuggingFace datasets
2. **Auto-detection**: Automatically detects labels, features, and task type
3. **Graph Construction**: Builds kNN graph from node features
4. **AutoML Search**: Tests hyperparameter combinations (quick search)
5. **Final Training**: Trains with best hyperparameters
6. **Evaluation**: Reports validation and test metrics
7. **Saving**: Saves model and results

## Evaluation Metrics

- **Classification**: Accuracy
- **Regression**: Negative MAE (higher is better)
- **Link Prediction**: AUC proxy

## Regularization Settings

The script uses:
- **Weight Decay**: `1e-4` (L2 regularization)
- **Dropout**: `0.2` (20% dropout in attention/conv layers)
- **Gradient Clipping**: `1.0` (max norm)
- **Layer Normalization**: After each layer

## Tips for Better Results

1. **Use GraphTransformer**: `--arch gtr` generally performs better
2. **Enable Positional Encodings**: `--lappe --rwse` helps with structural understanding
3. **Increase epochs**: For large datasets, use `--epochs 2000` or more
4. **Use neighbor loader**: `--use_neighbor_loader` for graphs with >10K nodes
5. **Tune k**: Higher `--k` values (e.g., 20-30) for sparser graphs
6. **Monitor validation**: Check validation metric trends in console output

## Expected Output

```
[info] device=cuda
[info] Combined 6 datasets into one with X total nodes
[choose] Best label = complexity (node_reg)
[choose] Best config: {'heads': 8, 'layers': 3, 'optim': 'adamw', 'sched': 'cosine', 'lr': 3e-4, 'val': 0.8234, 'test': 0.8101}

================ RESULT ================
Task: node_reg | Best Val: 0.8234 | Test -MAE (higher better): 0.8101
Config: heads=8 layers=3 optim=adamw sched=cosine lr=3e-4 arch=gtr edge_bias=no pos_enc=yes neighbor_loader=yes
=======================================
```

## Troubleshooting

- **Out of Memory**: Reduce `--k`, disable `--use_neighbor_loader`, or use smaller batch size
- **Poor Performance**: Try `--mutual` flag, increase `--epochs`, or enable positional encodings
- **Slow Training**: Enable `--use_neighbor_loader` for large graphs, use CUDA

