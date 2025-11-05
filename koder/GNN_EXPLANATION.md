# Understanding GNN Training Pipeline

## What is `build_homo_data` execution?

`build_homo_data` is the **data preparation phase** that happens BEFORE actual training. It's taking 50+ minutes because it's doing expensive operations:

### Phase 1: Label Candidate Selection (Lines 1412-1457)
**Purpose**: Automatically pick the best label column to predict

**What it does:**
1. **Finds candidate labels** (e.g., "complexity", "difficulty", "task_type")
2. **For each candidate** (up to 4 candidates):
   - Builds node features from all columns
   - **BUILDS kNN GRAPH** ⚠️ **VERY SLOW** (50K nodes × k=15 = ~750K similarity computations)
   - Runs **60 training steps** to see which label works best
   - Evaluates validation score

**Bottleneck**: With 50,631 nodes, building kNN graph takes ~10-15 minutes EACH TIME, and it does this 4 times (once per candidate) = **40-60 minutes just for graph building**

### Phase 2: Final Graph Construction (Lines 1459-1475)
**Purpose**: Build the final graph with positional encodings

**What it does:**
1. Rebuilds node features (fast)
2. **Rebuilds kNN graph AGAIN** (another 10-15 minutes)
3. **Computes Laplacian PE** ⚠️ **EXTREMELY SLOW**:
   - Builds dense 50K × 50K Laplacian matrix (2.5 billion elements!)
   - Computes eigen decomposition
   - For 50K nodes, this can take 30+ minutes or fail
4. Computes RWSE (skipped for graphs >4000 nodes)

### Phase 3: AutoML Hyperparameter Search (Lines 1522-1539)
**Purpose**: Find best hyperparameters

**What it does:**
- Tests 2×2×2×2×2 = **32 combinations** of:
  - Heads: [4, 8]
  - Layers: [2, 3]
  - Optimizers: [Adam, AdamW]
  - Schedulers: [None, Cosine]
  - Learning rates: [3e-4, 1e-3]
- Each runs **200 training steps**

### Phase 4: Final Training (Lines 1630-1638)
**Purpose**: Train with best hyperparameters

**What it does:**
- Runs **1000 training steps** (or your `--epochs` value)
- Uses best hyperparameters from AutoML

## Why It's So Slow

1. **kNN graph construction**: O(n²) or O(n×k) - done multiple times
2. **Laplacian PE**: O(n³) eigen decomposition - fails/skips for large graphs
3. **Multiple training loops**: 4 candidates × 60 steps + 32 AutoML × 200 steps + 1000 final = thousands of steps

## Solution: Optimize for Large Graphs

I'll optimize this by:
- Building kNN graph ONCE and caching it
- Skipping Laplacian PE for graphs >5000 nodes (it's already disabled >5000)
- Reducing candidate evaluation steps
- Using faster graph construction methods

