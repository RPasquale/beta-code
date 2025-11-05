# Using HuggingFace Datasets with GNN.py

The GNN.py script now supports loading HuggingFace datasets directly from your cache directory or by repository name.

## Supported Datasets

You can use any of these datasets:

1. `codeparrot___codecomplex` - Code complexity dataset
2. `deepmind___code_contests` - Code contest problems
3. `justus27___deepcoder-train` - DeepCoder training data
4. `mbpp` - Mostly Basic Python Problems
5. `PrimeIntellect___deepcoder-gold-standard-solutions` - Gold standard solutions
6. `princeton-nlp___swe-bench_lite` - SWE-Bench lite

## Usage Examples

### Single Dataset

```bash
uv run python GNN.py --path "C:\Users\Admin\.cache\huggingface\datasets\mbpp"
```

### Multiple Datasets (Comma-separated)

```bash
uv run python GNN.py --path "C:\Users\Admin\.cache\huggingface\datasets\mbpp,C:\Users\Admin\.cache\huggingface\datasets\codeparrot___codecomplex"
```

### All Your Datasets at Once

```bash
uv run python GNN.py --path "C:\Users\Admin\.cache\huggingface\datasets\codeparrot___codecomplex,C:\Users\Admin\.cache\huggingface\datasets\deepmind___code_contests,C:\Users\Admin\.cache\huggingface\datasets\justus27___deepcoder-train,C:\Users\Admin\.cache\huggingface\datasets\mbpp,C:\Users\Admin\.cache\huggingface\datasets\PrimeIntellect___deepcoder-gold-standard-solutions,C:\Users\Admin\.cache\huggingface\datasets\princeton-nlp___swe-bench_lite"
```

### With Additional Options

```bash
# Use GraphTransformer with positional encodings
uv run python GNN.py \
  --path "C:\Users\Admin\.cache\huggingface\datasets\mbpp" \
  --arch gtr \
  --lappe \
  --rwse \
  --label complexity

# Link prediction task
uv run python GNN.py \
  --path "C:\Users\Admin\.cache\huggingface\datasets\codeparrot___codecomplex" \
  --link_pred \
  --k 15
```

## How It Works

1. **Auto-detection**: The script automatically detects HuggingFace cache directories by looking for `.arrow` files or `dataset_dict.json`

2. **Feature Extraction**: The script automatically extracts:
   - Code/solution text
   - Problem/prompt text
   - Complexity/difficulty labels (if available)
   - Code length, prompt length as features

3. **Graph Construction**: 
   - Each code sample/problem becomes a node
   - Edges are created via kNN based on code/prompt similarity
   - Or you can provide explicit edge files

4. **Multiple Datasets**: When multiple datasets are provided, they're automatically combined into a single graph, with a `dataset_name` column to track which dataset each node came from.

## Requirements

Make sure you have the `datasets` library installed:

```bash
uv sync  # This will install datasets>=2.12.0 from pyproject.toml
```

## Troubleshooting

If a dataset fails to load:
- Check that the path exists and contains `.arrow` files or `dataset_dict.json`
- Try loading by repository name instead: `--path "mbpp"` (if the dataset is available on HuggingFace Hub)
- Check that the dataset has at least one of: code/solution columns, prompt/problem columns

