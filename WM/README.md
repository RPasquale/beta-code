# WM - World Model + Graph GRPO

This project uses the shared uv workspace from the parent directory.

## Setup

Since WM is part of the uv workspace (configured in `../pyproject.toml`), it will share the same virtual environment with the `koder` project.

### Option 1: Sync from workspace root (Recommended)
```powershell
cd C:\Users\Admin\beta-code
uv sync --package WM
```

### Option 2: Sync entire workspace
```powershell
cd C:\Users\Admin\beta-code
uv sync --all-packages
```

### Option 3: If you get permission errors
Close any Python processes/terminals that might be using the .venv, then:
```powershell
# Kill any Python processes
Get-Process python* | Stop-Process -Force

# Then sync
cd C:\Users\Admin\beta-code
uv sync --package WM
```

## Dependencies

The WM project requires:
- torch>=2.0.0
- transformers>=4.30.0
- numpy>=1.24.0
- scikit-learn>=1.3.0 (for ColBERT index)
- datasets>=2.12.0 (for HuggingFace datasets)
- tqdm>=4.65.0
- wandb>=0.15.0 (optional, for logging)
- peft>=0.5.0 (optional, for LoRA support)

These are already defined in `WM/pyproject.toml` and will be installed when you sync.

## Running Training

```powershell
cd C:\Users\Admin\beta-code\WM
python model.py code_train --dataset PrimeIntellect/deepcoder-gold-standard-solutions --epochs 3
```

