# UE-SEA Training Pipeline for RTX 4090

Complete training pipeline for UE-SEA (Unified Evolutionary Software Engineering Agent) optimized for RTX 4090 (24GB VRAM).

## Overview

This training pipeline provides:
- Safe training scripts optimized for RTX 4090
- Real-time monitoring tools
- Crash recovery and safety features
- Multiple monitoring interfaces (console, web)
- Complete UE-SEA system implementation

## Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv ue4090_env
ue4090_env\Scripts\activate  # Windows
# source ue4090_env/bin/activate  # Linux/Mac

# Install dependencies
pip install --upgrade pip
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.43.3 accelerate==0.33.0 datasets==2.20.0 peft==0.11.1 bitsandbytes==0.43.1 trl==0.9.6
pip install einops sentencepiece tiktoken tree_sitter==0.21.3 whoosh fastapi uvicorn pydantic pytest coverage hypothesis
```

### 2. Check System Requirements
```bash
python test_simple.py
```

### 3. Download Training Data
```bash
python scripts/download_datasets.py --data_dir data
```

### 4. Start Training
```bash
# Option 1: Training with monitoring
python start_monitoring.py --mode training

# Option 2: Manual training (with 10% validation holdout)
python train.py --output-dir outputs/unified_training --validation-ratio 0.1
```
The run writes training metrics to `outputs/unified_training/training_metrics.jsonl`,
validation results to `outputs/unified_training/eval_results.json`, and streams both
to Weights & Biases when `WANDB_MODE` is enabled.

### 5. Monitor Training
```bash
# Console monitoring
python monitor/simple_monitor.py --output_dir outputs/unified_training

# Web dashboard (requires: pip install fastapi uvicorn)
python monitor/web_dashboard.py --output_dir outputs/unified_training
# Open browser to http://localhost:8000
```

## System Requirements

### Hardware
- GPU: RTX 4090 (24GB VRAM) - Required
- CPU: 16GB RAM (recommended 32GB)
- Disk: 50GB free space
- OS: Windows/Linux with CUDA 12.1

### Software
- Python 3.11+
- CUDA 12.1
- NVIDIA drivers

## Training Configuration

### Memory Budget (Safe for 24GB)
```
RTX 4090 (24GB):
- Model (4-bit): ~8GB
- Gradients: ~4GB
- Optimizer: ~2GB
- Activations: ~4GB
- Buffer: ~2GB
- Safety margin: ~4GB
```

### Training Settings
- Model: Qwen2.5-Coder-3B (3B parameters)
- Quantization: 4-bit QLoRA (memory efficient)
- Batch Size: 1 (safe for 24GB)
- Sequence Length: 512 tokens (conservative)
- Learning Rate: 1.5e-4 (stable)

## Monitoring Tools

### Console Monitor
Real-time system monitoring with GPU/CPU metrics:
```bash
python monitor/simple_monitor.py --output_dir outputs/unified_training
```

Features:
- GPU memory usage and temperature
- CPU usage and memory
- Training progress tracking
- Automatic warnings for high usage

### Web Dashboard
Beautiful web interface with real-time updates:
```bash
pip install fastapi uvicorn
python monitor/web_dashboard.py --output_dir outputs/unified_training
# Open http://localhost:8000
```

Features:
- Real-time charts and progress bars
- Mobile-responsive design
- Historical data visualization
- WebSocket real-time updates

## Safety Features

### Automatic Warnings
- GPU Temperature > 80C: Overheating warning
- GPU Memory > 20GB: High memory usage
- CPU Usage > 90%: High CPU load
- Training Stopped: Unexpected training halt

### Memory Management
- Real-time monitoring of VRAM usage
- Automatic alerts when approaching limits
- Temperature monitoring to prevent overheating
- Process tracking to detect training status

### Crash Recovery
- Automatic checkpoint saving every 10 steps
- Recovery from training interruptions
- Progress tracking and state restoration

## File Structure

```
training/
├── README.md                    # This file
├── train.py                     # Unified training script
├── test_simple.py               # System requirements test
├── start_monitoring.py          # Monitoring launcher
├── configs/
│   └── sft_safe_4090.yaml       # Training configuration
├── monitor/
│   ├── simple_monitor.py        # Console monitoring
│   ├── web_dashboard.py         # Web dashboard
│   └── training_dashboard.py    # Advanced dashboard
├── sft/
│   └── train_diff_safe.py       # Safe training script
├── utils/
│   ├── memory_monitor.py        # Memory monitoring
│   ├── temperature_monitor.py   # Temperature monitoring
│   └── crash_recovery.py        # Crash recovery
├── scripts/
│   ├── download_datasets.py     # Dataset downloader
│   └── safe_training_launcher.py # Safe launcher
├── data/
│   ├── processed/               # Training datasets
│   └── raw/                     # Raw datasets
└── outputs/                     # Training outputs
```

## Training Scripts

### Main Training Script
`train.py` — unified entry point for every UE-SEA objective:
- Loads and balances all datasets (diff, evolutionary, reasoning, RL, AlphaCode)
- Applies QLoRA adapters to Qwen2.5-Coder-3B
- Streams metrics to JSONL + Weights & Biases
- Tracks GPU/CPU telemetry every training step
- Supports automatic validation splits and writes `eval_results.json` with loss/perplexity

### Safe Training Script
`sft/train_diff_safe.py` - Advanced safety features:
- Memory monitoring
- Temperature monitoring
- Crash recovery
- Automatic batch size adjustment

## Monitoring Scripts

### Console Monitor
`monitor/simple_monitor.py` - Lightweight monitoring:
- Real-time GPU/CPU metrics
- Training progress tracking
- Automatic warnings
- No dependencies required

### Web Dashboard
`monitor/web_dashboard.py` - Web interface:
- Beautiful responsive design
- Real-time charts and graphs
- Historical data visualization
- Mobile-friendly

## Validation & Evaluation

- Control the held-out split per objective with `--validation-ratio` (default `0.1`).
- Limit validation workload via `--eval-max-samples-per-objective` when running smoke tests.
- After training finishes, metrics are written to `outputs/<run>/eval_results.json` and logged to Weights & Biases (`eval_loss`, `eval_perplexity`, runtime stats).
- Set `--validation-ratio 0` to disable validation entirely when you only want to train.

## Expected Results

### Training Times
- Small test (3 samples): 5-10 minutes
- Full training (10K samples): 2-4 hours
- Complete pipeline: 12-20 hours

### Models Generated
- Diff Model: Generates SEARCH/REPLACE diffs
- Tools Model: Uses SearchEntity/TraverseGraph/RetrieveEntity
- Tests Model: Generates test cases
- RL Model: Improved code quality through reinforcement

## Troubleshooting

### Common Issues

**1. "nvidia-smi not found"**
- Install NVIDIA drivers
- Or use CPU-only monitoring

**2. "FastAPI not available"**
```bash
pip install fastapi uvicorn
```

**3. "Out of Memory"**
- Reduce batch size in training script
- Reduce sequence length
- Clear GPU cache: restart Python

**4. "GPU Overheating"**
- Stop training: Press Ctrl+C
- Wait for cooldown: 10-15 minutes
- Check cooling system
- Resume training: will continue from checkpoint

### Performance Tips

**1. Reduce Update Frequency**
```bash
python monitor/simple_monitor.py --interval 5
```

**2. Monitor Specific Directory**
```bash
python monitor/simple_monitor.py --output_dir outputs/sft_diff
```

**3. Web Dashboard on Network**
```bash
python monitor/web_dashboard.py --host 0.0.0.0 --port 8080
```

## Configuration

### Training Configuration
Edit `configs/sft_safe_4090.yaml`:
- Model name
- Batch size
- Learning rate
- Sequence length
- Memory limits

### Monitoring Configuration
Edit monitoring scripts:
- Update intervals
- Warning thresholds
- Output directories
- Host/port settings

## Success Checklist

Before starting:
- GPU memory >= 20GB available
- GPU temperature < 80C
- CPU memory >= 16GB available
- Environment activated
- Dependencies installed

During training:
- GPU temperature < 85C
- GPU memory usage < 90%
- No OOM errors in logs
- Checkpoints saving every 10 steps
- Training loss decreasing

## Next Steps

1. Start with small test: `python train.py --max-samples-per-objective 4`
2. Monitor progress: `python monitor/simple_monitor.py`
3. Scale up: increase samples and model size
4. Deploy: use trained models for UE-SEA system

## Support

If you encounter issues:
1. Check logs in `outputs/*/logs/`
2. Monitor system resources
3. Verify requirements are met
4. Try reducing batch size if OOM occurs
5. Check temperature if training stops

The system is designed to be crash-proof and self-recovering, but monitoring is still important for optimal performance.
