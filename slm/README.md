# SLM - Small Language Model with MuZero + Dr.GRPO

A modular implementation of a Small Language Model using MuZero planning and Dr.GRPO reinforcement learning.

## 📁 Structure

```
slm/
├── __init__.py          # Package initialization
├── slm.py              # Core model abstractions and components
├── train_slm.py        # Training functions and main training loop
└── inference_slm.py    # Inference with KV cache and beam search

slm_cli.py              # Easy CLI for inference
train_cli.py            # Easy CLI for training
```

## 🚀 Quick Start

### Training

```bash
# Train with default configuration
python train_cli.py

# Train with custom parameters
python train_cli.py --steps 10000 --batch_size 64 --d_model 256

# Train on specific device
python train_cli.py --device cuda --seed 123
```

### Inference

```bash
# Greedy generation
python slm_cli.py --model_path model.pth --prompt "The quick brown fox"

# Beam search generation
python slm_cli.py --model_path model.pth --prompt "Once upon a time" --method beam --beam_width 8

# Constrained generation with temperature
python slm_cli.py --model_path model.pth --prompt "The future of AI" --method constrained --temperature 0.8
```

## 🔧 Components

### Core Model (`slm.py`)
- **MuZeroTransformer**: Main model with representation, dynamics, and prediction heads
- **MuonClip**: QK-Clip optimizer for attention stability
- **SwiGLU**: Gated activation function
- **RoPE**: Rotary Position Embeddings
- **Causal Attention**: Proper causal masking for autoregressive generation

### Training (`train_slm.py`)
- **MuZero Loss**: Value, reward, policy, and behavior cloning losses
- **Dr.GRPO**: PPO-style auxiliary training for RL
- **Replay Buffer**: Experience replay for MuZero
- **Cosine Scheduler**: Learning rate warmup and decay
- **AMP Support**: Mixed precision training

### Inference (`inference_slm.py`)
- **KV Cache**: Efficient autoregressive generation
- **Beam Search**: High-quality text generation
- **Constrained Generation**: Top-k vocabulary sampling
- **Multiple Methods**: Greedy, beam search, and constrained sampling

## ⚙️ Configuration

The `Config` class in `slm.py` contains all hyperparameters:

```python
@dataclass
class Config:
    # Model architecture
    d_model: int = 512
    nhead: int = 8
    n_layers_rep: int = 6
    n_layers_dyn: int = 3
    
    # Training
    train_steps: int = 5000
    batch_size: int = 32
    learning_rate: float = 1e-4
    
    # Data
    vocab_topk: int = 2048
    max_seq_len: int = 128
    train_samples: int = 10000
    
    # And many more...
```

## 🎯 Key Features

### 1. **Causal Attention**
- Proper autoregressive masking prevents future token leakage
- Essential for correct next-token prediction

### 2. **MuZero Planning**
- MCTS-style search for better action selection
- Value and reward prediction heads
- Multi-step unrolling for temporal credit assignment

### 3. **Dr.GRPO Auxiliary Training**
- PPO-style RL for improved policy
- Entropy regularization for exploration
- Per-step and per-group advantage computation

### 4. **MuonClip Stabilization**
- QK-Clip prevents attention explosion
- Automatic scaling of query and key weights
- Maintains training stability

### 5. **Efficient Inference**
- KV cache for fast autoregressive generation
- Beam search for high-quality outputs
- Multiple sampling strategies

## 📊 Training Process

1. **Data Loading**: WikiText-2 dataset with top-k vocabulary
2. **Episode Collection**: Short rollouts using MCTS planning
3. **MuZero Updates**: Value, reward, policy, and BC losses
4. **Dr.GRPO Updates**: Auxiliary PPO training every N steps
5. **Evaluation**: Accuracy on held-out sequences

## 🔍 Usage Examples

### Programmatic Usage

```python
from slm import SLMInference, Config

# Load trained model
inference = SLMInference("model.pth")

# Generate text
result = inference.greedy_generate("The quick brown fox", max_new_tokens=50)
print(result)

# Beam search
result = inference.beam_search_generate(
    "Once upon a time", 
    beam_width=8, 
    max_new_tokens=100
)
```

### Training Programmatically

```python
from slm.train_slm import main as train_main
from slm.slm import Config, set_seed

# Set configuration
config = Config()
config.train_steps = 10000
config.batch_size = 64

# Set seed and train
set_seed(42)
train_main()
```

## 🎛️ CLI Options

### Training CLI (`train_cli.py`)
- `--steps`: Number of training steps
- `--batch_size`: Batch size
- `--learning_rate`: Learning rate
- `--device`: Device (cpu/cuda/mps)
- `--d_model`: Model dimension
- `--nhead`: Attention heads
- `--train_samples`: Training data size

### Inference CLI (`slm_cli.py`)
- `--model_path`: Path to model checkpoint
- `--prompt`: Input text prompt
- `--method`: Generation method (greedy/beam/constrained)
- `--max_tokens`: Maximum new tokens
- `--beam_width`: Beam width for beam search
- `--temperature`: Sampling temperature
- `--top_k`: Top-k for constrained generation

## 🏗️ Architecture

The model uses a dual-stack architecture:

1. **Representation Stack**: Processes input observations
2. **Dynamics Stack**: Predicts next states and rewards
3. **Prediction Heads**: Policy and value estimation

This design enables both supervised learning (behavior cloning) and reinforcement learning (MuZero + Dr.GRPO) for improved text generation.

## 📈 Performance

The model achieves:
- Stable training with gradient norms < 5.0
- High accuracy on next-token prediction
- Efficient inference with KV caching
- Quality text generation with beam search

## 🔧 Requirements

- PyTorch >= 2.0
- Transformers
- Datasets
- tqdm
- CUDA/MPS support (optional)

## 📝 Notes

- The model uses WikiText-2 for training
- Top-k vocabulary reduces computational cost
- Causal attention is crucial for correct autoregressive generation
- MuonClip prevents attention explosion in deeper models
- Dr.GRPO provides RL-based policy improvement
