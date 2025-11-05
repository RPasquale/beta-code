# slm package - Small Language Model with MuZero + Dr.GRPO
from .slm import (
    Config, MuZeroTransformer, MuonClip, MultiHeadSelfAttention,
    SwiGLU, FeedForward, TransformerBlock, PositionalEncoding,
    patch_model_for_qk_logging, set_seed, topk_by_frequency, grad_summary,
    causal_mask, precompute_rope_freqs, apply_rotary_pos_emb
)

from .train_slm import (
    NextTokenEnv, Replay, CosineWarmup, current_bc_coef,
    run_mcts, nstep_bootstrap, muzero_loss, drgrpo_step, load_data, main as train_main
)

from .inference_slm import (
    Hypothesis, SLMInference
)

__version__ = "1.0.0"
__all__ = [
    # Core model components
    "Config", "MuZeroTransformer", "MuonClip", "MultiHeadSelfAttention",
    "SwiGLU", "FeedForward", "TransformerBlock", "PositionalEncoding",
    
    # Training components
    "NextTokenEnv", "Replay", "CosineWarmup", "current_bc_coef",
    "run_mcts", "nstep_bootstrap", "muzero_loss", "drgrpo_step", "load_data",
    
    # Inference components
    "Hypothesis", "SLMInference",
    
    # Utilities
    "patch_model_for_qk_logging", "set_seed", "topk_by_frequency", "grad_summary",
    "causal_mask", "precompute_rope_freqs", "apply_rotary_pos_emb",
    
    # Main functions
    "train_main"
]
