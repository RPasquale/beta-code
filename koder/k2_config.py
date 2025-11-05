# k2_config.py
from k2_mini import K2Config  # uses the same dataclass your model imports. :contentReference[oaicite:2]{index=2}

# Faithful-but-fast for a single 4090 (Top-2 active experts)
k2_4090_fast = K2Config(
    vocab_size=32768,
    d_model=1024,
    n_heads=16,
    head_dim=64,          # 16*64 = 1024
    n_layers=12,
    n_experts_total=8,    # +1 shared internally => 9 experts total per layer
    top_k=2,              # Top-2 routing to keep compute low
    d_ff=2048,            # SwiGLU hidden per expert
    rope_max_len=4096,
    qk_clip_tau=100.0,
)

# Closer to the paper’s Top-8 (heavier compute), still 4090-friendly if you keep batch/K small
k2_4090_top8 = K2Config(
    vocab_size=32768,
    d_model=1024,
    n_heads=16,
    head_dim=64,
    n_layers=12,
    n_experts_total=8,    # +1 shared => 9 experts
    top_k=8,              # Top-8 routing
    d_ff=2048,
    rope_max_len=4096,
    qk_clip_tau=100.0,
)
