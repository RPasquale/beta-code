# slm.py - Model abstractions and core components
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable, Optional, Any, Callable
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim import AdamW
from transformers import AutoTokenizer


# =============================== MuonClip (QK-Clip) =============================== #
class _QKLogger:
    """Holds latest observed S_max per-module id for the current step."""
    def __init__(self):
        self._store: Dict[int, float] = {}

    @torch.no_grad()
    def update(self, module: nn.Module, smax: torch.Tensor):
        val = float(smax.max().detach().cpu())
        if val > self._store.get(id(module), float("-inf")):
            self._store[id(module)] = val

    def pop_all(self) -> Dict[int, float]:
        out = self._store
        self._store = {}
        return out

_QK = _QKLogger()


def _is_muzero_mha(m: nn.Module) -> bool:
    # Your attention block signature
    return (
        m.__class__.__name__ == "MultiHeadSelfAttention"
        and hasattr(m, "qkv") and isinstance(m.qkv, nn.Linear)
        and hasattr(m, "nhead") and hasattr(m, "d_head")
    )


def _wrap_muzero_mha_forward(attn: nn.Module):
    """
    Patch MultiHeadSelfAttention to log S_max. Recompute a minimal Q/K path
    (using attn.qkv) under no_grad, then delegate to original forward.
    """
    orig_forward = attn.forward
    if not hasattr(attn, "_muonclip_ctr"):
        attn._muonclip_ctr = 0
        attn._muonclip_log_every = getattr(attn, "_muonclip_log_every", 1)

    def wrapped_forward(x, *args, **kwargs):
        # optional downsample logging to reduce overhead
        attn._muonclip_ctr += 1
        try_log = (attn._muonclip_ctr % max(1, attn._muonclip_log_every)) == 0

        if try_log:
            with torch.no_grad():
                B, L, D = x.shape
                qkv = torch.nn.functional.linear(x, attn.qkv.weight, attn.qkv.bias)  # (B,L,3D)
                q, k, _ = torch.chunk(qkv, 3, dim=-1)
                # reshape to (B, nH, L, dH)
                q = q.view(B, L, attn.nhead, attn.d_head).transpose(1, 2)  # (B,H,L,dH)
                k = k.view(B, L, attn.nhead, attn.d_head).transpose(1, 2)
                logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(attn.d_head)  # (B,H,L,L)
                smax = logits.amax(dim=(-2, -1)).amax(dim=0)  # [H]
                _QK.update(attn, smax)

        return orig_forward(x, *args, **kwargs)

    return wrapped_forward


def patch_model_for_qk_logging(model: nn.Module,
                               selector: Optional[Callable[[nn.Module], bool]] = None):
    """Patch all MultiHeadSelfAttention modules to log S_max. Call once after model creation."""
    for m in model.modules():
        if selector is not None and not selector(m):
            continue
        if _is_muzero_mha(m) and not hasattr(m, "_muonclip_patched"):
            # Bind method properly to instance
            m.forward = _wrap_muzero_mha_forward(m)
            m._muonclip_patched = True


class MuonClip(Optimizer):
    """
    Wrap any base optimizer and apply global/naïve QK-Clip after each step.

    Args:
      params: iterable of parameters
      base_opt_ctor: optimizer class (default AdamW)
      base_opt_kwargs: kwargs for base optimizer (lr, betas, weight_decay, eps, ...)
      qk_tau: threshold τ (cap for max logits), default 100.0
      qk_alpha: exponent split α in (Wq *= γ^α, Wk *= γ^(1-α)), default 0.5
    """
    def __init__(self,
                 params: Iterable[nn.Parameter],
                 base_opt_ctor: Callable[..., Optimizer] = AdamW,
                 base_opt_kwargs: Optional[Dict[str, Any]] = None,
                 qk_tau: float = 100.0,
                 qk_alpha: float = 0.5):
        if base_opt_kwargs is None:
            base_opt_kwargs = {}
        self.base_opt: Optimizer = base_opt_ctor(params, **base_opt_kwargs)
        self.qk_tau = float(qk_tau)
        self.qk_alpha = float(qk_alpha)

        # Expose optimizer-like API
        self.param_groups = self.base_opt.param_groups
        self.state = self.base_opt.state

    def zero_grad(self, set_to_none: bool = False):
        return self.base_opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.base_opt.state_dict()

    def load_state_dict(self, state_dict):
        return self.base_opt.load_state_dict(state_dict)

    def step(self, closure=None):
        # Use step_with_qkclip(model) in your training loop.
        return self.base_opt.step(closure=closure)

    @torch.no_grad()
    def _apply_qkclip_on_model(self, model: nn.Module):
        smax_by_id = _QK.pop_all()
        if not smax_by_id:
            return

        tau = self.qk_tau
        a = self.qk_alpha

        for m in model.modules():
            mid = id(m)
            if mid not in smax_by_id:
                continue
            Smax = float(smax_by_id[mid])
            if Smax <= tau:
                continue

            gamma = min(1.0, tau / max(1e-6, Smax))
            q_scale = gamma ** a
            k_scale = gamma ** (1.0 - a)

            # Your attention: qkv is [3*D, D], bias [3*D]
            if _is_muzero_mha(m):
                D = m.qkv.out_features // 3
                # scale Q rows then K rows
                if m.qkv.weight is not None:
                    W = m.qkv.weight
                    W[:D].mul_(q_scale)
                    W[D:2*D].mul_(k_scale)
                if m.qkv.bias is not None:
                    b = m.qkv.bias
                    b[:D].mul_(q_scale)
                    b[D:2*D].mul_(k_scale)

    def step_with_qkclip(self, model: nn.Module, closure=None):
        out = self.base_opt.step(closure=closure)
        self._apply_qkclip_on_model(model)
        return out


# =============================== RoPE helpers =============================== #
def precompute_rope_freqs(head_dim: int, max_seq_len: int, base: float = 10000.0, device=None, dtype=None):
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # [T, Dh/2]
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)  # [T, Dh]
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)  # [T, Dh]
    return cos, sin

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # x: [B, H, L, Dh], cos/sin: [L, Dh] or [1,1,L,Dh]
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.cat([-x2, x1], dim=-1)
    return (x * cos) + (x_rot * sin)


def causal_mask(B: int, Lq: int, Lk: int, device, dtype=torch.bool):
    """Create causal attention mask: keep True where j <= i (lower-triangular)"""
    # keep True where j <= i  (lower-triangular)
    m = torch.ones(Lq, Lk, dtype=dtype, device=device).tril()
    # shape to [B, 1, Lq, Lk] for broadcasting over heads
    return m.unsqueeze(0).unsqueeze(1).expand(B, 1, Lq, Lk)


# ============================== Configuration ============================== #
@dataclass
class Config:
    model_name: str = "gpt2"
    vocab_topk: int = 2048       # increased vocab for better coverage
    max_seq_len: int = 128       # doubled sequence length
    latent_len: int = 16         # doubled latent representation
    d_model: int = 512           # 4x larger model dimension
    nhead: int = 8               # doubled attention heads
    n_layers_rep: int = 6        # 3x more representation layers
    n_layers_dyn: int = 3        # 3x more dynamics layers
    ff_mult: int = 4             # FFN hidden multiplier (kept same)
    unroll_K: int = 8            # increased unroll steps
    gamma: float = 0.99
    n_step_bootstrap: int = 8    # increased bootstrap steps

    batch_size: int = 32         # doubled batch size
    learning_rate: float = 1e-4  # slightly reduced for stability
    weight_decay: float = 1e-4
    replay_size: int = 20000     # 4x larger replay buffer
    train_steps: int = 5000      # 6x more training steps
    rollout_steps: int = 40      # doubled rollout length
    sims_per_move: int = 40      # doubled simulations
    expand_topk: int = 32        # doubled expansion
    train_samples: int = 10000   # 10x more training data
    eval_samples: int = 200      # 4x more eval samples
    eval_steps: int = 40         # doubled eval steps

    # Device
    device: str = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    # ===== Dr.GRPO aux training =====
    drgrpo_every: int = 10       # less frequent aux updates for stability
    drgrpo_batch: int = 16       # doubled batch size
    drgrpo_G: int = 8            # doubled rollouts per context
    drgrpo_T: int = 16           # doubled rollout length
    drgrpo_clip: float = 0.2     # PPO clip epsilon
    per_step_adv: bool = True    # per-step group baseline

    # Aux RL sampling / regularization
    aux_temp: float = 1.1        # sharper sampling for better focus
    aux_entropy_coef: float = 1e-3  # back up entropy coefficient for stability
    include_gt_in_aux: bool = True

    # Supervised warm-start (auxiliary CE on full vocab)
    bc_coef: float = 0.05        # much reduced BC coefficient to prevent explosion

    # Debug controls
    debug_level: int = 1         # 0 = quiet, 1 = summary per aux step, 2 = verbose per ctx
    debug_every: int = 50        # less frequent verbose logging
    seed: int = 42               # changed seed for reproducibility
    
    # Scaling-specific optimizations
    gradient_accumulation_steps: int = 2  # accumulate gradients for larger effective batch size
    warmup_steps: int = 100      # learning rate warmup
    save_every: int = 500        # save checkpoint every N steps
    log_every: int = 20          # log metrics every N steps
    use_mixed_precision: bool = True  # enable mixed precision training
    max_grad_norm: float = 1.0   # gradient clipping


# ============================== Model Components ============================== #
class SwiGLU(nn.Module):
    """SwiGLU activation function: Swish(xW + b) ⊙ (xV + c)"""
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)
        
    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, attn_dropout=0.0, proj_dropout=0.0, use_rope: bool = False):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.o = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)
        self.use_rope = use_rope

        # RoPE tables (set by parent model)
        self.register_buffer("rope_cos", None, persistent=False)
        self.register_buffer("rope_sin", None, persistent=False)

    def set_rope_tables(self, cos: torch.Tensor, sin: torch.Tensor):
        self.rope_cos = cos
        self.rope_sin = sin

    def forward(
        self,
        x: torch.Tensor,                       # [B,L,D]
        attn_mask: Optional[torch.Tensor] = None,
        *,
        position_ids: Optional[torch.Tensor] = None,  # [B,L] or [L]
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (Kpast,Vpast) [B,H,Tp,Dh]
        return_kv: bool = False
    ):
        B, L, D = x.shape
        qkv = self.qkv(x)  # (B, L, 3D)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # reshape to (B, H, L, Dh)
        q = q.view(B, L, self.nhead, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.nhead, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.nhead, self.d_head).transpose(1, 2)

        # ----- RoPE (Q & K only) -----
        if self.use_rope:
            assert self.rope_cos is not None and self.rope_sin is not None, "RoPE tables not set"
            if position_ids is None:
                pos = torch.arange(L, device=x.device)
            else:
                pos = position_ids
                if pos.dim() == 2:
                    pos = pos[0]
            cos = self.rope_cos[pos]  # [L, Dh]
            sin = self.rope_sin[pos]  # [L, Dh]
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)

        # ----- KV cache -----
        if past_kv is not None:
            Kpast, Vpast = past_kv  # [B,H,Tp,Dh]
            k = torch.cat([Kpast, k], dim=2)
            v = torch.cat([Vpast, v], dim=2)

        present_kv = (k, v)

        # ----- attention -----
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B,H,L,T)
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # (B,H,L,Dh)
        out = out.transpose(1, 2).contiguous().view(B, L, D)  # (B,L,D)
        out = self.o(out)
        out = self.proj_drop(out)
        return (out, present_kv) if return_kv else out


class FeedForward(nn.Module):
    def __init__(self, d_model, ff_mult=4, dropout=0.0):
        super().__init__()
        hidden = ff_mult * d_model
        self.swiglu = SwiGLU(d_model, hidden)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.dropout(self.swiglu(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, ff_mult=4, attn_dropout=0.0, resid_dropout=0.0, use_rope: bool = False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadSelfAttention(d_model, nhead, attn_dropout, resid_dropout, use_rope=use_rope)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, ff_mult, resid_dropout)

    def forward(self, x, attn_mask=None, *, position_ids=None, past_kv=None, return_kv: bool = False):
        h = self.ln1(x)
        mha_out = self.mha(h, attn_mask, position_ids=position_ids, past_kv=past_kv, return_kv=return_kv)
        if return_kv:
            h, present_kv = mha_out
        else:
            h = mha_out
            present_kv = None
        x = x + h
        h = self.ln2(x)
        h = self.ff(h)
        x = x + h
        return (x, present_kv) if return_kv else x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1,L,D)
    def forward(self, x):  # (B,L,D)
        return x + self.pe[:, :x.size(1), :]


class MuZeroTransformer(nn.Module):
    def __init__(self, vocab_size, cfg: Config):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model
        self.vocab_size = vocab_size
        self.latent_len = cfg.latent_len

        # Input embeddings
        self.tok_embed = nn.Embedding(vocab_size, D)
        self.cls = nn.Parameter(torch.randn(1,1,D))
        self.pos_rep = PositionalEncoding(D, max_len=cfg.max_seq_len + 1)
        self.rep_blocks = nn.ModuleList([
            TransformerBlock(D, cfg.nhead, cfg.ff_mult, use_rope=True) for _ in range(cfg.n_layers_rep)
        ])

        # Action + dynamics
        self.action_embed = nn.Embedding(vocab_size, D)
        self.pos_dyn = PositionalEncoding(D, max_len=cfg.latent_len + 2)
        self.dyn_blocks = nn.ModuleList([
            TransformerBlock(D, cfg.nhead, cfg.ff_mult, use_rope=True) for _ in range(cfg.n_layers_dyn)
        ])

        # Heads
        self.head_policy = nn.Linear(D, vocab_size)
        self.head_value  = nn.Sequential(nn.Linear(D, D), nn.SiLU(), nn.Linear(D, 1))
        self.head_reward = nn.Linear(D, 1)  # logits for BCE

        # ---- RoPE tables for rep & dyn stacks ----
        # Rep stack needs up to max_seq_len+1 (CLS + obs)
        cos_rep, sin_rep = precompute_rope_freqs(
            head_dim=D//self.rep_blocks[0].mha.nhead,
            max_seq_len=cfg.max_seq_len + 1,
            device=self.tok_embed.weight.device, dtype=self.tok_embed.weight.dtype
        )
        # Dyn stack: up to latent_len+2 (CLS+latent plus action slot)
        cos_dyn, sin_dyn = precompute_rope_freqs(
            head_dim=D//self.dyn_blocks[0].mha.nhead,
            max_seq_len=cfg.latent_len + 2,
            device=self.tok_embed.weight.device, dtype=self.tok_embed.weight.dtype
        )
        for blk in self.rep_blocks:
            blk.mha.set_rope_tables(cos_rep, sin_rep)
        for blk in self.dyn_blocks:
            blk.mha.set_rope_tables(cos_dyn, sin_dyn)

    # === original (training) APIs ===
    def h(self, obs_ids):  # (B,T)
        B = obs_ids.size(0)
        x = self.tok_embed(obs_ids)            # (B,T,D)
        cls = self.cls.expand(B,1,-1)          # (B,1,D)
        x = torch.cat([cls, x], dim=1)         # prepend CLS
        x = self.pos_rep(x)

        # positions for CLS..T (0..T) — for training we pass via kwargs but keep default behavior
        L = x.size(1)
        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        mask = causal_mask(B, L, L, x.device)  # causal

        for blk in self.rep_blocks:
            x = blk(x, attn_mask=mask, position_ids=pos_ids)  # causal self-attn

        # clamp length to 1+latent_len
        take = min(1 + self.latent_len, x.size(1))
        s0 = x[:, :take, :]
        if take < 1 + self.latent_len:
            pad = torch.zeros(B, 1 + self.latent_len - take, self.cfg.d_model, device=x.device, dtype=x.dtype)
            s0 = torch.cat([s0, pad], dim=1)
        return s0                               # (B, 1+L, D)

    def g(self, s_prev, a):                     # a: (B,)
        a_tok = self.action_embed(a).unsqueeze(1)  # (B,1,D)
        x = torch.cat([s_prev, a_tok], dim=1)      # ([CLS]+latent)+action
        x = self.pos_dyn(x)

        # positions for this stream start at 0
        B, L, _ = x.shape
        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        mask = causal_mask(B, L, L, x.device)

        for blk in self.dyn_blocks:
            x = blk(x, attn_mask=mask, position_ids=pos_ids)  # causal attn

        cls = x[:, :1, :]
        r_logit = self.head_reward(cls).squeeze(-1).squeeze(1)  # (B,)
        s_next = x[:, :1 + self.latent_len, :]
        return r_logit, s_next

    def f(self, s):
        cls = s[:, :1, :]
        p = self.head_policy(cls).squeeze(1)       # (B, V)
        v = self.head_value(cls).squeeze(-1).squeeze(1)  # (B,)
        return p, v

    # --------- new: cached prefill over rep stack ----------
    @torch.no_grad()
    def encode_prefill_with_cache(self, obs_ids: torch.Tensor):
        """
        obs_ids: [B,T]
        Returns:
          s0: [B, 1+latent_len, D]
          caches: list of (K,V) for each rep block
          pos_offset: next absolute position index for subsequent decode calls
        """
        B = obs_ids.size(0)
        x = self.tok_embed(obs_ids)            # (B,T,D)
        cls = self.cls.expand(B,1,-1)          # (B,1,D)
        x = torch.cat([cls, x], dim=1)         # prepend CLS
        x = self.pos_rep(x)

        L = x.size(1)
        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        mask = causal_mask(B, L, L, x.device)

        caches = []
        cur = x
        for blk in self.rep_blocks:
            cur, kv = blk(cur, attn_mask=mask, position_ids=pos_ids, past_kv=None, return_kv=True)
            caches.append(kv)

        take = min(1 + self.latent_len, cur.size(1))
        s0 = cur[:, :take, :]
        if take < 1 + self.latent_len:
            pad = torch.zeros(B, 1 + self.latent_len - take, self.cfg.d_model, device=cur.device, dtype=cur.dtype)
            s0 = torch.cat([s0, pad], dim=1)

        pos_offset = L  # next position index to use
        return s0, caches, pos_offset

    # --------- new: single-step decode over rep stack with cache ----------
    @torch.no_grad()
    def encode_decode_with_cache(self, last_token_ids: torch.Tensor, caches, pos_offset: int):
        """
        last_token_ids: [B] the newly appended token (not including CLS)
        caches: list of (K,V) from previous prefill/steps for rep blocks
        pos_offset: absolute position index for this token (>= previous L)

        Returns:
          new_hidden: [B, 1, D] representation for the new token position
          new_caches: updated caches
          pos_offset_next: pos_offset+1
        """
        B = last_token_ids.size(0)
        x = self.tok_embed(last_token_ids).unsqueeze(1)  # (B,1,D)
        x = self.pos_rep(x)  # add positional encoding (additive PE coexists with RoPE just fine here)

        pos_ids = torch.full((B,1), pos_offset, device=x.device, dtype=torch.long)
        mask = causal_mask(B, 1, pos_offset + 1, x.device)  # causal mask for new token

        new_caches = []
        cur = x
        for blk, kv in zip(self.rep_blocks, caches):
            cur, kv_new = blk(cur, attn_mask=mask, position_ids=pos_ids, past_kv=kv, return_kv=True)
            new_caches.append(kv_new)

        return cur, new_caches, (pos_offset + 1)


# ============================== Utility Functions ============================== #
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def topk_by_frequency(tokenized, tok, k):
    cnt = Counter()
    for ids in tokenized:
        cnt.update(ids)
    bad = {tok.eos_token_id} if tok.eos_token_id is not None else set()
    freq = [(tid, c) for tid, c in cnt.items() if tid not in bad]
    freq.sort(key=lambda x: x[1], reverse=True)
    return [tid for tid, _ in freq[:k]]


def grad_summary(model: nn.Module):
    total_norm = 0.0
    count_nonzero = 0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            total_norm += g.norm().item()
            count_nonzero += int((g.abs() > 0).sum().item())
    return total_norm, count_nonzero
