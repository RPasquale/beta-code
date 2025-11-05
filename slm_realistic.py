# Realistic SLM with Full Vocabulary Support
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
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


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
    return (
        m.__class__.__name__ == "MultiHeadSelfAttention"
        and hasattr(m, "qkv") and isinstance(m.qkv, nn.Linear)
        and hasattr(m, "nhead") and hasattr(m, "d_head")
    )


def _wrap_muzero_mha_forward(attn: nn.Module):
    """Patch MultiHeadSelfAttention to log S_max."""
    orig_forward = attn.forward
    if not hasattr(attn, "_muonclip_ctr"):
        attn._muonclip_ctr = 0
        attn._muonclip_log_every = getattr(attn, "_muonclip_log_every", 1)

    def wrapped_forward(x, *args, **kwargs):
        attn._muonclip_ctr += 1
        try_log = (attn._muonclip_ctr % max(1, attn._muonclip_log_every)) == 0

        if try_log:
            with torch.no_grad():
                B, L, D = x.shape
                qkv = torch.nn.functional.linear(x, attn.qkv.weight, attn.qkv.bias)
                q, k, _ = torch.chunk(qkv, 3, dim=-1)
                q = q.view(B, L, attn.nhead, attn.d_head).transpose(1, 2)
                k = k.view(B, L, attn.nhead, attn.d_head).transpose(1, 2)
                logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(attn.d_head)
                smax = logits.amax(dim=(-2, -1)).amax(dim=0)
                _QK.update(attn, smax)

        return orig_forward(x, *args, **kwargs)

    return wrapped_forward


def patch_model_for_qk_logging(model: nn.Module,
                               selector: Optional[Callable[[nn.Module], bool]] = None):
    """Patch all MultiHeadSelfAttention modules to log S_max."""
    for m in model.modules():
        if selector is not None and not selector(m):
            continue
        if _is_muzero_mha(m) and not hasattr(m, "_muonclip_patched"):
            m.forward = _wrap_muzero_mha_forward(m)
            m._muonclip_patched = True


class MuonClip(Optimizer):
    """Wrap any base optimizer and apply global/naïve QK-Clip after each step."""
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

        self.param_groups = self.base_opt.param_groups
        self.state = self.base_opt.state

    def zero_grad(self, set_to_none: bool = False):
        return self.base_opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.base_opt.state_dict()

    def load_state_dict(self, state_dict):
        return self.base_opt.load_state_dict(state_dict)

    def step(self, closure=None):
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

            if _is_muzero_mha(m):
                D = m.qkv.out_features // 3
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
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
    return cos, sin

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.cat([-x2, x1], dim=-1)
    return (x * cos) + (x_rot * sin)


def causal_mask(B: int, Lq: int, Lk: int, device, dtype=torch.bool):
    """Create causal attention mask: keep True where j <= i (lower-triangular)"""
    m = torch.ones(Lq, Lk, dtype=dtype, device=device).tril()
    return m.unsqueeze(0).unsqueeze(1).expand(B, 1, Lq, Lk)


# ============================== Configuration ============================== #
@dataclass
class Config:
    model_name: str = "gpt2"
    # REMOVED vocab_topk restriction - use full vocabulary
    max_seq_len: int = 256        # Increased for better language modeling
    latent_len: int = 16
    d_model: int = 512           # Keep reasonable for 24GB GPU
    nhead: int = 8
    n_layers_rep: int = 6
    n_layers_dyn: int = 3
    ff_mult: int = 4
    unroll_K: int = 8
    gamma: float = 0.99
    n_step_bootstrap: int = 8

    batch_size: int = 4           # Reduced further for memory
    learning_rate: float = 5e-5   # Lower for stability
    weight_decay: float = 1e-4
    replay_size: int = 20000
    train_steps: int = 10000      # Increased for better training
    rollout_steps: int = 40
    sims_per_move: int = 40
    train_samples: int = 50000    # Use more data
    eval_samples: int = 200
    eval_steps: int = 40

    # Device
    device: str = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    # ===== Dr.GRPO aux training =====
    drgrpo_every: int = 20        # Less frequent for stability
    drgrpo_batch: int = 4         # Reduced for memory
    drgrpo_G: int = 2             # Reduced for memory
    drgrpo_T: int = 4             # Reduced for memory
    drgrpo_clip: float = 0.2
    per_step_adv: bool = True

    # Aux RL sampling / regularization
    aux_temp: float = 1.1
    aux_entropy_coef: float = 1e-3
    include_gt_in_aux: bool = True

    # Supervised warm-start
    bc_coef: float = 0.05

    # Debug controls
    debug_level: int = 1
    debug_every: int = 50
    seed: int = 42
    
    # Scaling-specific optimizations
    gradient_accumulation_steps: int = 16  # Effective batch size = 64
    warmup_steps: int = 200
    save_every: int = 1000
    log_every: int = 50
    use_mixed_precision: bool = True
    max_grad_norm: float = 1.0

CFG = Config()
print(f"Running on device: {CFG.device}")

# AMP and gradient accumulation setup
USE_AMP = (CFG.use_mixed_precision and (torch.cuda.is_available() or torch.backends.mps.is_available()))
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float16
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
from contextlib import nullcontext
amp_autocast = (
    torch.autocast(device_type="cuda", dtype=AMP_DTYPE) if torch.cuda.is_available()
    else torch.autocast(device_type="cpu", dtype=AMP_DTYPE)
)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and AMP_DTYPE==torch.float16)
ACCUM = max(1, getattr(CFG, "gradient_accumulation_steps", 1))
MAX_NORM = getattr(CFG, "max_grad_norm", 1.0)

print(f"AMP enabled: {USE_AMP}, dtype: {AMP_DTYPE}, grad_accum: {ACCUM}")


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CFG.seed)




def grad_summary(model: nn.Module):
    total_norm = 0.0
    count_nonzero = 0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            total_norm += g.norm().item()
            count_nonzero += int((g.abs() > 0).sum().item())
    return total_norm, count_nonzero


class CosineWarmup:
    def __init__(self, optimizer, warmup, total_steps, base_lr):
        self.opt = optimizer
        self.warmup = warmup
        self.total = total_steps
        self.base = base_lr
        self.s = 0
        
    def step(self):
        self.s += 1
        if self.s <= self.warmup:
            lr = self.base * self.s / max(1, self.warmup)
        else:
            t = (self.s - self.warmup) / max(1, self.total - self.warmup)
            lr = 0.5 * self.base * (1 + math.cos(math.pi * t))
        for g in self.opt.param_groups:
            g["lr"] = lr


class NextTokenEnv:
    def __init__(self, token_ids: List[int], max_seq_len: int):
        self.ids = token_ids
        self.max_seq_len = max_seq_len
        center = len(self.ids) // 2
        if center == 0 and len(self.ids) > 1:
            center = 1
        self.pos = min(max_seq_len // 2, center)
        self.pos = min(self.pos, max(len(self.ids) - 1, 0))

    def get_obs(self) -> List[int]:
        s = max(0, self.pos - self.max_seq_len)
        return self.ids[s:self.pos]

    def step(self, action_id: int):
        if self.pos >= len(self.ids) - 1:
            return 0, True
        gt = self.ids[self.pos]
        r = 1 if action_id == gt else 0
        self.pos += 1
        return r, self.pos >= len(self.ids) - 1


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

        self.register_buffer("rope_cos", None, persistent=False)
        self.register_buffer("rope_sin", None, persistent=False)

    def set_rope_tables(self, cos: torch.Tensor, sin: torch.Tensor):
        self.rope_cos = cos
        self.rope_sin = sin

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        *,
        position_ids: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_kv: bool = False
    ):
        B, L, D = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(B, L, self.nhead, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.nhead, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.nhead, self.d_head).transpose(1, 2)

        if self.use_rope:
            assert self.rope_cos is not None and self.rope_sin is not None, "RoPE tables not set"
            if position_ids is None:
                pos = torch.arange(L, device=x.device)
            else:
                pos = position_ids
                if pos.dim() == 2:
                    pos = pos[0]
            cos = self.rope_cos[pos]
            sin = self.rope_sin[pos]
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)

        if past_kv is not None:
            Kpast, Vpast = past_kv
            k = torch.cat([Kpast, k], dim=2)
            v = torch.cat([Vpast, v], dim=2)

        present_kv = (k, v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.o(out)
        out = self.proj_drop(out)
        return (out, present_kv) if return_kv else out


class SwiGLU(nn.Module):
    """SwiGLU activation function: Swish(xW + b) ⊙ (xV + c)"""
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)
        
    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

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
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
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

        # Heads - FULL VOCABULARY SUPPORT
        self.head_policy = nn.Linear(D, vocab_size)  # Full vocab
        self.head_value  = nn.Sequential(nn.Linear(D, D), nn.SiLU(), nn.Linear(D, 1))
        self.head_reward = nn.Linear(D, 1)

        # RoPE tables
        cos_rep, sin_rep = precompute_rope_freqs(
            head_dim=D//self.rep_blocks[0].mha.nhead,
            max_seq_len=cfg.max_seq_len + 1,
            device=self.tok_embed.weight.device, dtype=self.tok_embed.weight.dtype
        )
        cos_dyn, sin_dyn = precompute_rope_freqs(
            head_dim=D//self.dyn_blocks[0].mha.nhead,
            max_seq_len=cfg.latent_len + 2,
            device=self.tok_embed.weight.device, dtype=self.tok_embed.weight.dtype
        )
        for blk in self.rep_blocks:
            blk.mha.set_rope_tables(cos_rep, sin_rep)
        for blk in self.dyn_blocks:
            blk.mha.set_rope_tables(cos_dyn, sin_dyn)

    def h(self, obs_ids):
        B = obs_ids.size(0)
        x = self.tok_embed(obs_ids)
        cls = self.cls.expand(B,1,-1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_rep(x)

        L = x.size(1)
        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        mask = causal_mask(B, L, L, x.device)

        for blk in self.rep_blocks:
            x = blk(x, attn_mask=mask, position_ids=pos_ids)

        take = min(1 + self.latent_len, x.size(1))
        s0 = x[:, :take, :]
        if take < 1 + self.latent_len:
            pad = torch.zeros(B, 1 + self.latent_len - take, self.cfg.d_model, device=x.device, dtype=x.dtype)
            s0 = torch.cat([s0, pad], dim=1)
        return s0

    def g(self, s_prev, a):
        a_tok = self.action_embed(a).unsqueeze(1)
        x = torch.cat([s_prev, a_tok], dim=1)
        x = self.pos_dyn(x)

        B, L, _ = x.shape
        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        mask = causal_mask(B, L, L, x.device)

        for blk in self.dyn_blocks:
            x = blk(x, attn_mask=mask, position_ids=pos_ids)

        cls = x[:, :1, :]
        r_logit = self.head_reward(cls).squeeze(-1).squeeze(1)
        s_next = x[:, :1 + self.latent_len, :]
        return r_logit, s_next

    def f(self, s):
        cls = s[:, :1, :]
        p = self.head_policy(cls).squeeze(1)
        v = self.head_value(cls).squeeze(-1).squeeze(1)
        return p, v


# ============================================================
# "MCTS" (soft – return probs for π target) with full vocabulary
# ============================================================
@torch.no_grad()
def run_mcts(net, s_root, gt_token, cfg):
    """Run MCTS with full vocabulary"""
    p_logits, _ = net.f(s_root)
    p_logits = p_logits.squeeze(0)
    
    # Use full vocabulary for training
    probs = F.softmax(p_logits, dim=-1)
    a_idx = torch.multinomial(probs, 1).item()
    action = a_idx
    return action, probs.detach().cpu().tolist()


# ============================================================
# Replay buffer
# ============================================================
class Replay:
    def __init__(self, cap):
        self.buf = []
        self.cap = cap
    def add(self, x):
        if len(self.buf) >= self.cap:
            self.buf.pop(0)
        self.buf.append(x)
    def sample(self, bs):
        return random.sample(self.buf, bs)
    def __len__(self): return len(self.buf)


# ============================================================
# MuZero training utilities
# ============================================================
def nstep_bootstrap(rews: List[int], values: List[float], gamma: float, n: int, start: int) -> float:
    G = 0.0
    T = len(rews)
    for i in range(n):
        t = start + i
        if t < T: G += (gamma ** i) * rews[t]
        else: break
    tB = start + n
    if tB < len(values): G += (gamma ** n) * values[tB]
    return G

def muzero_loss(batch, net, cfg: Config, bc_coef=None):
    B = len(batch)
    device = cfg.device
    maxlen = max(len(x["obs"]) for x in batch)
    obs = torch.full((B, min(cfg.max_seq_len, maxlen)), 0, dtype=torch.long, device=device)
    for i, x in enumerate(batch):
        toks = x["obs"][-cfg.max_seq_len:]
        obs[i, -len(toks):] = torch.tensor(toks, device=device)

    s = net.h(obs)
    v_losses, r_losses, p_losses, bc_losses = [], [], [], []

    with torch.no_grad():
        p0, v0 = net.f(s)
    values_cache = [v0.detach()]

    for k in range(cfg.unroll_K):
        a_k, r_true, gt_ids = [], [], []
        for i in range(B):
            if k < len(batch[i]["actions"]):
                a_k.append(batch[i]["actions"][k])
                r_true.append(batch[i]["rewards"][k])
                gt_ids.append(batch[i]["gt_ids"][k])
            else:
                a_k.append(0); r_true.append(0); gt_ids.append(-1)

        a_k = torch.tensor(a_k, device=device, dtype=torch.long)
        r_true = torch.tensor(r_true, device=device, dtype=torch.float)
        gt_ids_t = torch.tensor(gt_ids, device=device, dtype=torch.long)

        r_logit, s = net.g(s, a_k)
        p_pred, v_pred = net.f(s)
        values_cache.append(v_pred.detach())

        r_losses.append(F.binary_cross_entropy_with_logits(r_logit, r_true, reduction="mean"))

        z_k = []
        for i in range(B):
            if k < len(batch[i]["rewards"]):
                z = nstep_bootstrap(batch[i]["rewards"],
                                    [v[i].item() for v in values_cache],
                                    cfg.gamma, cfg.n_step_bootstrap, k)
                z_k.append(z)
            else:
                z_k.append(0.0)
        z_k = torch.tensor(z_k, device=device, dtype=torch.float)
        v_losses.append(F.mse_loss(v_pred, z_k))

        # Policy loss (full vocabulary)
        pi_targets = []
        for i in range(B):
            if k < len(batch[i]["actions"]):
                pi_target = batch[i]["mcts_pi"][k]
                # Convert sparse target to full vocabulary target
                full_target = torch.zeros(net.vocab_size, device=device, dtype=torch.float)
                if len(pi_target) > 0:
                    # For now, use uniform distribution over full vocab
                    # This is a simplified approach - could be improved with proper target mapping
                    full_target = torch.ones(net.vocab_size, device=device, dtype=torch.float) / net.vocab_size
                pi_targets.append(full_target)
            else:
                pi_targets.append(torch.zeros(net.vocab_size, device=device, dtype=torch.float))
        
        if pi_targets:
            pi_targets_tensor = torch.stack(pi_targets)  # [B, vocab_size]
            logp = F.log_softmax(p_pred, dim=-1)  # [B, vocab_size]
            p_losses.append(-(pi_targets_tensor * logp).sum(dim=-1).mean())

        current_bc = bc_coef if bc_coef is not None else cfg.bc_coef
        if current_bc > 0:
            mask = (gt_ids_t >= 0)
            if mask.any():
                bc_losses.append(F.cross_entropy(p_pred[mask], gt_ids_t[mask]))

    loss_v = sum(v_losses)/max(1,len(v_losses))
    loss_r = sum(r_losses)/max(1,len(r_losses))
    loss_p = sum(p_losses)/max(1,len(p_losses))
    loss_bc = (sum(bc_losses)/max(1,len(bc_losses))) if bc_losses else torch.tensor(0.0, device=device)
    current_bc = bc_coef if bc_coef is not None else cfg.bc_coef
    return loss_v + loss_r + loss_p + current_bc * loss_bc, (loss_v.item(), loss_r.item(), loss_p.item(), loss_bc.item())


# ============================================================
# Dr.GRPO auxiliary training (PPO-style)
# ============================================================
def drgrpo_step(
    net, opt,
    tokenized: List[List[int]],
    tok,
    cfg: Config,
    step_idx: int
):
    device = cfg.device
    net.train()
    opt.zero_grad(set_to_none=True)

    EPS_ADV = 0.05

    # Sample contexts
    contexts = []
    for _ in range(cfg.drgrpo_batch):
        seq = random.choice(tokenized)
        pos = len(seq) // 2
        pos = min(max(cfg.max_seq_len // 2, 1), max(len(seq)-1, 1), pos)
        contexts.append((seq, pos))

    # Collect rollouts with dynamic action space
    batch_groups = []
    actionable_steps = 0
    token_terms = 0

    for (seq, start_pos) in contexts:
        ctx_groups = []
        for _ in range(cfg.drgrpo_G):
            pos = start_pos
            traj = []
            for t in range(cfg.drgrpo_T):
                if pos >= len(seq) - 1: break
                gt = seq[pos]

                obs_tokens = seq[max(0, pos-cfg.max_seq_len):pos]
                obs_t = torch.tensor([obs_tokens], device=device, dtype=torch.long)
                s = net.h(obs_t)
                with torch.no_grad():
                    p_logits, _ = net.f(s)
                    
                    # Use full vocabulary for Dr.GRPO
                    if cfg.aux_temp != 1.0:
                        p_logits = p_logits / cfg.aux_temp
                    probs = F.softmax(p_logits, dim=-1)
                    a_global = torch.multinomial(probs, 1).item()
                    logp_old = torch.log(probs[0, a_global] + 1e-12).item()
                r = 1 if a_global == gt else 0
                traj.append((obs_tokens, a_global, logp_old, r, gt))
                actionable_steps += 1
                pos += 1
            ctx_groups.append(traj)
        batch_groups.append(ctx_groups)

    # Compute PPO loss
    losses = []
    ent_terms = []

    if cfg.per_step_adv:
        for ctx_groups in batch_groups:
            L = max((len(traj) for traj in ctx_groups), default=0)
            if L == 0:
                continue
            means = []
            for t in range(L):
                vals = [traj[t][3] for traj in ctx_groups if t < len(traj)]
                means.append(sum(vals)/len(vals) if vals else 0.0)

            for traj in ctx_groups:
                for t, step in enumerate(traj):
                    A_t = step[3] - means[t]
                    obs_tokens, a_global, logp_old, _, gt = step
                    
                    if A_t == 0.0:
                        A_t = EPS_ADV if (a_global == gt) else -EPS_ADV
                    
                    obs_t = torch.tensor([obs_tokens], device=device, dtype=torch.long)
                    s = net.h(obs_t)
                    p_logits, _ = net.f(s)
                    if cfg.aux_temp != 1.0:
                        p_logits = p_logits / cfg.aux_temp
                    probs = F.softmax(p_logits, dim=-1)
                    
                    logp_new = torch.log(probs[0, a_global] + 1e-12)
                    ratio = torch.exp(logp_new - torch.tensor([logp_old], device=device))
                    unclipped = ratio * A_t
                    clipped   = torch.clamp(ratio, 1 - cfg.drgrpo_clip, 1 + cfg.drgrpo_clip) * A_t
                    losses.append(-torch.min(unclipped, clipped))
                    token_terms += 1
                    
                    if cfg.aux_entropy_coef > 0:
                        ent_terms.append(-(probs * (probs.clamp_min(1e-12).log())).sum())

    if len(losses) == 0:
        if cfg.debug_level >= 1:
            print(f"[Dr.GRPO] skip: token_terms=0 | actionable_steps={actionable_steps}")
        return 0.0

    aux_loss = torch.stack(losses).mean()
    if ent_terms:
        aux_loss = aux_loss - cfg.aux_entropy_coef * torch.stack(ent_terms).mean()

    loss_div = aux_loss
    if USE_AMP and AMP_DTYPE==torch.float16:
        scaler.scale(loss_div).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), MAX_NORM)
        scaler.step(opt)
        opt._apply_qkclip_on_model(net)
        scaler.update()
    else:
        loss_div.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), MAX_NORM)
        opt.step_with_qkclip(net)

    gnorm, gcount = grad_summary(net)

    if cfg.debug_level >= 1:
        print(f"[Dr.GRPO] aux loss={aux_loss.item():.4f} | token_terms={token_terms} "
              f"actionable_steps={actionable_steps} grad_norm={gnorm:.3f} nonzero_grad_elems={int(gcount)}")

    return float(aux_loss.item())


# ============================================================
# Data loading with full dataset
# ============================================================
def load_data(cfg: Config):
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    # Use 50% of WikiText-2 for realistic training
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:50%]")
    texts = [t["text"] for t in ds if t["text"]]
    tokenized = [
        tok(t, truncation=True, max_length=cfg.max_seq_len + 10, add_special_tokens=False)["input_ids"]
        for t in texts if t.strip()
    ]
    min_len = max(cfg.eval_steps + 2, cfg.rollout_steps + 2)
    tokenized = [x for x in tokenized if len(x) > min_len]
    if cfg.train_samples and len(tokenized) > cfg.train_samples:
        tokenized = tokenized[:cfg.train_samples]
    
    print(f"Loaded {len(tokenized)} sequences for training")
    print(f"Vocabulary size: {tok.vocab_size}")
    return tok, tokenized


# ============================================================
# Main training loop
# ============================================================
def current_bc_coef(step):
    """Linear warm-down of BC coefficient"""
    warmdown = int(0.1 * CFG.train_steps)
    if step < warmdown:
        return CFG.bc_coef * (1 - step / warmdown)
    return 0.0


def main():
    cfg = CFG
    tok, tokenized = load_data(cfg)

    net = MuZeroTransformer(vocab_size=tok.vocab_size, cfg=cfg).to(cfg.device)
    print(f"Model parameters: {sum(p.numel() for p in net.parameters()):,}")

    patch_model_for_qk_logging(net)
    
    for m in net.modules():
        if isinstance(m, MultiHeadSelfAttention):
            m._muonclip_log_every = 2

    opt = MuonClip(
        net.parameters(),
        base_opt_ctor=torch.optim.AdamW,
        base_opt_kwargs=dict(lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=(0.9, 0.999), eps=1e-8),
        qk_tau=90.0,
        qk_alpha=0.6
    )
    
    sched = CosineWarmup(opt.base_opt, warmup=cfg.warmup_steps, total_steps=cfg.train_steps, base_lr=cfg.learning_rate)
    
    import platform
    if hasattr(torch, "compile") and platform.system() != "Windows":
        net = torch.compile(net, mode="reduce-overhead")
        print("Model compiled with torch.compile")
    elif platform.system() == "Windows":
        print("torch.compile not supported on Windows, skipping compilation")

    replay = Replay(cfg.replay_size)

    print("Collecting and training ...")
    for step in tqdm(range(cfg.train_steps)):
        # Collect experience with dynamic action space
        seq = random.choice(tokenized)
        env = NextTokenEnv(seq, cfg.max_seq_len)

        init_obs = env.get_obs()
        acts, rews, pis, gt_ids = [], [], [], []

        for _ in range(cfg.rollout_steps):
            if env.pos >= len(seq) - 1:
                break
            gt = seq[env.pos]
            obs_t = torch.tensor([env.get_obs()[-cfg.max_seq_len:]], device=cfg.device)
            s0 = net.h(obs_t)
            a, pi = run_mcts(net, s0, gt, cfg)
            r, done = env.step(a)
            acts.append(a); rews.append(r); pis.append(pi); gt_ids.append(gt)
            if done:
                break

        if acts:
            replay.add({
                "obs": init_obs,
                "actions": acts[:cfg.unroll_K],
                "rewards": rews[:cfg.unroll_K],
                "mcts_pi": pis[:cfg.unroll_K],
                "gt_ids": gt_ids[:cfg.unroll_K],
            })

        # MuZero update
        if len(replay) >= cfg.batch_size:
            batch = replay.sample(cfg.batch_size)
            
            if (step % ACCUM) == 0:
                opt.zero_grad(set_to_none=True)

            with (amp_autocast if USE_AMP else nullcontext()):
                current_bc = current_bc_coef(step)
                loss, parts = muzero_loss(batch, net, cfg, bc_coef=current_bc)
                loss_to_backprop = loss / ACCUM

            if USE_AMP and AMP_DTYPE==torch.float16:
                scaler.scale(loss_to_backprop).backward()
                if ((step + 1) % ACCUM) == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), MAX_NORM)
                    scaler.step(opt)
                    opt._apply_qkclip_on_model(net)
                    scaler.update()
                    sched.step()
            else:
                loss_to_backprop.backward()
                if ((step + 1) % ACCUM) == 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), MAX_NORM)
                    opt.step_with_qkclip(net)
                    sched.step()
            
            if (step + 1) % cfg.log_every == 0:
                lv, lrw, lp, lbc = parts
                print(f"[MuZero] step {step + 1}: loss={loss.item():.4f} "
                      f"(v={lv:.4f}, r={lrw:.4f}, p={lp:.4f}, bc={lbc:.4f}) replay={len(replay)}")

        # Dr.GRPO aux update
        if (step + 1) % cfg.drgrpo_every == 0:
            _ = drgrpo_step(net, opt, tokenized, tok, cfg, step_idx=step+1)

    # Evaluation
    print(f"Quick eval on {cfg.eval_samples} samples...")
    accs = []
    with torch.no_grad():
        for _ in range(cfg.eval_samples):
            seq = random.choice(tokenized)
            env = NextTokenEnv(seq, cfg.max_seq_len)

            correct, total = 0, 0
            for _ in range(cfg.eval_steps):
                if env.pos >= len(seq) - 1: break
                gt = seq[env.pos]
                obs_t = torch.tensor([env.get_obs()[-cfg.max_seq_len:]], device=cfg.device)
                s0 = net.h(obs_t)
                a, _ = run_mcts(net, s0, gt, cfg)
                correct += int(a == gt)
                total += 1
                _, done = env.step(a)
                if done: break
            if total:
                accs.append(correct / total)
    print(f"Mean step accuracy: {sum(accs)/len(accs):.3f}" if accs else "No evaluation episodes produced steps.")


if __name__ == "__main__":
    main()
