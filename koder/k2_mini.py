# k2_mini.py
# A faithful miniature of Kimi K2 architecture in PyTorch:
# - MoE Transformer with Top-k (k=8), one shared expert, SwiGLU, RMSNorm
# - MLA-style attention with head-specific (q_c, k_c, q_r) + shared k_r (rotary)
# - QK-Clip: per-head scaling of Q/K weights when max attention logit exceeds tau
# References: Kimi K2 Technical Report (Table 2 p.6 for arch; §2.1 & Alg.1 pp.3–4 for QK-Clip; §3.2.3 for RL)
# NOTE: This is a scaled-down, code-faithful miniature intended for 1–3B activated parameters.

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Utilities
# -------------------------

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        # x: (B, T, C)
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x

def build_rope_cache(head_dim: int, max_seq_len: int, base: float = 10000.0, device=None, dtype=torch.float32):
    # Classic RoPE cache for even head_dim (q_r, k_r apply rotary)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # (T, head_dim/2)
    cos = torch.cos(freqs)  # (T, D/2)
    sin = torch.sin(freqs)  # (T, D/2)
    return cos, sin

def apply_rope(x, cos, sin):
    # x: (B, T, H, D)
    B, T, H, D = x.shape
    x_ = x.view(B, T, H, D // 2, 2)
    x1, x2 = x_[..., 0], x_[..., 1]
    cos = cos[:T, :].view(1, T, 1, -1)
    sin = sin[:T, :].view(1, T, 1, -1)
    # rotary: [x1 * cos - x2 * sin, x1 * sin + x2 * cos]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    out = torch.stack([out1, out2], dim=-1).view(B, T, H, D)
    return out

# -------------------------
# MLA-style Attention
# -------------------------

class MLAAttention(nn.Module):
    """
    MLA-style attention with:
      - head-specific q_c, k_c (clipped by sqrt(gamma))
      - head-specific q_r (rotary, clipped by gamma)
      - shared k_r (rotary; NOT clipped per K2 paper note)
    K2 QK-Clip (§2.1) says in MLA clip unshared head parts q_c,k_c by sqrt(gamma), q_r by gamma, and leave shared k_r untouched.
    """
    def __init__(self, d_model: int, n_heads: int, head_dim: int, rope_max_len: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim

        # head-specific projections
        self.Wq_c = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.Wk_c = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.Wq_r = nn.Linear(d_model, n_heads * head_dim, bias=False)   # rotary, per-head
        # shared rotary K
        self.Wk_r_shared = nn.Linear(d_model, head_dim, bias=False)      # rotary, shared across heads

        self.Wv = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.Wo = nn.Linear(n_heads * head_dim, d_model, bias=False)

        self.register_buffer("rope_cos", None, persistent=False)
        self.register_buffer("rope_sin", None, persistent=False)
        self.rope_max_len = rope_max_len

        # runtime cache for QK-Clip: store last-step per-head max logits
        self.last_max_logits: Optional[torch.Tensor] = None  # (n_heads,)

    def maybe_init_rope(self, device, dtype):
        if self.rope_cos is None or self.rope_cos.device != device or self.rope_cos.dtype != dtype:
            cos, sin = build_rope_cache(self.head_dim, self.rope_max_len, device=device, dtype=dtype)
            self.rope_cos = cos
            self.rope_sin = sin

    def forward(self, x, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, T, C)
        mask: (B, 1, T, T) additive mask (0 for keep, -inf for block)
        """
        B, T, C = x.shape
        self.maybe_init_rope(x.device, x.dtype)

        q_c = self.Wq_c(x).view(B, T, self.n_heads, self.head_dim)      # head-specific
        k_c = self.Wk_c(x).view(B, T, self.n_heads, self.head_dim)
        q_r = self.Wq_r(x).view(B, T, self.n_heads, self.head_dim)      # head-specific rotary
        k_r_shared = self.Wk_r_shared(x).view(B, T, 1, self.head_dim).expand(B, T, self.n_heads, self.head_dim)  # shared

        # Apply rotary to q_r and k_r_shared
        q_r = apply_rope(q_r, self.rope_cos, self.rope_sin)
        k_r = apply_rope(k_r_shared, self.rope_cos, self.rope_sin)

        # Compose final Q,K and V
        Q = q_c + q_r                         # (B,T,H,D)
        K = k_c + k_r                         # (B,T,H,D)
        V = self.Wv(x).view(B, T, self.n_heads, self.head_dim)

        # scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_logits = torch.einsum("bthd,bshd->bhts", Q, K).float() * scale  # (B,H,T,T)
        # record per-head max logit for QK-Clip guidance
        with torch.no_grad():
            max_per_head = attn_logits.amax(dim=(0, 2, 3))  # (H,)
            self.last_max_logits = max_per_head.detach()

        if mask is not None:
            attn_logits = attn_logits + mask.to(attn_logits.dtype)  # additive mask

        attn = F.softmax(attn_logits, dim=-1)
        out = torch.einsum("bhts,bshd->bthd", attn, V.float()).contiguous()  # (B,T,H,D)
        out = out.to(V.dtype)
        out = out.view(B, T, self.n_heads * self.head_dim)
        return self.Wo(out)

    # QK-Clip hook: rescale weights after optimizer step
    @torch.no_grad()
    def qk_clip_step(self, tau: float):
        """
        If last_max_logits[h] > tau, scale:
            q_c[h] *= sqrt(gamma),  k_c[h] *= sqrt(gamma),  q_r[h] *= gamma,
        where gamma = tau / S_h_max. Do NOT touch k_r_shared per K2.
        """
        if self.last_max_logits is None:
            return
        H = self.n_heads
        device = self.Wq_c.weight.device
        S = self.last_max_logits.to(device)  # (H,)
        # gamma per head
        gamma = torch.minimum(torch.ones_like(S), tau / (S + 1e-6))  # min(1, tau / S_h)
        # Only adjust heads where S_h > tau (i.e., gamma < 1)
        needs = (gamma < 0.9999)
        if not needs.any():
            return

        # reshape matrices to (H, D_in, D_head) view so we can scale per-head slices
        def per_head_scale_linear(linear: nn.Linear, scale: torch.Tensor, sqrt: bool = False):
            w = linear.weight  # (out, in)
            out, din = w.shape
            assert out == H * self.head_dim
            w = w.view(H, self.head_dim, din)  # (H, D_h, D_in)
            factor = scale.sqrt() if sqrt else scale
            w[needs] *= factor[needs].view(-1, 1, 1)
            linear.weight.copy_(w.view(H * self.head_dim, din))

        # q_c & k_c by sqrt(gamma)
        per_head_scale_linear(self.Wq_c, gamma, sqrt=True)
        per_head_scale_linear(self.Wk_c, gamma, sqrt=True)
        # q_r by gamma
        per_head_scale_linear(self.Wq_r, gamma, sqrt=False)
        # DO NOT touch self.Wk_r_shared

# -------------------------
# MoE (Top-k, one shared expert, SwiGLU)
# -------------------------

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ff = SwiGLU(d_model, d_ff)
    def forward(self, x):
        return self.ff(x)

class TopKRouter(nn.Module):
    def __init__(self, d_model, n_experts, k_active):
        super().__init__()
        self.n_experts = n_experts
        self.k = k_active
        self.w = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x):
        # x: (B, T, C)
        logits = self.w(x)  # (B,T,E)
        topk_val, topk_idx = torch.topk(logits, self.k, dim=-1)  # (B,T,k)
        gates = F.softmax(topk_val, dim=-1)  # routed softmax among top-k
        return topk_idx, gates, logits

class MoE(nn.Module):
    """
    MoE feed-forward:
      - N experts + 1 shared expert (index 0 reserved for shared)
      - Top-k routing (expert load-balancing aux loss)
    """
    def __init__(self, d_model, n_experts, k_active, d_ff, use_shared_expert=True):
        super().__init__()
        self.n_experts = n_experts + (1 if use_shared_expert else 0)
        self.k = k_active
        self.use_shared = use_shared_expert
        self.shared_idx = 0 if use_shared_expert else None

        self.router = TopKRouter(d_model, self.n_experts, self.k)
        # Experts: index 0 is shared, the rest are normal
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(self.n_experts)])

    def forward(self, x):
        # x: (B,T,C)
        B, T, C = x.shape
        topk_idx, gates_init, logits = self.router(x)  # gates_init unused (BF16)
        logits_fp32 = logits.float()
        topk_val = torch.gather(logits_fp32, -1, topk_idx)
        gates = F.softmax(topk_val, dim=-1)
        gates = torch.nan_to_num(gates, nan=0.0, posinf=0.0, neginf=0.0).to(x.dtype)  # (B,T,k)

        # Auxiliary load balance loss (GShard-style)
        probs = F.softmax(logits_fp32, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)  # (B,T,E)
        me = probs.mean(dim=(0,1))                     # usage per expert
        ce = (probs > (1.0 / self.n_experts)).float().mean(dim=(0,1))
        aux_loss = (me * ce).sum() * self.n_experts  # keep in fp32 for stability

        # Dispatch
        out = torch.zeros_like(x)
        for i in range(self.k):
            idx = topk_idx[..., i]         # (B,T)
            gate = gates[..., i].unsqueeze(-1)  # (B,T,1)
            # Gather tokens for expert calls
            # Efficient implementations use token-->expert permutation. Simpler version here:
            chunk_out = torch.zeros_like(x)
            for e in range(self.n_experts):
                sel = (idx == e)                 # (B,T)  <-- note: 2D mask
                if sel.any():
                    # select tokens for this expert: (N, C)
                    x_sel = x[sel]
                    y = self.experts[e](x_sel)   # (N, C)
                    if y.dtype != chunk_out.dtype:
                        y = y.to(chunk_out.dtype)
                    # write results back into the same token positions
                    chunk_out[sel] = y
            out += gate * chunk_out

        return out, aux_loss

# -------------------------
# Transformer Block
# -------------------------

class K2Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = MLAAttention(cfg.d_model, cfg.n_heads, cfg.head_dim, cfg.rope_max_len)
        self.ff_norm = RMSNorm(cfg.d_model)
        self.moe = MoE(cfg.d_model, cfg.n_experts_total, cfg.top_k, cfg.d_ff, use_shared_expert=True)

    def forward(self, x, mask=None):
        h = x + self.attn(self.attn_norm(x), mask=mask)
        ff, aux = self.moe(self.ff_norm(h))
        h = h + ff
        return h, aux

    @torch.no_grad()
    def qk_clip_step(self, tau: float):
        self.attn.qk_clip_step(tau)

# -------------------------
# Model
# -------------------------

@dataclass
class K2Config:
    vocab_size: int = 32000
    d_model: int = 2048
    n_heads: int = 32
    head_dim: int = 64
    n_layers: int = 24
    n_experts_total: int = 96     # without the "shared" we add internally
    top_k: int = 8
    d_ff: int = 9216              # SwiGLU hidden
    rope_max_len: int = 4096
    tie_word_embeddings: bool = False
    qk_clip_tau: float = 100.0    # per paper default τ=100 used in large run

class K2Mini(nn.Module):
    def __init__(self, cfg: K2Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([K2Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.tok_emb.weight

    def forward(self, input_ids, attention_mask=None, labels=None):
        # input_ids: (B,T), attention_mask: (B,T) -> causal mask combined
        B, T = input_ids.shape
        x = self.tok_emb(input_ids)   # (B,T,C)
        # build causal mask (additive, -inf above diagonal)
        causal = torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype).triu(1)
        causal = causal.unsqueeze(0).unsqueeze(0)  # (1,1,T,T)
        if attention_mask is not None:
            # convert to additive (0 keep / -inf mask)
            attn = (1 - attention_mask[:, None, None, :]).to(x.dtype) * float("-inf")
            mask = causal + attn
        else:
            mask = causal

        aux_losses = []
        for block in self.blocks:
            x, aux = block(x, mask=mask)
            aux_losses.append(aux)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                                   labels[:, 1:].contiguous().view(-1),
                                   ignore_index=-100)
        return logits, (torch.stack(aux_losses).mean() if aux_losses else torch.tensor(0.0, device=x.device)), loss

    @torch.no_grad()
    def qk_clip_step(self, tau: float):
        for blk in self.blocks:
            blk.qk_clip_step(tau)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=128, temperature=0.8, top_p=0.95):
        self.eval()
        out = input_ids
        for _ in range(max_new_tokens):
            logits, _, _ = self.forward(out)
            next_logits = logits[:, -1, :]
            if temperature > 0:
                next_logits = next_logits / temperature
                probs = F.softmax(next_logits, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                    cum = torch.cumsum(sorted_probs, dim=-1)
                    cutoff = (cum - sorted_probs) > top_p
                    sorted_probs = torch.where(cutoff, torch.zeros_like(sorted_probs), sorted_probs)
                    denom = sorted_probs.sum(dim=-1, keepdim=True)
                    valid = denom.squeeze(-1) > 1e-8
                    next_token = torch.empty((probs.size(0), 1), dtype=torch.long, device=probs.device)
                    if valid.any():
                        norm = sorted_probs[valid] / denom[valid]
                        sampled = torch.multinomial(norm, num_samples=1)
                        next_token[valid] = sorted_idx[valid].gather(-1, sampled)
                    if (~valid).any():
                        next_token[~valid] = torch.argmax(probs[~valid], dim=-1, keepdim=True)
                else:
                    next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            out = torch.cat([out, next_token], dim=1)
        return out

    def logprobs(self, input_ids):
        logits, _, _ = self.forward(input_ids)
        logits = logits.float()
        logp = F.log_softmax(logits[:, :-1, :], dim=-1)
        tgt = input_ids[:, 1:].unsqueeze(-1)
        token_logp = logp.gather(-1, tgt).squeeze(-1)  # (B,T-1)
        return token_logp

# -------------------------
# MuonClip "wrapper": base optimizer + QK-Clip
# -------------------------

class MuonLikePreconditioner:
    """
    VERY light Muon-like preconditioning via Newton-Schulz matrix sign on 2D params.
    Disabled by default; enable via use_muon_like=True for experiments (heavy).
    """
    def __init__(self, iters=5):
        self.iters = iters

    @torch.no_grad()
    def msign(self, M):
        # M: (out, in). Compute sign(M) ~ M * (M^T M)^-1/2
        if M.ndim != 2:
            return M
        MtM = M.t().mm(M)
        # Newton-Schulz for inverse sqrt
        I = torch.eye(MtM.size(0), device=MtM.device, dtype=MtM.dtype)
        Y = MtM / MtM.norm()  # scale for stability
        Z = I.clone()
        for _ in range(self.iters):
            T = 0.5 * (3*I - Z.mm(Y))
            Y = Y.mm(T)
            Z = T.mm(Z)
        inv_sqrt = Z / math.sqrt(MtM.norm() + 1e-12)
        return M.mm(inv_sqrt)

class MuonClip(torch.optim.Optimizer):
    """
    Wrap any base optimizer (e.g., AdamW) and apply K2-style QK-Clip (Algorithm 1 §2.1).
    Optionally apply a very light Muon-like preconditioning (off by default).
    """
    def __init__(self, model: K2Mini, base_optim: torch.optim.Optimizer, tau: float = 100.0,
                 use_muon_like: bool = False, muon_match_scale: float = 0.2):
        self.model = model
        self.base_optim = base_optim
        self.tau = tau
        self.use_muon_like = use_muon_like
        self.muon = MuonLikePreconditioner() if use_muon_like else None
        self.muon_match_scale = muon_match_scale
        # state needed to emulate momentum buffer for muon-like (optional)
        self.momentum: Dict[int, torch.Tensor] = {}

    def zero_grad(self, set_to_none: bool = False):
        self.base_optim.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self):
        # Optional: apply Muon-like preconditioning by modifying .grad in-place
        if self.use_muon_like:
            for group in self.base_optim.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    if g.ndim == 2:
                        # simple momentum
                        st = self.momentum.setdefault(id(p), torch.zeros_like(g))
                        st.mul_(0.9).add_(g)
                        g_pre = self.muon.msign(st) * self.muon_match_scale
                        p.grad.copy_(g_pre)

        # Take base optimizer step
        loss = self.base_optim.step()

        # Apply QK-Clip to every attention module
        self.model.qk_clip_step(self.tau)
        return loss
