# k2_all_in_one.py
# Single-file training: K2-mini (MoE + MLA + QK-Clip) + SFT warm-start + RL with Verifiable Rewards.
# Auto-detects CUDA / MPS / CPU. Includes numerics guards to avoid NaN/Inf.
#
# Usage:
#   python k2_all_in_one.py
#   python k2_all_in_one.py --preset top8 --steps 400 --data-path "C:\...\deepcoder-cache-dir"
#
# Dependencies: torch, datasets
#   pip install torch datasets

from __future__ import annotations

import os, glob, math, json, random, argparse, subprocess, sys, tempfile, textwrap, time
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from datasets import load_from_disk, load_dataset, Dataset, concatenate_datasets

# -------------------------
# Device / Precision setup
# -------------------------

def pick_device(preferred: Optional[str] = None) -> str:
    """
    Pick the compute device.

    preferred:
        - "auto" (or None): use CUDA if available, then MPS, else CPU.
        - Explicit device string such as "cpu", "cuda", "cuda:1", "mps".
    """
    if preferred is not None and preferred.lower() == "auto":
        preferred = None

    if preferred:
        normalized = preferred.lower()
        if normalized.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
            if ":" in normalized:
                index_str = normalized.split(":", 1)[1]
                if index_str:
                    index = int(index_str)
                    if index < 0 or index >= torch.cuda.device_count():
                        raise RuntimeError(f"CUDA device index {index} is out of range.")
                    torch.cuda.set_device(index)
            return "cuda"
        if normalized == "mps":
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
            raise RuntimeError("MPS requested but not available.")
        if normalized == "cpu":
            return "cpu"
        raise RuntimeError(f"Unknown device requested: {preferred}")

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_autocast_dtype(device: str):
    # Force full precision to maximize numerical stability across devices.
    return None  # autocast disabled


def format_seconds(seconds: float) -> str:
    """Render durations with coarse-friendly precision."""
    seconds = max(0.0, float(seconds))
    if seconds >= 3600:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if seconds >= 60:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs:02d}s"
    if seconds >= 10:
        return f"{seconds:.1f}s"
    return f"{seconds:.2f}s"

# -------------------------
# Model: K2-mini with MoE + MLA + QK-Clip
# -------------------------

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        # x: (B, T, C)
        orig_dtype = x.dtype
        x_float = x.to(torch.float32)
        norm = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_float = x_float * torch.rsqrt(norm + self.eps)
        out = self.weight.to(torch.float32) * x_float
        return out.to(orig_dtype)

def build_rope_cache(head_dim: int, max_seq_len: int, base: float = 10000.0, device=None, dtype=torch.float32):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # (T, D/2)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope(x, cos, sin):
    # x: (B, T, H, D)
    B, T, H, D = x.shape
    x_ = x.view(B, T, H, D // 2, 2)
    x1, x2 = x_[..., 0], x_[..., 1]
    cos = cos[:T, :].view(1, T, 1, -1)
    sin = sin[:T, :].view(1, T, 1, -1)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack([out1, out2], dim=-1).view(B, T, H, D)

def masked_softmax_fp32(logits: torch.Tensor, mask: Optional[torch.Tensor], dim: int = -1) -> torch.Tensor:
    """
    logits: (..., T)
    mask: additive mask; same shape; 0 keep, -inf block
    Returns fp32-softmax with rows of all-masked positions -> all zeros (no NaN).
    """
    x = logits.float()
    if mask is not None:
        x = x + mask
    valid = torch.isfinite(x)
    row_has_valid = valid.any(dim=dim, keepdim=True)
    x = torch.where(row_has_valid, x, torch.zeros_like(x))
    x = x - x.max(dim=dim, keepdim=True).values
    exp = torch.exp(x)
    exp = torch.where(valid, exp, torch.zeros_like(exp))
    denom = exp.sum(dim=dim, keepdim=True)
    return exp / (denom + 1e-12)

class MLAAttention(nn.Module):
    """
    MLA-style attention:
      - head-specific q_c, k_c (clip by sqrt(gamma))
      - head-specific q_r (rotary, clip by gamma)
      - shared k_r (rotary, NOT clipped)
    """
    def __init__(self, d_model: int, n_heads: int, head_dim: int, rope_max_len: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim

        self.Wq_c = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.Wk_c = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.Wq_r = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.Wk_r_shared = nn.Linear(d_model, head_dim, bias=False)

        self.Wv = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.Wo = nn.Linear(n_heads * head_dim, d_model, bias=False)

        self.register_buffer("rope_cos", None, persistent=False)
        self.register_buffer("rope_sin", None, persistent=False)
        self.rope_max_len = rope_max_len
        self.last_max_logits: Optional[torch.Tensor] = None  # (H,)

    def maybe_init_rope(self, device, dtype):
        if self.rope_cos is None or self.rope_cos.device != device or self.rope_cos.dtype != dtype:
            cos, sin = build_rope_cache(self.head_dim, self.rope_max_len, device=device, dtype=dtype)
            self.rope_cos, self.rope_sin = cos, sin

    def forward(self, x, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        self.maybe_init_rope(x.device, x.dtype)

        q_c = self.Wq_c(x).view(B, T, self.n_heads, self.head_dim)
        k_c = self.Wk_c(x).view(B, T, self.n_heads, self.head_dim)
        q_r = self.Wq_r(x).view(B, T, self.n_heads, self.head_dim)
        k_r_shared = self.Wk_r_shared(x).view(B, T, 1, self.head_dim).expand(B, T, self.n_heads, self.head_dim)

        q_r = apply_rope(q_r, self.rope_cos, self.rope_sin)
        k_r = apply_rope(k_r_shared, self.rope_cos, self.rope_sin)

        Q = q_c + q_r
        K = k_c + k_r
        V = self.Wv(x).view(B, T, self.n_heads, self.head_dim)

        # scaled dot-product attention in fp32
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_logits = torch.einsum("bthd,bshd->bhts", Q, K).float() * scale  # (B,H,T,T)

        # record per-head max for QK-Clip
        with torch.no_grad():
            s = torch.nan_to_num(attn_logits, nan=0.0, posinf=0.0, neginf=0.0)
            self.last_max_logits = s.amax(dim=(0, 2, 3)).detach()

        attn = masked_softmax_fp32(attn_logits, mask, dim=-1).to(Q.dtype)
        out = torch.einsum("bhts,bshd->bthd", attn, V).contiguous()
        out = out.view(B, T, self.n_heads * self.head_dim)
        return self.Wo(out)

    @torch.no_grad()
    def qk_clip_step(self, tau: float):
        if self.last_max_logits is None:
            return
        H = self.n_heads
        device = self.Wq_c.weight.device
        S = torch.nan_to_num(self.last_max_logits.to(device), nan=0.0, posinf=tau, neginf=0.0)
        tau_t = torch.tensor(tau, device=S.device, dtype=S.dtype)
        gamma = torch.minimum(torch.ones_like(S), tau_t / (S + 1e-6))
        needs = (gamma < 0.9999)
        if not needs.any():
            return

        def per_head_scale_linear(linear: nn.Linear, scale: torch.Tensor, sqrt: bool = False):
            w = linear.weight  # (out, in)
            out, din = w.shape
            assert out == H * self.head_dim
            w = w.view(H, self.head_dim, din)
            factor = scale.sqrt() if sqrt else scale
            w[needs] *= factor[needs].view(-1, 1, 1)
            linear.weight.copy_(w.view(H * self.head_dim, din))

        per_head_scale_linear(self.Wq_c, gamma, sqrt=True)
        per_head_scale_linear(self.Wk_c, gamma, sqrt=True)
        per_head_scale_linear(self.Wq_r, gamma, sqrt=False)
        # shared k_r is not clipped

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
        logits = self.w(x.float())  # (B,T,E) in fp32
        topk_val, topk_idx = torch.topk(logits, self.k, dim=-1)
        gates = F.softmax(topk_val, dim=-1).to(x.dtype)  # (B,T,k)
        return topk_idx, gates, logits

class MoE(nn.Module):
    def __init__(self, d_model, n_experts, k_active, d_ff, use_shared_expert=True):
        super().__init__()
        self.n_experts = n_experts + (1 if use_shared_expert else 0)
        self.k = k_active
        self.router = TopKRouter(d_model, self.n_experts, self.k)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(self.n_experts)])
    def forward(self, x):
        B, T, C = x.shape
        topk_idx, gates, logits = self.router(x)  # (B,T,k), (B,T,k), (B,T,E)
        probs = F.softmax(logits.float(), dim=-1)
        me = probs.mean(dim=(0,1))
        ce = (probs > (1.0 / self.n_experts)).float().mean(dim=(0,1))
        aux_loss = (me * ce).sum() * self.n_experts

        out = torch.zeros_like(x)
        for i in range(self.k):
            idx = topk_idx[..., i]             # (B,T)
            gate = gates[..., i].unsqueeze(-1) # (B,T,1)
            chunk_out = torch.zeros_like(x)
            for e in range(self.n_experts):
                sel = (idx == e)               # 2D mask (B,T)
                if sel.any():
                    x_sel = x[sel]             # (N, C)
                    y = self.experts[e](x_sel) # (N, C)
                    chunk_out[sel] = y.to(chunk_out.dtype)
            out += gate.to(chunk_out.dtype) * chunk_out
        return out, aux_loss

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

@dataclass
class K2Config:
    vocab_size: int = 32768
    d_model: int = 1024
    n_heads: int = 16
    head_dim: int = 64
    n_layers: int = 12
    n_experts_total: int = 8   # +1 shared internally
    top_k: int = 2             # 'fast' preset; can switch to 8
    d_ff: int = 2048
    rope_max_len: int = 4096
    tie_word_embeddings: bool = False
    qk_clip_tau: float = 100.0

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
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        self.apply(_init_weights)
        self._ascii_masks: Dict[str, torch.Tensor] = {}

    def _ascii_mask(self, device) -> torch.Tensor:
        """
        Return a boolean mask over the vocabulary that keeps printable ASCII plus tab/newline/CR.
        Cached per device to avoid repeated allocations during decoding.
        """
        dev = torch.device(device)
        key = f"{dev.type}:{dev.index if dev.index is not None else 'default'}"
        mask = self._ascii_masks.get(key)
        if mask is None or mask.device != dev:
            allowed = [9, 10, 13] + list(range(32, 127))
            base = torch.zeros(self.cfg.vocab_size, dtype=torch.bool)
            base[allowed] = True
            mask = base.to(dev)
            self._ascii_masks[key] = mask
        return mask
    def forward(self, input_ids, attention_mask=None, labels=None):
        B, T = input_ids.shape
        x = self.tok_emb(input_ids)
        # causal + padding mask (additive, fp32)
        causal = torch.full((T, T), float("-inf"), device=x.device, dtype=torch.float32).triu(1)
        causal = causal.unsqueeze(0).unsqueeze(0)
        if attention_mask is not None:
            attn = (1 - attention_mask[:, None, None, :]).to(torch.float32) * float("-inf")
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
            ce_logits = logits[:, :-1].contiguous().view(-1, logits.size(-1)).float()
            ce_labels = labels[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(ce_logits, ce_labels, ignore_index=-100)
        aux_mean = torch.stack(aux_losses).mean() if aux_losses else torch.tensor(0.0, device=x.device)
        return logits, aux_mean, loss
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
            next_logits = logits[:, -1, :].float()
            ascii_mask = self._ascii_mask(next_logits.device)
            next_logits = next_logits.masked_fill(~ascii_mask, float("-inf"))
            if temperature > 0:
                next_logits = next_logits / temperature
                probs = F.softmax(next_logits, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                    cum = torch.cumsum(sorted_probs, dim=-1)
                    cutoff = (cum > top_p).float().cumsum(dim=-1) >= 1
                    sorted_probs[cutoff] = 0
                    zero_rows = (sorted_probs.sum(dim=-1, keepdim=True) <= 0)
                    if zero_rows.any():
                        sorted_probs[zero_rows.expand_as(sorted_probs)] = 0
                        sorted_probs[zero_rows.squeeze(-1), 0] = 1.0
                    denom = sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    sorted_probs = sorted_probs / denom
                    next_id = torch.multinomial(sorted_probs, num_samples=1)
                    next_token = sorted_idx.gather(-1, next_id)
                else:
                    next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            out = torch.cat([out, next_token], dim=1)
        return out
    @torch.no_grad()
    def generate_beam(
        self,
        input_ids,
        beam_width: int = 4,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        per_beam_topk: Optional[int] = 50,
        length_penalty_alpha: float = 0.6,
        length_penalty_k: float = 5.0,
        diversity_penalty: float = 0.0,
        entropy_bonus: float = 0.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        eos_token_id: Optional[int] = None,
        return_full_beam: bool = False,
    ):
        """
        Beam-style decoding that keeps multiple candidate continuations in parallel.
        Applies GNMT-style length normalization via `length_penalty_alpha` / `length_penalty_k`,
        with optional repetition penalty and n-gram blocking.
        Returns the highest scoring sequence (optionally with full beam metadata).
        """
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if per_beam_topk is not None and per_beam_topk < 1:
            raise ValueError("per_beam_topk must be >= 1 when provided")
        if repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty must be > 0")
        if no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size must be >= 0")

        self.eval()
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        device = input_ids.device
        batch_size, base_len = input_ids.shape
        beam_width = int(beam_width)
        vocab_size = self.cfg.vocab_size
        top_k = vocab_size if per_beam_topk is None else min(per_beam_topk, vocab_size)

        # Beam state: tokens, cumulative log-prob, entropy accumulator, lengths, finished flag
        sequences = input_ids.unsqueeze(1).expand(batch_size, beam_width, base_len).contiguous()
        sequences = sequences.clone()
        beam_logprobs = torch.full((batch_size, beam_width), float("-inf"), device=device)
        beam_logprobs[:, 0] = 0.0
        beam_entropy = torch.zeros((batch_size, beam_width), device=device)
        beam_lengths = torch.full((batch_size, beam_width), base_len, device=device, dtype=torch.long)
        beam_finished = torch.zeros((batch_size, beam_width), device=device, dtype=torch.bool)
        beam_adjusted_scores = beam_logprobs.clone()

        def _collect_ngram_bans(seq_tokens: torch.Tensor, n: int) -> set:
            if n <= 1:
                return set()
            total_len = seq_tokens.numel()
            prefix_len = n - 1
            if total_len < n:
                return set()
            ngram_dict: Dict[Tuple[int, ...], set] = {}
            seq_list = seq_tokens.tolist()
            for i in range(total_len - n + 1):
                prefix = tuple(seq_list[i : i + prefix_len])
                next_token = int(seq_list[i + prefix_len])
                bucket = ngram_dict.setdefault(prefix, set())
                bucket.add(next_token)
            current_prefix = tuple(seq_list[-prefix_len:])
            return ngram_dict.get(current_prefix, set()) or set()

        for _ in range(max_new_tokens):
            flat_sequences = sequences.view(batch_size * beam_width, -1)
            logits, _, _ = self.forward(flat_sequences)
            next_logits = logits[:, -1, :].float()
            if temperature > 0:
                scaled_logits = next_logits / temperature
            else:
                scaled_logits = next_logits

            ascii_mask = self._ascii_mask(scaled_logits.device)
            scaled_logits.masked_fill_(~ascii_mask, float("-inf"))

            if repetition_penalty != 1.0:
                for row_idx in range(flat_sequences.size(0)):
                    row_logits = scaled_logits[row_idx]
                    seen_tokens = torch.unique(flat_sequences[row_idx])
                    for tok in seen_tokens.tolist():
                        tok = int(tok)
                        logit = row_logits[tok]
                        if logit > 0:
                            row_logits[tok] = logit / repetition_penalty
                        else:
                            row_logits[tok] = logit * repetition_penalty

            log_probs = F.log_softmax(scaled_logits, dim=-1)
            probs = log_probs.exp()

            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                keep_mask = cumulative <= top_p
                keep_mask[..., 0] = True
                filtered_probs = torch.zeros_like(probs)
                filtered_probs.scatter_(dim=-1, index=sorted_idx, src=sorted_probs * keep_mask)
                norm = filtered_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                probs = filtered_probs / norm
                mask = probs > 0
                log_probs = torch.where(mask, torch.log(probs), torch.full_like(probs, float("-inf")))

            if no_repeat_ngram_size > 1:
                flat_sequences = sequences.view(batch_size * beam_width, -1)
                for row_idx in range(flat_sequences.size(0)):
                    banned = _collect_ngram_bans(flat_sequences[row_idx], no_repeat_ngram_size)
                    if not banned:
                        continue
                    row_log_probs = log_probs[row_idx]
                    row_probs = probs[row_idx]
                    prev_log_probs = row_log_probs.clone()
                    prev_probs = row_probs.clone()
                    banned_idx = torch.tensor(list(banned), device=row_log_probs.device, dtype=torch.long)
                    row_log_probs[banned_idx] = float("-inf")
                    row_probs[banned_idx] = 0.0
                    if not torch.isfinite(row_log_probs).any():
                        row_log_probs.copy_(prev_log_probs)
                        row_probs.copy_(prev_probs)

            probs_sum = probs.sum(dim=-1, keepdim=True)
            probs = probs / probs_sum.clamp_min(1e-12)
            log_probs = torch.where(probs > 0, torch.log(probs), torch.full_like(probs, float("-inf")))

            beam_finished_flat = beam_finished.view(-1)
            if eos_token_id is not None and beam_finished_flat.any():
                finished_rows = torch.nonzero(beam_finished_flat, as_tuple=False).squeeze(-1)
                if finished_rows.numel() > 0:
                    log_probs[finished_rows] = float("-inf")
                    log_probs[finished_rows, eos_token_id] = 0.0
                    probs[finished_rows] = 0.0
                    probs[finished_rows, eos_token_id] = 1.0

            safe_log_probs = torch.where(probs > 0, torch.log(probs), torch.zeros_like(probs))
            step_entropy = -(probs * safe_log_probs).sum(dim=-1)

            k = min(top_k, log_probs.size(-1))
            if k <= 0:
                break
            top_log_probs, top_tokens = torch.topk(log_probs, k=k, dim=-1)
            k_actual = top_log_probs.size(-1)

            beam_logprobs_flat = beam_logprobs.view(-1)
            beam_entropy_flat = beam_entropy.view(-1)
            beam_lengths_flat = beam_lengths.view(-1)

            candidate_logprob_totals = beam_logprobs_flat.unsqueeze(-1) + top_log_probs
            candidate_entropy_totals = beam_entropy_flat.unsqueeze(-1) + step_entropy.unsqueeze(-1).expand(step_entropy.size(0), k_actual)
            length_increment = (~beam_finished_flat).long()
            candidate_lengths = beam_lengths_flat.unsqueeze(-1) + length_increment.unsqueeze(-1).expand(length_increment.size(0), k_actual)

            parent_indices = torch.arange(beam_width, device=device).view(1, beam_width, 1)
            parent_indices = parent_indices.expand(batch_size, beam_width, k_actual)

            candidate_logprob_totals = candidate_logprob_totals.view(batch_size, beam_width, k_actual)
            candidate_entropy_totals = candidate_entropy_totals.view(batch_size, beam_width, k_actual)
            candidate_lengths = candidate_lengths.view(batch_size, beam_width, k_actual)
            step_entropy = step_entropy.view(batch_size, beam_width)
            top_tokens = top_tokens.view(batch_size, beam_width, k_actual)

            score_with_entropy = candidate_logprob_totals + entropy_bonus * candidate_entropy_totals
            if diversity_penalty != 0.0:
                diversity_term = diversity_penalty * parent_indices.to(score_with_entropy.dtype)
                score_with_entropy = score_with_entropy - diversity_term
            if length_penalty_alpha > 0.0:
                lengths_fp = candidate_lengths.float().clamp_min(1.0)
                norm = ((length_penalty_k + lengths_fp) / (length_penalty_k + 1.0)) ** length_penalty_alpha
                score_for_ranking = score_with_entropy / norm
            else:
                score_for_ranking = score_with_entropy

            if eos_token_id is not None:
                candidate_finished = beam_finished.unsqueeze(-1).expand(-1, -1, k_actual) | (top_tokens == eos_token_id)
            else:
                candidate_finished = beam_finished.unsqueeze(-1).expand(-1, -1, k_actual)

            flat_scores = score_for_ranking.reshape(batch_size, -1)
            flat_tokens = top_tokens.reshape(batch_size, -1)
            flat_logprob_totals = candidate_logprob_totals.reshape(batch_size, -1)
            flat_entropy_totals = candidate_entropy_totals.reshape(batch_size, -1)
            flat_lengths = candidate_lengths.reshape(batch_size, -1)
            flat_parents = parent_indices.reshape(batch_size, -1)
            flat_finished = candidate_finished.reshape(batch_size, -1).long()

            top_scores, best_indices = flat_scores.topk(beam_width, dim=-1)
            best_tokens = torch.gather(flat_tokens, 1, best_indices)
            best_logprob_totals = torch.gather(flat_logprob_totals, 1, best_indices)
            best_entropy_totals = torch.gather(flat_entropy_totals, 1, best_indices)
            best_lengths = torch.gather(flat_lengths, 1, best_indices)
            best_parents = torch.gather(flat_parents, 1, best_indices)
            best_finished = torch.gather(flat_finished, 1, best_indices).bool()

            gather_idx = best_parents.unsqueeze(-1).expand(-1, -1, sequences.size(-1))
            selected_sequences = torch.gather(sequences, 1, gather_idx)
            sequences = torch.cat([selected_sequences, best_tokens.unsqueeze(-1)], dim=-1)

            beam_logprobs = best_logprob_totals
            beam_entropy = best_entropy_totals
            beam_lengths = best_lengths
            beam_finished = best_finished
            beam_adjusted_scores = top_scores

            if beam_finished.all():
                break

        final_scores = beam_logprobs + entropy_bonus * beam_entropy
        if length_penalty_alpha > 0.0:
            lengths_fp = beam_lengths.float().clamp_min(1.0)
            norm = ((length_penalty_k + lengths_fp) / (length_penalty_k + 1.0)) ** length_penalty_alpha
            final_scores = final_scores / norm

        best_indices = final_scores.argmax(dim=-1)
        pad_value = eos_token_id if eos_token_id is not None else 0
        best_sequences = []
        best_lengths_list: List[int] = []
        for batch_idx in range(batch_size):
            parent_len = int(beam_lengths[batch_idx, best_indices[batch_idx]].item())
            seq = sequences[batch_idx, best_indices[batch_idx], :parent_len].clone()
            best_sequences.append(seq)
            best_lengths_list.append(parent_len)
        best_batch = torch.nn.utils.rnn.pad_sequence(best_sequences, batch_first=True, padding_value=pad_value)

        if not return_full_beam:
            return best_batch

        beam_snapshot = {
            "sequences": sequences.clone(),
            "logprobs": beam_logprobs.clone(),
            "entropy": beam_entropy.clone(),
            "lengths": beam_lengths.clone(),
            "scores": beam_adjusted_scores.clone(),
            "finished": beam_finished.clone(),
            "best_lengths": best_lengths_list,
            # Length statistics help with alpha/k sweep logging.
            "mean_length_all": beam_lengths.float().mean().item(),
            "mean_length_finished": beam_lengths[beam_finished].float().mean().item()
                if beam_finished.any() else None,
            "mean_length_best": float(sum(best_lengths_list) / max(len(best_lengths_list), 1)),
            "length_penalty_alpha": float(length_penalty_alpha),
            "length_penalty_k": float(length_penalty_k),
            "final_scores": final_scores.clone(),
        }
        return best_batch, beam_snapshot

    def logprobs(self, input_ids):
        logits, _, _ = self.forward(input_ids)
        logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
        tgt = input_ids[:, 1:].unsqueeze(-1)
        return logp.gather(-1, tgt).squeeze(-1)  # (B,T-1)
    @torch.no_grad()
    def generate_with_strategy(self, input_ids, strategy: str = "sample", **kwargs):
        """
        Convenience wrapper to choose between sampling-based and beam-style decoding.
        """
        if strategy == "sample":
            return self.generate(input_ids, **kwargs)
        if strategy == "beam":
            return self.generate_beam(input_ids, **kwargs)
        raise ValueError(f"Unknown decoding strategy: {strategy}")

class MuonLikePreconditioner:
    def __init__(self, iters=5):
        self.iters = iters
    @torch.no_grad()
    def msign(self, M):
        if M.ndim != 2:
            return M
        MtM = M.t().mm(M)
        I = torch.eye(MtM.size(0), device=MtM.device, dtype=MtM.dtype)
        Y = MtM / (MtM.norm() + 1e-12)
        Z = I.clone()
        for _ in range(self.iters):
            T = 0.5 * (3*I - Z.mm(Y))
            Y = Y.mm(T); Z = T.mm(Z)
        inv_sqrt = Z / math.sqrt(MtM.norm() + 1e-12)
        return M.mm(inv_sqrt)

class MuonClip(torch.optim.Optimizer):
    def __init__(self, model: K2Mini, base_optim: torch.optim.Optimizer, tau: float = 100.0,
                 use_muon_like: bool = False, muon_match_scale: float = 0.2):
        self.model = model
        self.base_optim = base_optim
        self.tau = tau
        self.use_muon_like = use_muon_like
        self.muon = MuonLikePreconditioner() if use_muon_like else None
        self.muon_match_scale = muon_match_scale
        self.momentum: Dict[int, torch.Tensor] = {}
    def zero_grad(self, set_to_none: bool = False):
        self.base_optim.zero_grad(set_to_none=set_to_none)
    @torch.no_grad()
    def step(self):
        if self.use_muon_like:
            for group in self.base_optim.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    if g.ndim == 2:
                        st = self.momentum.setdefault(id(p), torch.zeros_like(g))
                        st.mul_(0.9).add_(g)
                        g_pre = self.muon.msign(st) * self.muon_match_scale
                        p.grad.copy_(g_pre)
        loss = self.base_optim.step()
        self.model.qk_clip_step(self.tau)
        return loss

# -------------------------
# Config presets (4090-friendly). 'fast' = Top-2, 'top8' = Top-8.
# -------------------------

PRESETS = {
    "fast":  dict(vocab_size=32768, d_model=1024, n_heads=16, head_dim=64, n_layers=12,
                  n_experts_total=8, top_k=2, d_ff=2048, rope_max_len=4096, qk_clip_tau=100.0),
    "top8":  dict(vocab_size=32768, d_model=1024, n_heads=16, head_dim=64, n_layers=12,
                  n_experts_total=8, top_k=8, d_ff=2048, rope_max_len=4096, qk_clip_tau=100.0),
}

# -------------------------
# Data adapter: DeepCoder (robust to local cache or HF)
# -------------------------

class DeepCoderAdapter:
    """
    Columns expected:
      - 'prompt' (str)
      - 'gold_standard_solution' (str)
      - 'verification_info' (str; often JSON)
    """
    def __init__(self, path_or_repo: str):
        self.split = self._load_any(path_or_repo)
        self.prompt_key = "prompt"
        self.solution_key = "gold_standard_solution"
        self.verif_key   = "verification_info"
    def _is_saved_dataset_dir(self, p: str) -> bool:
        return os.path.exists(os.path.join(p, "dataset_dict.json")) or \
               os.path.exists(os.path.join(p, "state.json"))
    def _try_load_arrow_shards(self, p: str):
        shard_paths = sorted(glob.glob(os.path.join(p, "*train-*.arrow")))
        if not shard_paths:
            return None
        parts = [Dataset.from_file(sp) for sp in shard_paths]
        return concatenate_datasets(parts)
    def _load_any(self, p: str):
        if os.path.isdir(p) and self._is_saved_dataset_dir(p):
            return load_from_disk(p)
        if os.path.isdir(p):
            ds = self._try_load_arrow_shards(p)
            if ds is not None:
                return ds
        try:
            return load_dataset("PrimeIntellect/deepcoder-gold-standard-solutions",
                                split="train", local_files_only=True)
        except Exception:
            return load_dataset("PrimeIntellect/deepcoder-gold-standard-solutions",
                                split="train")
    def __len__(self):
        return len(self.split)
    def get_prompt(self, idx: int) -> str:
        return str(self.split[idx][self.prompt_key])
    def get_reference(self, idx: int) -> Dict[str, any]:
        rec = self.split[idx]
        ref = {"solution": rec.get(self.solution_key)}
        vi  = rec.get(self.verif_key)
        if isinstance(vi, str) and vi.strip():
            try:
                ref["verification_info"] = json.loads(vi)
            except Exception:
                ref["verification_info"] = vi
        return ref

# -------------------------
# Verifier + Reward shaping
# -------------------------

def run_python_code_with_tests(code: str, tests: List[Dict[str, any]], timeout_sec: int = 2) -> bool:
    code = textwrap.dedent(code)
    with tempfile.TemporaryDirectory() as tmp:
        main_py = os.path.join(tmp, "main.py")
        with open(main_py, "w", encoding="utf-8") as f:
            f.write(code)
        driver = os.path.join(tmp, "driver.py")
        driver_code = """
import importlib, json, sys
m = importlib.import_module("main")
ok = True
tests = json.loads(sys.stdin.read())
for t in tests:
    func = getattr(m, t.get("entry", "solve"), None)
    if func is None:
        ok = False; break
    out = func(*t.get("input", []))
    if out != t.get("output"):
        ok = False; break
print("OK" if ok else "FAIL")
"""
        with open(driver, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(driver_code))
        p = subprocess.Popen([sys.executable, driver],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             cwd=tmp)
        out, err = p.communicate(input=json.dumps(tests).encode("utf-8"), timeout=timeout_sec)
        return out.decode("utf-8").strip() == "OK"

def deepcoder_reward(generated: str, reference: Dict[str, any]) -> float:
    """
    Graded reward:
      +0.10 if parses and defines correct entry (default 'solve')
      +0.20 if executes cleanly on at least 1 test
      +0.70 * fraction of tests passed
    Fallback: exact match to gold solution.
    """
    vi = reference.get("verification_info")
    try:
        if isinstance(vi, str):
            vi = json.loads(vi)
    except Exception:
        vi = None
    entry = "solve"
    tests = None
    if isinstance(vi, dict):
        entry = vi.get("entry", entry)
        if isinstance(vi.get("tests"), list):
            tests = vi["tests"]
    bonus = 0.0
    try:
        import ast
        tree = ast.parse(generated)
        has_entry = any(isinstance(n, ast.FunctionDef) and n.name == entry for n in tree.body)
        if has_entry: bonus += 0.10
    except Exception:
        sol = reference.get("solution")
        return 1.0 if isinstance(sol, str) and generated.strip() == sol.strip() else 0.0
    if tests:
        try:
            ok_first = run_python_code_with_tests(generated, tests[:1])
            if ok_first: bonus += 0.20
            passed = 0
            for t in tests:
                try:
                    ok = run_python_code_with_tests(generated, [t])
                    if ok: passed += 1
                except Exception:
                    pass
            frac = passed / max(1, len(tests))
            return min(1.0, bonus + 0.70 * frac)
        except Exception:
            return bonus
    else:
        sol = reference.get("solution")
        return max(bonus, 1.0 if isinstance(sol, str) and generated.strip() == sol.strip() else 0.0)

def debug_reward(generated: str, reference: Dict[str, any]) -> Dict[str, any]:
    """
    Return diagnostics for why a generated snippet is or is not receiving reward.
    Intended for debugging early training steps.
    """
    info = {
        "parse_ok": False,
        "has_entry": False,
        "first_test_ok": None,
        "n_passed": 0,
        "n_tests": 0,
        "entry": "solve",
    }
    vi = reference.get("verification_info")
    try:
        if isinstance(vi, str):
            vi = json.loads(vi)
    except Exception:
        vi = None
    entry = info["entry"] = (
        vi.get("entry") if isinstance(vi, dict) and isinstance(vi.get("entry"), str) else "solve"
    )
    tests = vi.get("tests") if isinstance(vi, dict) else None
    try:
        import ast
        ast.parse(generated)
        info["parse_ok"] = True
        info["has_entry"] = (f"def {entry}" in generated)
    except Exception:
        return info
    if tests:
        info["n_tests"] = len(tests)
        try:
            ok_first = run_python_code_with_tests(generated, tests[:1])
            info["first_test_ok"] = bool(ok_first)
            passed = 0
            for t in tests:
                try:
                    if run_python_code_with_tests(generated, [t]):
                        passed += 1
                except Exception:
                    pass
            info["n_passed"] = passed
        except Exception:
            pass
    return info

# -------------------------
# Tokenizer, prompting, batching
# -------------------------

class ByteTokenizer:
    def __init__(self, vocab_size=32768):
        self.vocab_size = vocab_size
    def encode(self, text: str, max_len: int = None) -> List[int]:
        ids = [c for c in text.encode("utf-8")]
        if max_len is not None:
            ids = ids[:max_len]
        return ids
    def decode(self, ids: List[int]) -> str:
        raw = bytes([(i & 0xFF) for i in ids])
        if b"\x00" in raw:
            raw = raw.replace(b"\x00", b"")
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError:
            decoded = raw.decode("utf-8", errors="ignore")
        cleaned = []
        for ch in decoded:
            code = ord(ch)
            if ch in ("\n", "\t", "\r") or 32 <= code <= 126:
                cleaned.append(ch)
            else:
                cleaned.append(" ")
        return "".join(cleaned)

def make_prompt(raw_prompt: str, entry: str = "solve", anchor: bool = True) -> str:
    prompt = (
        "You are a Python coding assistant.\n"
        f"Write valid Python 3 code that defines a function `{entry}` and returns the answer.\n"
        "Do not print; do not use input(); just return the result.\n"
        "Use only the standard library. Keep it concise.\n"
        "Task:\n"
        f"{raw_prompt}\n\n"
        "# Your code below:\n"
    )
    if anchor:
        prompt += f"def {entry}("
    return prompt

def prepare_batch(tokenizer, prompts: List[str], max_prompt_len: int, device: str):
    ids = [tokenizer.encode(p, max_len=max_prompt_len) for p in prompts]
    maxlen = max(len(x) for x in ids)
    B = len(ids)
    input_ids = torch.full((B, maxlen), 0, dtype=torch.long, device=device)
    attn = torch.zeros((B, maxlen), dtype=torch.long, device=device)
    for i, seq in enumerate(ids):
        L = len(seq)
        input_ids[i, :L] = torch.tensor(seq, device=device)
        attn[i, :L] = 1
    return input_ids, attn

def sample_and_logprobs(model: K2Mini, input_ids, rl_cfg: RLConfig, autocast_dtype):
    """
    Roll out K candidates from the policy under the configured decoding strategy and
    return both the generated tokens and their per-token log-probabilities.
    """
    all_gen_tokens, all_logprobs = [], []
    aux = None

    if input_ids.is_cuda:
        ctx = torch.amp.autocast
        ctx_kwargs = dict(device_type="cuda", enabled=(autocast_dtype is not None), dtype=autocast_dtype)
    else:
        device_type = "mps" if input_ids.device.type == "mps" else "cpu"
        ctx = torch.amp.autocast
        ctx_kwargs = dict(device_type=device_type, enabled=(device_type == "mps" and autocast_dtype is not None), dtype=autocast_dtype)

    if rl_cfg.decode_strategy == "beam":
        with torch.no_grad(), ctx(**ctx_kwargs):
            _, beam_snapshot = model.generate_beam(
                input_ids,
                beam_width=rl_cfg.beam_width,
                max_new_tokens=rl_cfg.max_gen_len,
                temperature=rl_cfg.beam_temperature,
                top_p=rl_cfg.beam_top_p,
                per_beam_topk=rl_cfg.beam_per_beam_topk,
                length_penalty_alpha=rl_cfg.beam_length_alpha,
                length_penalty_k=rl_cfg.beam_length_k,
                diversity_penalty=rl_cfg.beam_diversity_penalty,
                entropy_bonus=rl_cfg.beam_entropy_bonus,
                repetition_penalty=rl_cfg.beam_repetition_penalty,
                no_repeat_ngram_size=rl_cfg.beam_no_repeat_ngram,
                return_full_beam=True,
            )

        aux = beam_snapshot
        sequences = beam_snapshot["sequences"]
        final_scores = beam_snapshot["final_scores"]
        beam_width = sequences.size(1)
        take = min(rl_cfg.K, beam_width)
        if rl_cfg.K > beam_width:
            print(f"[Beam] K={rl_cfg.K} exceeds beam_width={beam_width}; repeating best beams to fill K.")
        if take <= 0:
            raise ValueError("Beam decoding requires K >= 1.")

        top_scores, top_indices = final_scores.topk(take, dim=1)
        del top_scores  # not used further; kept for clarity

        base_generated, base_logprobs = [], []
        for rank in range(take):
            idx = top_indices[:, rank]
            gather_idx = idx.view(-1, 1, 1).expand(-1, 1, sequences.size(-1))
            beam_tokens = torch.gather(sequences, 1, gather_idx).squeeze(1)
            with ctx(**ctx_kwargs):
                logp = model.logprobs(beam_tokens)
            all_gen_tokens.append(beam_tokens)
            all_logprobs.append(logp)
            base_generated.append(beam_tokens)
            base_logprobs.append(logp)

        base_count = len(base_generated)
        if base_count == 0:
            raise RuntimeError("Beam search returned no hypotheses.")
        while len(all_gen_tokens) < rl_cfg.K:
            src = len(all_gen_tokens) % base_count
            all_gen_tokens.append(base_generated[src].clone())
            all_logprobs.append(base_logprobs[src].clone())

    else:
        for _ in range(rl_cfg.K):
            with torch.no_grad(), ctx(**ctx_kwargs):
                gen_ids = model.generate(
                    input_ids,
                    max_new_tokens=rl_cfg.max_gen_len,
                    temperature=rl_cfg.temperature,
                    top_p=rl_cfg.top_p,
                )
            with ctx(**ctx_kwargs):
                logp = model.logprobs(gen_ids)
            all_gen_tokens.append(gen_ids)
            all_logprobs.append(logp)
        aux = all_gen_tokens  # legacy behaviour (unused by callers)

    return all_gen_tokens, all_logprobs, aux

# -------------------------
# Training (SFT warm-start + RLVR)
# -------------------------

@dataclass
class RLConfig:
    lr: float = 1e-4
    batch_size: int = 1
    max_prompt_len: int = 512
    max_gen_len: int = 128
    temperature: float = 0.8
    top_p: float = 0.95
    K: int = 2
    tau_reg: float = 0.005
    token_budget: int = 128
    steps: int = 200
    sft_steps: int = 500
    use_muon_like: bool = False
    decode_strategy: str = "sample"
    beam_width: int = 4
    beam_temperature: float = 1.0
    beam_top_p: float = 1.0
    beam_per_beam_topk: int = 50
    beam_length_alpha: float = 0.6
    beam_length_k: float = 5.0
    beam_diversity_penalty: float = 0.0
    beam_entropy_bonus: float = 0.0
    beam_repetition_penalty: float = 1.0
    beam_no_repeat_ngram: int = 0

def build_param_groups(model):
    decay, nodecay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: 
            continue
        is_bias = n.endswith(".bias")
        is_norm = "norm" in n.lower()
        is_emb  = "tok_emb" in n or "embedding" in n.lower()
        if p.ndim == 1 or is_bias or is_norm or is_emb:
            nodecay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay,   "weight_decay": 0.1},
        {"params": nodecay, "weight_decay": 0.0},
    ]

def run_sft(model, tokenizer, data, device, autocast_dtype, steps=500, lr=5e-5, max_prompt_len=512, max_label_len=256):
    model.train()
    opt = AdamW(build_param_groups(model), lr=lr, betas=(0.9, 0.95), eps=1e-8)
    loop_start = time.time()
    last_log = loop_start
    log_every = max(1, steps // 20)
    for step in range(steps):
        step_start = time.time()
        i = random.randrange(len(data))
        ref = data.get_reference(i)
        vi = ref.get("verification_info")
        entry = "solve" if not (isinstance(vi, dict) and isinstance(vi.get("entry"), str)) else vi["entry"]
        raw_prompt = data.get_prompt(i)
        label = ref.get("solution") or ""
        anchor = f"def {entry}("
        use_anchor = label.startswith(anchor)
        prompt = make_prompt(raw_prompt, entry=entry, anchor=use_anchor)
        if use_anchor:
            label = label[len(anchor):]

        prompt_ids = tokenizer.encode(prompt, max_len=max_prompt_len)
        label_ids = tokenizer.encode(label, max_len=max_label_len)
        ids = (prompt_ids + label_ids)[:max_prompt_len + max_label_len]
        if not ids:
            print(f"[SFT WARN] step {step+1}: empty tokenized sample; skipping")
            continue
        input_ids = torch.tensor(ids, device=device).unsqueeze(0)
        labels = input_ids.clone()
        cutoff = min(len(prompt_ids), len(ids))
        labels[:, :cutoff] = -100

        if device == "cuda":
            with torch.amp.autocast(device_type="cuda", enabled=(autocast_dtype is not None), dtype=autocast_dtype):
                _, _, ce = model(input_ids, labels=labels)
        else:
            with torch.amp.autocast(device_type=device, enabled=(device in ("mps",) and autocast_dtype is not None), dtype=autocast_dtype):
                _, _, ce = model(input_ids, labels=labels)

        if not torch.isfinite(ce):
            print(f"[SFT WARN] step {step+1}: non-finite loss; skipping")
            opt.zero_grad(set_to_none=True)
            continue

        ce_val = float(ce.detach().cpu())
        opt.zero_grad()
        ce.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        bad = False
        for p in model.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                bad = True
                break
        if bad:
            print(f"[SFT WARN] step {step+1}: non-finite grad; skipping")
            opt.zero_grad(set_to_none=True)
            continue

        opt.step()
        step_time = time.time() - step_start

        should_log = (
            step == 0
            or (step + 1) % log_every == 0
            or (step + 1) == steps
            or (time.time() - last_log) >= 30
        )
        if should_log:
            elapsed = time.time() - loop_start
            avg_step = elapsed / max(1, step + 1)
            remaining = max(0, steps - (step + 1))
            eta = avg_step * remaining
            prompt_tokens = cutoff
            total_tokens = len(ids)
            print(
                f"[SFT] step {step+1}/{steps} | loss={ce_val:.4f} | prompt_tokens={prompt_tokens} | "
                f"total_tokens={total_tokens} | step_time={format_seconds(step_time)} | "
                f"elapsed={format_seconds(elapsed)} | eta={format_seconds(eta)}"
            )
            last_log = time.time()

def quick_eval(model, tokenizer, data, device, autocast_dtype, rl_cfg: RLConfig, n=16):
    model.eval()
    ok = 0
    for _ in range(n):
        i = random.randrange(len(data))
        ref = data.get_reference(i)
        vi = ref.get("verification_info")
        entry = "solve" if not (isinstance(vi, dict) and isinstance(vi.get("entry"), str)) else vi["entry"]
        prompt = make_prompt(data.get_prompt(i), entry=entry)
        inp, _ = prepare_batch(tokenizer, [prompt], 512, device)
        device_type = "cuda" if device == "cuda" else ("mps" if device == "mps" else "cpu")
        ctx_kwargs = dict(
            device_type=device_type,
            enabled=(device_type in ("cuda", "mps") and autocast_dtype is not None),
            dtype=autocast_dtype,
        )
        with torch.no_grad(), torch.amp.autocast(**ctx_kwargs):
            if rl_cfg.decode_strategy == "beam":
                out = model.generate_with_strategy(
                    inp,
                    strategy="beam",
                    max_new_tokens=rl_cfg.max_gen_len,
                    beam_width=rl_cfg.beam_width,
                    temperature=rl_cfg.beam_temperature,
                    top_p=rl_cfg.beam_top_p,
                    per_beam_topk=rl_cfg.beam_per_beam_topk,
                    length_penalty_alpha=rl_cfg.beam_length_alpha,
                    length_penalty_k=rl_cfg.beam_length_k,
                    diversity_penalty=rl_cfg.beam_diversity_penalty,
                    entropy_bonus=rl_cfg.beam_entropy_bonus,
                    repetition_penalty=rl_cfg.beam_repetition_penalty,
                    no_repeat_ngram_size=rl_cfg.beam_no_repeat_ngram,
                )
            else:
                out = model.generate_with_strategy(
                    inp,
                    strategy="sample",
                    max_new_tokens=rl_cfg.max_gen_len,
                    temperature=rl_cfg.temperature,
                    top_p=rl_cfg.top_p,
                )
        completion = tokenizer.decode(out[0].tolist()[inp.size(1):])
        generated_code = f"def {entry}(" + completion
        r = deepcoder_reward(generated_code, ref)
        ok += (r > 0.0)
    model.train()
    return ok / max(1, n)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=r"C:\Users\Admin\.cache\huggingface\datasets\PrimeIntellect___deepcoder-gold-standard-solutions\default\0.0.0\146e11ac3b13abc18109ce95cc117f8ce5e61ac3",
                        help="Path to local HF cache dir with *.arrow shards; falls back to HF download if missing.")
    parser.add_argument("--preset", type=str, default="fast", choices=list(PRESETS.keys()), help="Model preset: fast (Top-2) or top8.")
    parser.add_argument("--steps", type=int, default=200, help="RL steps after SFT warm-start.")
    parser.add_argument("--sft-steps", type=int, default=500, help="Warm-start SFT steps.")
    parser.add_argument("--use-top8", action="store_true", help="Shortcut to use Top-8 preset.")
    parser.add_argument("--decode-strategy", type=str, default="sample", choices=("sample", "beam"),
                        help="Decoding strategy for RL rollouts and eval (sample or beam).")
    parser.add_argument("--beam-width", type=int, default=4, help="Beam width when using beam decoding.")
    parser.add_argument("--beam-temperature", type=float, default=1.0, help="Temperature applied before beam softmax.")
    parser.add_argument("--beam-top-p", type=float, default=1.0, help="Top-p filter inside beam decoding.")
    parser.add_argument("--beam-per-beam-topk", type=int, default=50, help="Top-k expansion candidates per beam.")
    parser.add_argument("--beam-length-alpha", type=float, default=0.6, help="Length penalty alpha (GNMT-style).")
    parser.add_argument("--beam-length-k", type=float, default=5.0, help="Length penalty k (GNMT-style).")
    parser.add_argument("--beam-diversity-penalty", type=float, default=0.0, help="Per-beam diversity penalty.")
    parser.add_argument("--beam-entropy-bonus", type=float, default=0.0, help="Entropy bonus added to beam scores.")
    parser.add_argument("--beam-repetition-penalty", type=float, default=1.0, help="Penalize previously used tokens (>1.0).")
    parser.add_argument("--beam-no-repeat-ngram", type=int, default=0, help="Block repeats of size-n ngrams (0 disables).")
    parser.add_argument("--device", type=str, default="auto", help="Device override: auto, cpu, cuda, cuda:<idx>, mps.")
    args = parser.parse_args()

    device = pick_device(args.device)
    autocast_dtype = get_autocast_dtype(device)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        current_idx = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_idx)
        device_str = f"cuda:{current_idx} ({gpu_name})"
    elif device == "mps":
        device_str = "mps"
    else:
        device_str = "cpu"
    print(f"[Device] {device_str} | running in full precision (autocast disabled)")

    # Config
    preset_key = "top8" if args.use_top8 else args.preset
    cfg = K2Config(**PRESETS[preset_key])
    rl = RLConfig(
        steps=args.steps,
        sft_steps=args.sft_steps,
        decode_strategy=args.decode_strategy,
        beam_width=args.beam_width,
        beam_temperature=args.beam_temperature,
        beam_top_p=args.beam_top_p,
        beam_per_beam_topk=args.beam_per_beam_topk,
        beam_length_alpha=args.beam_length_alpha,
        beam_length_k=args.beam_length_k,
        beam_diversity_penalty=args.beam_diversity_penalty,
        beam_entropy_bonus=args.beam_entropy_bonus,
        beam_repetition_penalty=args.beam_repetition_penalty,
        beam_no_repeat_ngram=args.beam_no_repeat_ngram,
    )
    if rl.decode_strategy == "beam":
        print(
            f"[Decode] strategy=beam | width={rl.beam_width} | alpha={rl.beam_length_alpha} | "
            f"k={rl.beam_length_k} | repeat_penalty={rl.beam_repetition_penalty} | "
            f"no_repeat_ngram={rl.beam_no_repeat_ngram}"
        )
    else:
        print(
            f"[Decode] strategy=sample | temperature={rl.temperature} | top_p={rl.top_p}"
        )

    # Data
    data = DeepCoderAdapter(args.data_path)
    print(f"[Data] Loaded {len(data)} samples")

    # Model
    model = K2Mini(cfg).to(device)
    # Keep full precision for stability regardless of device.
    target_dtype = torch.float32
    model.to(dtype=target_dtype)
    print(f"[Model] parameters dtype: {target_dtype}")
    # keep RMSNorm in FP32
    for m in model.modules():
        if m.__class__.__name__ in ("RMSNorm",):
            m.to(dtype=torch.float32)

    tokenizer = ByteTokenizer(vocab_size=cfg.vocab_size)

    base_optim = AdamW(build_param_groups(model), lr=rl.lr, betas=(0.9, 0.95), eps=1e-8)
    optim = MuonClip(model, base_optim, tau=cfg.qk_clip_tau, use_muon_like=rl.use_muon_like)
    warmup_steps = 100

    # SFT warm-start
    print("[Run] SFT warm-start...")
    run_sft(model, tokenizer, data, device, autocast_dtype,
            steps=rl.sft_steps, lr=5e-5, max_prompt_len=rl.max_prompt_len, max_label_len=rl.max_gen_len)
    print("[Run] SFT done. Switching to RL.")

    # RL training
    rl_start = time.time()
    rl_last_log = rl_start
    rl_log_every = max(1, rl.steps // 20)
    for step in range(rl.steps):
        rl_step_start = time.time()
        idxs = random.sample(range(len(data)), rl.batch_size)
        prompts, refs, entries = [], [], []
        for i in idxs:
            ref = data.get_reference(i)
            vi = ref.get("verification_info")
            entry = "solve" if not (isinstance(vi, dict) and isinstance(vi.get("entry"), str)) else vi["entry"]
            prompts.append(make_prompt(data.get_prompt(i), entry=entry))
            refs.append(ref)
            entries.append(entry)

        input_ids, attn = prepare_batch(tokenizer, prompts, rl.max_prompt_len, device)

        gen_seqs, seq_logps, decode_meta = sample_and_logprobs(
            model, input_ids, rl, autocast_dtype
        )
        beam_meta = decode_meta if isinstance(decode_meta, dict) and rl.decode_strategy == "beam" else None

        # rewards
        rewards = torch.zeros((rl.batch_size, rl.K), device=device)
        for ki in range(rl.K):
            for b in range(rl.batch_size):
                full_ids = gen_seqs[ki][b].tolist()
                gen_only = full_ids[len(input_ids[b]):]
                completion = tokenizer.decode(gen_only)
                code = f"def {entries[b]}(" + completion
                rewards[b, ki] = deepcoder_reward(code, refs[b])
                if step < 3 and ki == 0 and b == 0:
                    diag = debug_reward(code, refs[b])
                    snippet = code[:400]
                    if len(code) > 400:
                        snippet = snippet + "..."
                    print("[DBG] reward diagnostics:", diag)
                    print("[DBG] snippet:\n", snippet)

        r_bar = rewards.mean(dim=1, keepdim=True)
        advantages = rewards - r_bar

        # objective
        if device == "cuda":
            with torch.amp.autocast(device_type="cuda", enabled=(autocast_dtype is not None), dtype=autocast_dtype):
                obj = 0.0
                for ki in range(rl.K):
                    lp = seq_logps[ki]  # (B, L-1)
                    used = lp[:, -rl.token_budget:].mean(dim=-1)  # (B,)
                    obj = obj + (advantages[:, ki] - rl.tau_reg * used).pow(2).mean()
                obj = obj / max(1, rl.K)
        else:
            with torch.amp.autocast(device_type=device, enabled=(device in ("mps",) and autocast_dtype is not None), dtype=autocast_dtype):
                obj = 0.0
                for ki in range(rl.K):
                    lp = seq_logps[ki]  # (B, L-1)
                    used = lp[:, -rl.token_budget:].mean(dim=-1)  # (B,)
                    obj = obj + (advantages[:, ki] - rl.tau_reg * used).pow(2).mean()
                obj = obj / max(1, rl.K)

        if not torch.isfinite(obj):
            print(f"[RL WARN] step {step+1}: non-finite obj; skipping & halving LR")
            optim.zero_grad(set_to_none=True)
            for g in base_optim.param_groups:
                g['lr'] = max(g['lr'] / 2, 1e-6)
            continue

        # MoE aux (prompt-only)
        with torch.no_grad():
            _, aux_loss, _ = model(input_ids)
        loss = obj + 1e-3 * aux_loss

        loss_val = float(loss.detach().cpu())
        obj_val = float(obj.detach().cpu())
        aux_val = float(aux_loss.detach().cpu())
        rewards_float = rewards.detach().float()
        reward_mean = float(rewards_float.mean().cpu())
        reward_std = float(rewards_float.std(unbiased=False).cpu())
        lr_val = base_optim.param_groups[0].get("lr", rl.lr)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Grad finite check
        bad = False
        for p in model.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                bad = True
                break
        if bad:
            print(f"[RL WARN] step {step+1}: non-finite grad; skipping & halving LR")
            optim.zero_grad(set_to_none=True)
            for g in base_optim.param_groups:
                g['lr'] = max(g['lr'] / 2, 1e-6)
            continue

        optim.step()
        step_time = time.time() - rl_step_start

        # linear warmup
        if step < warmup_steps:
            scale = (step + 1) / warmup_steps
            for g in base_optim.param_groups:
                g['lr'] = rl.lr * scale
            lr_val = base_optim.param_groups[0].get("lr", rl.lr)

        should_log = (
            step == 0
            or (step + 1) % rl_log_every == 0
            or (step + 1) == rl.steps
            or (time.time() - rl_last_log) >= 60
        )
        if should_log:
            elapsed = time.time() - rl_start
            avg_step = elapsed / max(1, step + 1)
            eta = avg_step * max(0, rl.steps - (step + 1))
            length_msg = ""
            if beam_meta:
                best_len = beam_meta.get("mean_length_best")
                all_len = beam_meta.get("mean_length_all")
                if best_len is not None and all_len is not None:
                    length_msg = f" | beam_len_best={best_len:.1f} | beam_len_all={all_len:.1f}"
            print(
                f"[RL] step {step+1}/{rl.steps} | loss={loss_val:.4f} | obj={obj_val:.4f} | aux={aux_val:.4f} | "
                f"reward_mean={reward_mean:.3f} | reward_std={reward_std:.3f} | lr={lr_val:.2e} | "
                f"step_time={format_seconds(step_time)} | elapsed={format_seconds(elapsed)} | eta={format_seconds(eta)}"
                f"{length_msg}"
            )
            rl_last_log = time.time()

        if (step + 1) % 20 == 0:
            b = 0
            gen_completion = tokenizer.decode((gen_seqs[0][b].tolist())[len(input_ids[b]):])
            gen_only = f"def {entries[b]}(" + gen_completion
            print(f"\n--- SAMPLE @ step {step+1} ---\nPROMPT:\n", (prompts[b][:400] + "...") if len(prompts[b])>400 else prompts[b])
            print("GEN:\n", (gen_only[:800] + "...") if len(gen_only)>800 else gen_only, "\n")

        if (step + 1) % 100 == 0:
            acc = quick_eval(model, tokenizer, data, device, autocast_dtype, rl, n=16)
            print(f"[EVAL] pass@1 ≈ {acc:.2f}")
            ckpt = {"model": model.state_dict(), "step": step + 1}
            torch.save(ckpt, f"ckpt_rl_step_{step+1}.pt")

if __name__ == "__main__":
    main()
