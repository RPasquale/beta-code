"""
k_small_big.py
---------------
Self-contained training pipeline for the compact K2-mini model that runs:
  1. Supervised fine-tuning (SFT) warm-up on DeepCoder.
  2. Reinforcement learning (RLVR-style) on DeepCoder with verifiable rewards.
  3. SPCT (Self-Principled Critique Tuning) on DeepMind CodeFeedback.

The script keeps everything in one file so you can schedule long overnight runs
without juggling multiple entry points. Each stage can be enabled/disabled via
CLI flags, and checkpoints are saved between stages for safety.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict, concatenate_datasets
import numpy as np

# Optional: teacher model via Hugging Face
try:
    from transformers import AutoTokenizer as HFAutoTokenizer, AutoModelForCausalLM as HFAutoModel
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# -----------------------------------------------------------------------------
# Device helpers
# -----------------------------------------------------------------------------


def pick_device(preferred: Optional[str] = None) -> str:
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


def format_seconds(seconds: float) -> str:
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Model definition (K2-mini with ASCII-safe decoding)
# -----------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_float = x.to(torch.float32)
        norm = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_float = x_float * torch.rsqrt(norm + self.eps)
        out = self.weight.to(torch.float32) * x_float
        return out.to(orig_dtype)


def build_rope_cache(head_dim: int, max_seq_len: int, base: float = 10000.0, device=None, dtype=torch.float32):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    B, T, H, D = x.shape
    x_ = x.view(B, T, H, D // 2, 2)
    x1, x2 = x_[..., 0], x_[..., 1]
    cos = cos[:T, :].view(1, T, 1, -1)
    sin = sin[:T, :].view(1, T, 1, -1)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack([out1, out2], dim=-1).view(B, T, H, D)


def masked_softmax_fp32(logits: torch.Tensor, mask: Optional[torch.Tensor], dim: int = -1) -> torch.Tensor:
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
        self.last_max_logits: Optional[torch.Tensor] = None

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

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_logits = torch.einsum("bthd,bshd->bhts", Q, K).float() * scale

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
            w = linear.weight
            out, din = w.shape
            assert out == H * self.head_dim
            w = w.view(H, self.head_dim, din)
            factor = scale.sqrt() if sqrt else scale
            w[needs] *= factor[needs].view(-1, 1, 1)
            linear.weight.copy_(w.view(H * self.head_dim, din))

        per_head_scale_linear(self.Wq_c, gamma, sqrt=True)
        per_head_scale_linear(self.Wk_c, gamma, sqrt=True)
        per_head_scale_linear(self.Wq_r, gamma, sqrt=False)


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
        logits = self.w(x.float())
        topk_val, topk_idx = torch.topk(logits, self.k, dim=-1)
        gates = F.softmax(topk_val, dim=-1).to(x.dtype)
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
        topk_idx, gates, logits = self.router(x)
        probs = F.softmax(logits.float(), dim=-1)
        me = probs.mean(dim=(0, 1))
        ce = (probs > (1.0 / self.n_experts)).float().mean(dim=(0, 1))
        aux_loss = (me * ce).sum() * self.n_experts

        out = torch.zeros_like(x)
        for i in range(self.k):
            idx = topk_idx[..., i]
            gate = gates[..., i].unsqueeze(-1)
            chunk_out = torch.zeros_like(x)
            for e in range(self.n_experts):
                sel = (idx == e)
                if sel.any():
                    x_sel = x[sel]
                    y = self.experts[e](x_sel)
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
    n_experts_total: int = 8
    top_k: int = 2
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
        self._code_masks: Dict[str, torch.Tensor] = {}

    def _ascii_mask(self, device) -> torch.Tensor:
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

    def _code_mask(self, device) -> torch.Tensor:
        """More restrictive than ASCII: allow only bytes commonly used in Python code.
        Excludes curly braces and exotic punctuation that often breaks parsing.
        """
        dev = torch.device(device)
        key = f"{dev.type}:{dev.index if dev.index is not None else 'default'}"
        mask = self._code_masks.get(key)
        if mask is None or mask.device != dev:
            allowed = set([9, 10, 13, 32, 34, 39, 40, 41, 44, 46, 58, 61, 91, 93, 43, 45, 42, 47, 37, 60, 62, 33])
            # digits
            allowed.update(range(48, 58))
            # uppercase A-Z
            allowed.update(range(65, 91))
            # underscore
            allowed.add(95)
            # lowercase a-z
            allowed.update(range(97, 123))
            base = torch.zeros(self.cfg.vocab_size, dtype=torch.bool)
            idxs = [i for i in allowed if 0 <= i < self.cfg.vocab_size]
            base[idxs] = True
            mask = base.to(dev)
            self._code_masks[key] = mask
        return mask

    def forward(self, input_ids, attention_mask=None, labels=None):
        B, T = input_ids.shape
        x = self.tok_emb(input_ids)
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

    def logprobs(self, input_ids):
        logits, _, _ = self.forward(input_ids)
        logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
        tgt = input_ids[:, 1:].unsqueeze(-1)
        return logp.gather(-1, tgt).squeeze(-1)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=128, temperature=0.8, top_p=0.95, allowed_mask: Optional[torch.Tensor] = None):
        self.eval()
        out = input_ids
        mask = allowed_mask if allowed_mask is not None else self._ascii_mask(out.device)
        for _ in range(max_new_tokens):
            logits, _, _ = self.forward(out)
            next_logits = logits[:, -1, :].float().masked_fill(~mask, float("-inf"))
            if temperature > 0:
                next_logits = next_logits / temperature
                probs = F.softmax(next_logits, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                    cum = torch.cumsum(sorted_probs, dim=-1)
                    keep = cum <= top_p
                    keep[..., 0] = True
                    filtered = torch.zeros_like(probs)
                    filtered.scatter_(dim=-1, index=sorted_idx, src=sorted_probs * keep)
                    denom = filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    probs = filtered / denom
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            out = torch.cat([out, next_token], dim=1)
        return out


# -----------------------------------------------------------------------------
# DeepCoder data adapter & reward
# -----------------------------------------------------------------------------


class DeepCoderAdapter:
    def __init__(self, path_or_repo: str):
        self.split = self._load_any(path_or_repo)
        self.prompt_key = "prompt"
        self.solution_key = "gold_standard_solution"
        self.verif_key = "verification_info"

    def _is_saved_dataset_dir(self, p: str) -> bool:
        return os.path.exists(os.path.join(p, "dataset_dict.json")) or os.path.exists(os.path.join(p, "state.json"))

    def _try_load_arrow_shards(self, p: str):
        shard_paths = sorted(Path(p).glob("*train-*.arrow"))
        if not shard_paths:
            return None
        parts = [Dataset.from_file(str(sp)) for sp in shard_paths]
        return concatenate_datasets(parts)

    def _load_any(self, p: str):
        if os.path.isdir(p) and self._is_saved_dataset_dir(p):
            return load_from_disk(p)
        if os.path.isdir(p):
            ds = self._try_load_arrow_shards(p)
            if ds is not None:
                return ds
        try:
            return load_dataset(
                "PrimeIntellect/deepcoder-gold-standard-solutions",
                split="train",
                local_files_only=True,
            )
        except Exception:
            return load_dataset("PrimeIntellect/deepcoder-gold-standard-solutions", split="train")

    def __len__(self):
        return len(self.split)

    def get_prompt(self, idx: int) -> str:
        return str(self.split[idx][self.prompt_key])

    def get_reference(self, idx: int) -> Dict[str, Any]:
        rec = self.split[idx]
        ref: Dict[str, Any] = {"solution": rec.get(self.solution_key)}
        vi = rec.get(self.verif_key)
        if isinstance(vi, str) and vi.strip():
            try:
                ref["verification_info"] = json.loads(vi)
            except Exception:
                ref["verification_info"] = vi
        return ref


def run_python_code_with_tests(code: str, tests: List[Dict[str, Any]], timeout_sec: int = 2) -> bool:
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
        p = subprocess.Popen([sys.executable, driver], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmp)
        out, err = p.communicate(input=json.dumps(tests).encode("utf-8"), timeout=timeout_sec)
        return out.decode("utf-8").strip() == "OK"


def deepcoder_reward(generated: str, reference: Dict[str, Any]) -> float:
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
    # String-level heuristics (helpful when AST parse fails)
    has_entry_str = False
    has_return_str = False
    try:
        has_entry_str = (f"def {entry}" in generated)
        has_return_str = ("return" in generated)
    except Exception:
        pass

    try:
        import ast
        tree = ast.parse(generated)
        has_entry = any(isinstance(n, ast.FunctionDef) and n.name == entry for n in tree.body)
        if has_entry:
            bonus += 0.10
        # Small shaping to create variance: reward a "return" anywhere.
        has_return = any(isinstance(n, ast.Return) for n in ast.walk(tree))
        if has_return:
            bonus += 0.05
    except Exception:
        # Try to salvage by trimming trailing incomplete lines.
        try:
            import ast
            lines = generated.splitlines()
            salvaged = None
            for _ in range(min(3, len(lines))):
                try:
                    code2 = "\n".join(lines)
                    tree = ast.parse(code2)
                    salvaged = code2
                    break
                except Exception:
                    lines = lines[:-1]
            if salvaged is not None:
                # Parsed after salvage
                has_entry = any(isinstance(n, ast.FunctionDef) and n.name == entry for n in tree.body)
                if has_entry:
                    bonus += 0.10
                has_return = any(isinstance(n, ast.Return) for n in ast.walk(tree))
                if has_return:
                    bonus += 0.05
                generated = salvaged
            else:
                # No salvage possible: heuristic bonus even when parse fails to encourage structure.
                if has_entry_str:
                    bonus += 0.05
                if has_return_str:
                    bonus += 0.03
                sol = reference.get("solution")
                match = 1.0 if isinstance(sol, str) and generated.strip() == sol.strip() else 0.0
                return max(bonus, match)
        except Exception:
            if has_entry_str:
                bonus += 0.05
            if has_return_str:
                bonus += 0.03
            sol = reference.get("solution")
            match = 1.0 if isinstance(sol, str) and generated.strip() == sol.strip() else 0.0
            return max(bonus, match)
    if tests:
        try:
            ok_first = run_python_code_with_tests(generated, tests[:1])
            if ok_first:
                bonus += 0.20
            passed = 0
            for t in tests:
                try:
                    if run_python_code_with_tests(generated, [t]):
                        passed += 1
                except Exception:
                    pass
            frac = passed / max(1, len(tests))
            return min(1.0, bonus + 0.70 * frac)
        except Exception:
            return bonus
    else:
        sol = reference.get("solution")
        return max(bonus, 1.0 if isinstance(sol, str) and generated.strip() == sol.strip() else 0.0)


def debug_reward(generated: str, reference: Dict[str, Any]) -> Dict[str, Any]:
    info = {
        "parse_ok": False,
        "has_entry": False,
        "first_test_ok": None,
        "n_passed": 0,
        "n_tests": 0,
        "entry": "solve",
        "has_return": False,
        "has_entry_str": False,
        "has_return_str": False,
        "parse_error": None,
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
        tree = ast.parse(generated)
        info["parse_ok"] = True
        info["has_entry"] = (f"def {entry}" in generated)
        info["has_return"] = any(isinstance(n, ast.Return) for n in ast.walk(tree))
    except Exception as e:
        # String-level checks even if parse failed
        info["has_entry_str"] = (f"def {entry}" in generated)
        info["has_return_str"] = ("return" in generated)
        try:
            msg = str(e)
            if len(msg) > 160:
                msg = msg[:157] + "..."
            info["parse_error"] = msg
        except Exception:
            info["parse_error"] = "syntax"
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


# -----------------------------------------------------------------------------
# Tokenizer & prompt helpers
# -----------------------------------------------------------------------------


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


def make_prompt(raw_prompt: str, entry: str = "solve") -> str:
    prompt = (
        "You are a Python coding assistant.\n"
        f"Write valid Python 3 code that defines a function `{entry}` and returns the answer.\n"
        "Do not print; do not use input(); just return the result.\n"
        "Use only the standard library. Keep it concise.\n"
        "Task:\n"
        f"{raw_prompt}\n\n"
        "# Your code below:\n"
    )
    return prompt


def clean_solution_text(raw: Optional[str], entry: str) -> Tuple[str, bool]:
    if not isinstance(raw, str):
        return "", False
    txt = raw.strip()
    if not txt:
        return "", False
    if txt.startswith("```"):
        parts = txt.split("```")
        cleaned_part = ""
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if part.lower().startswith("python"):
                part = part[len("python"):].lstrip("\n")
            if part:
                cleaned_part = part
                break
        if cleaned_part:
            txt = cleaned_part
        else:
            txt = txt.lstrip("`")
    txt = txt.strip()
    txt = txt.replace("```python", "").replace("```", "")
    txt = txt.replace("`", "")
    anchor = f"def {entry}"
    idx = txt.find(anchor)
    has_anchor = False
    if idx != -1:
        txt = txt[idx:]
        has_anchor = True
    else:
        has_anchor = txt.startswith("def ")
    return txt.strip(), has_anchor


def _strip_code_fences(raw: Optional[str]) -> str:
    if not isinstance(raw, str):
        return ""
    txt = raw.strip()
    if not txt:
        return ""
    if txt.startswith("```"):
        parts = txt.split("```")
        # prefer first non-empty block (skip possible language tag)
        for part in parts:
            p = part.strip()
            if not p:
                continue
            if p.lower().startswith("python"):
                p = p[len("python"):].lstrip("\n")
            if p:
                txt = p
                break
    txt = txt.replace("```python", "").replace("```", "")
    txt = txt.replace("`", "")
    return txt.strip()


def _assemble_code_from_body(entry: str, body: str) -> str:
    # Remove any fences/ticks and normalize whitespace
    body_clean = _strip_code_fences(body)
    if not body_clean:
        body_clean = "pass"
    lines = body_clean.splitlines()
    indented = []
    for ln in lines:
        if ln.strip():
            indented.append("    " + ln)
        else:
            indented.append("")
    indented_block = "\n".join(indented)
    return f"def {entry}(*args):\n{indented_block}\n"


def sanitize_generated_python(body: str) -> str:
    """Remove characters outside a conservative Python subset to reduce syntax noise.
    This is applied only to RL-generated bodies before assembly.
    """
    if not isinstance(body, str):
        return ""
    allowed = set("\t\r\n "
                  "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
                  "()[]:,.=+-*/%<>!\'\"")
    # Note: intentionally exclude { } ` ~ | ; @ # to avoid common parse-breaking noise
    cleaned_chars = []
    for ch in body:
        if ch in allowed:
            cleaned_chars.append(ch)
        else:
            # Replace disallowed with space to keep token alignment reasonably intact
            cleaned_chars.append(" ")
    # Collapse obvious overlong spaces
    text = "".join(cleaned_chars)
    return text


# -----------------------------------------------------------------------------
# Teacher model wrapper (Hugging Face)
# -----------------------------------------------------------------------------


class TeacherModel:
    def __init__(self, model_id_or_path: str, device: str = "cuda", use_chat_template: bool = False, local_only: bool = False):
        if not HF_AVAILABLE:
            raise RuntimeError("transformers not available; install to use --teacher-id/--teacher-path")
        self.device = device
        self.use_chat_template = use_chat_template
        # Normalize a few common aliases
        alias_map = {
            "qwen-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "deepseek-r1-distill-qwen-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        }
        key = model_id_or_path.strip().lower()
        model_id = alias_map.get(key, model_id_or_path)

        tok = None
        mdl = None
        # First try local cache if requested, else permit remote
        def _load(local_only_flag: bool):
            _tok = HFAutoTokenizer.from_pretrained(model_id, local_files_only=local_only_flag)
            _mdl = HFAutoModel.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device.startswith("cuda") else None,
                local_files_only=local_only_flag,
            )
            return _tok, _mdl

        try:
            tok, mdl = _load(local_only)
        except Exception as e1:
            # Fallback: attempt remote download if local load failed
            try:
                print(f"[Teacher] Local load failed; attempting remote download for {model_id}...")
                tok, mdl = _load(False)
            except Exception as e2:
                raise e2 from e1

        self.tok = tok
        self.model = mdl
        if device.startswith("cuda"):
            self.model.to("cuda")
        elif device == "mps":
            self.model.to("mps")
        else:
            self.model.to("cpu")
        self.model.eval()

    @torch.no_grad()
    def generate_text(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.2, top_p: float = 0.9) -> str:
        if self.use_chat_template and hasattr(self.tok, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            inputs = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt")
        else:
            inputs = self.tok(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        gen = self.model.generate(
            **inputs,
            do_sample=(temperature > 0.0),
            temperature=max(1e-5, float(temperature)),
            top_p=float(top_p),
            max_new_tokens=int(max_new_tokens),
            pad_token_id=self.tok.eos_token_id if self.tok.eos_token_id is not None else 0,
        )
        # Extract only the newly generated portion
        start = inputs["input_ids"].shape[-1]
        text = self.tok.decode(gen[0][start:], skip_special_tokens=True)
        return text.strip()


# -----------------------------------------------------------------------------
# FineWeb Edu adapter (local .npy shards of text or token ids)
# -----------------------------------------------------------------------------


class HFTextDecoder:
    def __init__(self, tok_id: str, local_only: bool = False):
        if not HF_AVAILABLE:
            raise RuntimeError("transformers not available; install to use HFTextDecoder")
        self.tok = HFAutoTokenizer.from_pretrained(tok_id, local_files_only=local_only)
    def decode(self, ids: List[int]) -> str:
        try:
            return self.tok.decode(ids, skip_special_tokens=True).strip()
        except Exception:
            return ""


class FineWebEduAdapter:
    def __init__(self, root_dir: str, decoder: Optional[HFTextDecoder] = None, token_keys: Optional[List[str]] = None, debug: bool = False):
        self.root = Path(root_dir)
        if not self.root.exists():
            raise FileNotFoundError(f"FineWeb Edu path not found: {root_dir}")
        self.npy_files = sorted([p for p in self.root.rglob("*.npy") if p.is_file()])
        if not self.npy_files:
            print(f"[Edu] No .npy files found under {root_dir}")
        else:
            print(f"[Edu] Found {len(self.npy_files)} .npy shards under {root_dir}")
        self._cache: Dict[str, Any] = {}
        self.decoder = decoder
        self.token_keys = token_keys or ["input_ids", "tokens", "ids"]
        self.debug = debug
        self._debug_prints = 0

    def __len__(self):
        # Unknown without loading; return rough estimate (files count)
        return len(self.npy_files) * 1_000_000

    def _to_text(self, x) -> Optional[str]:
        try:
            if x is None:
                return None
            if isinstance(x, (bytes, bytearray)):
                return bytes(x).decode("utf-8", errors="ignore").strip()
            if isinstance(x, str):
                return x.strip()
            if isinstance(x, dict):
                # Prefer explicit text keys first
                for k in ("text", "content", "document", "body"):
                    v = x.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
                # Fallback: token ids
                for k in self.token_keys:
                    ids = x.get(k)
                    if ids is None:
                        continue
                    arr = np.asarray(ids)
                    if arr.ndim == 0:
                        continue
                    if self.decoder is not None:
                        return self.decoder.decode([int(t) for t in arr.tolist()]) or None
                    # No decoder: treat as bytes only if safe range
                    if arr.dtype.kind in ("i", "u") and arr.size > 0 and int(arr.max(initial=0)) < 256:
                        raw = bytes(int(t) & 0xFF for t in arr.tolist())
                        return raw.decode("utf-8", errors="ignore").strip()
            if isinstance(x, (list, tuple, np.ndarray)):
                arr = np.asarray(x)
                # If we have a decoder and values look like token ids, decode via decoder
                if self.decoder is not None and arr.dtype.kind in ("i", "u") and arr.size > 0 and int(arr.max(initial=0)) >= 256:
                    return self.decoder.decode([int(t) for t in arr.tolist()]) or None
                # Else only treat as bytes if values are in 0..255
                if arr.dtype.kind in ("i", "u") and arr.size > 0 and int(arr.max(initial=0)) < 256:
                    raw = bytes(int(t) & 0xFF for t in arr.tolist())
                    return raw.decode("utf-8", errors="ignore").strip()
        except Exception:
            return None
        return None

    def sample_text(self, max_attempts: int = 3) -> Optional[str]:
        if not self.npy_files:
            return None
        for _ in range(max_attempts):
            f = random.choice(self.npy_files)
            try:
                # Use memmap to avoid loading entire shard into memory
                arr = self._cache.get(str(f))
                if arr is None:
                    arr = np.load(str(f), allow_pickle=True, mmap_mode='r')
                    self._cache[str(f)] = arr
                if arr is None or len(arr) == 0:
                    continue
                # Object arrays likely hold per-sample records (text/dicts)
                if arr.dtype == object:
                    idx = random.randrange(len(arr))
                    x = arr[idx]
                    s = self._to_text(x)
                    if isinstance(s, str) and len(s) > 0:
                        return s
                # Numeric 1D arrays likely hold a long token stream (e.g., uint16 GPT-2 ids)
                elif arr.ndim == 1 and arr.dtype.kind in ('i','u'):
                    # Choose a window of token ids and decode
                    if len(arr) < 64:
                        continue
                    L = min(max(256, len(arr) // 2048), 2048)  # 256..2048 tokens
                    start = random.randrange(0, max(1, len(arr) - L))
                    window = arr[start:start+L]
                    if self.decoder is not None:
                        s = self.decoder.decode([int(t) for t in window.tolist()])
                        if isinstance(s, str) and len(s) > 0:
                            return s
                    else:
                        # Try auto-load a default GPT-2 decoder if available
                        if HF_AVAILABLE:
                            try:
                                self.decoder = HFTextDecoder('openai-community/gpt2', local_only=False)
                                if self.debug:
                                    print("[EDU] Auto-loaded decoder tokenizer: openai-community/gpt2")
                                s = self.decoder.decode([int(t) for t in window.tolist()])
                                if s:
                                    return s
                            except Exception:
                                try:
                                    self.decoder = HFTextDecoder('gpt2', local_only=False)
                                    if self.debug:
                                        print("[EDU] Auto-loaded decoder tokenizer: gpt2")
                                    s = self.decoder.decode([int(t) for t in window.tolist()])
                                    if s:
                                        return s
                                except Exception:
                                    pass
                        # Best-effort fallback: if in byte range, try raw bytes
                        if int(window.max(initial=0)) < 256:
                            raw = bytes(int(t) & 0xFF for t in window.tolist())
                            s = raw.decode('utf-8', errors='ignore').strip()
                            if s:
                                return s
                # 2D arrays: treat each row as a sequence of ids
                elif arr.ndim == 2 and arr.dtype.kind in ('i','u'):
                    ridx = random.randrange(arr.shape[0])
                    row = arr[ridx]
                    if row.size >= 32:
                        if self.decoder is not None:
                            s = self.decoder.decode([int(t) for t in row.tolist()])
                            if s:
                                return s
                        else:
                            # Try auto-load default decoder
                            if HF_AVAILABLE:
                                try:
                                    self.decoder = HFTextDecoder('openai-community/gpt2', local_only=False)
                                    if self.debug:
                                        print("[EDU] Auto-loaded decoder tokenizer: openai-community/gpt2")
                                    s = self.decoder.decode([int(t) for t in row.tolist()])
                                    if s:
                                        return s
                                except Exception:
                                    try:
                                        self.decoder = HFTextDecoder('gpt2', local_only=False)
                                        if self.debug:
                                            print("[EDU] Auto-loaded decoder tokenizer: gpt2")
                                        s = self.decoder.decode([int(t) for t in row.tolist()])
                                        if s:
                                            return s
                                    except Exception:
                                        pass
                            elif int(row.max(initial=0)) < 256:
                                raw = bytes(int(t) & 0xFF for t in row.tolist())
                                s = raw.decode('utf-8', errors='ignore').strip()
                                if s:
                                    return s
                if self.debug and self._debug_prints < 5:
                    self._debug_prints += 1
                    kind = type(x).__name__
                    if isinstance(arr, np.ndarray):
                        if arr.dtype == object:
                            print(f"[EDU DBG] shard={f.name} object array; sample keys/preview may vary")
                        else:
                            vmax = 'n/a'
                            try:
                                vmax = int(arr.max()) if arr.size>0 and arr.dtype.kind in ('i','u','f') else 'n/a'
                            except Exception:
                                pass
                            print(f"[EDU DBG] shard={f.name} dtype={arr.dtype} shape={arr.shape} vmax={vmax}")
                    else:
                        rx = repr(arr)
                        print(f"[EDU DBG] shard={f.name} kind={kind} sample={rx[:160]}{'...' if len(rx)>160 else ''}")
            except Exception as e:
                if self.debug and self._debug_prints < 5:
                    self._debug_prints += 1
                    print(f"[EDU DBG] failed to read {f.name}: {e}")
                continue
        return None


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


# -----------------------------------------------------------------------------
# SFT warm-up
# -----------------------------------------------------------------------------


@dataclass
class SFTConfig:
    steps: int = 1500
    lr: float = 5e-5
    max_prompt_len: int = 512
    max_label_len: int = 256
    grad_clip: float = 1.0
    log_every: int = 100
    kd_weight: float = 0.0
    teacher_max_new: int = 256
    teacher_temp: float = 0.2
    teacher_top_p: float = 0.9


# -----------------------------------------------------------------------------
# EDU KD stage (distill general knowledge from teacher on local corpus)
# -----------------------------------------------------------------------------


@dataclass
class EduKDConfig:
    steps: int = 2000
    max_prompt_len: int = 512
    max_target_len: int = 256
    lr: float = 2e-4
    grad_clip: float = 1.0
    log_every: int = 100
    teacher_max_new: int = 256
    teacher_temp: float = 0.2
    teacher_top_p: float = 0.9


def edu_kd_loop(model: K2Mini, tok: ByteTokenizer, adapter: FineWebEduAdapter, device: str, cfg: EduKDConfig):
    model.train()
    opt = build_optimizer(model, cfg.lr, (0.9, 0.95), 1e-8)
    t0 = time.time()
    last_log = t0
    teacher: Optional[TeacherModel] = getattr(cfg, "teacher", None)
    if teacher is None:
        print("[EDU] No teacher provided; skipping EDU KD stage.")
        return
    updates = 0
    skips_no_text = 0
    skips_teacher_empty = 0
    skips_nonfinite = 0
    for step in range(1, cfg.steps + 1):
        src = adapter.sample_text()
        if not isinstance(src, str) or len(src) < 32:
            skips_no_text += 1
            if step == 1 or step % max(1, cfg.log_every) == 0:
                print(f"[EDU KD] step {step}/{cfg.steps} | updates={updates} | skips(no_text)={skips_no_text}")
            continue
        # Random prompt slice from document
        Lp = min(cfg.max_prompt_len, max(32, len(src) // 8))
        start = random.randrange(max(1, len(src) - Lp)) if len(src) > Lp else 0
        prompt = src[start:start + Lp]
        try:
            t_text = teacher.generate_text(prompt, max_new_tokens=cfg.teacher_max_new, temperature=cfg.teacher_temp, top_p=cfg.teacher_top_p)
        except Exception:
            skips_teacher_empty += 1
            if step == 1 or step % max(1, cfg.log_every) == 0:
                print(f"[EDU KD] step {step}/{cfg.steps} | updates={updates} | skips(teacher)={skips_teacher_empty}")
            continue
        if not t_text:
            skips_teacher_empty += 1
            if step == 1 or step % max(1, cfg.log_every) == 0:
                print(f"[EDU KD] step {step}/{cfg.steps} | updates={updates} | skips(teacher)={skips_teacher_empty}")
            continue
        p_ids = tok.encode(prompt, cfg.max_prompt_len)
        t_ids = tok.encode(t_text, cfg.max_target_len)
        ids = (p_ids + t_ids)[: cfg.max_prompt_len + cfg.max_target_len]
        if not ids:
            skips_no_text += 1
            if step == 1 or step % max(1, cfg.log_every) == 0:
                print(f"[EDU KD] step {step}/{cfg.steps} | updates={updates} | skips(no_text)={skips_no_text}")
            continue
        x = torch.tensor(ids, device=device).unsqueeze(0)
        labels = x.clone()
        cutoff = min(len(p_ids), len(ids))
        labels[:, :cutoff] = -100
        _, aux, loss = model(x, labels=labels)
        if loss is None or not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            skips_nonfinite += 1
            if step == 1 or step % max(1, cfg.log_every) == 0:
                print(f"[EDU KD] step {step}/{cfg.steps} | updates={updates} | skips(nonfinite)={skips_nonfinite}")
            continue
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        updates += 1
        if step == 1 or step % cfg.log_every == 0 or time.time() - last_log >= 60:
            elapsed = time.time() - t0
            eta = elapsed / step * max(0, cfg.steps - step)
            print(f"[EDU KD] step {step}/{cfg.steps} | updates={updates} | loss={float(loss):.4f} | aux={float(aux):.4f} | elapsed={format_seconds(elapsed)} | eta={format_seconds(eta)}")
            last_log = time.time()
    # Final summary
    print(f"[EDU KD] complete | updates={updates} | skips(no_text)={skips_no_text} | skips(teacher)={skips_teacher_empty} | skips(nonfinite)={skips_nonfinite}")


def build_param_groups(model: nn.Module):
    decay, nodecay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_bias = n.endswith(".bias")
        is_norm = "norm" in n.lower()
        is_emb = "tok_emb" in n or "embedding" in n.lower()
        if p.ndim == 1 or is_bias or is_norm or is_emb:
            nodecay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": 0.1},
        {"params": nodecay, "weight_decay": 0.0},
    ]


def run_sft(model: K2Mini, tokenizer: ByteTokenizer, data: DeepCoderAdapter, device: str, cfg: SFTConfig, autocast_dtype=None):
    model.train()
    opt = AdamW(build_param_groups(model), lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8)
    loop_start = time.time()
    last_log = loop_start
    for step in range(cfg.steps):
        step_start = time.time()
        idx = random.randrange(len(data))
        ref = data.get_reference(idx)
        vi = ref.get("verification_info")
        entry = "solve" if not (isinstance(vi, dict) and isinstance(vi.get("entry"), str)) else vi["entry"]
        raw_prompt = data.get_prompt(idx)
        label_raw = ref.get("solution")
        label, has_def = clean_solution_text(label_raw, entry)
        if not label:
            continue
        prompt = make_prompt(raw_prompt, entry=entry)

        prompt_ids = tokenizer.encode(prompt, max_len=cfg.max_prompt_len)
        label_ids = tokenizer.encode(label, max_len=cfg.max_label_len)
        ids = (prompt_ids + label_ids)[: cfg.max_prompt_len + cfg.max_label_len]
        if not ids:
            continue
        input_ids = torch.tensor(ids, device=device).unsqueeze(0)
        labels = input_ids.clone()
        cutoff = min(len(prompt_ids), len(ids))
        labels[:, :cutoff] = -100

        if device == "cuda":
            ctx = torch.amp.autocast
            ctx_kwargs = dict(device_type="cuda", enabled=(autocast_dtype is not None), dtype=autocast_dtype)
        else:
            device_type = "mps" if device == "mps" else "cpu"
            ctx = torch.amp.autocast
            ctx_kwargs = dict(device_type=device_type, enabled=(device_type == "mps" and autocast_dtype is not None), dtype=autocast_dtype)

        with ctx(**ctx_kwargs):
            _, _, ce = model(input_ids, labels=labels)
            kd_loss = None
            teacher = getattr(cfg, "teacher", None)
            if teacher is not None and cfg.kd_weight > 0.0:
                try:
                    t_text = teacher.generate_text(prompt, max_new_tokens=min(cfg.max_label_len, cfg.teacher_max_new), temperature=cfg.teacher_temp, top_p=cfg.teacher_top_p)
                    t_ids = tokenizer.encode(t_text, max_len=cfg.max_label_len)
                    if t_ids:
                        t_all = (prompt_ids + t_ids)[: cfg.max_prompt_len + cfg.max_label_len]
                        t_x = torch.tensor(t_all, device=device).unsqueeze(0)
                        t_labels = t_x.clone()
                        cutoff2 = min(len(prompt_ids), len(t_all))
                        t_labels[:, :cutoff2] = -100
                        _, _, kd_ce = model(t_x, labels=t_labels)
                        kd_loss = kd_ce
                except Exception:
                    kd_loss = None
        if not torch.isfinite(ce):
            opt.zero_grad(set_to_none=True)
            continue

        opt.zero_grad()
        total = ce
        if kd_loss is not None and torch.isfinite(kd_loss):
            total = total + cfg.kd_weight * kd_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step == 0 or (step + 1) % cfg.log_every == 0 or (time.time() - last_log) >= 60:
            elapsed = time.time() - loop_start
            avg_step = elapsed / max(1, step + 1)
            eta = avg_step * max(0, cfg.steps - (step + 1))
            print(
                f"[SFT] step {step+1}/{cfg.steps} | loss={float(ce):.4f} | "
                f"step_time={format_seconds(time.time() - step_start)} | "
                f"elapsed={format_seconds(elapsed)} | eta={format_seconds(eta)}"
            )
            last_log = time.time()

    del opt
    model.train()


# -----------------------------------------------------------------------------
# RL stage (sample-based, verifiable rewards)
# -----------------------------------------------------------------------------


@dataclass
class RLConfig:
    steps: int = 600
    batch_size: int = 1
    K: int = 2
    max_prompt_len: int = 512
    max_gen_len: int = 128
    temperature: float = 0.85
    top_p: float = 0.95
    lr: float = 1e-4
    tau_reg: float = 0.005
    token_budget: int = 128
    grad_clip: float = 1.0
    warmup_steps: int = 100
    debug_first_n: int = 3
    kd_weight: float = 0.0
    teacher_max_new: int = 128
    teacher_temp: float = 0.2
    teacher_top_p: float = 0.9


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
            T = 0.5 * (3 * I - Z.mm(Y))
            Y = Y.mm(T)
            Z = T.mm(Z)
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


def sample_candidates(model: K2Mini, input_ids: torch.Tensor, rl_cfg: RLConfig, autocast_dtype):
    all_gen_tokens, all_logprobs = [], []
    ctx = torch.amp.autocast
    ctx_kwargs = dict(device_type=("cuda" if input_ids.is_cuda else input_ids.device.type),
                      enabled=(input_ids.is_cuda and autocast_dtype is not None),
                      dtype=autocast_dtype)
    for _ in range(rl_cfg.K):
        with torch.no_grad(), ctx(**ctx_kwargs):
            gen_ids = model.generate(
                input_ids,
                max_new_tokens=rl_cfg.max_gen_len,
                temperature=rl_cfg.temperature,
                top_p=rl_cfg.top_p,
                allowed_mask=model._code_mask(input_ids.device),
            )
        with ctx(**ctx_kwargs):
            logp = model.logprobs(gen_ids)
        all_gen_tokens.append(gen_ids)
        all_logprobs.append(logp)
    return all_gen_tokens, all_logprobs


def run_rl(model: K2Mini, tokenizer: ByteTokenizer, data: DeepCoderAdapter, device: str, cfg: RLConfig, autocast_dtype=None):
    model.train()
    base_optim = AdamW(build_param_groups(model), lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8)
    optim = MuonClip(model, base_optim, tau=model.cfg.qk_clip_tau, use_muon_like=False)
    rl_start = time.time()
    rl_last_log = rl_start
    for step in range(cfg.steps):
        step_start = time.time()
        idxs = random.sample(range(len(data)), cfg.batch_size)
        prompts, refs, entries = [], [], []
        for i in idxs:
            ref = data.get_reference(i)
            vi = ref.get("verification_info")
            entry = "solve" if not (isinstance(vi, dict) and isinstance(vi.get("entry"), str)) else vi["entry"]
            # Preseed a valid function header so generated text is likely a function body.
            # We reconstruct the full code before scoring to ensure a defined entry function.
            seeded = make_prompt(data.get_prompt(i), entry=entry) + f"\n\ndef {entry}(*args):\n    "
            prompts.append(seeded)
            refs.append(ref)
            entries.append(entry)

        input_ids, attn = prepare_batch(tokenizer, prompts, cfg.max_prompt_len, device)
        gen_seqs, seq_logps = sample_candidates(model, input_ids, cfg, autocast_dtype)

        rewards = torch.zeros((cfg.batch_size, cfg.K), device=device)
        for ki in range(cfg.K):
            for b in range(cfg.batch_size):
                full_ids = gen_seqs[ki][b].tolist()
                gen_only = full_ids[len(input_ids[b]):]
                # Generated text is a body continuation after the preseeded header.
                completion_raw = tokenizer.decode(gen_only).strip()
                completion = sanitize_generated_python(completion_raw)
                code_candidate, has_def = clean_solution_text(completion, entries[b])
                if has_def:
                    code = code_candidate
                else:
                    code = _assemble_code_from_body(entries[b], completion)
                rewards[b, ki] = deepcoder_reward(code, refs[b])
                if step < cfg.debug_first_n and ki == 0 and b == 0:
                    diag = debug_reward(code, refs[b])
                    snippet = code[:400] + ("..." if len(code) > 400 else "")
                    print("[DBG] reward diagnostics:", diag)
                    print(f"[DBG] assembled_from_body={not has_def}")
                    if completion_raw != completion:
                        print(f"[DBG] sanitized_body_diff=removed {len(completion_raw)-len(completion)} chars")
                    print("[DBG] snippet:\n", snippet)

        r_bar = rewards.mean(dim=1, keepdim=True)
        advantages = rewards - r_bar

        if device == "cuda":
            with torch.amp.autocast(device_type="cuda", enabled=(autocast_dtype is not None), dtype=autocast_dtype):
                obj = 0.0
                for ki in range(cfg.K):
                    lp = seq_logps[ki]
                    used = lp[:, -cfg.token_budget:].mean(dim=-1)
                    obj = obj + (advantages[:, ki] - cfg.tau_reg * used).pow(2).mean()
                obj = obj / max(1, cfg.K)
        else:
            device_type = "mps" if device == "mps" else "cpu"
            with torch.amp.autocast(device_type=device_type, enabled=(device_type == "mps" and autocast_dtype is not None), dtype=autocast_dtype):
                obj = 0.0
                for ki in range(cfg.K):
                    lp = seq_logps[ki]
                    used = lp[:, -cfg.token_budget:].mean(dim=-1)
                    obj = obj + (advantages[:, ki] - cfg.tau_reg * used).pow(2).mean()
                obj = obj / max(1, cfg.K)

        if not torch.isfinite(obj):
            print(f"[RL WARN] step {step+1}: non-finite objective; skipping")
            optim.zero_grad(set_to_none=True)
            for g in base_optim.param_groups:
                g["lr"] = max(g["lr"] / 2, 1e-6)
            continue

        _, aux_loss, _ = model(input_ids)
        loss = obj + 1e-3 * aux_loss
        # Teacher imitation auxiliary: 1 completion per prompt
        teacher = getattr(cfg, "teacher", None)
        if teacher is not None and cfg.kd_weight > 0.0:
            kd_losses = []
            for b, ptxt in enumerate(prompts):
                try:
                    t_text = teacher.generate_text(ptxt, max_new_tokens=min(cfg.max_gen_len, cfg.teacher_max_new), temperature=cfg.teacher_temp, top_p=cfg.teacher_top_p)
                    t_ids = tokenizer.encode(t_text, max_len=cfg.teacher_max_new)
                    if t_ids:
                        p_ids_loc = tokenizer.encode(ptxt, max_len=cfg.max_prompt_len)
                        t_all = (p_ids_loc + t_ids)[: cfg.max_prompt_len + cfg.teacher_max_new]
                        t_x = torch.tensor(t_all, device=device).unsqueeze(0)
                        t_labels = t_x.clone()
                        cutoff2 = min(len(p_ids_loc), len(t_all))
                        t_labels[:, :cutoff2] = -100
                        _, _, kd_ce = model(t_x, labels=t_labels)
                        if torch.isfinite(kd_ce):
                            kd_losses.append(kd_ce)
                except Exception:
                    continue
            if kd_losses:
                kd_term = torch.stack(kd_losses).mean()
                loss = loss + cfg.kd_weight * kd_term

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        bad_grad = False
        for p in model.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                bad_grad = True
                break
        if bad_grad:
            print(f"[RL WARN] step {step+1}: non-finite grad; skipping & halving LR")
            optim.zero_grad(set_to_none=True)
            for g in base_optim.param_groups:
                g["lr"] = max(g["lr"] / 2, 1e-6)
            continue

        optim.step()

        if step < cfg.warmup_steps:
            scale = (step + 1) / cfg.warmup_steps
            for g in base_optim.param_groups:
                g["lr"] = cfg.lr * scale

        should_log = (
            step == 0
            or (step + 1) % max(1, cfg.steps // 20) == 0
            or (time.time() - rl_last_log) >= 60
            or (step + 1) == cfg.steps
        )
        if should_log:
            elapsed = time.time() - rl_start
            avg_step = elapsed / max(1, step + 1)
            eta = avg_step * max(0, cfg.steps - (step + 1))
            reward_mean = float(rewards.mean().cpu())
            reward_std = float(rewards.std(unbiased=False).cpu())
            lr_val = base_optim.param_groups[0].get("lr", cfg.lr)
            print(
                f"[RL] step {step+1}/{cfg.steps} | loss={float(loss):.4f} | obj={float(obj):.4f} | aux={float(aux_loss):.4f} | "
                f"reward_mean={reward_mean:.3f} | reward_std={reward_std:.3f} | lr={lr_val:.2e} | "
                f"step_time={format_seconds(time.time() - step_start)} | elapsed={format_seconds(elapsed)} | eta={format_seconds(eta)}"
            )
            rl_last_log = time.time()

    del optim
    model.train()


# -----------------------------------------------------------------------------
# SPCT stage (DeepMind CodeFeedback RFT + GRPO)
# -----------------------------------------------------------------------------


SPCT_DEFAULT_PRINCIPLES = [
    "Reward responses that solve the task correctly and pass all tests.",
    "Prefer code that is concise, readable, and adheres to the requested interface.",
    "Penalize logic errors, missing edge-case handling, or unintended side effects.",
]


class CodeFeedbackAdapter:
    DEFAULT_REPO = "deepmind/code_contests_codefeedback"

    def __init__(self, path_or_repo: Optional[str] = None, split: str = "train", sample_limit: Optional[int] = None):
        repo = path_or_repo or self.DEFAULT_REPO
        self.dataset = self._load(repo, split)
        if sample_limit:
            self.dataset = self.dataset.select(range(min(sample_limit, len(self.dataset))))
        self.records = self._normalize_all(self.dataset)
        if not self.records:
            print("[SPCT] WARN: No usable records parsed.")

    def _load(self, path, split):
        if os.path.isdir(path):
            ds = load_from_disk(path)
            if isinstance(ds, DatasetDict):
                return ds[split]
            return ds
        return load_dataset(path, split=split, trust_remote_code=True)

    def _normalize_all(self, ds):
        out = []
        for rec in ds:
            task = rec.get("prompt") or rec.get("question") or rec.get("problem") or rec.get("instruction") or ""
            if not isinstance(task, str):
                task = str(task)
            raw = rec.get("responses") or rec.get("solutions") or rec.get("candidates") \
                or rec.get("completions") or rec.get("samples") or rec.get("model_outputs")
            if raw is None:
                continue
            cands = []
            best_idx = None
            best_score = -1e9
            for i, c in enumerate(raw):
                if isinstance(c, str):
                    code = c
                    fb = ""
                    score = None
                    corr = None
                else:
                    code = c.get("solution") or c.get("code") or c.get("completion") or c.get("response") or c.get("text") or ""
                    fb = c.get("feedback") or c.get("critique") or c.get("rationale") or c.get("explanation") or ""
                    sc = c.get("score") or c.get("rating") or c.get("reward") or c.get("label") or c.get("quality")
                    if isinstance(sc, (list, tuple)) and sc:
                        sc = sc[0]
                    score = float(sc) if isinstance(sc, (int, float)) else None
                    v = c.get("verdict") or c.get("status") or c.get("result") or c.get("passed") or c.get("is_correct")
                    if isinstance(v, bool):
                        corr = v
                    elif isinstance(v, (int, float)):
                        corr = v > 0
                    else:
                        corr = None
                code = code or ""
                fb = fb or ""
                corr = bool(corr) if corr is not None else False
                cands.append({"code": code, "feedback": fb, "score": score, "is_correct": corr})
                ns = score if isinstance(score, (int, float)) else (1.0 if corr else 0.0)
                if ns > best_score:
                    best_score = ns
                    best_idx = i
            if best_idx is None:
                best_idx = 0
            pr = rec.get("principles")
            if isinstance(pr, list):
                pr = [str(p).strip() for p in pr if str(p).strip()]
            else:
                pr = []
            out.append({"task": task, "candidates": cands, "best_index": int(best_idx), "principles": pr})
        return out

    def __len__(self):
        return len(self.records)

    def sample(self):
        return random.choice(self.records) if self.records else None


def spct_prompt(record: Dict[str, Any]) -> str:
    lines = [
        "You are a self-principled reward model for coding tasks.",
        "Review each candidate solution, derive guiding principles, critique them, and assign integer scores.",
        "When you finish, identify the single best response.",
        "",
        "### Task",
        record.get("task", "").strip(),
        "",
        "### Candidate Responses",
    ]
    for i, c in enumerate(record.get("candidates", []), start=1):
        lines.append(f"[Response {i}]")
        lines.append(c.get("code", "").rstrip() or "(no code provided)")
        lines.append("")
    lines.append("Respond with a structured analysis listing principles, per-response critiques, integer scores, and the index of the best response.")
    return "\n".join(lines)


def spct_target(record: Dict[str, Any]) -> str:
    principles = record.get("principles") or SPCT_DEFAULT_PRINCIPLES
    cands = record.get("candidates", [])
    best = int(record.get("best_index", 0))
    crit = []
    scores = []
    for i, c in enumerate(cands, start=1):
        fb = c.get("feedback") or ""
        corr = bool(c.get("is_correct"))
        if fb:
            ct = fb.strip()
        elif corr:
            ct = "Implementation passes tests and returns the expected outputs."
        else:
            ct = "Fails to produce the required result or violates task constraints."
        crit.append(f"{i}. Response {i}: {ct}")
        sc = c.get("score")
        scores.append(f"{i}. Response {i}: {int(sc) if isinstance(sc, (int, float)) else (10 if corr else 0)}")
    parts = [
        "Principles:", *[f"- {p}" for p in principles], "",
        "Critiques:", *crit, "",
        "Scores:", *scores, "",
        f"Best response: Response {best + 1}"
    ]
    return "\n".join(parts)


@dataclass
class SPCTRFTConfig:
    steps: int = 800
    max_prompt_len: int = 768
    max_target_len: int = 512
    lr: float = 5e-5
    grad_clip: float = 1.0
    log_every: int = 50
    kd_weight: float = 0.0
    teacher_max_new: int = 512
    teacher_temp: float = 0.2
    teacher_top_p: float = 0.9


@dataclass
class SPCTGRPOConfig:
    steps: int = 200
    samples_per_prompt: int = 4
    max_prompt_len: int = 768
    max_target_len: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    lr: float = 1e-4
    grad_clip: float = 1.0
    kl_beta: float = 0.02
    ema_decay: float = 0.999
    kd_weight: float = 0.0
    teacher_max_new: int = 512
    teacher_temp: float = 0.2
    teacher_top_p: float = 0.9


def build_optimizer(model, lr, betas, eps):
    decay, nodecay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_bias = n.endswith(".bias")
        is_norm = "norm" in n.lower()
        is_emb = "tok" in n.lower() and "weight" in n
        if p.ndim == 1 or is_bias or is_norm or is_emb:
            nodecay.append(p)
        else:
            decay.append(p)
    return AdamW(
        [
            {"params": decay, "weight_decay": 0.1},
            {"params": nodecay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=betas,
        eps=eps,
    )


def spct_rft_loop(model: K2Mini, tok: ByteTokenizer, adapter: CodeFeedbackAdapter, device: str, cfg: SPCTRFTConfig):
    model.train()
    opt = build_optimizer(model, cfg.lr, (0.9, 0.95), 1e-8)
    t0 = time.time()
    last_log = t0
    for step in range(1, cfg.steps + 1):
        rec = adapter.sample()
        if rec is None:
            continue
        prompt = spct_prompt(rec)
        target = spct_target(rec)
        p_ids = tok.encode(prompt, cfg.max_prompt_len)
        t_ids = tok.encode(target, cfg.max_target_len)
        ids = (p_ids + t_ids)[: cfg.max_prompt_len + cfg.max_target_len]
        if not ids:
            continue
        x = torch.tensor(ids, device=device).unsqueeze(0)
        labels = x.clone()
        cutoff = min(len(p_ids), len(ids))
        labels[:, :cutoff] = -100

        logits, aux, loss = model(x, labels=labels)
        teacher = getattr(cfg, "teacher", None)
        if teacher is not None and cfg.kd_weight > 0.0:
            try:
                t_text = teacher.generate_text(prompt, max_new_tokens=min(cfg.max_target_len, cfg.teacher_max_new), temperature=cfg.teacher_temp, top_p=cfg.teacher_top_p)
                t_ids = tok.encode(t_text, cfg.max_target_len)
                if t_ids:
                    t_all = (p_ids + t_ids)[: cfg.max_prompt_len + cfg.max_target_len]
                    t_x = torch.tensor(t_all, device=device).unsqueeze(0)
                    t_labels = t_x.clone()
                    cutoff2 = min(len(p_ids), len(t_all))
                    t_labels[:, :cutoff2] = -100
                    _, _, kd_ce = model(t_x, labels=t_labels)
                    if torch.isfinite(kd_ce):
                        loss = loss + cfg.kd_weight * kd_ce
            except Exception:
                pass
        if loss is None or not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            continue

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step == 1 or step % cfg.log_every == 0 or time.time() - last_log >= 60:
            elapsed = time.time() - t0
            eta = elapsed / step * max(0, cfg.steps - step)
            print(f"[SPCT RFT] step {step}/{cfg.steps} | loss={float(loss):.4f} | aux={float(aux):.4f} | elapsed={format_seconds(elapsed)} | eta={format_seconds(eta)}")
            last_log = time.time()


class EMAPolicy:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for t in self.shadow.values():
            t.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=(1.0 - self.decay))


def spct_grpo_loop(model: K2Mini, tok: ByteTokenizer, adapter: CodeFeedbackAdapter, device: str, cfg: SPCTGRPOConfig):
    model.train()
    opt = build_optimizer(model, cfg.lr, (0.9, 0.95), 1e-8)
    ema = EMAPolicy(model, decay=cfg.ema_decay) if cfg.kl_beta > 0 else None
    t0 = time.time()
    last_log = t0
    for step in range(1, cfg.steps + 1):
        rec = adapter.sample()
        if rec is None:
            continue
        prompt = spct_prompt(rec)
        p_ids = tok.encode(prompt, cfg.max_prompt_len)
        if not p_ids:
            continue
        x = torch.tensor(p_ids, device=device).unsqueeze(0)

        samples = []
        logps = []
        with torch.no_grad():
            for _ in range(cfg.samples_per_prompt):
                gen = model.generate(x, max_new_tokens=cfg.max_target_len, temperature=cfg.temperature, top_p=cfg.top_p)
                samples.append(gen)
                logp = model.logprobs(gen)
                logps.append(logp)

        best = int(rec.get("best_index", 0)) + 1
        rewards = []
        for g in samples:
            text = tok.decode(g[0].tolist()[len(p_ids):]).lower()
            if f"best response: response {best}" in text:
                rewards.append(1.0)
            else:
                digits = "".join([ch for ch in text if ch.isdigit()])
                rewards.append(0.5 if str(best) in digits else -1.0)
        R = torch.tensor(rewards, device=device, dtype=torch.float32)
        adv = R - R.mean()

        loss = 0.0
        for a, lp in zip(adv, logps):
            seq_lp = lp[:, -cfg.max_target_len:].mean()
            loss = loss - a * seq_lp
        loss = loss / max(1, len(logps))

        # Teacher imitation term
        teacher = getattr(cfg, "teacher", None)
        if teacher is not None and cfg.kd_weight > 0.0:
            try:
                t_text = teacher.generate_text(prompt, max_new_tokens=min(cfg.max_target_len, cfg.teacher_max_new), temperature=cfg.teacher_temp, top_p=cfg.teacher_top_p)
                t_ids = tok.encode(t_text, cfg.max_target_len)
                if t_ids:
                    t_all = (p_ids + t_ids)[: cfg.max_prompt_len + cfg.max_target_len]
                    t_x = torch.tensor(t_all, device=device).unsqueeze(0)
                    t_labels = t_x.clone()
                    cutoff2 = min(len(p_ids), len(t_all))
                    t_labels[:, :cutoff2] = -100
                    _, _, kd_ce = model(t_x, labels=t_labels)
                    if torch.isfinite(kd_ce):
                        loss = loss + cfg.kd_weight * kd_ce
            except Exception:
                pass

        if cfg.kl_beta > 0 and ema is not None:
            teacher = K2Mini(model.cfg).to(device)
            teacher.load_state_dict(ema.shadow, strict=True)
            with torch.no_grad():
                t_logits, _, _ = teacher(x)
            s_logits, _, _ = model(x)
            p = F.log_softmax(s_logits[:, -1, :].float(), dim=-1)
            q = F.softmax(t_logits[:, -1, :].float(), dim=-1)
            kl = F.kl_div(p, q, reduction="batchmean")
            loss = loss + cfg.kl_beta * kl

        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            continue

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if ema:
            ema.update(model)

        if step == 1 or step % max(1, cfg.steps // 10) == 0 or time.time() - last_log >= 60:
            elapsed = time.time() - t0
            print(f"[SPCT GRPO] step {step}/{cfg.steps} | reward_mean={float(R.mean()):.3f} | reward_std={float(R.std(unbiased=False)):.3f} | loss={float(loss):.4f} | elapsed={format_seconds(elapsed)}")
            last_log = time.time()


# -----------------------------------------------------------------------------
# Checkpoint utilities
# -----------------------------------------------------------------------------


def save_checkpoint(model: K2Mini, path: Path) -> None:
    ckpt = {
        "model": model.state_dict(),
        "config": model.cfg.__dict__,
    }
    torch.save(ckpt, str(path))


def load_checkpoint(model: K2Mini, path: Path) -> None:
    ckpt = torch.load(str(path), map_location="cpu")
    state = ckpt.get("model")
    if state is None:
        raise ValueError(f"Checkpoint {path} missing 'model' key.")
    model.load_state_dict(state, strict=True)


# -----------------------------------------------------------------------------
# CLI orchestration
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Unified SFT + RL + SPCT pipeline for K2-mini.")
    parser.add_argument("--device", type=str, default="auto", help="Device override: auto, cpu, cuda, cuda:<idx>, mps.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility.")

    # EDU KD
    parser.add_argument("--run-edu", action="store_true", help="Run initial EDU distillation using teacher over local .npy corpus.")
    parser.add_argument("--edu-path", type=str, default=r"C:\\Users\\Admin\\GPT_2\\edu_fineweb10B", help="Path to FineWeb Edu .npy shards.")
    parser.add_argument("--edu-steps", type=int, default=2000)
    parser.add_argument("--edu-max-prompt", type=int, default=512)
    parser.add_argument("--edu-max-target", type=int, default=256)
    parser.add_argument("--edu-log-every", type=int, default=50)
    parser.add_argument("--edu-tokenizer-id", type=str, default=None, help="HF tokenizer to decode token-id corpora (e.g., openai-community/gpt2)")
    parser.add_argument("--edu-tokenizer-local", action="store_true", help="Load EDU tokenizer locally only (no network)")
    parser.add_argument("--edu-token-key", type=str, default="input_ids", help="Dict key for token ids in npy samples (comma-separated to try multiple)")
    parser.add_argument("--edu-debug", action="store_true", help="Log a few sample decodings/dtypes for EDU shards")

    # SFT
    parser.add_argument("--run-sft", action="store_true", help="Run the SFT warm-up stage.")
    parser.add_argument("--sft-steps", type=int, default=1500)
    parser.add_argument("--sft-lr", type=float, default=5e-5)
    parser.add_argument("--sft-max-prompt", type=int, default=512)
    parser.add_argument("--sft-max-label", type=int, default=256)

    # RL
    parser.add_argument("--run-rl", action="store_true", help="Run the RL stage after SFT.")
    parser.add_argument("--rl-steps", type=int, default=600)
    parser.add_argument("--rl-batch-size", type=int, default=1)
    parser.add_argument("--rl-K", type=int, default=2)
    parser.add_argument("--rl-lr", type=float, default=1e-4)
    parser.add_argument("--rl-temperature", type=float, default=0.85)
    parser.add_argument("--rl-top-p", type=float, default=0.95)

    # Teacher distillation (optional)
    parser.add_argument("--teacher-id", type=str, default=None, help="HF repo id or local path for teacher model (e.g., deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)")
    parser.add_argument("--teacher-local", action="store_true", help="Load teacher from local cache/files only (no network)")
    parser.add_argument("--teacher-chat-template", action="store_true", help="Use tokenizer.apply_chat_template for teacher prompts")
    parser.add_argument("--kd-sft", type=float, default=0.0, help="KD weight during SFT")
    parser.add_argument("--kd-rl", type=float, default=0.0, help="KD weight during RL")
    parser.add_argument("--kd-rft", type=float, default=0.0, help="KD weight during SPCT RFT")
    parser.add_argument("--kd-grpo", type=float, default=0.0, help="KD weight during SPCT GRPO")
    parser.add_argument("--teacher-max-new", type=int, default=256, help="Teacher max new tokens for distillation")
    parser.add_argument("--teacher-temp", type=float, default=0.2, help="Teacher sampling temperature for distillation")
    parser.add_argument("--teacher-top-p", type=float, default=0.9, help="Teacher top_p for distillation")

    # SPCT
    parser.add_argument("--run-spct", action="store_true", help="Run SPCT (RFT + GRPO) after RL.")
    parser.add_argument("--spct-dataset", type=str, default=None, help="Local path to CodeFeedback dataset (optional).")
    parser.add_argument("--spct-rft-steps", type=int, default=800)
    parser.add_argument("--spct-grpo-steps", type=int, default=200)
    parser.add_argument("--spct-samples-per-prompt", type=int, default=4)
    parser.add_argument("--spct-kl-beta", type=float, default=0.02)

    # Data paths
    parser.add_argument("--deepcoder-path", type=str,
                        default=r"C:\Users\Admin\.cache\huggingface\datasets\PrimeIntellect___deepcoder-gold-standard-solutions\default\0.0.0\146e11ac3b13abc18109ce95cc117f8ce5e61ac3",
                        help="Local DeepCoder dataset path (falls back to HF cache if missing).")

    # Checkpointing
    parser.add_argument("--load", type=str, default=None, help="Load model weights from checkpoint before training.")
    parser.add_argument("--save-dir", type=str, default="checkpoints_big", help="Directory to store stage checkpoints.")
    parser.add_argument("--preview-samples", type=int, default=0, help="Generate this many samples after SFT for inspection.")

    args = parser.parse_args()

    device = pick_device(args.device)
    autocast_dtype = None
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

    set_seed(args.seed)

    cfg = K2Config()
    model = K2Mini(cfg).to(device)
    model.to(dtype=torch.float32)
    for m in model.modules():
        if isinstance(m, RMSNorm):
            m.to(dtype=torch.float32)

    tokenizer = ByteTokenizer(vocab_size=cfg.vocab_size)

    # Optional teacher
    teacher = None
    if args.teacher_id:
        if not HF_AVAILABLE:
            print("[WARN] transformers not installed; teacher distillation disabled.")
        else:
            try:
                print(f"[Teacher] Loading teacher: {args.teacher_id} (local_only={args.teacher_local})")
                teacher = TeacherModel(args.teacher_id, device=device, use_chat_template=args.teacher_chat_template, local_only=args.teacher_local)
            except Exception as e:
                print(f"[WARN] Failed to load teacher: {e}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.load:
        load_path = Path(args.load)
        print(f"[Load] Loading checkpoint from {load_path}")
        load_checkpoint(model, load_path)

    # Stage 0: EDU KD (teacher distillation on general text)
    if args.run_edu:
        if not args.teacher_id:
            print("[Stage] EDU requested but no --teacher-id provided; skipping.")
        else:
            print("[Stage] EDU KD starting...")
            decoder = None
            token_keys = None
            if args.edu_tokenizer_id:
                try:
                    decoder = HFTextDecoder(args.edu_tokenizer_id, local_only=args.edu_tokenizer_local)
                    print(f"[EDU] Loaded decoder tokenizer: {args.edu_tokenizer_id}")
                except Exception as e:
                    print(f"[EDU WARN] Failed to load EDU tokenizer: {e}")
            if args.edu_token_key:
                token_keys = [k.strip() for k in str(args.edu_token_key).split(',') if k.strip()]
            edu_adapter = FineWebEduAdapter(args.edu_path, decoder=decoder, token_keys=token_keys, debug=args.edu_debug)
            edu_cfg = EduKDConfig(
                steps=args.edu_steps,
                max_prompt_len=args.edu_max_prompt,
                max_target_len=args.edu_max_target,
                log_every=args.edu_log_every,
                teacher_max_new=args.teacher_max_new,
                teacher_temp=args.teacher_temp,
                teacher_top_p=args.teacher_top_p,
            )
            if teacher is not None:
                setattr(edu_cfg, "teacher", teacher)
                edu_kd_loop(model, tokenizer, edu_adapter, device, edu_cfg)
                edu_ckpt = save_dir / f"edu_{int(time.time())}.pt"
                save_checkpoint(model, edu_ckpt)
                print(f"[Save] EDU checkpoint written to {edu_ckpt}")
            else:
                print("[Stage] EDU skipped (teacher failed to load).")

    # Stage 1: SFT
    if args.run_sft:
        print("[Stage] SFT warm-up starting...")
        deepcoder = DeepCoderAdapter(args.deepcoder_path)
        print(f"[Data] DeepCoder samples: {len(deepcoder)}")
        sft_cfg = SFTConfig(
            steps=args.sft_steps,
            lr=args.sft_lr,
            max_prompt_len=args.sft_max_prompt,
            max_label_len=args.sft_max_label,
        )
        if teacher is not None and args.kd_sft > 0.0:
            sft_cfg.kd_weight = args.kd_sft
            sft_cfg.teacher_max_new = args.teacher_max_new
            sft_cfg.teacher_temp = args.teacher_temp
            sft_cfg.teacher_top_p = args.teacher_top_p
            setattr(sft_cfg, "teacher", teacher)
        run_sft(model, tokenizer, deepcoder, device, sft_cfg, autocast_dtype=autocast_dtype)
        sft_ckpt = save_dir / f"sft_{int(time.time())}.pt"
        save_checkpoint(model, sft_ckpt)
        print(f"[Save] SFT checkpoint written to {sft_ckpt}")
        if args.preview_samples > 0:
            print(f"[Preview] Generating {args.preview_samples} samples after SFT...")
            sample_indices = random.sample(range(len(deepcoder)), min(args.preview_samples, len(deepcoder)))
            model.eval()
            for idx in sample_indices:
                ref = deepcoder.get_reference(idx)
                vi = ref.get("verification_info")
                entry = "solve" if not (isinstance(vi, dict) and isinstance(vi.get("entry"), str)) else vi["entry"]
                prompt_text = make_prompt(deepcoder.get_prompt(idx), entry=entry)
                prompt_ids = tokenizer.encode(prompt_text, max_len=sft_cfg.max_prompt_len)
                if not prompt_ids:
                    continue
                inp = torch.tensor(prompt_ids, device=device).unsqueeze(0)
                with torch.no_grad():
                    gen = model.generate(inp, max_new_tokens=sft_cfg.max_label_len, temperature=0.2, top_p=0.9)
                completion_ids = gen[0].tolist()[len(prompt_ids):]
                completion = tokenizer.decode(completion_ids)
                cleaned, _ = clean_solution_text(completion, entry)
                diag = debug_reward(cleaned or completion, ref)
                snippet = (cleaned or completion)[:400]
                if len(cleaned or completion) > 400:
                    snippet += "..."
                print(f"[Preview idx={idx}] diagnostics={diag}")
                print(snippet)
            model.train()
    else:
        print("[Stage] SFT skipped.")

    # Stage 2: RL
    if args.run_rl:
        print("[Stage] RL training starting...")
        deepcoder = DeepCoderAdapter(args.deepcoder_path)
        rl_cfg = RLConfig(
            steps=args.rl_steps,
            batch_size=args.rl_batch_size,
            K=args.rl_K,
            lr=args.rl_lr,
            temperature=args.rl_temperature,
            top_p=args.rl_top_p,
        )
        if teacher is not None and args.kd_rl > 0.0:
            rl_cfg.kd_weight = args.kd_rl
            rl_cfg.teacher_max_new = args.teacher_max_new
            rl_cfg.teacher_temp = args.teacher_temp
            rl_cfg.teacher_top_p = args.teacher_top_p
            setattr(rl_cfg, "teacher", teacher)
        run_rl(model, tokenizer, deepcoder, device, rl_cfg, autocast_dtype=autocast_dtype)
        rl_ckpt = save_dir / f"rl_{int(time.time())}.pt"
        save_checkpoint(model, rl_ckpt)
        print(f"[Save] RL checkpoint written to {rl_ckpt}")
    else:
        print("[Stage] RL skipped.")

    # Stage 3: SPCT
    if args.run_spct:
        print("[Stage] SPCT training starting...")
        adapter = CodeFeedbackAdapter(args.spct_dataset, split="train")
        print(f"[Data] CodeFeedback records: {len(adapter)} (source={'local' if args.spct_dataset else 'hf'})")
        rft_cfg = SPCTRFTConfig(steps=args.spct_rft_steps)
        grpo_cfg = SPCTGRPOConfig(
            steps=args.spct_grpo_steps,
            samples_per_prompt=args.spct_samples_per_prompt,
            kl_beta=args.spct_kl_beta,
        )
        if teacher is not None and args.kd_rft > 0.0:
            rft_cfg.kd_weight = args.kd_rft
            rft_cfg.teacher_max_new = args.teacher_max_new
            rft_cfg.teacher_temp = args.teacher_temp
            rft_cfg.teacher_top_p = args.teacher_top_p
            setattr(rft_cfg, "teacher", teacher)
        if teacher is not None and args.kd_grpo > 0.0:
            grpo_cfg.kd_weight = args.kd_grpo
            grpo_cfg.teacher_max_new = args.teacher_max_new
            grpo_cfg.teacher_temp = args.teacher_temp
            grpo_cfg.teacher_top_p = args.teacher_top_p
            setattr(grpo_cfg, "teacher", teacher)
        spct_rft_loop(model, tokenizer, adapter, device, rft_cfg)
        spct_grpo_loop(model, tokenizer, adapter, device, grpo_cfg)
        spct_ckpt = save_dir / f"spct_{int(time.time())}.pt"
        save_checkpoint(model, spct_ckpt)
        print(f"[Save] SPCT checkpoint written to {spct_ckpt}")
    else:
        print("[Stage] SPCT skipped.")

    final_path = save_dir / f"final_{uuid.uuid4().hex[:8]}.pt"
    save_checkpoint(model, final_path)
    print(f"[Final] Saved final combined checkpoint to {final_path}")


if __name__ == "__main__":
    main()
