#!/usr/bin/env python3
# Combined ColBERT + World Model Graph GRPO
#
# Part 1: Advanced ColBERT-style late-interaction retriever
# Part 2: World Model + Graph Memory + GRPO (R1) + GFlowNet (DB/TB) + BPTT + Coding Action Space + ColBERT RAG

import os, sys, json, math, argparse, random, struct, io, time, tempfile, subprocess, textwrap, shlex, signal
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Iterable, Callable
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForCausalLM,
        PreTrainedTokenizerBase, PreTrainedModel
    )
except Exception as e:
    print("Please install transformers: pip install transformers", file=sys.stderr)
    raise

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x

try:
    from sklearn.cluster import KMeans
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False

try:
    from datasets import load_dataset
    _HAVE_DATASETS = True
except Exception:
    _HAVE_DATASETS = False
    def load_dataset(*args, **kwargs):
        raise ImportError("Please install datasets: pip install datasets")

try:
    import wandb
    _HAVE_WANDB = True
except Exception:
    _HAVE_WANDB = False


# =========================
# Utils + Determinism
# =========================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def device_of(_="cpu"):
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)

# --- Determinism & misc perf helpers (NEW) ---
def set_global_determinism(seed: int = 42):
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch.backends.cudnn as cudnn
        cudnn.deterministic = True
        cudnn.benchmark = False
    except Exception:
        pass
    set_seed(seed)

# stable 64-bit FNV-1a (avoids randomized built-in hash)
def stable_u64(s: str) -> int:
    h = 0xcbf29ce484222325
    for ch in s.encode("utf-8"):
        h ^= ch
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


# =========================
# Logging helpers
# =========================

def _format_stat(value):
    if value is None:
        return "None"
    if isinstance(value, (int, str)):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Inf"
        return f"{value:.4f}"
    if torch.is_tensor(value):
        if not torch.isfinite(value).all():
            return "Tensor(non-finite)"
        if value.numel() == 1:
            return f"{value.item():.4f}"
        return f"Tensor(mean={value.float().mean().item():.4f})"
    return str(value)


def log_component_stats(tag: str, stats: Dict[str, Optional[float]]):
    parts = []
    for k, v in stats.items():
        parts.append(f"{k}={_format_stat(v)}")
    print(f"[DEBUG][{tag}] " + " | ".join(parts))


def gradient_norm(module: nn.Module) -> Tuple[float, int]:
    total_sq = 0.0
    count = 0
    for p in module.parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return float("nan"), count
        norm = p.grad.data.norm(2)
        total_sq += norm.item() ** 2
        count += 1
    return math.sqrt(total_sq), count


def mean_dict_list(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    if not dicts:
        return {}
    sums = defaultdict(float)
    counts = defaultdict(int)
    for d in dicts:
        if not d:
            continue
        for k, v in d.items():
            if v is None:
                continue
            if isinstance(v, (int, float)):
                val = float(v)
                if math.isfinite(val):
                    sums[k] += val
                    counts[k] += 1
    return {k: sums[k] / counts[k] for k in sums if counts[k] > 0}

# -------- Graph stats (NEW) --------
def graph_stats(g: "GraphMemory") -> Dict[str, float]:
    n_nodes = len(g.nodes)
    n_edges = sum(len(v) for v in g.adj.values())
    if n_nodes == 0:
        return {
            "graph_nodes": 0, "graph_edges": 0,
            "deg_in_mean": 0.0, "deg_out_mean": 0.0,
            "deg_in_p95": 0.0, "deg_out_p95": 0.0,
        }
    deg_in = np.array([len(g.rev.get(nid, [])) for nid in g.nodes.keys()], dtype=np.float32)
    deg_out= np.array([len(g.adj.get(nid, [])) for nid in g.nodes.keys()], dtype=np.float32)
    def p95(x): 
        return float(np.percentile(x, 95)) if x.size else 0.0
    return {
        "graph_nodes": float(n_nodes),
        "graph_edges": float(n_edges),
        "deg_in_mean": float(deg_in.mean()) if deg_in.size else 0.0,
        "deg_out_mean": float(deg_out.mean()) if deg_out.size else 0.0,
        "deg_in_p95": p95(deg_in),
        "deg_out_p95": p95(deg_out),
    }


# =========================
# ColBERT encoder + tokenizer
# =========================

@dataclass
class ColBERTConfig:
    model_name: str = "bert-base-uncased"
    d_model: int = 768
    dim: int = 128
    max_query_len: int = 32
    max_doc_len: int = 180
    learn_token_gate: bool = True
    gate_dropout: float = 0.1
    normalize: bool = True
    fp16: bool = True

class ColBERTEncoder(nn.Module):
    def __init__(self, cfg: ColBERTConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = AutoModel.from_pretrained(cfg.model_name)
        hidden = self.backbone.config.hidden_size
        self.proj = nn.Linear(hidden, cfg.dim, bias=False)
        if cfg.learn_token_gate:
            self.gate = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
        else:
            self.gate = None
        self.dropout = nn.Dropout(cfg.gate_dropout)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        H = out.last_hidden_state  # [B,T,H]
        Z = self.proj(H)           # [B,T,d]
        if self.cfg.normalize:
            Z = l2norm(Z, dim=-1)
        g = self.gate(self.dropout(H)).squeeze(-1) if self.gate is not None else None
        return Z, g

class Tokenizer:
    def __init__(self, name: str):
        self.tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    def encode(self, texts: List[str], max_len: int) -> Dict[str, torch.Tensor]:
        return self.tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")

def maxsim_score(q_tok: torch.Tensor, d_tok: torch.Tensor) -> torch.Tensor:
    sims = torch.matmul(q_tok, d_tok.transpose(0, 1))   # [Q,D]
    return sims.max(dim=1).values.sum()


# =========================
# ColBERT training
# =========================

@dataclass
class TrainConfig:
    lr: float = 2e-5
    wd: float = 0.01
    epochs: int = 2
    batch_size: int = 8
    accum_steps: int = 1
    max_grad_norm: float = 1.0
    temperature: float = 0.05
    teacher_lambda: float = 0.5
    seed: int = 42
    fp16: bool = True
    max_query_len: int = 32
    max_doc_len: int = 180

class ColBERTTrainer:
    def __init__(self, model: ColBERTEncoder, tokenizer: Tokenizer, cfg: ColBERTConfig, tcfg: TrainConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.tcfg = tcfg
        self.device = device_of()
        self.model.to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=tcfg.lr, weight_decay=tcfg.wd)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(tcfg.fp16 and torch.cuda.is_available()))
        self.hard_neg_callback: Optional[Callable[[], None]] = None

    def set_hard_negative_provider(self, fn: Callable[[], None]):
        self.hard_neg_callback = fn

    def _encode_texts(self, texts: List[str], max_len: int) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        b = self.tokenizer.encode(texts, max_len)
        b = {k: v.to(self.device) for k, v in b.items()}
        with torch.cuda.amp.autocast(enabled=(self.tcfg.fp16 and torch.cuda.is_available())):
            Z, g = self.model(b["input_ids"], b["attention_mask"])
        mask = b["attention_mask"].bool()
        token_batches = []
        for i in range(Z.size(0)):
            tok = Z[i][mask[i]]
            gate = g[i][mask[i]] if g is not None else None
            token_batches.append((tok, gate))
        return token_batches

    def _compute_scores_matrix(self, q_list, d_list) -> torch.Tensor:
        B = len(q_list)
        scores = torch.zeros(B, B, device=self.device, dtype=torch.float32)
        for i in range(B):
            q_tok, _ = q_list[i]
            for j in range(B):
                d_tok, _ = d_list[j]
                scores[i, j] = maxsim_score(q_tok, d_tok)
        return scores

    def _contrastive_loss(self, scores: torch.Tensor, temperature: float) -> torch.Tensor:
        logits = scores / temperature
        labels = torch.arange(scores.size(0), device=scores.device)
        return F.cross_entropy(logits, labels)

    def _kd_loss(self, student_scores: torch.Tensor, teacher_scores: torch.Tensor) -> torch.Tensor:
        s = F.log_softmax(student_scores, dim=1)
        t = F.softmax(teacher_scores, dim=1).detach()
        return F.kl_div(s, t, reduction="batchmean")

    def fit(self, train_path: str, out_ckpt: str):
        set_global_determinism(self.tcfg.seed)
        data = []
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                if not ex.get("positives"): continue
                pos = random.choice(ex["positives"])
                negs = ex.get("negatives", [])
                if not negs: continue
                neg = random.choice(negs)
                teacher = ex.get("teacher_scores", None)
                data.append((ex["query"], pos, neg, teacher))

        self.model.train()
        step = 0
        for epoch in range(self.tcfg.epochs):
            random.shuffle(data)
            it = tqdm(range(0, len(data), self.tcfg.batch_size), desc=f"epoch {epoch+1}/{self.tcfg.epochs}")
            for st in it:
                batch = data[st: st + self.tcfg.batch_size]
                queries = [b[0] for b in batch]
                docs = [b[1] for b in batch]
                q_enc = self._encode_texts(queries, self.tcfg.max_query_len)
                d_enc = self._encode_texts(docs, self.tcfg.max_doc_len)
                scores = self._compute_scores_matrix(q_enc, d_enc)
                loss_c = self._contrastive_loss(scores, self.tcfg.temperature)

                if any(b[3] is not None for b in batch) and self.tcfg.teacher_lambda > 0:
                    with torch.no_grad():
                        T = torch.full_like(scores, -10.0)
                        T = T + torch.eye(T.size(0), device=T.device) * 10.0
                    loss_kd = self._kd_loss(scores, T)
                    loss = (1.0 - self.tcfg.teacher_lambda) * loss_c + self.tcfg.teacher_lambda * loss_kd
                else:
                    loss = loss_c

                self.scaler.scale(loss / self.tcfg.accum_steps).backward()
                if (step + 1) % self.tcfg.accum_steps == 0:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tcfg.max_grad_norm)
                    self.scaler.step(self.opt); self.scaler.update()
                    self.opt.zero_grad(set_to_none=True)
                step += 1
                it.set_postfix(loss=float(loss.detach().cpu()))
            if callable(self.hard_neg_callback):
                self.hard_neg_callback()

        os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)
        torch.save({"config": asdict(self.cfg), "state_dict": self.model.state_dict()}, out_ckpt)
        print(f"Saved checkpoint to {out_ckpt}")


# =========================
# Index (ResidualIndex upgraded)
# =========================

@dataclass
class IndexConfig:
    dim: int = 128
    n_centroids: int = 4096
    residual_bits: int = 8
    prune_topk: int = 96
    max_doc_len: int = 180
    max_query_len: int = 32
    centroid_retries: int = 3
    seed: int = 42
    stage1_centroids: int = 8
    cand_multiplier: int = 50  # beam = k * this

class ResidualIndex:
    """
    Packed record per token:
      - doc_id:    uint64 (8)
      - token_idx: uint32 (4)
      - scale:     float16 (2)
      - resid:     int8[dim] (dim)
    Total = 14 + dim bytes
    """
    def __init__(self, cfg: IndexConfig):
        self.cfg = cfg
        self.centroids = None
        self.centroid_blobs = defaultdict(bytearray)
        self.doc_lengths = {}
        self._mmaps = {}

    @staticmethod
    def _pack_record(doc_int: int, token_int: int, scale_f16: np.float16, residual_q8: np.ndarray) -> bytes:
        return (
            struct.pack("<Q", doc_int) +
            struct.pack("<I", token_int) +
            np.asarray(scale_f16, dtype=np.float16).tobytes() +
            residual_q8.astype(np.int8).tobytes()
        )

    @staticmethod
    def _unpack_record(buf: memoryview, dim: int):
        doc_id = struct.unpack_from("<Q", buf, 0)[0]
        token_idx = struct.unpack_from("<I", buf, 8)[0]
        scale = np.frombuffer(buf[12:14], dtype=np.float16).astype(np.float32)[0]
        resid = np.frombuffer(buf[14:14+dim], dtype=np.int8).astype(np.float32)
        return doc_id, token_idx, scale, resid

    def train_centroids(self, all_tokens: np.ndarray):
        C = self.cfg.n_centroids
        if _HAVE_SK:
            km = KMeans(n_clusters=C, random_state=self.cfg.seed, n_init=self.cfg.centroid_retries)
            km.fit(all_tokens)
            self.centroids = km.cluster_centers_.astype(np.float32)
        else:
            print("[WARN] sklearn not found; using naive k-means init.")
            N = all_tokens.shape[0]
            rng = np.random.default_rng(self.cfg.seed)
            centroids = [all_tokens[rng.integers(0, N)]]
            for _ in range(C-1):
                d2 = ((all_tokens[:, None, :] - np.array(centroids)[None, :, :])**2).sum(axis=-1).min(axis=1)
                probs = d2 / (d2.sum() + 1e-9)
                centroids.append(all_tokens[rng.choice(N, p=probs)])
            centroids = np.array(centroids)
            for _ in range(10):
                d2 = ((all_tokens[:, None, :] - centroids[None, :, :])**2).sum(axis=-1)
                assign = d2.argmin(axis=1)
                for c in range(C):
                    pts = all_tokens[assign == c]
                    if len(pts) > 0:
                        centroids[c] = pts.mean(axis=0)
            self.centroids = centroids.astype(np.float32)

    def _assign_centroid(self, T: np.ndarray) -> np.ndarray:
        return np.argmax(self.centroids @ T.T, axis=0).astype(np.int32)

    def add_doc(self, doc_int: int, tok_matrix: np.ndarray, gates: Optional[np.ndarray]):
        T = tok_matrix.shape[0]
        if gates is not None and self.cfg.prune_topk and self.cfg.prune_topk < T:
            idx = np.argpartition(-gates, self.cfg.prune_topk-1)[:self.cfg.prune_topk]
            tok_matrix = tok_matrix[idx]
            kept_positions = idx
        else:
            kept_positions = np.arange(T, dtype=np.int32)

        self.doc_lengths[doc_int] = int(tok_matrix.shape[0])
        cids = self._assign_centroid(tok_matrix)
        cents = self.centroids[cids]
        resid = tok_matrix - cents
        scale = np.maximum(1e-6, np.max(np.abs(resid), axis=1, keepdims=True))
        q8 = np.clip(np.round((resid / scale) * 127.0), -127, 127).astype(np.int8)
        scale_f16 = scale.astype(np.float16).squeeze(1)

        for local_idx in range(tok_matrix.shape[0]):
            c = int(cids[local_idx])
            rec = self._pack_record(int(doc_int), int(kept_positions[local_idx]), scale_f16[local_idx], q8[local_idx])
            self.centroid_blobs[c].extend(rec)

    def finalize(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "centroids.npy"), self.centroids)
        meta = {}
        for cid, blob in self.centroid_blobs.items():
            path = os.path.join(out_dir, f"c_{cid:08d}.bin")
            with open(path, "wb") as f:
                f.write(blob)
            meta[cid] = {"bytes": len(blob), "path": os.path.basename(path)}
        with open(os.path.join(out_dir, "postings.json"), "w") as f:
            json.dump(meta, f)
        with open(os.path.join(out_dir, "doc_lengths.json"), "w") as f:
            json.dump({str(k): int(v) for k, v in self.doc_lengths.items()}, f)
        print(f"[index] wrote centroids + postings to {out_dir}")

    @classmethod
    def load(cls, index_dir: str, cfg: IndexConfig):
        idx = cls(cfg)
        idx.centroids = np.load(os.path.join(index_dir, "centroids.npy")).astype(np.float32)
        with open(os.path.join(index_dir, "postings.json"), "r") as f:
            meta = json.load(f)
        idx._mmaps = {}
        for cid, m in meta.items():
            path = os.path.join(index_dir, m["path"])
            idx._mmaps[int(cid)] = np.memmap(path, mode="r", dtype=np.uint8)
        with open(os.path.join(index_dir, "doc_lengths.json"), "r") as f:
            idx.doc_lengths = {int(k): int(v) for k, v in json.load(f).items()}
        return idx

    def _iter_records(self, cid: int, dim: int):
        mm = self._mmaps[cid]
        rec_size = 14 + dim
        total = mm.size
        mv = memoryview(mm)
        for off in range(0, total, rec_size):
            yield self._unpack_record(mv[off:off+rec_size], dim)

    def _top_centroids_for_query_token(self, qv: np.ndarray, topC: int) -> np.ndarray:
        dots = self.centroids @ qv
        return np.argpartition(-dots, topC-1)[:topC]

    def search(self, query_tokens: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        Q, dim = query_tokens.shape
        topC = self.cfg.stage1_centroids
        beam = max(k * self.cfg.cand_multiplier, 200)

        doc_ub = defaultdict(float)
        cand_docs_set = set()
        max_per_q = np.zeros(Q, dtype=np.float32)
        for i in range(Q):
            qv = query_tokens[i]
            cands = self._top_centroids_for_query_token(qv, topC)
            c_scores = (self.centroids[cands] @ qv)
            max_per_q[i] = np.max(c_scores)
            m = float(max_per_q[i])
            if m <= 0:
                continue
            for c in cands:
                for doc_id, token_idx, scale_f, resid_q8 in self._iter_records(int(c), dim):
                    doc_ub[doc_id] += m
                    cand_docs_set.add(doc_id)

        if not doc_ub:
            return []

        cand_docs = sorted(doc_ub.items(), key=lambda x: -x[1])[:beam]
        cand_ids = set(d for d,_ in cand_docs)

        doc_to_tokens = defaultdict(list)
        C = self.centroids
        for c, _mm in self._mmaps.items():
            for doc_id, token_idx, scale_f, resid_q8 in self._iter_records(int(c), dim):
                if doc_id not in cand_ids:
                    continue
                tok = C[c] + (float(scale_f) * (resid_q8 / 127.0))
                tok = tok / (np.linalg.norm(tok) + 1e-8)
                doc_to_tokens[doc_id].append(tok)

        results = []
        q_t = torch.from_numpy(query_tokens).float()
        for doc_id, toks in doc_to_tokens.items():
            if not toks: continue
            d_t = torch.from_numpy(np.stack(toks, axis=0)).float()
            s = maxsim_score(q_t, d_t).item()
            results.append((doc_id, s))
        results.sort(key=lambda x: -x[1])
        return results[:k]


# =========================
# LateInteractionRetriever
# =========================

class LateInteractionRetriever:
    def __init__(self, enc: ColBERTEncoder, tok: Tokenizer, cfg: ColBERTConfig, icfg: IndexConfig):
        self.enc = enc
        self.tok = tok
        self.cfg = cfg
        self.icfg = icfg
        self.device = device_of()
        self.enc.to(self.device)
        self.index: Optional[ResidualIndex] = None

    @torch.no_grad()
    def encode_text(self, texts: List[str], max_len: int) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]]]:
        self.enc.eval()
        batch = self.tok.encode(texts, max_len)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        Z, g = self.enc(batch["input_ids"], batch["attention_mask"])
        mask = batch["attention_mask"].bool()
        tok_list, gate_list = [], []
        for i in range(Z.size(0)):
            t = Z[i][mask[i]].detach().cpu().numpy().astype(np.float32)
            if self.cfg.normalize:
                n = np.linalg.norm(t, axis=1, keepdims=True) + 1e-8
                t = t / n
            tok_list.append(t)
            gate_list.append(g[i][mask[i]].detach().cpu().numpy().astype(np.float32) if g is not None else None)
        return tok_list, gate_list

    def build_index(self, texts: List[Tuple[int, str]], out_dir: str, sample_for_kmeans: int = 200000):
        all_tok, all_gate = [], []
        B = 64
        for i in tqdm(range(0, len(texts), B), desc="encode docs"):
            chunk = [t[1] for t in texts[i:i+B]]
            toks, gates = self.encode_text(chunk, self.icfg.max_doc_len)
            all_tok.extend(toks); all_gate.extend(gates)

        pool = [t for t in all_tok if t is not None and len(t) > 0]
        if not pool: raise RuntimeError("No tokens to index.")
        cat = np.concatenate(pool, axis=0)
        if cat.shape[0] > sample_for_kmeans:
            idx = np.random.choice(cat.shape[0], size=sample_for_kmeans, replace=False)
            cat = cat[idx]

        idx = ResidualIndex(self.icfg)
        idx.train_centroids(cat)

        pbar = tqdm(total=len(texts), desc="add docs")
        for (doc_int, _), t, g in zip(texts, all_tok, all_gate):
            if t is None or len(t) == 0:
                pbar.update(1); continue
            idx.add_doc(doc_int, t, g)
            pbar.update(1)
        pbar.close()
        idx.finalize(out_dir)
        self.index = idx

    def load_index(self, index_dir: str):
        self.index = ResidualIndex.load(index_dir, self.icfg)

    def search(self, queries: List[str], k: int = 10) -> List[List[Tuple[int, float]]]:
        if self.index is None:
            raise RuntimeError("Index not loaded.")
        q_toks, _ = self.encode_text(queries, self.icfg.max_query_len)
        results = []
        for qt in q_toks:
            res = self.index.search(qt, k=k)
            results.append(res)
        return results


# =========================
# I/O Helpers
# =========================

def read_train_jsonl(path: str) -> List[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def read_index_jsonl(path: str) -> List[Tuple[int, str]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            doc_id = ex.get("doc_id")
            text = ex.get("text", "")
            if isinstance(doc_id, str):
                doc_int = int(stable_u64(doc_id))
            else:
                doc_int = int(doc_id)
            out.append((doc_int, text))
    return out


# =========================
# World Model + Graph memory + encoders
# =========================

SPECIAL_TOKENS = {
    "BOS": "<BOS>",
    "EOS": "<EOS>",
    "STATE": "<STATE>",
    "ACTION": "<ACTION>",
    "NEXT_STATE": "<NEXT_STATE>",
    "REWARD": "<REWARD>",
    "SEP": "<SEP>",
    "CTX_START": "<CTX>",
    "CTX_END": "</CTX>",
}
ADDITIONAL_TOKENS = list(SPECIAL_TOKENS.values())

def pack_world_io(state_text: str, action_text: Optional[str] = None,
                  next_state_text: Optional[str] = None, include_reward_tag: bool = False) -> str:
    bos, eos = SPECIAL_TOKENS["BOS"], SPECIAL_TOKENS["EOS"]
    s, a, ns, sep, r = SPECIAL_TOKENS["STATE"], SPECIAL_TOKENS["ACTION"], SPECIAL_TOKENS["NEXT_STATE"], SPECIAL_TOKENS["SEP"], SPECIAL_TOKENS["REWARD"]
    out = f"{bos} {s} {state_text} {sep} {a} {action_text or ''} {sep} {ns} "
    if next_state_text is not None:
        out += f"{next_state_text} "
    if include_reward_tag:
        out += f"{sep} {r} "
    out += eos
    return out.strip()

@dataclass
class GNode: id: str; typ: str; props: Dict[str, str]
@dataclass
class GEdge: src: str; rel: str; dst: str; props: Dict[str, str]

class GraphMemory:
    def __init__(self):
        self.nodes: Dict[str, GNode] = {}
        self.adj: Dict[str, List[GEdge]] = {}
        self.rev: Dict[str, List[GEdge]] = {}
        self.clock = 0

    def clone(self) -> "GraphMemory":
        g = GraphMemory()
        g.nodes = {k: GNode(v.id, v.typ, dict(v.props)) for k,v in self.nodes.items()}
        g.adj = {k: [GEdge(e.src, e.rel, e.dst, dict(e.props)) for e in v] for k,v in self.adj.items()}
        g.rev = {k: [GEdge(e.src, e.rel, e.dst, dict(e.props)) for e in v] for k,v in self.rev.items()}
        g.clock = self.clock
        return g

    def upsert_node(self, nid: str, typ: str, **props):
        n = self.nodes.get(nid)
        if n is None:
            self.nodes[nid] = GNode(nid, typ, dict(props))
        else:
            n.props.update(props)

    def add_edge(self, src: str, rel: str, dst: str, **props):
        e = GEdge(src, rel, dst, dict(props))
        self.adj.setdefault(src, []).append(e)
        self.rev.setdefault(dst, []).append(e)
        self.clock += 1

    def neighbors(self, nid: str, rel: Optional[str] = None) -> List[GEdge]:
        es = self.adj.get(nid, [])
        return [e for e in es if rel is None or e.rel == rel]

    def remove_edge(self, src: str, rel: str, dst: str) -> bool:
        changed = False
        if src in self.adj:
            keep = []
            for e in self.adj[src]:
                if e.rel == rel and e.dst == dst: changed = True
                else: keep.append(e)
            self.adj[src] = keep
        if dst in self.rev:
            keep = []
            for e in self.rev[dst]:
                if e.src == src and e.rel == rel: continue
                keep.append(e)
            self.rev[dst] = keep
        if changed: self.clock += 1
        return changed

    def serialize_subgraph(self, focus: Iterable[str], max_hops: int = 2) -> str:
        seen=set(); frontier=list(focus); depth={nid:0 for nid in frontier}
        nodes, edges = set(), []
        while frontier:
            nid = frontier.pop(0)
            if nid in seen: continue
            seen.add(nid); nodes.add(nid)
            h = depth[nid]
            if h >= max_hops: continue
            for e in sorted(self.adj.get(nid, []), key=lambda x: (x.rel, x.dst)):
                edges.append(e)
                if e.dst not in seen:
                    depth[e.dst] = h+1
                    frontier.append(e.dst)
        lines = ["KB:"]
        for nid in sorted(nodes):
            n = self.nodes.get(nid, GNode(nid,"Unknown",{}))
            props = " ".join(f"{k}={repr(v)}" for k,v in sorted(n.props.items()))
            lines.append(f" - NODE({n.typ},{nid}) {props}".strip())
        for e in sorted(edges, key=lambda x: (x.src, x.rel, x.dst)):
            props = " ".join(f"{k}={repr(v)}" for k,v in sorted(e.props.items()))
            lines.append(f" - EDGE({e.rel},{e.src}->{e.dst}) {props}".strip())
        return "\n".join(lines)


# =========================
# GraphEncoder (upgraded)
# =========================

class HashVocab:
    def __init__(self, num_buckets: int = 1_000_003):
        self.num_buckets = num_buckets
    def __call__(self, s: str) -> int:
        return (abs(hash("##" + s + "##")) % self.num_buckets)

class GraphEncoder(nn.Module):
    def __init__(self, d_model: int, buckets: int = 1_000_003, layers: int = 2, d_hidden: int = 512, num_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.hv = HashVocab(buckets)
        self.emb_id = nn.Embedding(buckets, d_hidden)
        self.emb_typ = nn.Embedding(buckets, d_hidden)
        self.emb_rel = nn.Embedding(buckets, d_hidden)
        self.emb_deg_in = nn.Embedding(512, d_hidden)
        self.emb_deg_out = nn.Embedding(512, d_hidden)
        self.emb_dir = nn.Embedding(3, d_hidden)  # 0: undirected, 1: out, 2: in
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_hidden, num_heads, dim_feedforward=4*d_hidden, batch_first=True) for _ in range(layers)])
        self.proj_out = nn.Linear(d_hidden, d_model)
        self.pool = nn.Parameter(torch.randn(d_hidden))

    def _deg(self, g: GraphMemory, n: str):
        d_in = len(g.rev.get(n, []))
        d_out = len(g.adj.get(n, []))
        return min(d_in, 511), min(d_out, 511)

    def encode_single(self, g: GraphMemory, focus: Iterable[str], device: torch.device, max_hops: int = 2):
        seen=set(); frontier=list(focus); depth={nid:0 for nid in frontier}
        nodes, edges = [], []
        while frontier:
            nid = frontier.pop(0)
            if nid in seen: continue
            seen.add(nid); nodes.append(nid)
            if depth[nid] >= max_hops: continue
            for e in g.neighbors(nid):
                edges.append(e)
                if e.dst not in seen:
                    depth[e.dst] = depth[nid] + 1
                    frontier.append(e.dst)
        if not nodes: nodes = ["__DUMMY__"]

        nid_ids = torch.tensor([self.hv(n) for n in nodes], device=device)
        ntyp_ids = torch.tensor([self.hv(g.nodes[n].typ) if n in g.nodes else self.hv("__DUMMY_TYP__") for n in nodes], device=device)
        deg_in = torch.tensor([self._deg(g, n)[0] for n in nodes], device=device)
        deg_out= torch.tensor([self._deg(g, n)[1] for n in nodes], device=device)

        nvec = self.emb_id(nid_ids) + self.emb_typ(ntyp_ids) + self.emb_deg_in(deg_in) + self.emb_deg_out(deg_out)

        agg = torch.zeros_like(nvec)
        idx_of = {n:i for i,n in enumerate(nodes)}
        for e in edges:
            if e.src in idx_of and e.dst in idx_of:
                j = idx_of[e.dst]
                agg[j] = agg[j] + self.emb_rel(torch.tensor([self.hv(e.rel)], device=device))[0] + self.emb_dir(torch.tensor([1], device=device))[0]

        x = (nvec + agg).unsqueeze(0)
        for layer in self.layers:
            x = layer(x)
        x = x.squeeze(0)
        att = torch.softmax((x @ self.pool), dim=0)
        g_vec = (att.unsqueeze(-1) * x).sum(dim=0)
        g_vec = self.proj_out(g_vec)
        x = self.proj_out(x)
        return g_vec, x, nodes

    def forward(self, graphs: List[Tuple[GraphMemory, List[str]]], device: torch.device, max_hops: int = 2):
        g_reprs, node_packs, node_maps = [], [], []
        for g, focus in graphs:
            Gv, Nv, nodes = self.encode_single(g, focus, device, max_hops)
            g_reprs.append(Gv); node_packs.append(Nv); node_maps.append(nodes)
        g_reprs = torch.stack(g_reprs, dim=0)
        return g_reprs, node_packs, node_maps


# =========================
# WorldModelGraphGRM
# =========================

@dataclass
class DynamicsSample:
    state_text: str
    action_text: str
    next_state_text: str
    graph_focus: List[str]

@dataclass
class Trajectory:
    states: List[str]
    actions: List[str]
    next_states: List[str]
    graph_focuses: List[List[str]]

class WorldModelGraphGRM(nn.Module):
    def __init__(self, model_name: str = "gpt2", train_backbone: bool = True,
                 use_gaussian_reward: bool = True, reward_hidden: int = 256,
                 graph_soft_tokens: int = 8, graph_layers: int = 2, graph_hidden: int = 512,
                 graph_heads: int = 4, graph_buckets: int = 1_000_003):
        super().__init__()
        self.model_name_str = model_name
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "</s>"
        self._install_tokens()
        # Load model (no quantization during training - use LoRA + reduced dims instead)
        self.backbone: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_name)
        self.backbone.resize_token_embeddings(len(self.tokenizer))
        for p in self.backbone.parameters(): p.requires_grad = train_backbone
        self.hidden_size = self.backbone.config.hidden_size

        # OPTIONAL: gradient checkpointing + LoRA hooks
        self.backbone.config.use_cache = False
        if os.environ.get("WM_GRAD_CKPT", "0") == "1":
            try: self.backbone.gradient_checkpointing_enable()
            except Exception: pass
        self._lora_enabled = os.environ.get("WM_USE_LORA", "0") == "1"
        if self._lora_enabled:
            try:
                from peft import LoraConfig, get_peft_model
                # Minimal LoRA for memory: r=4 instead of 8
                lcfg = LoraConfig(
                    r=4,  # Reduced rank for memory
                    lora_alpha=8,  # Reduced alpha
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Only attention layers
                )
                self.backbone = get_peft_model(self.backbone, lcfg)
            except Exception:
                self._lora_enabled = False

        self.graph_encoder = GraphEncoder(d_model=self.hidden_size, buckets=graph_buckets,
                                          layers=graph_layers, d_hidden=graph_hidden, num_heads=graph_heads)
        self.graph_token_proj = nn.Linear(self.hidden_size, graph_soft_tokens * self.hidden_size)
        self.graph_soft_tokens = graph_soft_tokens

        out_dim = 2 if use_gaussian_reward else 1
        self.reward_head = nn.Sequential(nn.Linear(self.hidden_size, reward_hidden), nn.Tanh(),
                                         nn.Linear(reward_hidden, out_dim))
        self.use_gaussian_reward = use_gaussian_reward

        self.plan_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.graph_transition = nn.Sequential(nn.Linear(2*self.hidden_size, self.hidden_size),
                                              nn.Tanh(), nn.Linear(self.hidden_size, self.hidden_size))

        self.id_reward = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["REWARD"])
        self.id_eos = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["EOS"])

    def _install_tokens(self):
        add = []; vocab = self.tokenizer.get_vocab()
        for t in ADDITIONAL_TOKENS:
            if t not in vocab: add.append(t)
        if add: self.tokenizer.add_special_tokens({"additional_special_tokens": add})

    def _graph_soft_prompt(self, g_vec: torch.Tensor) -> torch.Tensor:
        B, H = g_vec.shape
        x = self.graph_token_proj(g_vec)
        return x.view(B, self.graph_soft_tokens, H)

    def _text(self, texts: List[str], device: torch.device):
        enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        return {k: v.to(device) for k,v in enc.items()}

    def _reward_from_hidden(self, hidden: torch.Tensor, input_ids: torch.Tensor):
        B, T, H = hidden.size()
        with torch.no_grad():
            pos = (input_ids == self.id_reward)
            idxs = []
            for i in range(B):
                w = torch.where(pos[i])[0]
                idxs.append(w[-1].item() if w.numel() else T-1)
            idxs = torch.tensor(idxs, device=hidden.device, dtype=torch.long)
        pooled = hidden[torch.arange(B, device=hidden.device), idxs]
        out = self.reward_head(pooled)
        if self.use_gaussian_reward:
            return out[:, :1], out[:, 1:]
        else:
            return out[:, :1], None

    def forward_dynamics(self, graphs, inputs: List[str], targets: Optional[List[str]] = None,
                         device: Optional[torch.device] = None):
        device = device or next(self.parameters()).device
        # Ensure graphs and inputs have matching lengths
        assert len(graphs) == len(inputs), f"Graphs ({len(graphs)}) and inputs ({len(inputs)}) must have same length"
        g_vec, _, _ = self.graph_encoder(graphs, device=device)
        soft = self._graph_soft_prompt(g_vec)
        enc_in = self._text(inputs, device)
        
        # Tokenize targets early if provided, so we can align everything upfront
        if targets is not None:
            assert len(inputs) == len(targets), f"Batch size mismatch: inputs={len(inputs)}, targets={len(targets)}"
            enc_tg = self._text(targets, device)
        else:
            enc_tg = None
        
        # Align all batch sizes upfront: graph, inputs, and targets (if provided)
        B_graph = soft.size(0)
        B_input = enc_in["input_ids"].size(0)
        B_target = enc_tg["input_ids"].size(0) if enc_tg else None
        
        # Find minimum batch size across all tensors
        batch_sizes = [B_graph, B_input]
        if B_target is not None:
            batch_sizes.append(B_target)
        min_batch = min(batch_sizes)
        
        # Truncate all to minimum batch size
        if soft.size(0) > min_batch:
            soft = soft[:min_batch]
        for k in enc_in:
            if isinstance(enc_in[k], torch.Tensor) and enc_in[k].size(0) > min_batch:
                enc_in[k] = enc_in[k][:min_batch]
        if enc_tg:
            for k in enc_tg:
                if isinstance(enc_tg[k], torch.Tensor) and enc_tg[k].size(0) > min_batch:
                    enc_tg[k] = enc_tg[k][:min_batch]
        
        # Now create embeddings with aligned batch sizes
        input_embeds = self.backbone.get_input_embeddings()(enc_in["input_ids"])
        input_embeds = torch.cat([soft, input_embeds], dim=1)
        attn = torch.cat([torch.ones(soft.size()[:2], device=device, dtype=enc_in["attention_mask"].dtype),
                          enc_in["attention_mask"]], dim=1)
        
        if targets is None:
            out = self.backbone(inputs_embeds=input_embeds, attention_mask=attn)
            return {"logits": out.logits}
        
        # All batch sizes are already aligned, so create target embeddings
        tgt_embeds = self.backbone.get_input_embeddings()(enc_tg["input_ids"])
        in_full = torch.cat([input_embeds, tgt_embeds], dim=1)
        att_full = torch.cat([attn, enc_tg["attention_mask"]], dim=1)
        # Labels must account for soft prompt tokens, input tokens (ignore), and target tokens
        soft_pad = torch.full(
            (enc_in["input_ids"].size(0), soft.size(1)),
            -100,
            dtype=enc_in["input_ids"].dtype,
            device=device,
        )
        labels = torch.cat([soft_pad, torch.full_like(enc_in["input_ids"], -100), enc_tg["input_ids"]], dim=1)
        
        # Final sanity check: all should have batch size min_batch
        assert in_full.size(0) == min_batch, f"in_full batch size {in_full.size(0)} != {min_batch}"
        assert labels.size(0) == min_batch, f"labels batch size {labels.size(0)} != {min_batch}"
        
        out = self.backbone(inputs_embeds=in_full, attention_mask=att_full, labels=labels)
        return {"loss": out.loss, "logits": out.logits}

    def forward_reward(self, graphs, texts: List[str], rewards: Optional[torch.Tensor] = None,
                       device: Optional[torch.device] = None, reduction: str = "mean"):
        device = device or next(self.parameters()).device
        g_vec, _, _ = self.graph_encoder(graphs, device=device)
        soft = self._graph_soft_prompt(g_vec)
        enc = self._text(texts, device)
        embeds = self.backbone.get_input_embeddings()(enc["input_ids"])
        embeds = torch.cat([soft, embeds], dim=1)
        attn = torch.cat([torch.ones(soft.size()[:2], device=device, dtype=enc["attention_mask"].dtype),
                          enc["attention_mask"]], dim=1)
        out = self.backbone(inputs_embeds=embeds, attention_mask=attn, output_hidden_states=True)
        hidden = out.hidden_states[-1]
        pad_prefix = torch.full((enc["input_ids"].size(0), soft.size(1)),
                                fill_value=self.tokenizer.pad_token_id, device=device, dtype=enc["input_ids"].dtype)
        ids_aug = torch.cat([pad_prefix, enc["input_ids"]], dim=1)
        mu, logvar = self._reward_from_hidden(hidden, ids_aug)
        outd = {"reward_mu": mu}
        if self.use_gaussian_reward: outd["reward_logvar"] = logvar
        if rewards is not None:
            rewards = rewards.to(device).unsqueeze(1)
            if self.use_gaussian_reward:
                nll = 0.5*((rewards-mu)**2)*torch.exp(-logvar) + 0.5*logvar
                loss = nll
            else:
                loss = F.mse_loss(mu, rewards, reduction="none")
            loss = loss.mean() if reduction == "mean" else loss.sum()
            outd["loss"] = loss
        return outd

    def graph_transition_loss(self, graphs_t, plans_t: List[str], graphs_tp1, device: Optional[torch.device]=None, loss_type="l2"):
        device = device or next(self.parameters()).device
        g_t, _, _ = self.graph_encoder(graphs_t, device=device)
        g_tp1, _, _ = self.graph_encoder(graphs_tp1, device=device)
        enc = self._text(plans_t, device)
        out = self.backbone(**enc, output_hidden_states=True)
        h = out.hidden_states[-1][:, -1, :]
        p = self.plan_proj(h)
        pred = self.graph_transition(torch.cat([g_t, p], dim=-1))
        if loss_type == "l2":
            return F.mse_loss(pred, g_tp1)
        pred_n, tgt_n = F.normalize(pred, dim=-1), F.normalize(g_tp1, dim=-1)
        return 1.0 - (pred_n * tgt_n).sum(dim=-1).mean()

    @torch.no_grad()
    def generate_next_state(self, graph, state_text: str, action_text: str,
                            max_new_tokens=128, temperature=0.7, top_p=0.9, device=None):
        device = device or next(self.parameters()).device
        g_vec, _, _ = self.graph_encoder([graph], device=device)
        soft = self._graph_soft_prompt(g_vec)
        prompt = pack_world_io(state_text, action_text, None)
        enc = self._text([prompt], device)
        embeds = self.backbone.get_input_embeddings()(enc["input_ids"])
        embeds = torch.cat([soft, embeds], dim=1)
        attn = torch.cat([torch.ones(1, soft.size(1), device=device, dtype=enc["attention_mask"].dtype),
                          enc["attention_mask"]], dim=1)
        do_sample = temperature is not None and temperature > 0.0
        gen_kwargs = dict(
            input_ids=None,
            inputs_embeds=embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.id_eos,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if do_sample:
            gen_kwargs.update(dict(do_sample=True, temperature=temperature, top_p=top_p))
        else:
            gen_kwargs.update(dict(do_sample=False))
        out_ids = self.backbone.generate(**gen_kwargs)
        txt = self.tokenizer.decode(out_ids[0], skip_special_tokens=False)
        marker, eos = SPECIAL_TOKENS["NEXT_STATE"], SPECIAL_TOKENS["EOS"]
        if marker in txt: txt = txt.split(marker, 1)[1]
        if eos in txt: txt = txt.split(eos, 1)[0]
        return txt.strip()


# =========================
# RAG wrapper (ColBERT)
# =========================

DOC_TAG = "[d@{}]"
CTX_START, CTX_END = SPECIAL_TOKENS["CTX_START"], SPECIAL_TOKENS["CTX_END"]

class RetrieverAugmentor:
    def __init__(self, retriever: LateInteractionRetriever, corpus_lookup: Callable[[int], str], k_ctx: int = 4, max_chars: int = 600):
        self.retriever = retriever
        self.corpus_lookup = corpus_lookup
        self.k_ctx = k_ctx
        self.max_chars = max_chars

    def retrieve_snippets(self, query_text: str) -> List[str]:
        res = self.retriever.search([query_text], k=self.k_ctx)[0]
        out = []
        for doc_id, _ in res:
            t = self.corpus_lookup(int(doc_id)) or ""
            out.append(t[:self.max_chars].replace("\n", " "))
        return out

    def inject_ctx(self, state_text: str) -> str:
        snippets = self.retrieve_snippets(state_text)
        if not snippets: return state_text
        hints = "\n".join([f"{DOC_TAG.format(i+1)} {s}" for i, s in enumerate(snippets)])
        return f"{state_text}\n{CTX_START}\n{hints}\n{CTX_END}"


# =========================
# GRPO (R1-style) w/ adaptive KL
# =========================

@dataclass
class GRPOConfig:
    K: int = 4
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    clip_eps: float = 0.2
    ref_kl_coef: float = 0.02
    ent_coef: float = 0.0
    grad_clip: float = 1.0
    lr: float = 1e-5
    weight_decay: float = 0.0
    normalize_group: bool = True
    eps_norm: float = 1e-6
    use_internal_reward: bool = True
    use_reference_kl: bool = True
    use_rag: bool = False
    target_kl: float = 0.08
    kl_adapt_rate: float = 1.5
    teacher_kl_weight: float = 0.0
    min_advantage: float = 0.01

@dataclass
class GRPOItem:
    state_text: str
    action_text: str
    graph_focus: List[str]

def _seq_logprob(model: WorldModelGraphGRM, inputs: List[str], targets: List[str], graphs, device: torch.device):
    enc_in = model.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
    enc_tg = model.tokenizer(targets, padding=True, truncation=True, return_tensors="pt")
    model_device = next(model.parameters()).device
    for k in enc_in: enc_in[k] = enc_in[k].to(model_device)
    for k in enc_tg: enc_tg[k] = enc_tg[k].to(model_device)
    g_vec, _, _ = model.graph_encoder(graphs, device=model_device)
    B = g_vec.size(0); H = model.hidden_size
    soft = model.graph_token_proj(g_vec).view(B, model.graph_soft_tokens, H)
    in_emb = model.backbone.get_input_embeddings()(enc_in["input_ids"])
    tg_emb = model.backbone.get_input_embeddings()(enc_tg["input_ids"])
    embs = torch.cat([soft, in_emb, tg_emb], dim=1)
    attn = torch.cat([
        torch.ones(soft.size()[:2], device=model_device, dtype=enc_in["attention_mask"].dtype),
        enc_in["attention_mask"], enc_tg["attention_mask"]
    ], dim=1)
    labels = torch.cat([torch.full_like(enc_in["input_ids"], -100), enc_tg["input_ids"]], dim=1)
    out = model.backbone(inputs_embeds=embs, attention_mask=attn)
    logp = F.log_softmax(out.logits, dim=-1)
    pad_id = model.tokenizer.pad_token_id
    lab = labels.clone(); lab[lab==-100] = pad_id
    gathered = logp.gather(-1, lab.unsqueeze(-1)).squeeze(-1)
    mask = (labels != -100) & (labels != pad_id)
    tok_sum = (gathered * mask).sum(dim=1); tok_cnt = mask.sum(dim=1).clamp_min(1)
    result = tok_sum / tok_cnt
    # Move result back to original device if model was on different device
    if model_device != device:
        result = result.to(device)
    return result

class GRPOTrainer:
    def __init__(self, model: WorldModelGraphGRM, cfg: GRPOConfig, reward_fn: Optional[Callable[[str,str,str], float]]=None,
                 reference_model: Optional[WorldModelGraphGRM]=None, retriever_augmentor: Optional[RetrieverAugmentor]=None):
        self.model = model; self.cfg = cfg; self.reward_fn = reward_fn
        self.device = next(model.parameters()).device
        self.opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.rag = retriever_augmentor if cfg.use_rag else None
        # Use checkpointing approach: don't store full old model, just snapshot weights when needed
        # This saves GPU memory by avoiding duplicate model copies
        # Old state is snapshotted at the start of each grpo_step
        self._ref_state = None
        
        self._teacher_state = None

        if cfg.use_reference_kl:
            # Take snapshot of reference weights (store on CPU to save GPU memory)
            self._ref_state = {k: v.clone().detach().cpu() for k, v in model.backbone.state_dict().items()}
            self.ref = None  # We'll use weight swapping instead of full model
        else:
            self.ref = None

        if cfg.teacher_kl_weight > 0:
            self._teacher_state = {k: v.clone().detach().cpu() for k, v in model.backbone.state_dict().items()}
        self._step_count = 0

    @torch.no_grad()
    def _sample_group(self, item: GRPOItem, graph: GraphMemory) -> List[Dict]:
        outs = []
        state_for_sampling = self.rag.inject_ctx(item.state_text) if self.rag is not None else item.state_text
        self._step_count = getattr(self, "_step_count", 0) + 1
        for i in range(self.cfg.K):
            ns = self.model.generate_next_state((graph, item.graph_focus), state_for_sampling, item.action_text,
                                                max_new_tokens=self.cfg.max_new_tokens,
                                                temperature=self.cfg.temperature, top_p=self.cfg.top_p, device=self.device)
            outs.append({"state_text": state_for_sampling, "action_text": item.action_text,
                         "next_state_text": ns, "graph_focus": item.graph_focus})
            if i == 0 and self._step_count % 50 == 0:
                print(f"[DEBUG GRPO] sample: {ns[:120]}")
        return outs

    def _score_group(self, group: List[Dict], graph: GraphMemory) -> torch.Tensor:
        if self.cfg.use_internal_reward and self.reward_fn is None:
            texts = [pack_world_io(g["state_text"], g["action_text"], g["next_state_text"], include_reward_tag=True) for g in group]
            graphs = [(graph, g["graph_focus"]) for g in group]
            out = self.model.forward_reward(graphs, texts)
            return out["reward_mu"].squeeze(1).detach()
        vals = [self.reward_fn(g["state_text"], g["action_text"], g["next_state_text"]) for g in group]
        return torch.tensor(vals, dtype=torch.float32, device=self.device)

    def grpo_step(self, batch: List[Tuple[GRPOItem, GraphMemory]]) -> Dict[str,float]:
        self.model.train()
        # Save old state BEFORE sampling (only save trainable params to save memory)
        # If LoRA enabled, only save LoRA adapter weights (much smaller)
        trainable_params = {name: param for name, param in self.model.backbone.named_parameters() if param.requires_grad}
        if hasattr(self.model.backbone, 'peft_config'):
            # LoRA: only save LoRA adapter weights (tiny compared to full model)
            old_state_snapshot = {k: v.clone().detach().cpu() for k, v in trainable_params.items() if 'lora' in k.lower()}
        else:
            # No LoRA: save all trainable params
            old_state_snapshot = {k: v.clone().detach().cpu() for k, v in trainable_params.items()}
        
        all_inputs, all_targets, all_graphs, all_rewards, group_bounds = [], [], [], [], []
        cursor = 0
        for item, g in batch:
            group = self._sample_group(item, g)
            rewards = self._score_group(group, g)
            inputs  = [pack_world_io(x["state_text"], x["action_text"], None) for x in group]
            targets = [pack_world_io(x["state_text"], x["action_text"], x["next_state_text"]) for x in group]
            graphs  = [(g, item.graph_focus) for _ in group]
            all_inputs += inputs; all_targets += targets; all_graphs += graphs
            all_rewards.append(rewards); group_bounds.append((cursor, cursor+len(group))); cursor += len(group)

        rewards = torch.cat(all_rewards, dim=0)
        adv = torch.empty_like(rewards)
        for s, e in group_bounds:
            grp = rewards[s:e]
            mu = grp.mean()
            std = grp.std(unbiased=False)
            if std < 1e-6:
                adv[s:e] = torch.ones_like(grp) * self.cfg.min_advantage
            else:
                adv[s:e] = (grp - mu) / (std + self.cfg.eps_norm)
        adv = adv.detach()

        # Use autocast for memory efficiency
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            logp_curr = _seq_logprob(self.model, all_inputs, all_targets, all_graphs, self.device)
        
        # Compute old logprobs using snapshot (reuse same model, swap only necessary weights)
        with torch.no_grad():
            # Only swap LoRA adapter weights if LoRA is enabled (much smaller than full model)
            if hasattr(self.model.backbone, 'peft_config'):
                # LoRA: only swap adapter weights
                current_state = {}
                for name, param in self.model.backbone.named_parameters():
                    if 'lora' in name.lower() and name in old_state_snapshot:
                        current_state[name] = param.data.detach().cpu()
                        param.data.copy_(old_state_snapshot[name].to(device=self.device, dtype=param.data.dtype))
                logp_old = _seq_logprob(self.model, all_inputs, all_targets, all_graphs, self.device)
                # Restore
                for name, param in self.model.backbone.named_parameters():
                    if name in current_state:
                        param.data.copy_(current_state[name].to(device=self.device, dtype=param.data.dtype))
                del current_state
            else:
                # No LoRA: swap all trainable parameters (but still use same model)
                current_state = {}
                for name, param in self.model.backbone.named_parameters():
                    if param.requires_grad and name in old_state_snapshot:
                        current_state[name] = param.data.detach().cpu()
                        param.data.copy_(old_state_snapshot[name].to(device=self.device, dtype=param.data.dtype))
                logp_old = _seq_logprob(self.model, all_inputs, all_targets, all_graphs, self.device)
                # Restore
                for name, param in self.model.backbone.named_parameters():
                    if name in current_state:
                        param.data.copy_(current_state[name].to(device=self.device, dtype=param.data.dtype))
                del current_state
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        teacher_kl = torch.tensor(0.0, device=self.device)
        if self.cfg.teacher_kl_weight > 0 and self._teacher_state is not None:
            with torch.no_grad():
                current_state = {}
                for name, param in self.model.backbone.named_parameters():
                    if name in self._teacher_state:
                        current_state[name] = param.data.detach().cpu()
                        param.data.copy_(self._teacher_state[name].to(device=self.device, dtype=param.data.dtype))
                logp_teacher = _seq_logprob(self.model, all_inputs, all_targets, all_graphs, self.device)
                for name, param in self.model.backbone.named_parameters():
                    if name in current_state:
                        param.data.copy_(current_state[name].to(device=self.device, dtype=param.data.dtype))
                del current_state
            teacher_kl = (logp_curr - logp_teacher).mean()
            del logp_teacher

        ratios = torch.exp(logp_curr - logp_old)
        unclipped = -ratios * adv
        clipped   = -torch.clamp(ratios, 1.0-self.cfg.clip_eps, 1.0+self.cfg.clip_eps) * adv
        policy_loss = torch.max(unclipped, clipped).mean()

        kl_loss = torch.tensor(0.0, device=self.device)
        curr_kl = torch.tensor(0.0, device=self.device)
        # Reference KL disabled for memory efficiency (use_reference_kl=False)
        # If enabled in future, use same in-place weight swapping as old model

        ent_loss = torch.tensor(0.0, device=self.device)
        if self.cfg.ent_coef > 0:
            ent_loss = -self.cfg.ent_coef * (-(logp_curr)).mean()

        loss = policy_loss + kl_loss + ent_loss + self.cfg.teacher_kl_weight * teacher_kl
        
        # Store metrics before cleanup
        metrics = {
            "loss_total": float(loss.item()),
            "loss_policy": float(policy_loss.item()),
            "loss_kl": float(kl_loss.item()),
            "loss_teacher_kl": float((teacher_kl * self.cfg.teacher_kl_weight).item()) if self.cfg.teacher_kl_weight > 0 else 0.0,
            "reward_mean": float(rewards.mean().item()),
            "reward_std": float(rewards.std(unbiased=False).item()),
            "kl_curr": float(curr_kl.item()),
            "kl_coef": float(self.cfg.ref_kl_coef),
        }
        
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.opt.step()
        
        # Cleanup memory
        del logp_curr, logp_old, ratios, unclipped, clipped, policy_loss, kl_loss, ent_loss, loss, teacher_kl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return metrics


# =========================
# BPTT trainer
# =========================

@dataclass
class BPTTConfig:
    lr: float = 1e-5
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    lambda_graph: float = 1.0
    lambda_dyn: float = 1.0
    window: int = 4

class BPTTTrainer:
    def __init__(self, model: WorldModelGraphGRM, cfg: BPTTConfig):
        self.model = model; self.cfg = cfg
        self.device = next(model.parameters()).device
        self.opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)

    def step(self, traj_batch: List[Tuple[Trajectory, GraphMemory]]) -> Dict[str,float]:
        self.model.train()
        total_loss, total_steps = 0.0, 0
        dyn_losses, graph_losses = [], []
        for traj, gmem in traj_batch:
            T = min(self.cfg.window, len(traj.actions))
            for t in range(T):
                inputs = [pack_world_io(traj.states[t], traj.actions[t], None)]
                targets= [pack_world_io(traj.states[t], traj.actions[t], traj.next_states[t])]
                graphs = [(gmem, traj.graph_focuses[t])]
                out = self.model.forward_dynamics(graphs, inputs, targets, device=self.device)
                loss_dyn = out["loss"]
                graph_loss = self.model.graph_transition_loss(
                    graphs_t=[(gmem, traj.graph_focuses[t])],
                    plans_t=[traj.next_states[t][:256]],
                    graphs_tp1=[(gmem, traj.graph_focuses[t+1] if t+1 < len(traj.graph_focuses) else traj.graph_focuses[t])],
                    device=self.device, loss_type="l2"
                )
                loss = self.cfg.lambda_dyn * loss_dyn + self.cfg.lambda_graph * graph_loss
                self.opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.opt.step()
                total_loss += float(loss.item()); total_steps += 1
                dyn_losses.append(float(loss_dyn.detach().cpu()))
                graph_losses.append(float(graph_loss.detach().cpu()))
        return {
            "loss_bptt": total_loss / max(1,total_steps),
            "loss_dyn_mean": (float(np.mean(dyn_losses)) if dyn_losses else 0.0),
            "loss_graph_mean": (float(np.mean(graph_losses)) if graph_losses else 0.0)
        }


# =========================
# GFlowNet (losses updated)
# =========================

class GFNHead(nn.Module):
    def __init__(self, d_model: int, act_dim: int = 256):
        super().__init__()
        self.flow_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1))
        self.fwd_mlp  = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())
        self.bwd_mlp  = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())
        self.act_proj = nn.Linear(d_model, act_dim, bias=False)
        self.state_proj = nn.Linear(d_model, act_dim, bias=False)

    def forward_state(self, g_vec: torch.Tensor):
        logF_s = self.flow_mlp(g_vec)
        state_q = self.state_proj(self.fwd_mlp(g_vec))
        return logF_s, state_q

    def score_actions(self, state_q: torch.Tensor, act_feats: torch.Tensor):
        act_k = self.act_proj(act_feats)
        return (state_q[:,None,:] * act_k).sum(-1)

    def backward_logits(self, g_vec_child: torch.Tensor, parent_feats: torch.Tensor):
        child_q = self.state_proj(self.bwd_mlp(g_vec_child))
        par_k = self.act_proj(parent_feats)
        return (child_q[:,None,:] * par_k).sum(-1)

def detailed_balance_loss(logF_s, logF_sp, logits_fwd, logits_bwd, a_index, p_index, R_sp: torch.Tensor=None, delta: float=1e-4):
    logPF = (logits_fwd - logits_fwd.logsumexp(-1, keepdim=True)).gather(-1, a_index[...,None]).squeeze(-1)
    logPB = (logits_bwd - logits_bwd.logsumexp(-1, keepdim=True)).gather(-1, p_index[...,None]).squeeze(-1)
    if R_sp is None:
        lhs = torch.logaddexp(logF_s.squeeze(-1) + logPF, torch.log(torch.tensor(delta, device=logF_s.device)))
        rhs = torch.logaddexp(logF_sp.squeeze(-1) + logPB, torch.log(torch.tensor(delta, device=logF_s.device)))
    else:
        lhs = torch.logaddexp(logF_s.squeeze(-1) + logPF, torch.log(torch.tensor(delta, device=logF_s.device)))
        rhs = torch.logaddexp(torch.log(R_sp + delta), torch.log(torch.tensor(delta, device=logF_s.device)))
    return (lhs - rhs).pow(2).mean()

def trajectory_balance_loss(Z_log: torch.Tensor, logPF_sum: torch.Tensor, logPB_sum: torch.Tensor, logR_term: torch.Tensor):
    resid = (Z_log + logPF_sum) - (logR_term + logPB_sum)
    return (resid ** 2).mean()


# =========================
# Action space (token-grounded seam)
# =========================

@dataclass
class Action:
    kind: str
    args: Tuple[str, ...]      # see handlers below

class ActionSpace:
    def __init__(self, patch_arg_fn: Optional[Callable[[GraphMemory, str], Tuple[str,str]]] = None):
        """
        patch_arg_fn(g, file) -> (patch_id, diff_text)
        If None, falls back to stub diff.
        """
        self.patch_arg_fn = patch_arg_fn

    def legal_actions(self, g: GraphMemory) -> List[Action]:
        acts = []
        files = [n.id.split(":",1)[1] for n in g.nodes.values() if n.typ=="File"]
        tests = [n.id.split(":",1)[1] for n in g.nodes.values() if n.typ=="Test"]
        builds= [n.id.split(":",1)[1] for n in g.nodes.values() if n.typ=="BuildRun"]
        if not builds:
            bid = "current"; g.upsert_node(f"BuildRun:{bid}", "BuildRun", ok="False", time_ms="0", reason="init")
            g.add_edge(f"BuildRun:{bid}", "touches", "File:main.py"); builds = ["current"]

        for f in files:
            acts.append(Action("touch_file", (f,)))

        for f in files:
            if self.patch_arg_fn:
                pid, diff = self.patch_arg_fn(g, f)
            else:
                pid = f"patch#{abs(hash(str(g.clock)+f))%100000}"
                diff = f"@@ diff for {f} at {g.clock} @@"
            acts.append(Action("apply_patch", (f, pid, diff)))

        for b in builds:
            acts += [Action("set_build_ok", (b, "True", "150", "green")),
                     Action("set_build_ok", (b, "False", "200", "red"))]

        for t in tests:
            for b in builds:
                acts += [Action("mark_pass", (t, b)), Action("mark_fail", (t, b))]
        return acts

    def apply(self, g: GraphMemory, a: Action) -> GraphMemory:
        g2 = g.clone()
        if a.kind == "touch_file":
            f, = a.args
            g2.add_edge("BuildRun:current", "touches", f"File:{f}")
        elif a.kind == "apply_patch":
            f, pid, diff = a.args
            patch_nid = f"Patch:{pid}"
            g2.upsert_node(patch_nid, "Patch", diff=diff, ts=str(int(time.time())))
            g2.add_edge(patch_nid, "modifies", f"File:{f}")
            g2.add_edge("BuildRun:current", "applies", patch_nid)
            g2.upsert_node(f"File:{f}", "File", last_patch=pid)
        elif a.kind == "set_build_ok":
            b, ok, tms, reason = a.args
            g2.upsert_node(f"BuildRun:{b}", "BuildRun", ok=ok, time_ms=tms, reason=reason)
        elif a.kind == "mark_pass":
            t, b = a.args
            g2.add_edge(f"Test:{t}", "passes", f"BuildRun:{b}")
            g2.remove_edge(f"Test:{t}", "fails", f"BuildRun:{b}")
        elif a.kind == "mark_fail":
            t, b = a.args
            g2.add_edge(f"Test:{t}", "fails", f"BuildRun:{b}")
            g2.remove_edge(f"Test:{t}", "passes", f"BuildRun:{b}")
        return g2


# =========================
# GFN Trainer + Replay
# =========================

@dataclass
class GFNConfig:
    lr: float = 1e-5
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    lambda_db: float = 1.0
    lambda_tb: float = 1.0
    Z_learn: bool = True

@dataclass
class Transition:
    s: GraphMemory
    a: Action
    s_next: GraphMemory
    reward: float
    terminal: bool

class ReplayBuffer:
    def __init__(self, cap: int = 20000):
        self.cap = cap; self.buf: List[Transition] = []
    def add(self, t: Transition):
        if len(self.buf) >= self.cap: self.buf.pop(0)
        self.buf.append(t)
    def sample(self, n: int) -> List[Transition]:
        if not self.buf: return []
        n = min(n, len(self.buf))
        return random.sample(self.buf, n)

class GFNTrainer:
    def __init__(self, model: WorldModelGraphGRM, cfg: GFNConfig, act_space: ActionSpace):
        self.model = model; self.cfg = cfg; self.act_space = act_space
        self.device = next(model.parameters()).device
        self.head = GFNHead(d_model=model.hidden_size).to(self.device)
        self.act_encoder = ActionEncoder(model.graph_encoder, d_model=model.hidden_size).to(self.device)
        params = list(self.head.parameters()) + list(self.model.graph_encoder.parameters())
        if cfg.Z_learn: self.Z_log = nn.Parameter(torch.zeros(1, device=self.device))
        else: self.Z_log = None
        self.opt = torch.optim.AdamW(params + ([self.Z_log] if self.Z_log is not None else []),
                                     lr=cfg.lr, weight_decay=cfg.weight_decay)

    def _reward_fn(self, g: GraphMemory) -> float:
        passes = sum(1 for edges in g.adj.values() for e in edges if e.rel=="passes")
        fails  = sum(1 for edges in g.adj.values() for e in edges if e.rel=="fails")
        ok_nodes= [n for n in g.nodes.values() if n.typ=="BuildRun" and n.props.get("ok","False")=="True"]
        ok_bonus = len(ok_nodes)
        patch_pen = sum(1 for n in g.nodes.values() if n.typ=="Patch") * 0.05
        return max(0.0, passes - 0.5*fails + 0.5*ok_bonus - patch_pen)

    def db_step(self, batch: List[Transition]) -> Dict[str,float]:
        if not batch: return {"gfn_loss_db": 0.0, "gfn_flow_mean": 0.0, "gfn_flow_std": 0.0}
        self.model.train(); self.head.train()
        loss_vals = []
        flow_vals = []

        for tr in batch:
            gv_s,_,_= self.model.graph_encoder([(tr.s, ["File:main.py"])], device=self.device)
            gv_sp,_,_= self.model.graph_encoder([(tr.s_next, ["File:main.py"])], device=self.device)
            logF_s, state_q = self.head.forward_state(gv_s)
            logF_sp,_ = self.head.forward_state(gv_sp)

            acts_s = self.act_space.legal_actions(tr.s)
            if not acts_s: 
                continue
            try: a_index = torch.tensor([acts_s.index(tr.a)], device=self.device)
            except ValueError: a_index = torch.tensor([0], device=self.device)
            logits_fwd = self.head.score_actions(state_q, self.act_encoder.encode_actions(tr.s, acts_s, self.device))

            parents = self.act_space.legal_actions(tr.s_next) or [tr.a]
            p_index = torch.tensor([0], device=self.device)
            logits_bwd = self.head.backward_logits(gv_sp, self.act_encoder.encode_actions(tr.s_next, parents, self.device))

            R_sp = torch.tensor([tr.reward], device=self.device) if tr.terminal else None
            loss_db = detailed_balance_loss(logF_s, logF_sp, logits_fwd, logits_bwd, a_index, p_index, R_sp)

            self.opt.zero_grad(set_to_none=True); loss_db.backward()
            torch.nn.utils.clip_grad_norm_(self.head.parameters(), self.cfg.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.model.graph_encoder.parameters(), self.cfg.grad_clip)
            if self.Z_log is not None: torch.nn.utils.clip_grad_norm_([self.Z_log], self.cfg.grad_clip)
            self.opt.step()

            loss_vals.append(float(loss_db.detach().cpu()))
            flow_vals.append(float(logF_s.detach().cpu().mean()))

        flow_mean = float(np.mean(flow_vals)) if flow_vals else 0.0
        flow_std  = float(np.std(flow_vals)) if flow_vals else 0.0
        return {"gfn_loss_db": (float(np.mean(loss_vals)) if loss_vals else 0.0),
                "gfn_flow_mean": flow_mean, "gfn_flow_std": flow_std}

    def tb_step(self, trajs: List[List[Transition]]) -> Dict[str,float]:
        if not trajs: return {"gfn_loss_tb": 0.0}
        self.model.train(); self.head.train()
        loss_vals = []

        for traj in trajs:
            if not traj or len(traj) == 0:
                continue
            logPF_sum = torch.zeros(1, device=self.device)
            logPB_sum = torch.zeros(1, device=self.device)
            for tr in traj:
                gv_s,_,_  = self.model.graph_encoder([(tr.s, ["File:main.py"])], device=self.device)
                gv_sp,_,_ = self.model.graph_encoder([(tr.s_next, ["File:main.py"])], device=self.device)
                logF_s, q_s = self.head.forward_state(gv_s)
                acts_s = self.act_space.legal_actions(tr.s) or [tr.a]
                try: a_index = torch.tensor([acts_s.index(tr.a)], device=self.device)
                except ValueError: a_index = torch.tensor([0], device=self.device)
                logits_f = self.head.score_actions(q_s, self.act_encoder.encode_actions(tr.s, acts_s, self.device))
                logPF_sum = logPF_sum + (logits_f - logits_f.logsumexp(-1, keepdim=True)).gather(-1, a_index[...,None]).squeeze(-1)

                parents = self.act_space.legal_actions(tr.s_next) or [tr.a]
                logits_b = self.head.backward_logits(gv_sp, self.act_encoder.encode_actions(tr.s_next, parents, self.device))
                p_index = torch.tensor([0], device=self.device)
                logPB_sum = logPB_sum + (logits_b - logits_b.logsumexp(-1, keepdim=True)).gather(-1, p_index[...,None]).squeeze(-1)

            R = self._reward_fn(traj[-1].s_next)
            logR = torch.log(torch.tensor([max(1e-6, R)], device=self.device))
            Zl = getattr(self, "Z_log", None) if hasattr(self, "Z_log") else None
            Zl = Zl if Zl is not None else torch.zeros(1, device=self.device)
            loss_tb = trajectory_balance_loss(Zl, logPF_sum, logPB_sum, logR)

            self.opt.zero_grad(set_to_none=True); loss_tb.backward()
            torch.nn.utils.clip_grad_norm_(self.head.parameters(), self.cfg.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.model.graph_encoder.parameters(), self.cfg.grad_clip)
            if hasattr(self, "Z_log") and self.Z_log is not None:
                torch.nn.utils.clip_grad_norm_([self.Z_log], self.cfg.grad_clip)
            self.opt.step()

            loss_vals.append(float(loss_tb.detach().cpu()))

        return {"gfn_loss_tb": (float(np.mean(loss_vals)) if loss_vals else 0.0)}


# =========================
# ActionEncoder (GFN)
# =========================

class ActionEncoder(nn.Module):
    def __init__(self, graph_encoder: GraphEncoder, d_model: int):
        super().__init__()
        self.node_proj = nn.Linear(d_model, d_model, bias=False)
        self.rel_embed = nn.Embedding(256, d_model)
        self.hv = HashVocab(256)
        self.graph_encoder = graph_encoder

    @torch.no_grad()
    def _node_vec(self, g: GraphMemory, nid: str, device: torch.device):
        Gv, _, _ = self.graph_encoder.encode_single(g, [nid], device=device, max_hops=1)
        return Gv

    def encode_actions(self, g: GraphMemory, acts: List[Action], device: torch.device) -> torch.Tensor:
        vecs = []
        for a in acts:
            if a.kind in ("mark_pass","mark_fail"):
                t, b = a.args
                sv = self._node_vec(g, f"Test:{t}", device)
                dv = self._node_vec(g, f"BuildRun:{b}", device)
                rv = self.rel_embed(torch.tensor([self.hv("passes" if a.kind=="mark_pass" else "fails")], device=device))
                v = self.node_proj(sv) + rv + self.node_proj(dv)
            elif a.kind == "touch_file":
                f, = a.args
                sv = self._node_vec(g, f"BuildRun:current", device)
                dv = self._node_vec(g, f"File:{f}", device)
                rv = self.rel_embed(torch.tensor([self.hv("touches")], device=device))
                v = self.node_proj(sv) + rv + self.node_proj(dv)
            elif a.kind == "apply_patch":
                f, pid, _diff = a.args
                sv = self._node_vec(g, f"Patch:{pid}", device)
                dv = self._node_vec(g, f"File:{f}", device)
                rv = self.rel_embed(torch.tensor([self.hv("modifies")], device=device))
                v = self.node_proj(sv) + rv + self.node_proj(dv)
            elif a.kind == "set_build_ok":
                b, _ok, _tms, _reason = a.args
                sv = self._node_vec(g, f"BuildRun:{b}", device)
                rv = self.rel_embed(torch.tensor([self.hv("ok")], device=device))
                v = self.node_proj(sv) + rv
            else:
                v = torch.zeros(1, self.graph_encoder.d_model, device=device)
            vecs.append(v)
        return torch.stack(vecs, dim=1)  # [1,A,d_model]


# =========================
# ColBERT loader + corpus lookup
# =========================

def build_corpus_lookup(corpus_jsonl: str) -> Dict[int, str]:
    m = {}
    with open(corpus_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            doc_id = ex.get("doc_id")
            text = ex.get("text", "")
            if isinstance(doc_id, str):
                doc_int = int(stable_u64(doc_id))
            else:
                doc_int = int(doc_id)
            m[doc_int] = text
    return m

def load_colbert_retriever(ckpt_path: str, index_dir: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ColBERTConfig(**ckpt["config"])
    tok = Tokenizer(cfg.model_name)
    enc = ColBERTEncoder(cfg)
    enc.load_state_dict(ckpt["state_dict"])
    retr = LateInteractionRetriever(enc, tok, cfg, IndexConfig(dim=cfg.dim))
    retr.load_index(index_dir)
    return retr, cfg


# =========================
# Code task: dataset adapter + executor + end-to-end trainer
# =========================

class CodeSandbox:
    def __init__(self, py_bin: str = sys.executable, time_limit_s: float = 2.5, mem_limit_mb: int = 512):
        self.py_bin = py_bin
        self.time_limit_s = time_limit_s
        self.mem_limit_mb = mem_limit_mb

    def _limits(self):
        # UNIX-only soft limits (best-effort); on Windows we'll just use timeout.
        try:
            import resource
        except Exception:
            return None

        def set_limits():
            try:
                # CPU time
                resource.setrlimit(resource.RLIMIT_CPU, (int(self.time_limit_s) + 1, int(self.time_limit_s) + 1))
                # Address space
                bytes_limit = self.mem_limit_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))
                # File size/NOFILE (paranoid)
                resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
            except Exception:
                pass
        return set_limits

    def run_code(self, code_text: str, stdin_data: str) -> Tuple[str, str, int]:
        """Return (stdout, stderr, returncode) with time/mem constraints."""
        # write temp file
        with tempfile.TemporaryDirectory() as td:
            main_path = os.path.join(td, "main.py")
            with open(main_path, "w", encoding="utf-8") as f:
                f.write(code_text)
            preexec = self._limits()
            try:
                proc = subprocess.Popen(
                    [self.py_bin, main_path],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    preexec_fn=preexec if preexec is not None else None,
                )
                try:
                    out, err = proc.communicate(stdin_data, timeout=self.time_limit_s)
                    returncode = proc.returncode
                    return out, err, returncode
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    return "", "TIMEOUT", -9
            except Exception as e:
                return "", f"EXEC_ERROR:{e}", 127

    def eval_testcases(self, code_text: str, tests: List[Tuple[str, str]]) -> Dict[str, float]:
        passed = 0
        compile_ok = 0
        runtime_fail = 0
        for tin, tout in tests:
            out, err, rc = self.run_code(code_text, tin)
            ok = (rc == 0) and (out.strip() == tout.strip())
            passed += 1 if ok else 0
            compile_ok += 1 if rc == 0 else 0
            runtime_fail += 1 if rc != 0 else 0
        total = max(1, len(tests))
        return {
            "passed": float(passed),
            "total": float(total),
            "pass_rate": float(passed) / total,
            "compile_rate": float(compile_ok) / total,
            "runtime_failures": float(runtime_fail),
        }


# ---- Hugging Face dataset adapter ----
class CodeDatasetAdapter:
    """
    Adapts generic HF datasets to (problem prompt, tests, optional gold).
    For PrimeIntellect/deepcoder-gold-standard-solutions:

      - 'prompt' : natural language problem (string)
      - 'verification_info' : JSON with a 'ground_truth' list of stdin_stdout tests
          e.g. [{"type":"stdin_stdout","input":"3 6\n","output":"3\n"}, ...]

    Optional: 'gold_standard_solution' (string) if provided by a dataset.
    """
    def __init__(self, ds_name: str, split: str = "train", text_col: str = "prompt",
                 verify_col: str = "verification_info", gold_col: Optional[str] = None, 
                 streaming: bool = False, val_ratio: float = 0.05, test_ratio: float = 0.05, 
                 split_seed: int = 42):
        if not _HAVE_DATASETS:
            raise ImportError("Please install datasets: pip install datasets")
        self.ds_name = ds_name
        self._ds_cache = {}
        self._split = split
        self.text_col = text_col
        self.verify_col = verify_col
        self.gold_col = gold_col
        self.streaming = streaming
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_seed = split_seed
        self._auto_split_done = False
        self._load_split(split)

    def _load_split(self, split: str):
        if split not in self._ds_cache:
            # Check if we need to auto-split
            if split == "train" and not self._auto_split_done:
                # First, try to see what splits exist
                available_native = self._check_native_splits()
                # If only train exists (or train is the only one we can load), auto-split it
                if "train" in available_native and len(available_native) == 1:
                    self._auto_split_train()
                    self._auto_split_done = True
                    # After auto-splitting, check cache again
                    if split in self._ds_cache:
                        return
            
            # Normal loading
            try:
                self._ds_cache[split] = load_dataset(self.ds_name, split=split, streaming=self.streaming)
            except Exception as e:
                # If split doesn't exist and we haven't auto-split yet, try auto-splitting train
                if split in ["val", "validation", "test"] and not self._auto_split_done:
                    if "train" not in self._ds_cache:
                        try:
                            self._ds_cache["train"] = load_dataset(self.ds_name, split="train", streaming=self.streaming)
                        except Exception:
                            raise e
                    self._auto_split_train()
                    self._auto_split_done = True
                    if split not in self._ds_cache:
                        raise ValueError(f"Split '{split}' not available even after auto-splitting train")
                else:
                    raise e
    
    def _check_native_splits(self) -> List[str]:
        """Check what splits are natively available in the dataset."""
        available = []
        common_splits = ["train", "validation", "val", "test", "dev"]
        for split_name in common_splits:
            try:
                # Try loading with streaming to check existence without downloading
                test_ds = load_dataset(self.ds_name, split=split_name, streaming=True, num_proc=None)
                available.append(split_name)
            except Exception:
                continue
        return available
    
    def _auto_split_train(self):
        """Automatically split train dataset into train/val/test."""
        # Ensure train is loaded (non-streaming)
        if "train" not in self._ds_cache:
            try:
                train_ds = load_dataset(self.ds_name, split="train", streaming=False)
                self._ds_cache["train"] = train_ds
            except Exception:
                return  # Can't auto-split if we can't load train
        
        train_ds = self._ds_cache.get("train")
        if train_ds is None:
            return
        if self.streaming:
            # Can't split streaming datasets
            print("[WARN] Cannot auto-split streaming dataset")
            return
        
        # Calculate split sizes
        total_size = len(train_ds)
        test_size = max(1, int(total_size * self.test_ratio))
        val_size = max(1, int(total_size * self.val_ratio))
        train_size = total_size - val_size - test_size
        
        if train_size <= 0:
            # Not enough data to split meaningfully
            self._ds_cache["train"] = train_ds
            return
        
        # Use HF datasets train_test_split - first split train from (val+test)
        # Then split val from test
        splits1 = train_ds.train_test_split(
            test_size=val_size + test_size,
            shuffle=True,
            seed=self.split_seed
        )
        
        # Split the test portion into val and test
        val_test_ds = splits1["test"]
        splits2 = val_test_ds.train_test_split(
            test_size=test_size,
            shuffle=True,
            seed=self.split_seed + 1
        )
        
        # Store all splits
        self._ds_cache["train"] = splits1["train"]
        self._ds_cache["val"] = splits2["train"]
        self._ds_cache["test"] = splits2["test"]
        
        print(f"[INFO] Auto-split dataset: train={len(self._ds_cache['train'])}, "
              f"val={len(self._ds_cache['val'])}, test={len(self._ds_cache['test'])}")

    def get_split(self, split: str):
        self._load_split(split)
        return self._ds_cache.get(split)

    def size(self, split: str) -> int:
        ds = self.get_split(split)
        if hasattr(ds, '__len__'):
            return len(ds)
        return 0

    def available_splits(self) -> List[str]:
        """Return list of available splits for this dataset."""
        # Check what we've successfully loaded/cached
        if self._ds_cache:
            return list(self._ds_cache.keys())
        # If we haven't loaded anything yet, trigger auto-split if needed
        # by checking native splits first
        native_splits = self._check_native_splits()
        if "train" in native_splits and len(native_splits) == 1:
            # Only train exists, so we'll auto-split it
            return ["train", "val", "test"]
        return native_splits if native_splits else ["train"]  # Fallback to train as default

    def _parse_tests(self, verify_blob: str) -> List[Tuple[str, str]]:
        try:
            v = json.loads(verify_blob) if isinstance(verify_blob, str) else verify_blob
            # Some datasets store directly as list; some as dict{'ground_truth': ...}
            gt = v.get("ground_truth", v) if isinstance(v, dict) else v
            tests = []
            for t in gt:
                if t.get("type") == "stdin_stdout":
                    tests.append((t.get("input",""), t.get("output","")))
            return tests
        except Exception:
            return []

    def iter_examples(self, split: Optional[str] = None):
        split = split or self._split
        self._load_split(split)
        ds = self._ds_cache[split]
        # Supports streaming or in-memory
        iterator = ds if isinstance(ds, Iterable) else iter(ds)
        for ex in iterator:
            prompt = ex.get(self.text_col, "")
            verify_blob = ex.get(self.verify_col, "")
            tests = self._parse_tests(verify_blob)
            gold = None
            if self.gold_col:
                gold = ex.get(self.gold_col, None)
            yield {
                "prompt": prompt,
                "tests": tests,
                "gold": gold
            }


# ---- Code reward (extract code from NEXT_STATE text and run tests) ----
def extract_code_from_next_state(text: str) -> str:
    """
    Heuristics:
      - if triple backticks present, take the first fenced block
      - else use the whole text verbatim
    """
    s = text.strip()
    if "```" in s:
        parts = s.split("```")
        if len(parts) >= 3:
            # parts[1] could be "python\ncode" or just "code"
            block = parts[1]
            if "\n" in block:
                lang, code = block.split("\n", 1)
                return code
            return block
    return s

def make_code_reward_fn(sandbox: CodeSandbox, tests: List[Tuple[str,str]]) -> Callable[[str,str,str], float]:
    def _reward(_state_text: str, _action_text: str, next_state_text: str) -> float:
        code = extract_code_from_next_state(next_state_text)
        intrinsic = 0.0
        if any(tok in code for tok in ("def ", "import ", "=", "print(")):
            intrinsic = 0.2
        if "```python" in next_state_text or code.strip():
            intrinsic += 0.1

        res = sandbox.eval_testcases(code, tests)
        reward = float(res["pass_rate"])
        reward += 0.2 * float(res.get("compile_rate", 0.0))
        if res.get("passed", 0.0) > 0 and reward < 1.0:
            reward += 0.05
        reward += intrinsic
        reward = max(0.0, min(1.0, reward))
        return reward
    return _reward

def greedy_code_from_model(model: WorldModelGraphGRM, graph: GraphMemory, focus: List[str],
                           prompt: str, max_new_tokens: int = 256, device: Optional[torch.device] = None) -> str:
    """Generate code greedily (temperature=0) for evaluation."""
    device = device or next(model.parameters()).device
    state_text = (
        "You are a Python 3 coding agent. "
        "Solve the following problem. Print the answer to stdout. "
        "Return ONLY the full Python program, wrapped in triple backticks.\n\n"
        f"{prompt}"
    )
    action_text = "write_python3_program"
    with torch.no_grad():
        next_state = model.generate_next_state(
            (graph, focus), state_text, action_text,
            max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0, device=device
        )
    return next_state


# ---- Code task trainer (SFT warmup + GRPO) ----
def cmd_code_train(args):
    """
    Example:
      python model.py code_train \
        --dataset PrimeIntellect/deepcoder-gold-standard-solutions \
        --split train \
        --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --epochs 3 \
        --sft_steps 200 \
        --max_new_tokens 256
    """
    set_global_determinism(42)
    device = device_of()
    
    # Setup save directory and CSV
    os.makedirs(args.save_dir, exist_ok=True)
    csv_path = os.path.join(args.save_dir, "metrics.csv")
    
    # Enable memory optimizations
    import torch.backends.cuda
    # Enable TF32 for faster computation (new API)
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
    os.environ.setdefault("WM_GRAD_CKPT", "1")  # Enable gradient checkpointing
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    
    # Clear any existing cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Enable LoRA for memory efficiency (trains only small adapters, not full model)
    os.environ.setdefault("WM_USE_LORA", "1")
    
    # Backbone world model - aggressively optimized for 24GB GPU
    wm = WorldModelGraphGRM(model_name=args.model_name,
                            train_backbone=True,
                            use_gaussian_reward=False,  # Disable Gaussian reward (saves memory)
                            reward_hidden=64,  # Minimal
                            graph_soft_tokens=1,  # Minimal
                            graph_layers=1,
                            graph_hidden=64,  # Minimal
                            graph_heads=1).to(device)  # Minimal
    teacher_backbone_state = {k: v.clone().detach().cpu() for k, v in wm.backbone.state_dict().items()}
    print("[INFO] Saved teacher model snapshot for knowledge distillation")
    
    # Enable mixed precision training
    use_amp = torch.cuda.is_available()

    # Dataset
    adapter = CodeDatasetAdapter(
        ds_name=args.dataset,
        split=args.split,
        text_col=args.text_col,
        verify_col=args.verify_col,
        gold_col=args.gold_col,
        streaming=args.streaming,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=42  # Use deterministic seed for reproducible splits
    )

    # Sandbox
    sandbox = CodeSandbox(time_limit_s=args.time_limit, mem_limit_mb=args.mem_limit)

    # Trainers - aggressively optimized for memory
    grpo = GRPOTrainer(wm, GRPOConfig(
        K=max(1, args.K),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        clip_eps=0.2,
        ref_kl_coef=0.02,
        ent_coef=0.0,
        lr=args.lr,
        use_internal_reward=False,  # we supply reward_fn below
        use_rag=False,              # RAG off by default for code
        use_reference_kl=args.use_reference_kl,
        target_kl=0.08,
        teacher_kl_weight=args.teacher_kl_weight,
        min_advantage=args.min_advantage,
    ))
    if args.teacher_kl_weight > 0:
        grpo._teacher_state = teacher_backbone_state

    bptt = BPTTTrainer(wm, BPTTConfig(lambda_graph=args.bptt_lambda_graph, lambda_dyn=args.bptt_lambda_dyn, window=1))
    act_space = ActionSpace()
    gfn = GFNTrainer(wm, GFNConfig(lr=1e-5, lambda_db=1.0, lambda_tb=1.0, Z_learn=True), act_space=act_space)
    replay = ReplayBuffer(cap=5000)

    # A tiny supervised warm start if gold is present
    opt_sft = torch.optim.AdamW([p for p in wm.parameters() if p.requires_grad], lr=args.sft_lr)
    sft_done = 0
    sft_buffer: List[Tuple[str, str, str]] = []

    # One tiny graph (not essential for code, but keeps API uniform)
    gmem = make_demo_graph()
    focus = ["File:main.py", "Test:test_sum_large"]

    def seed_replay_buffer(num_steps: int):
        if num_steps <= 0:
            return
        cur = gmem.clone()
        for _ in range(num_steps):
            acts = act_space.legal_actions(cur)
            if not acts:
                break
            act = random.choice(acts)
            nxt = act_space.apply(cur, act)
            reward_seed = 0.0
            if act.kind == "mark_pass":
                reward_seed = 1.0
            elif act.kind == "set_build_ok" and act.args[1] == "True":
                reward_seed = 0.5
            replay.add(Transition(cur, act, nxt, reward_seed, terminal=True))
            cur = nxt

    seed_replay_buffer(args.replay_init_samples)
    
    # Initialize W&B if requested
    if args.wandb and _HAVE_WANDB:
        wandb.init(project=args.wandb_project, config=vars(args))
    
    best_val = 0.0
    
    # CSV logger header
    if not os.path.exists(csv_path):
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join([
                "epoch","phase","step",
                # world model
                "loss_sft","loss_policy","loss_kl","kl_curr","kl_coef",
                "reward_mean","reward_std","pass_rate",
                # BPTT / graph learning
                "wm_dyn_loss","wm_graph_loss",
                # GFlowNet
                "gfn_loss_db","gfn_loss_tb","gfn_flow_mean","gfn_flow_std",
                # Graph stats
                "graph_nodes","graph_edges","deg_in_mean","deg_out_mean","deg_in_p95","deg_out_p95"
            ]) + "\n")

    def log_row(epoch, phase, step, sft, stats_grpo, pass_rate, bptt_stats=None, gfn_db=None, gfn_tb=None, gstats=None):
        wm_dyn = bptt_stats.get("loss_dyn_mean") if bptt_stats else None
        wm_gl  = bptt_stats.get("loss_graph_mean") if bptt_stats else None
        gdb = gfn_db.get("gfn_loss_db") if gfn_db else None
        gtb = gfn_tb.get("gfn_loss_tb") if gfn_tb else None
        gfm = gfn_db.get("gfn_flow_mean") if gfn_db else None
        gfs = gfn_db.get("gfn_flow_std") if gfn_db else None
        gs  = gstats or {}
        row = [
            epoch, phase, step,
            (f"{sft:.6f}" if sft is not None else ""),
            stats_grpo.get("loss_policy",""),
            stats_grpo.get("loss_kl",""),
            stats_grpo.get("kl_curr",""),
            stats_grpo.get("kl_coef",""),
            stats_grpo.get("reward_mean",""),
            stats_grpo.get("reward_std",""),
            (f"{pass_rate:.6f}" if pass_rate is not None else ""),
            (f"{wm_dyn:.6f}" if wm_dyn is not None else ""),
            (f"{wm_gl:.6f}" if wm_gl is not None else ""),
            (f"{gdb:.6f}" if gdb is not None else ""),
            (f"{gtb:.6f}" if gtb is not None else ""),
            (f"{gfm:.6f}" if gfm is not None else ""),
            (f"{gfs:.6f}" if gfs is not None else ""),
            gs.get("graph_nodes",""),
            gs.get("graph_edges",""),
            gs.get("deg_in_mean",""),
            gs.get("deg_out_mean",""),
            gs.get("deg_in_p95",""),
            gs.get("deg_out_p95",""),
        ]
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(",".join(map(str, row)) + "\n")

    def compute_kd_loss_student_teacher(
        student_model: WorldModelGraphGRM,
        teacher_state: Dict[str, torch.Tensor],
        inputs: List[str],
        targets: List[str],
        graphs: List,
        device: torch.device,
        student_logits: torch.Tensor,
        temperature: float = 2.0,
    ) -> torch.Tensor:
        if not teacher_state:
            return torch.tensor(0.0, device=device)

        current_state = {}
        for name, param in student_model.backbone.named_parameters():
            if name in teacher_state:
                current_state[name] = param.data.clone()
                param.data.copy_(teacher_state[name].to(device=device, dtype=param.data.dtype))

        with torch.no_grad():
            teacher_out = student_model.forward_dynamics(graphs, inputs, targets, device=device)
            teacher_logits = teacher_out["logits"]

        for name, param in student_model.backbone.named_parameters():
            if name in current_state:
                param.data.copy_(current_state[name])
        del current_state

        enc_in = student_model._text(inputs, device)
        enc_tg = student_model._text(targets, device)
        target_len = enc_tg["input_ids"].size(1)
        input_len = enc_in["input_ids"].size(1)
        soft_tokens = student_model.graph_soft_tokens
        start_idx = soft_tokens + input_len

        student_slice = student_logits[:, start_idx:start_idx + target_len, :]
        teacher_slice = teacher_logits[:, start_idx:start_idx + target_len, :]

        student_slice = student_slice.reshape(-1, student_slice.size(-1))
        teacher_slice = teacher_slice.reshape(-1, teacher_slice.size(-1))

        student_log_probs = F.log_softmax(student_slice / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_slice / temperature, dim=-1)
        kd = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        return kd * (temperature ** 2)

    def run_sft_minibatch() -> Optional[Tuple[float, float]]:
        nonlocal sft_done, sft_buffer
        remaining = args.sft_steps - sft_done
        if remaining <= 0 or not sft_buffer:
            return None
        take = min(len(sft_buffer), remaining)
        batch = sft_buffer[:take]
        inputs = [pack_world_io(s, a, None) for s, a, _ in batch]
        targets = [pack_world_io(s, a, ns) for s, a, ns in batch]
        graphs = [(gmem, focus) for _ in batch]
        out = wm.forward_dynamics(graphs, inputs, targets, device=device)
        ce_loss = out["loss"]
        kd_loss = torch.tensor(0.0, device=device)
        if args.kd_weight > 0 and teacher_backbone_state:
            kd_loss = compute_kd_loss_student_teacher(
                wm,
                teacher_backbone_state,
                inputs,
                targets,
                graphs,
                device,
                out["logits"],
                temperature=args.kd_temperature,
            )
        total_loss = ce_loss + args.kd_weight * kd_loss
        opt_sft.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), args.grad_clip)
        opt_sft.step()
        sft_done += take
        del sft_buffer[:take]
        ce_val = float(ce_loss.detach().cpu())
        kd_val = float(kd_loss.detach().cpu()) if args.kd_weight > 0 else 0.0
        return ce_val, kd_val

    # ===== Epoch loops =====
    for epoch in range(1, args.epochs + 1):
        # === TRAIN ===
        step = 0
        train_pass_accum = []
        train_size = adapter.size("train") if adapter.size("train") > 0 else None
        pbar = tqdm(adapter.iter_examples("train"), total=train_size, desc=f"train epoch {epoch}")
        metrics_interval = max(1, args.metrics_interval if args.metrics_interval else args.log_every or 1)
        sft_accum: List[Dict[str, float]] = []
        grpo_accum: List[Dict[str, float]] = []
        gfn_db_accum: List[Dict[str, float]] = []
        gfn_tb_accum: List[Dict[str, float]] = []
        bptt_accum: List[Dict[str, float]] = []
        gstats_accum: List[Dict[str, float]] = []
        pass_interval_accum: List[float] = []
        steps_since_log = 0

        def flush_metrics(force: bool = False):
            nonlocal sft_accum, grpo_accum, gfn_db_accum, gfn_tb_accum, bptt_accum, gstats_accum, pass_interval_accum, steps_since_log
            if not force and steps_since_log < metrics_interval:
                return
            if not grpo_accum:
                steps_since_log = 0
                pass_interval_accum = []
                return
            sft_mean = mean_dict_list(sft_accum)
            grpo_mean = mean_dict_list(grpo_accum)
            gfn_db_mean = mean_dict_list(gfn_db_accum)
            gfn_tb_mean = mean_dict_list(gfn_tb_accum)
            bptt_mean = mean_dict_list(bptt_accum)
            gstats_mean = mean_dict_list(gstats_accum)
            pr_avg = float(np.mean(pass_interval_accum)) if pass_interval_accum else 0.0

            if sft_mean:
                log_component_stats("SFT", {
                    "ce_loss": sft_mean.get("ce"),
                    "kd_loss": sft_mean.get("kd"),
                    "total": sft_mean.get("total")
                })
            else:
                log_component_stats("SFT", {"total": None})

            log_component_stats("GRPO", {
                "loss_total": grpo_mean.get("loss_total"),
                "loss_policy": grpo_mean.get("loss_policy"),
                "loss_kl": grpo_mean.get("loss_kl"),
                "loss_teacher_kl": grpo_mean.get("loss_teacher_kl"),
                "reward_mean": grpo_mean.get("reward_mean"),
                "reward_std": grpo_mean.get("reward_std"),
                "kl": grpo_mean.get("kl_curr")
            })
            log_component_stats("GFN_DB", gfn_db_mean)
            log_component_stats("GFN_TB", gfn_tb_mean)
            log_component_stats("BPTT", bptt_mean)

            pbar.set_postfix(
                rew=f"{grpo_mean.get('reward_mean', 0.0):.3f}",
                pr=f"{pr_avg:.2f}",
                gdb=f"{gfn_db_mean.get('gfn_loss_db', 0.0):.3f}",
                gtb=f"{gfn_tb_mean.get('gfn_loss_tb', 0.0):.3f}",
            )

            sft_total = sft_mean.get("total") if sft_mean else None
            stats_for_row = {
                "loss_policy": grpo_mean.get("loss_policy"),
                "loss_kl": grpo_mean.get("loss_kl"),
                "kl_curr": grpo_mean.get("kl_curr"),
                "kl_coef": grpo_mean.get("kl_coef"),
                "reward_mean": grpo_mean.get("reward_mean"),
                "reward_std": grpo_mean.get("reward_std"),
            }
            log_row(epoch, "train", step, sft_total, stats_for_row, pr_avg,
                    bptt_stats=bptt_mean, gfn_db=gfn_db_mean, gfn_tb=gfn_tb_mean, gstats=gstats_mean)

            if args.wandb and _HAVE_WANDB:
                wandb.log({
                    "epoch": epoch, "phase": "train", "step": step,
                    "sft_loss_total": sft_total if sft_total is not None else 0,
                    "sft_loss_ce": sft_mean.get("ce", 0.0) if sft_mean else 0.0,
                    "sft_loss_kd": sft_mean.get("kd", 0.0) if sft_mean else 0.0,
                    "grpo/loss_policy": grpo_mean.get("loss_policy", 0.0),
                    "grpo/loss_kl": grpo_mean.get("loss_kl", 0.0),
                    "grpo/loss_teacher_kl": grpo_mean.get("loss_teacher_kl", 0.0),
                    "grpo/kl_curr": grpo_mean.get("kl_curr", 0.0),
                    "grpo/kl_coef": grpo_mean.get("kl_coef", 0.0),
                    "grpo/reward_mean": grpo_mean.get("reward_mean", 0.0),
                    "grpo/reward_std": grpo_mean.get("reward_std", 0.0),
                    "pass_rate/train_single": pr_avg,
                    "bptt/loss_dyn_mean": bptt_mean.get("loss_dyn_mean", 0.0),
                    "bptt/loss_graph_mean": bptt_mean.get("loss_graph_mean", 0.0),
                    "gfn/loss_db": gfn_db_mean.get("gfn_loss_db", 0.0),
                    "gfn/loss_tb": gfn_tb_mean.get("gfn_loss_tb", 0.0),
                    "gfn/flow_mean": gfn_db_mean.get("gfn_flow_mean", 0.0),
                    "gfn/flow_std": gfn_db_mean.get("gfn_flow_std", 0.0),
                    **{f"graph/{k}": v for k, v in gstats_mean.items()},
                })
            if args.log_grad_every and (step % max(1, args.log_grad_every) == 0):
                gn, gc = gradient_norm(wm)
                log_component_stats("GRAD", {"norm": gn, "count": gc})

            sft_accum.clear()
            grpo_accum.clear()
            gfn_db_accum.clear()
            gfn_tb_accum.clear()
            bptt_accum.clear()
            gstats_accum.clear()
            pass_interval_accum.clear()
            steps_since_log = 0
        for ex in pbar:
            prompt, tests, gold = ex["prompt"], ex["tests"], ex["gold"]

            # Graph snapshot stats (before updates)
            gstats_now = graph_stats(gmem)

            # 1) SFT (code tokens)
            sft_loss_val = None
            if sft_done < args.sft_steps and gold and isinstance(gold, str) and len(gold.strip()) > 0:
                state_text = f"Problem:\n{prompt}\n\nOutput only the full Python 3 program that solves it.\nWrap the program in triple backticks."
                action_text = "write_python3_program"
                next_state_text = f"```python\n{gold}\n```"
                sft_buffer.append((state_text, action_text, next_state_text))
                if len(sft_buffer) >= args.sft_batch or sft_done + len(sft_buffer) >= args.sft_steps:
                    sft_loss_val = run_sft_minibatch()

            # 2) GFN global step(s)
            # Only train GFN if replay buffer has data
            if len(replay.buf) > 0:
                db_batch = replay.sample(args.gfn_db_batch) if args.gfn_db_batch > 0 else []
                gfn_db_stats = gfn.db_step(db_batch)
                tb_sample = replay.sample(args.gfn_tb_traj_len) if args.gfn_tb_traj_len > 0 else []
                gfn_tb_stats = gfn.tb_step([tb_sample]) if tb_sample else {"gfn_loss_tb": 0.0}
            else:
                gfn_db_stats = {"gfn_loss_db": 0.0, "gfn_flow_mean": 0.0, "gfn_flow_std": 0.0}
                gfn_tb_stats = {"gfn_loss_tb": 0.0}

            # 3) GRPO (execution aligned)
            reward_fn = make_code_reward_fn(sandbox, tests)
            grpo.reward_fn = reward_fn
            state_text = (
                "You are a Python 3 coding agent. "
                "Solve the following problem. Print the answer to stdout. "
                "Return ONLY the full Python program, wrapped in triple backticks.\n\n"
                f"{prompt}"
            )
            stats_grpo = {}
            for _ in range(max(1, args.grpo_inner_steps)):
                stats_grpo = grpo.grpo_step([(GRPOItem(state_text=state_text, action_text="write_python3_program", graph_focus=focus), gmem)])
            
            # Cleanup GPU memory after GRPO step
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 4) BPTT (world-model consistency)
            bptt_stats = bptt.step([(Trajectory(
                states=[f"KB:\n - NODE(File,File:main.py)\n - NODE(Test,Test:test_sum_large)"],
                actions=["noop"], next_states=["noop"], graph_focuses=[focus, focus]
            ), gmem)])

            # 5) Immediate greedy eval for this row
            code_text = extract_code_from_next_state(
                greedy_code_from_model(wm, gmem, focus, prompt, max_new_tokens=args.max_new_tokens, device=device)
            )
            pr = sandbox.eval_testcases(code_text, tests)["pass_rate"]
            train_pass_accum.append(pr)
            sft_total_for_row = None
            sft_ce = None
            sft_kd = None
            if sft_loss_val is not None:
                if isinstance(sft_loss_val, tuple):
                    sft_ce, sft_kd = sft_loss_val
                    sft_total_for_row = sft_ce + args.kd_weight * sft_kd
                else:
                    sft_total_for_row = sft_loss_val
            if sft_total_for_row is not None:
                entry = {"total": sft_total_for_row}
                if sft_ce is not None:
                    entry["ce"] = sft_ce
                if sft_kd is not None:
                    entry["kd"] = sft_kd
                sft_accum.append(entry)
            grpo_accum.append(stats_grpo)
            gfn_db_accum.append(gfn_db_stats)
            gfn_tb_accum.append(gfn_tb_stats)
            bptt_accum.append(bptt_stats)
            gstats_accum.append(gstats_now)
            pass_interval_accum.append(pr)
            steps_since_log += 1
            flush_metrics()

            step += 1
            if args.steps_per_epoch and step >= args.steps_per_epoch:
                break
        flush_metrics(force=True)

        pending_loss = run_sft_minibatch()
        if pending_loss is not None:
            sft_loss_val = pending_loss

        # === VALIDATE ===
        val_passes = []
        val_sft_losses = []
        val_graphs = []
        val_split = args.val_split or "val"
        available = adapter.available_splits()
        if val_split and val_split in available:
            try:
                val_ds = adapter.get_split(val_split)
                if val_ds is not None:
                    val_size = adapter.size(val_split) if adapter.size(val_split) > 0 else None
                    vbar = tqdm(adapter.iter_examples(val_split), total=val_size, desc=f"val epoch {epoch}")
                    for ex in vbar:
                        prompt, tests, gold = ex["prompt"], ex["tests"], ex["gold"]
                        code_text = extract_code_from_next_state(
                            greedy_code_from_model(wm, gmem, focus, prompt, max_new_tokens=args.max_new_tokens, device=device)
                        )
                        pr = sandbox.eval_testcases(code_text, tests)["pass_rate"]
                        val_passes.append(pr)
                        if gold and sft_done > 0:
                            state_text = f"Problem:\n{prompt}\n\nOutput only the full Python 3 program that solves it.\nWrap the program in triple backticks."
                            action_text = "write_python3_program"
                            next_state_text = f"```python\n{gold}\n```"
                            inputs  = [pack_world_io(state_text, action_text, None)]
                            targets = [pack_world_io(state_text, action_text, next_state_text)]
                            with torch.no_grad():
                                out = wm.forward_dynamics([(gmem, focus)], inputs, targets, device=device)
                                val_sft_losses.append(float(out["loss"].detach().cpu()))
                        val_graphs.append(graph_stats(gmem))

                    val_pr = float(np.mean(val_passes)) if len(val_passes) else 0.0
                    val_sft = float(np.mean(val_sft_losses)) if len(val_sft_losses) else None
                    mean_gstats = {}
                    if val_graphs:
                        keys = val_graphs[0].keys()
                        mean_gstats = {k: float(np.mean([d[k] for d in val_graphs])) for k in keys}

                    # record a single row for epoch-level val
                    log_row(epoch, "val", step, val_sft,
                            {"loss_policy": "", "loss_kl":"", "kl_curr":"", "kl_coef":"", "reward_mean":"", "reward_std":""},
                            val_pr, bptt_stats=None, gfn_db=None, gfn_tb=None, gstats=mean_gstats)
                    print(f"[epoch {epoch}] val pass-rate: {val_pr:.4f} | val sft: {val_sft}")
                    
                    if args.wandb and _HAVE_WANDB:
                        wandb.log({
                            "epoch": epoch, "phase": "val", "pass_rate/val": val_pr,
                            "val_sft_loss": val_sft if val_sft is not None else 0,
                            **{f"graph/val_{k}": v for k,v in mean_gstats.items()}
                        })

                    # checkpoint on best val
                    if val_pr > best_val:
                        best_val = val_pr
                        best_ckpt = os.path.join(args.save_dir, f"best-epoch{epoch}-val{best_val:.4f}.pt")
                        torch.save({"model": wm.state_dict(), "epoch": epoch, "val_pass_rate": best_val}, best_ckpt)
                        print(f"  -> saved new best to {best_ckpt}")
            except Exception as e:
                print(f"[INFO] Skipping validation; error loading split '{val_split}': {e}")
        else:
            if val_split:
                print(f"[INFO] Skipping validation; split '{val_split}' not present (available: {available})")

        # === TEST (optional) ===
        test_split = args.test_split or "test"
        available_test = adapter.available_splits()
        if test_split and test_split in available_test and (args.eval_test_each_epoch or epoch == args.epochs):
            try:
                test_ds = adapter.get_split(test_split)
                if test_ds is not None:
                    test_passes = []
                    test_size = adapter.size(test_split) if adapter.size(test_split) > 0 else None
                    tbar = tqdm(adapter.iter_examples(test_split), total=test_size, desc=f"test epoch {epoch}")
                    for ex in tbar:
                        prompt, tests, _ = ex["prompt"], ex["tests"], ex["gold"]
                        code_text = extract_code_from_next_state(
                            greedy_code_from_model(wm, gmem, focus, prompt, max_new_tokens=args.max_new_tokens, device=device)
                        )
                        pr = sandbox.eval_testcases(code_text, tests)["pass_rate"]
                        test_passes.append(pr)
                    test_pr = float(np.mean(test_passes)) if len(test_passes) else 0.0
                    log_row(epoch, "test", step, None,
                            {"loss_policy":"","loss_kl":"","kl_curr":"","kl_coef":"","reward_mean":"","reward_std":""},
                            test_pr)
                    print(f"[epoch {epoch}] test pass-rate: {test_pr:.4f}")
                    
                    if args.wandb and _HAVE_WANDB:
                        wandb.log({"epoch": epoch, "phase": "test", "pass_rate/test": test_pr})
            except Exception as e:
                print(f"[INFO] Skipping test; error loading split '{test_split}': {e}")
        else:
            if test_split and (args.eval_test_each_epoch or epoch == args.epochs):
                print(f"[INFO] Skipping test; split '{test_split}' not present (available: {available_test})")

    print(f"[code_train] done. best val pass-rate={best_val:.4f}")


# =========================
# Demo graph + reward
# =========================

def make_demo_graph() -> GraphMemory:
    g = GraphMemory()
    g.upsert_node("File:main.py", "File", lang="py", sha1="abc", lines="42")
    g.upsert_node("Test:test_sum_small", "Test")
    g.upsert_node("Test:test_sum_large", "Test")
    g.upsert_node("BuildRun:current", "BuildRun", ok="False", time_ms="200", reason="init")
    g.add_edge("BuildRun:current", "touches", "File:main.py")
    g.add_edge("Test:test_sum_large", "fails", "BuildRun:current")
    return g

def reward_textual(state, action, next_state):
    s = next_state.lower()
    score = 0.0
    if "pass" in s: score += 1.0
    if "fail" in s: score -= 0.5
    if "minimal patch" in state.lower(): score += 0.1
    return max(0.0, score)


# =========================
# CLI commands
# =========================

def cmd_train(args):
    set_global_determinism(42)
    cfg = ColBERTConfig(model_name=args.model_name, dim=args.dim,
                        max_query_len=args.max_query_len, max_doc_len=args.max_doc_len)
    tcfg = TrainConfig(lr=args.lr, wd=args.wd, epochs=args.epochs, batch_size=args.batch_size,
                       temperature=args.temperature, teacher_lambda=args.teacher_lambda,
                       fp16=not args.no_fp16, max_query_len=args.max_query_len, max_doc_len=args.max_doc_len)
    tok = Tokenizer(cfg.model_name)
    model = ColBERTEncoder(cfg)
    trainer = ColBERTTrainer(model, tok, cfg, tcfg)
    trainer.fit(args.train_jsonl, args.ckpt)

def cmd_index(args):
    set_global_determinism(42)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ColBERTConfig(**ckpt["config"])
    tok = Tokenizer(cfg.model_name)
    model = ColBERTEncoder(cfg)
    model.load_state_dict(ckpt["state_dict"])
    retr = LateInteractionRetriever(model, tok, cfg, IndexConfig(
        dim=cfg.dim, n_centroids=args.n_centroids, prune_topk=args.prune_topk,
        max_doc_len=args.max_doc_len, max_query_len=args.max_query_len))
    docs = read_index_jsonl(args.index_jsonl)
    retr.build_index(docs, args.index_dir)

def cmd_search(args):
    set_global_determinism(42)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ColBERTConfig(**ckpt["config"])
    tok = Tokenizer(cfg.model_name)
    model = ColBERTEncoder(cfg)
    model.load_state_dict(ckpt["state_dict"])
    retr = LateInteractionRetriever(model, tok, cfg, IndexConfig(dim=cfg.dim,
                                                                 max_doc_len=args.max_doc_len,
                                                                 max_query_len=args.max_query_len))
    retr.load_index(args.index_dir)
    res = retr.search([args.query], k=args.k)[0]
    for rank, (doc_id, score) in enumerate(res, 1):
        print(f"{rank:2d}. doc_id={doc_id}  score={score:.4f}")

def cmd_world_model(args):
    set_global_determinism(42)
    device = device_of()

    model = WorldModelGraphGRM(model_name=args.model_name, train_backbone=True,
                               use_gaussian_reward=True, reward_hidden=128,
                               graph_soft_tokens=6, graph_layers=1, graph_hidden=256, graph_heads=4).to(device)

    rag = None
    if args.use_rag:
        assert args.colbert_ckpt and args.colbert_index_dir and args.colbert_corpus_jsonl, \
            "--use_rag requires --colbert_ckpt, --colbert_index_dir, and --colbert_corpus_jsonl"
        retr, _ = load_colbert_retriever(args.colbert_ckpt, args.colbert_index_dir)
        doc_lookup = build_corpus_lookup(args.colbert_corpus_jsonl)
        def corpus_lookup(doc_id: int) -> str: return doc_lookup.get(int(doc_id), "")
        rag = RetrieverAugmentor(retr, corpus_lookup, k_ctx=args.rag_k, max_chars=args.rag_chars)

    grpo = GRPOTrainer(model, GRPOConfig(K=3, max_new_tokens=64, use_internal_reward=False, use_rag=bool(rag)),
                       reward_fn=reward_textual, retriever_augmentor=rag)
    bptt = BPTTTrainer(model, BPTTConfig(lambda_graph=1.0, lambda_dyn=1.0, window=1))
    act_space = ActionSpace()
    gfn = GFNTrainer(model, GFNConfig(lr=1e-5, lambda_db=1.0, lambda_tb=1.0, Z_learn=True), act_space=act_space)
    replay = ReplayBuffer(cap=5000)

    g = make_demo_graph()
    focus = ["File:main.py","Test:test_sum_large"]
    s0_txt = g.serialize_subgraph(focus, max_hops=2)
    a0_txt = "apply minimal patch to main.py"
    ns0_txt = "All tests pass."
    traj = Trajectory(states=[s0_txt], actions=[a0_txt], next_states=[ns0_txt], graph_focuses=[focus, focus])

    for _ in range(8):
        acts = act_space.legal_actions(g)
        if not acts: break
        a = random.choice(acts)
        g2 = act_space.apply(g, a)
        rew = 0.0
        if a.kind == "mark_pass": rew = 1.0
        if a.kind == "set_build_ok" and a.args[1] == "True": rew = 0.5
        replay.add(Transition(g, a, g2, rew, terminal=True))

    for step in range(1, args.steps+1):
        db_batch = replay.sample(8)
        tb_batch = [replay.sample(3)]
        stats_db = gfn.db_step(db_batch)
        stats_tb = gfn.tb_step(tb_batch)

        item = GRPOItem(state_text=s0_txt, action_text=a0_txt, graph_focus=focus)
        stats_grpo = grpo.grpo_step([(item, g)])

        stats_bptt = bptt.step([(traj, g)])

        print(f"[step {step}] GFN_DB {stats_db} | GFN_TB {stats_tb} | GRPO {stats_grpo} | BPTT {stats_bptt}")


# =========================
# CLI
# =========================

def build_argparser():
    p = argparse.ArgumentParser("ColBERT + World Model Graph GRPO")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("colbert_train", help="Train ColBERT retriever")
    pt.add_argument("--train_jsonl", required=True)
    pt.add_argument("--model_name", default="bert-base-uncased")
    pt.add_argument("--dim", type=int, default=128)
    pt.add_argument("--epochs", type=int, default=2)
    pt.add_argument("--batch_size", type=int, default=8)
    pt.add_argument("--lr", type=float, default=2e-5)
    pt.add_argument("--wd", type=float, default=0.01)
    pt.add_argument("--temperature", type=float, default=0.05)
    pt.add_argument("--teacher_lambda", type=float, default=0.5)
    pt.add_argument("--max_query_len", type=int, default=32)
    pt.add_argument("--max_doc_len", type=int, default=180)
    pt.add_argument("--ckpt", default="ckpts/colbert.pt")
    pt.add_argument("--no_fp16", action="store_true")
    pt.set_defaults(func=cmd_train)

    pi = sub.add_parser("colbert_index", help="Build ColBERT index")
    pi.add_argument("--index_jsonl", required=True)
    pi.add_argument("--ckpt", required=True)
    pi.add_argument("--index_dir", required=True)
    pi.add_argument("--n_centroids", type=int, default=4096)
    pi.add_argument("--prune_topk", type=int, default=96)
    pi.add_argument("--max_doc_len", type=int, default=180)
    pi.add_argument("--max_query_len", type=int, default=32)
    pi.set_defaults(func=cmd_index)

    ps = sub.add_parser("colbert_search", help="Search ColBERT index")
    ps.add_argument("--ckpt", required=True)
    ps.add_argument("--index_dir", required=True)
    ps.add_argument("--query", required=True)
    ps.add_argument("-k", type=int, default=10)
    ps.add_argument("--max_doc_len", type=int, default=180)
    ps.add_argument("--max_query_len", type=int, default=32)
    ps.set_defaults(func=cmd_search)

    pw = sub.add_parser("world_model", help="Train World Model with Graph GRPO")
    pw.add_argument("--model_name", default="sshleifer/tiny-gpt2", help="HF causal LM model name")
    pw.add_argument("--steps", type=int, default=3, help="Training steps")
    pw.add_argument("--use_rag", action="store_true", help="Enable ColBERT RAG")
    pw.add_argument("--colbert_ckpt", type=str, default="", help="ColBERT checkpoint path")
    pw.add_argument("--colbert_index_dir", type=str, default="", help="ColBERT index directory")
    pw.add_argument("--colbert_corpus_jsonl", type=str, default="", help="ColBERT corpus JSONL")
    pw.add_argument("--rag_k", type=int, default=3, help="Number of retrieved snippets")
    pw.add_argument("--rag_chars", type=int, default=500, help="Max chars per snippet")
    pw.set_defaults(func=cmd_world_model)

    pc = sub.add_parser("code_train", help="End-to-end code task training (any HF dataset, any HF causal LLM)")
    pc.add_argument("--dataset", required=True, help="HF dataset name, e.g. PrimeIntellect/deepcoder-gold-standard-solutions")
    pc.add_argument("--split", default="train")
    pc.add_argument("--model_name", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    pc.add_argument("--text_col", default="prompt")
    pc.add_argument("--verify_col", default="verification_info")
    pc.add_argument("--gold_col", default=None)
    pc.add_argument("--streaming", action="store_true")

    pc.add_argument("--steps", type=int, default=500)
    pc.add_argument("--sft_steps", type=int, default=100)
    pc.add_argument("--sft_batch", type=int, default=4, help="Mini-batch size for SFT warmup updates")
    pc.add_argument("--lr", type=float, default=1e-5)
    pc.add_argument("--sft_lr", type=float, default=5e-6)
    pc.add_argument("--K", type=int, default=2, help="GRPO samples per step (reduced from 3 for 24GB GPU)")
    pc.add_argument("--grpo_inner_steps", type=int, default=1, help="Number of GRPO updates per training example")
    pc.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens to generate (reduced for 24GB GPU)")
    pc.add_argument("--temperature", type=float, default=0.7)
    pc.add_argument("--top_p", type=float, default=0.9)
    pc.add_argument("--use_reference_kl", action="store_true", help="Enable reference KL regularization during GRPO")
    pc.add_argument("--kd_weight", type=float, default=0.5, help="Weight for knowledge distillation loss during SFT warmup")
    pc.add_argument("--kd_temperature", type=float, default=2.0, help="Knowledge distillation temperature during SFT warmup")
    pc.add_argument("--teacher_kl_weight", type=float, default=0.1, help="Weight for teacher KL in GRPO (0 disables)")

    pc.add_argument("--time_limit", type=float, default=2.5)
    pc.add_argument("--mem_limit", type=int, default=512)
    pc.add_argument("--log_every", type=int, default=10)
    pc.add_argument("--metrics_interval", type=int, default=5, help="Aggregate metrics and log every N training steps")
    pc.add_argument("--log_grad_every", type=int, default=25, help="How often to log gradient norms (in steps)")
    pc.add_argument("--min_advantage", type=float, default=0.01, help="Minimum advantage when reward variance collapses (0 disables)")
    
    # Logging & evaluation
    pc.add_argument("--save_dir", type=str, default="runs/code", help="Where to write metrics and best checkpoint")
    pc.add_argument("--epochs", type=int, default=3)
    pc.add_argument("--steps_per_epoch", type=int, default=0, help="0 = iterate whole split; else cap per epoch")
    pc.add_argument("--val_split", type=str, default=None)
    pc.add_argument("--test_split", type=str, default=None)
    pc.add_argument("--val_ratio", type=float, default=0.05, help="Used only if val_split is None")
    pc.add_argument("--test_ratio", type=float, default=0.05, help="Used only if test_split is None")
    pc.add_argument("--eval_test_each_epoch", action="store_true")
    # GFlowNet step sizes per batch
    pc.add_argument("--gfn_db_batch", type=int, default=8, help="Num transitions sampled for DB per train step (0=skip)")
    pc.add_argument("--gfn_tb_traj_len", type=int, default=3, help="Num transitions in one TB trajectory per step (0=skip)")
    pc.add_argument("--replay_init_samples", type=int, default=16, help="Number of random transitions to seed replay buffer with")
    pc.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value for SFT updates")
    pc.add_argument("--bptt_lambda_graph", type=float, default=0.5, help="Weight for graph transition loss")
    pc.add_argument("--bptt_lambda_dyn", type=float, default=1.0, help="Weight for dynamics loss")
    # W&B
    pc.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging (if installed)")
    pc.add_argument("--wandb_project", type=str, default="wm-gfn-code", help="W&B project")

    pc.set_defaults(func=cmd_code_train)

    return p

if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    args.func(args)
