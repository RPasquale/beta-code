# autognn_linkpos.py
# Universal, auto-ingest GNN trainer with:
# - Multi-file ingest (nodes/edges or directory)
# - Homo + Hetero (bipartite & multi-relation) graphs
# - Node / Graph / Edge tasks + Link Prediction (neg sampling)
# - Graph Positional Encodings: LapPE + RWSE (edge-relative bias in GraphTransformer)
# - Edge features -> attention bias (GraphTransformer)
# - Optional NeighborLoader mini-batching for large graphs
# - Tiny AutoML over {heads,layers,lr,optim,scheduler}; AMP, grad-clip, early-stop
#
# Requirements: torch, torch_geometric, pandas, numpy
# Optional: scikit-learn (kNN), pyarrow (parquet)
# (No SciPy required; Laplacian built with PyG utils; eigen done with torch where feasible)

import argparse, os, glob, math, sys
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import (
    GATv2Conv, HeteroConv, global_mean_pool
)
from torch_geometric.utils import (
    softmax, get_laplacian, to_undirected, negative_sampling, subgraph
)
from torch_geometric.loader import NeighborLoader

# ----------------------- Optional deps -----------------------
try:
    from sklearn.neighbors import NearestNeighbors
    HAVE_SK = True
except Exception:
    HAVE_SK = False

try:
    from datasets import load_dataset, load_from_disk, Dataset, concatenate_datasets
    from pathlib import Path
    HAVE_HF = True
except Exception:
    HAVE_HF = False

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

try:
    from torch.sparse.linalg import lobpcg as sparse_lobpcg
    HAVE_TORCH_LOBPCG = True
except Exception:
    HAVE_TORCH_LOBPCG = False

SUPPORTED = (".csv", ".tsv", ".parquet", ".pq", ".json", ".jsonl")

# ----------------------- HuggingFace Dataset Support -----------------------
def _convert_windows_path(path: str) -> str:
    """Convert Windows path to WSL/Linux path if running in WSL."""
    # Check if path looks like Windows path but we're in WSL
    if path.startswith("C:\\") or path.startswith("C:/"):
        # Convert C:\Users\... to /mnt/c/Users/...
        wsl_path = "/mnt/" + path[0].lower() + path[2:].replace("\\", "/")
        if os.path.exists(wsl_path):
            return wsl_path
    return path

def _is_hf_cache_dir(path: str) -> bool:
    """Check if path is a HuggingFace dataset cache directory."""
    converted_path = _convert_windows_path(path)
    if not os.path.isdir(converted_path):
        return False
    
    # Check for arrow shards - HF cache always has these
    arrow_files = list(glob.glob(os.path.join(converted_path, "*.arrow")))
    if arrow_files:
        return True
    
    # Check subdirectories recursively for arrow files
    try:
        for root, dirs, files in os.walk(converted_path):
            if any(f.endswith('.arrow') for f in files):
                return True
    except Exception:
        pass
    
    return False

def _load_hf_dataset(path: str) -> pd.DataFrame:
    """Load a HuggingFace dataset FAST - work directly with Arrow format, minimal pandas conversion."""
    if not HAVE_HF:
        raise ImportError("HuggingFace datasets library not installed. Install with: pip install datasets")
    
    converted_path = _convert_windows_path(path)
    print(f"[info] Loading HF dataset from: {converted_path}")
    
    # Collect all .arrow files recursively
    arrow_shards = []
    arrow_shards.extend(sorted(glob.glob(os.path.join(converted_path, "*.arrow"))))
    if not arrow_shards:
        try:
            for root, dirs, files in os.walk(converted_path):
                arrow_shards.extend(sorted(glob.glob(os.path.join(root, "*.arrow"))))
        except Exception as e:
            print(f"[warn] Error walking directory: {e}")
    
    if not arrow_shards:
        raise ValueError(f"No .arrow files found in: {converted_path}")
    
    print(f"[info] Found {len(arrow_shards)} arrow file(s)")
    
    # Load arrow files (FAST - Arrow format is already loaded)
    import time
    start_time = time.time()
    try:
        parts = [Dataset.from_file(shard) for shard in arrow_shards]
        dataset = concatenate_datasets(parts)
        load_time = time.time() - start_time
        print(f"[info] Loaded dataset with {len(dataset)} rows in {load_time:.2f}s")
    except Exception as e:
        raise ValueError(f"Failed to load arrow files from {converted_path}: {e}")
    
    # FAST PATH: Convert only needed columns to pandas (much faster than full conversion)
    print(f"[info] Extracting only needed columns (fast columnar conversion)...")
    extract_start = time.time()
    
    # Get column names
    sample = dataset[0] if len(dataset) > 0 else {}
    all_cols = list(sample.keys())
    
    # Find code/prompt/label columns - handle specific dataset schemas
    code_col = None
    prompt_col = None
    label_col = None
    
    # Specific column mappings for known datasets
    dataset_name_lower = os.path.basename(converted_path.rstrip("/\\")).lower()
    
    # codeparrot___codecomplex: src, problem, complexity
    if "codecomplex" in dataset_name_lower:
        if "src" in all_cols:
            code_col = "src"
        if "problem" in all_cols:
            prompt_col = "problem"
        if "complexity" in all_cols:
            label_col = "complexity"
    
    # deepmind___code_contests: description, solutions (nested!), difficulty
    elif "code_contests" in dataset_name_lower or "codecontests" in dataset_name_lower:
        if "description" in all_cols:
            prompt_col = "description"
        # Skip nested "solutions" - too complex
        # Use first solution if we need code later
        if "difficulty" in all_cols:
            label_col = "difficulty"
        elif "cf_rating" in all_cols:
            label_col = "cf_rating"
    
    # PrimeIntellect___deepcoder-gold-standard-solutions: prompt, gold_standard_solution, task_type
    elif "deepcoder" in dataset_name_lower:
        if "gold_standard_solution" in all_cols:
            code_col = "gold_standard_solution"
        if "prompt" in all_cols:
            prompt_col = "prompt"
        if "task_type" in all_cols:
            label_col = "task_type"
    
    # Generic fallback
    if not code_col or not prompt_col:
        for col in all_cols:
            col_lower = col.lower()
            if not code_col and (col_lower in ["code", "solution", "gold_standard_solution", "canonical_solution", "source_code"] or "solution" in col_lower or ("code" in col_lower and "complexity" not in col_lower)):
                # Skip nested sequences
                sample_val = sample.get(col)
                if not isinstance(sample_val, (list, dict)) or (isinstance(sample_val, list) and len(sample_val) == 0):
                    code_col = col
            if not prompt_col and (col_lower in ["prompt", "problem", "description", "task", "question"] or "prompt" in col_lower or "problem" in col_lower):
                sample_val = sample.get(col)
                if not isinstance(sample_val, (list, dict)):
                    prompt_col = col
            if not label_col and col_lower in ["complexity", "difficulty", "score", "rating", "label", "y", "target", "task_type"]:
                sample_val = sample.get(col)
                if not isinstance(sample_val, (list, dict)):
                    label_col = col
    
    # Select only needed columns from dataset (Arrow handles this efficiently)
    needed_cols = [c for c in [code_col, prompt_col, label_col] if c is not None and c in all_cols]
    
    print(f"[info] Detected columns - code: {code_col}, prompt: {prompt_col}, label: {label_col}")
    print(f"[info] Extracting {len(needed_cols)} columns from {len(all_cols)} total columns")
    
    if needed_cols:
        # Select only needed columns - Arrow handles this efficiently
        try:
            dataset_subset = dataset.select_columns(needed_cols)
        except Exception:
            # Fallback: select columns manually by creating new dataset
            dataset_subset = dataset.remove_columns([c for c in all_cols if c not in needed_cols])
    else:
        # Fallback: use full dataset
        dataset_subset = dataset
    
    # Now convert only the subset (much faster!)
    try:
        df = dataset_subset.to_pandas()
        convert_time = time.time() - extract_start
        print(f"[info] Converted {len(needed_cols)} columns to pandas in {convert_time:.2f}s")
    except Exception as e:
        # Fallback: convert full dataset but keep only needed columns
        print(f"[warn] Column selection failed: {e}, using full conversion")
        df = dataset.to_pandas()
        # Keep only needed columns
        if needed_cols:
            df = df[[c for c in needed_cols if c in df.columns]].copy()
        convert_time = time.time() - extract_start
        print(f"[info] Converted full dataset, kept {len(needed_cols)} columns in {convert_time:.2f}s")
    
    # Add ID column if missing
    if "__id__" not in df.columns:
        df["__id__"] = range(len(df))
    
    # Extract dataset name
    dataset_name = os.path.basename(converted_path.rstrip("/\\"))
    df["dataset_name"] = dataset_name
    
    # Rename label column to "label" for easier detection (if we found one)
    if label_col and label_col in df.columns:
        if label_col != "label":
            df["label"] = df[label_col]
            df = df.drop(columns=[label_col], errors="ignore")
        else:
            df["label"] = df[label_col]
    
    extract_time = time.time() - extract_start
    print(f"[info] Feature extraction complete in {extract_time:.2f}s ({len(df)} rows)")
    
    return df

def _extract_code_features_from_hf(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and normalize features from code datasets for GNN."""
    # Check if label column already exists (from optimized loading)
    if "label" in df.columns:
        # Label already extracted, just add derived features
        pass
    else:
        # Fallback: try to find label columns
        label_cols = ["complexity", "difficulty", "score", "rating", "label", "y", "target", "task_type"]
        for col in label_cols:
            if col in df.columns:
                df["label"] = df[col]
                break
    
    # Find code/prompt columns if not already extracted
    code_col = None
    prompt_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ["code", "solution", "gold_standard_solution", "canonical_solution", "source_code", "src"] or "solution" in col_lower or "code" in col_lower:
            if code_col is None:
                code_col = col
        if col_lower in ["prompt", "problem", "description", "task", "question"]:
            if prompt_col is None:
                prompt_col = col
    
    # Create normalized features
    if code_col:
        df["code_text"] = df[code_col].astype(str)
        df["code_length"] = df["code_text"].str.len()
    else:
        df["code_text"] = ""
        df["code_length"] = 0
    
    if prompt_col:
        df["prompt_text"] = df[prompt_col].astype(str)
        df["prompt_length"] = df["prompt_text"].str.len()
    else:
        df["prompt_text"] = ""
        df["prompt_length"] = 0
    
    if "label" in df.columns:
        if pd.api.types.is_numeric_dtype(df["label"]):
            df["label"] = df["label"].fillna(-1)
        else:
            df["label"] = df["label"].astype(str).fillna("__missing__")

    drop_cols = []
    for col in [code_col, prompt_col]:
        if col and col in df.columns and col not in ["code_text", "prompt_text"]:
            drop_cols.append(col)
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    keep_order = []
    for col in ["id", "__id__", "label", "code_text", "code_length", "prompt_text", "prompt_length", "dataset_name"]:
        if col in df.columns and col not in keep_order:
            keep_order.append(col)
    if keep_order:
        df = df.loc[:, keep_order]

    return df

# ----------------------- IO helpers -----------------------
def _load_one(path: str) -> pd.DataFrame:
    # Check if it's a HuggingFace Hub marker (load from Hub)
    if path.startswith("__HF_HUB__:"):
        repo_name = path.replace("__HF_HUB__:", "")
        print(f"[info] Loading {repo_name} from HuggingFace Hub...")
        if not HAVE_HF:
            raise ImportError("HuggingFace datasets library not installed. Install with: pip install datasets")
        try:
            dataset = load_dataset(repo_name, split="train", trust_remote_code=True)
        except Exception:
            try:
                ds_dict = load_dataset(repo_name, trust_remote_code=True)
                if "train" in ds_dict:
                    dataset = ds_dict["train"]
                elif len(ds_dict) > 0:
                    dataset = list(ds_dict.values())[0]
                else:
                    raise ValueError(f"Could not find train split in {repo_name}")
            except Exception as e:
                raise ValueError(f"Could not load dataset {repo_name} from HuggingFace Hub: {e}")
        
        # Convert to DataFrame - optimized
        print(f"[info] Converting {repo_name} to pandas DataFrame...")
        try:
            df = dataset.to_pandas()
        except Exception as e:
            print(f"[warn] Direct to_pandas failed: {e}, using batched conversion")
            # Batched conversion for large datasets
            batch_size = 10000
            batches = []
            total_rows = len(dataset)
            for i in range(0, total_rows, batch_size):
                end_idx = min(i + batch_size, total_rows)
                batch_data = [dataset[j] for j in range(i, end_idx)]
                batches.append(pd.DataFrame(batch_data))
                if (i // batch_size) % 10 == 0:
                    print(f"[info] Converted {end_idx}/{total_rows} rows...")
            df = pd.concat(batches, ignore_index=True)
        
        # Add node ID if missing
        if "id" not in df.columns and "__id__" not in df.columns:
            df["__id__"] = range(len(df))
        
        df["dataset_name"] = repo_name.replace("/", "_")
        df = _extract_code_features_from_hf(df)
        return df
    
    # Convert Windows path to WSL if needed
    converted_path = _convert_windows_path(path)
    # Check if it's a HuggingFace dataset cache directory
    if _is_hf_cache_dir(converted_path):
        df = _load_hf_dataset(converted_path)
        df = _extract_code_features_from_hf(df)
        return df
    
    # Standard file loading
    ext = os.path.splitext(converted_path)[1].lower()
    if ext in [".csv", ".tsv"]:
        return pd.read_csv(converted_path, sep="," if ext==".csv" else "\t")
    if ext in [".parquet", ".pq"]:
        return pd.read_parquet(converted_path)
    if ext in [".json", ".jsonl"]:
        return pd.read_json(converted_path, lines=True)
    raise ValueError(f"Unsupported file: {ext}")

def load_paths(path_arg: str) -> List[str]:
    """Load paths, supporting both file paths and HuggingFace cache directories."""
    valid_paths = []
    
    for tok in path_arg.split(","):
        tok = tok.strip()
        converted_tok = _convert_windows_path(tok)
        
        # Check if path exists
        if not os.path.exists(converted_tok):
            print(f"[warn] Path does not exist: {converted_tok}")
            continue
        
        # If it's a directory, check for HF dataset or regular files
        if os.path.isdir(converted_tok):
            # Check for .arrow files (HF dataset)
            arrow_files = list(glob.glob(os.path.join(converted_tok, "**/*.arrow"), recursive=True))
            if arrow_files:
                valid_paths.append(converted_tok)
                print(f"[info] Found HF dataset with {len(arrow_files)} arrow files: {converted_tok}")
            else:
                # Check for regular supported files
                for ext in SUPPORTED:
                    found_files = glob.glob(os.path.join(converted_tok, f"*{ext}"))
                    if found_files:
                        valid_paths.extend(found_files)
        # If it's a file, check if supported
        elif os.path.isfile(converted_tok) and os.path.splitext(converted_tok)[1].lower() in SUPPORTED:
            valid_paths.append(converted_tok)
    
    if not valid_paths:
        raise FileNotFoundError(f"No supported files or HuggingFace datasets found in: {path_arg}")
    
    return sorted(valid_paths)

# ----------------------- Schema inference -----------------------
def _first(mp, names):
    for n in names:
        if n in mp: return mp[n]
    return None

def guess_cols(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    mp = {c.lower(): c for c in df.columns}
    src = _first(mp, ["src","source","u","from","user","node_u","left","head"])
    dst = _first(mp, ["dst","target","v","to","item","node_v","right","tail"])
    nid = _first(mp, ["id","node_id","nid","index","_id"])
    y   = _first(mp, ["label","y","target","class","category"])
    weight = _first(mp, ["weight","w","edge_weight","rating","score"])
    etype  = _first(mp, ["type","edge_type","rel","relation"])
    uid = _first(mp, ["user_id","uid","u_id","customer_id","account_id"])
    iid = _first(mp, ["item_id","iid","i_id","product_id","movie_id"])
    gid = _first(mp, ["graph_id","gid","session_id","mol_id","molecule_id","group_id","component_id"])
    eyl  = _first(mp, ["edge_label","y_edge","edge_class","edge_target"])
    return {"src":src,"dst":dst,"nid":nid,"y":y,"weight":weight,"etype":etype,"uid":uid,"iid":iid,"gid":gid,"edge_label":eyl}

def is_edge_table(df: pd.DataFrame) -> bool:
    c = guess_cols(df)
    return bool(c["src"] and c["dst"]) or bool(c["uid"] and c["iid"])

def is_node_table(df: pd.DataFrame) -> bool:
    return bool(guess_cols(df)["nid"])

def split_node_edge_tables(dfs: List[pd.DataFrame]) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    nodes, edges = None, None
    for d in dfs:
        if is_edge_table(d) and edges is None:
            edges = d
        elif is_node_table(d) and nodes is None:
            nodes = d
    if nodes is None and edges is None:
        return dfs[0], None
    if nodes is None and edges is not None:
        return None, edges
    if nodes is not None and edges is None:
        return nodes, None
    return nodes, edges

def infer_label_candidates(df: pd.DataFrame, prefer: Optional[str]=None) -> List[Tuple[str,str]]:
    # Return [(column, task_type)], task_type in {"node_cls","node_reg","graph_cls","graph_reg"}
    cols = guess_cols(df)
    gid = cols["gid"]
    cands = []
    # Priority: explicit prefer
    if prefer and prefer in df.columns:
        t = "node_reg" if (pd.api.types.is_float_dtype(df[prefer]) or df[prefer].nunique()>100) else "node_cls"
        cands.append((prefer, t))
    # Named hints
    for nm in ["label","target","class","y","category"]:
        real = next((c for c in df.columns if c.lower()==nm), None)
        if real and real != prefer:
            t = "node_reg" if (pd.api.types.is_float_dtype(df[real]) or df[real].nunique()>100) else "node_cls"
            cands.append((real, t))
    # Other plausible columns
    for c in df.columns:
        if any(c==x for x,_ in cands): continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and s.nunique()>1:
            if s.nunique() <= 100 and pd.api.types.is_integer_dtype(s):
                cands.append((c,"node_cls"))
            else:
                cands.append((c,"node_reg"))
        elif s.dtype==object and 2 <= s.nunique() <= 1000:
            cands.append((c,"node_cls"))
    # Upgrade to graph-level if per-graph constant
    if gid and gid in df.columns:
        g = df.groupby(gid)
        out = []
        for col, t in cands:
            try:
                pg = g[col].nunique()
                if pg.max() == 1 and pg.min() == 1 and df[gid].nunique() > 1:
                    t = "graph_reg" if (pd.api.types.is_float_dtype(df[col]) or df[col].nunique()>100) else "graph_cls"
            except Exception:
                pass
            out.append((col, t))
        cands = out
    # unique preserve
    seen, uniq = set(), []
    for it in cands:
        if it[0] not in seen:
            uniq.append(it); seen.add(it[0])
    return uniq[:6]

# ----------------------- Feature encoders -----------------------
class HashEmbedder(nn.Module):
    def __init__(self, buckets=4096, dim=16):
        super().__init__()
        self.emb = nn.Embedding(buckets, dim)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        self.buckets = buckets
    @torch.no_grad()
    def encode(self, arr: np.ndarray) -> np.ndarray:
        idx = np.fromiter(((hash(str(v)) % self.buckets) for v in arr), count=len(arr), dtype=np.int64)
        return self.emb(torch.from_numpy(idx)).cpu().numpy().astype(np.float32)

def build_node_features(nodes: pd.DataFrame, drop_cols: List[str], add_cols: Optional[List[str]]=None) -> torch.Tensor:
    X_parts = []
    he = HashEmbedder(4096, 16)
    add_cols = add_cols or []
    for c in nodes.columns:
        if c in drop_cols and c not in add_cols: continue
        s = nodes[c]
        if pd.api.types.is_numeric_dtype(s):
            x = s.to_numpy(dtype=np.float32)
            mu, st = float(np.mean(x)), float(np.std(x) + 1e-8)
            x = (x-mu)/st
            X_parts.append(x.reshape(-1,1))
        else:
            X_parts.append(he.encode(s.astype(str).fillna("").to_numpy()))
    if not X_parts:
        return torch.zeros((len(nodes),1), dtype=torch.float32)
    X = np.concatenate(X_parts, axis=1)
    return torch.from_numpy(X.astype(np.float32))

# ----------------------- Graph builders -----------------------
def knn_edges(X: torch.Tensor, k: int=10, metric="euclidean", mutual=False) -> torch.Tensor:
    N = X.shape[0]
    print(f"[info] Computing kNN graph (k={k}) for {N} nodes...")
    
    if HAVE_SK:
        # Use parallel processing for large graphs
        n_jobs = -1 if N > 10000 else None
        nbrs = NearestNeighbors(n_neighbors=min(k+1, N), metric=metric, n_jobs=n_jobs, algorithm='auto').fit(X.numpy())
        print(f"[info] Finding neighbors...")
        _, idx = nbrs.kneighbors(X.numpy(), return_distance=True)
        print(f"[info] Building edge list...")
        src, dst = [], []
        for i in range(N):
            neigh = idx[i,1:]
            for j in neigh:
                src.append(int(j)); dst.append(i)
        edge = torch.tensor([src, dst], dtype=torch.long)
        print(f"[info] Initial edges: {edge.size(1)}")
    else:
        # Fallback: batch computation to avoid OOM
        if N > 10000:
            print(f"[warn] sklearn not available, using batched computation...")
            batch_size = 5000
            topk_indices = []
            Xn = F.normalize(X, p=2, dim=1)
            for i in range(0, N, batch_size):
                end = min(i + batch_size, N)
                batch = Xn[i:end]
                sims = batch @ Xn.T  # (batch_size, N)
                topk = torch.topk(sims, min(k+1, N), dim=1).indices
                topk_indices.append(topk)
                if (i // batch_size + 1) % 10 == 0:
                    print(f"[info] Processed {end}/{N} nodes...")
            topk_all = torch.cat(topk_indices, dim=0)
        else:
            Xn = F.normalize(X, p=2, dim=1)
            sims = Xn @ Xn.T
            topk_all = torch.topk(sims, min(k+1, N), dim=1).indices
        
        src, dst = [], []
        for i in range(N):
            for j in topk_all[i].tolist():
                if j==i: continue
                src.append(j); dst.append(i)
        edge = torch.tensor([src, dst], dtype=torch.long)
    
    if mutual:
        print(f"[info] Filtering to mutual edges...")
        E = set((int(s),int(d)) for s,d in zip(edge[0].tolist(), edge[1].tolist()))
        src2, dst2 = [], []
        for s,d in E:
            if (d,s) in E:
                src2.append(s); dst2.append(d)
        edge = torch.tensor([src2,dst2], dtype=torch.long)
        print(f"[info] Mutual edges: {edge.size(1)}")
    return edge

def degree_features(edge_index: torch.Tensor, N: int) -> torch.Tensor:
    deg = torch.zeros(N, dtype=torch.float32)
    deg.index_add_(0, edge_index[1], torch.ones(edge_index.size(1)))
    return deg.view(-1,1)

# ----------------------- Positional Encodings -----------------------
@torch.no_grad()
def laplacian_pe(edge_index: torch.Tensor, num_nodes: int, k: int = 16, norm: bool = True) -> Optional[torch.Tensor]:
    """Compute Laplacian eigenvectors (k) as node PE. Uses sparse solvers for large graphs."""
    if num_nodes <= 0:
        return None
    k = min(k, num_nodes)
    if k == 0:
        return None
    try:
        ei = to_undirected(edge_index, num_nodes=num_nodes)
        L_edge, L_weight = get_laplacian(ei, normalization='sym' if norm else None, num_nodes=num_nodes)
        # Small graph: dense eigendecomposition is ok
        if num_nodes <= 5000:
            L = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
            idx_i = L_edge[0].tolist()
            idx_j = L_edge[1].tolist()
            weights = L_weight.tolist()
            for i, j, w in zip(idx_i, idx_j, weights):
                L[i][j] += w
            L = (L + L.T) * 0.5
            evals, evecs = torch.linalg.eigh(L)
            return evecs[:, :k].float()

        # Large graph: prefer SciPy sparse eigensolver if available
        row = L_edge[0].cpu().numpy()
        col = L_edge[1].cpu().numpy()
        data = L_weight.cpu().numpy()
        last_eigsh_error = None
        if HAVE_SCIPY:
            try:
                L_sparse = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes)).tocsr()
                req_k = min(max(k + 5, 2), num_nodes - 1) if num_nodes > 1 else 1
                for tol in (1e-3, 5e-3, 1e-2):
                    for maxiter in (500, 1000, 2000):
                        try:
                            evals, evecs = spla.eigsh(L_sparse, k=req_k, which="SM", tol=tol, maxiter=maxiter)
                            order = np.argsort(evals)
                            evecs = evecs[:, order[:k]]
                            if evecs.size > 0:
                                return torch.from_numpy(evecs).float()
                        except Exception as inner_err:
                            last_eigsh_error = inner_err
                            continue
            except Exception as outer_err:
                last_eigsh_error = outer_err
            if last_eigsh_error is not None:
                print(f"[warn] SciPy eigsh failed: {last_eigsh_error}")

        last_lobpcg_error = None
        if HAVE_TORCH_LOBPCG:
            L_sparse = torch.sparse_coo_tensor(L_edge, L_weight, (num_nodes, num_nodes)).coalesce()
            init = torch.randn(num_nodes, k, dtype=L_sparse.dtype)
            for tol in (1e-3, 5e-3, 1e-2):
                for maxiter in (200, 400, 800):
                    try:
                        evals, evecs = sparse_lobpcg(L_sparse, k=k, largest=False, tol=tol, maxiter=maxiter, X=init)
                        if evecs is not None:
                            return evecs.float()
                    except Exception as lobpcg_err:
                        last_lobpcg_error = lobpcg_err
                        continue
            if last_lobpcg_error is not None:
                print(f"[warn] torch.lobpcg failed: {last_lobpcg_error}")

        if num_nodes <= 20000:
            try:
                L = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
                idx_i = L_edge[0].tolist()
                idx_j = L_edge[1].tolist()
                weights = L_weight.tolist()
                for i, j, w in zip(idx_i, idx_j, weights):
                    L[i][j] += w
                L = (L + L.T) * 0.5
                evals, evecs = torch.linalg.eigh(L)
                return evecs[:, :k].float()
            except Exception as dense_err:
                print(f"[warn] Dense Laplacian eigendecomposition failed: {dense_err}")
        print("[warn] Laplacian PE failed: falling back to None")
        return None
    except Exception:
        return None

@torch.no_grad()
def rwse_return_probs(edge_index: torch.Tensor, num_nodes: int, k: int = 8) -> Optional[torch.Tensor]:
    """Compute RWSE as k-step return probabilities (diag P^k). Uses sparse operations for large graphs."""
    try:
        # Build row-normalized transition matrix P (sparse)
        ei = to_undirected(edge_index, num_nodes=num_nodes)
        row = ei[0]; col = ei[1]
        deg = torch.bincount(row, minlength=num_nodes).clamp(min=1).float()
        val = torch.ones(ei.size(1), dtype=torch.float32) / deg[row]
        P = torch.sparse_coo_tensor(ei, val, size=(num_nodes, num_nodes)).coalesce()
        
        # For large graphs, use sparse-only computation (no dense conversion)
        if num_nodes > 10000:
            # Approximate RWSE using sparse power iteration on diagonal only
            # Compute diag(P^k) using sparse operations without converting to dense
            diag_list = []
            # Start with identity: diag(P^0) = ones
            diag_list.append(torch.ones(num_nodes, dtype=torch.float32).unsqueeze(1))
            
            # For very large graphs, use sampling-based approximation
            if num_nodes > 50000:
                print(f"[info] Using sampling-based RWSE approximation (graph size: {num_nodes})...")
                sample_size = min(10000, num_nodes // 5)
                sampled_nodes = torch.randperm(num_nodes)[:sample_size]
                
                for step in range(1, k+1):
                    diag_approx = torch.zeros(num_nodes, dtype=torch.float32)
                    for idx in sampled_nodes:
                        # Compute (P^step)_{idx,idx} via sparse power iteration
                        x = torch.zeros(num_nodes, dtype=torch.float32)
                        x[idx] = 1.0
                        for _ in range(step):
                            x = torch.sparse.mm(P, x.unsqueeze(1)).squeeze(1)
                        diag_approx[idx] = x[idx]
                    # Interpolate using degree similarity
                    deg_normalized = deg / (deg.max() + 1e-8)
                    diag_approx_full = diag_approx.mean() * deg_normalized
                    # Refine using sampled values where available
                    diag_approx_full[sampled_nodes] = diag_approx[sampled_nodes]
                    diag_list.append(diag_approx_full.unsqueeze(1))
            else:
                # For moderate graphs (10k-50k), compute exact diagonal via sparse operations
                for step in range(1, k+1):
                    diag_curr = torch.zeros(num_nodes, dtype=torch.float32)
                    print(f"[info] Computing RWSE step {step}/{k}...")
                    # Batch computation for efficiency
                    batch_size = 1000
                    for i in range(0, num_nodes, batch_size):
                        end = min(i + batch_size, num_nodes)
                        for idx in range(i, end):
                            x = torch.zeros(num_nodes, dtype=torch.float32)
                            x[idx] = 1.0
                            for _ in range(step):
                                x = torch.sparse.mm(P, x.unsqueeze(1)).squeeze(1)
                            diag_curr[idx] = x[idx]
                        if (i // batch_size + 1) % 10 == 0:
                            print(f"[info] Processed {end}/{num_nodes} nodes...")
                    diag_list.append(diag_curr.unsqueeze(1))
        else:
            # Small graph: use original dense method (exact)
            Pk = P
            diag_list = []
            diag_list.append(torch.ones(num_nodes, dtype=torch.float32).unsqueeze(1))  # P^0
            for step in range(1, k+1):
                Pk_dense = Pk.to_dense()
                diag_list.append(torch.diag(Pk_dense).unsqueeze(1))
                if step < k:
                    Pk = torch.sparse.mm(P, Pk)  # next power
        
        RW = torch.cat(diag_list, dim=1)  # [N, k+1] - includes P^0 through P^k
        # Take only P^1 through P^k for consistency (skip P^0)
        RW = RW[:, 1:] if RW.size(1) > k else RW
        # Standardize per-dim
        mu, st = RW.mean(0, keepdim=True), RW.std(0, keepdim=True) + 1e-8
        return ((RW - mu) / st).float()
    except Exception as e:
        print(f"[warn] RWSE computation failed: {e}")
        return None

def build_edge_pos_bias_from_node_pe(node_pe: Optional[torch.Tensor], edge_index: torch.Tensor, method: str = "diff") -> Optional[torch.Tensor]:
    """Construct per-edge positional features from node PE to be fed as attention bias."""
    if node_pe is None: return None
    src, dst = edge_index
    if method == "diff":
        pe = node_pe[dst] - node_pe[src]  # [E, d]
    else:
        pe = torch.cat([node_pe[dst], node_pe[src]], dim=1)
    return pe

# ----------------------- Homo Models -----------------------
class GATEncoder(nn.Module):
    def __init__(self, in_dim, hid=256, out_dim=256, n_layers=3, heads=4, drop=0.2):
        super().__init__()
        dims = [in_dim] + [hid]*(n_layers-1) + [out_dim]
        self.layers = nn.ModuleList([
            GATv2Conv(dims[i], dims[i+1]//heads, heads=heads, concat=False, dropout=drop)
            for i in range(n_layers)
        ])
        self.lns = nn.ModuleList([nn.LayerNorm(dims[i+1]) for i in range(n_layers)])
        self.drop = nn.Dropout(drop)
    def forward(self, x, edge_index):
        for conv, ln in zip(self.layers, self.lns):
            h = conv(x, edge_index)
            x = ln(x + self.drop(F.relu(h)))
        return x

class GraphTransformerLayer(nn.Module):
    def __init__(self, d_in, d_out, heads=8, edge_dim: Optional[int]=None, pos_dim: Optional[int]=None, drop=0.1):
        super().__init__()
        assert d_out % heads == 0
        self.h = heads
        self.dk = d_out // heads
        self.q = nn.Linear(d_in,  d_out, bias=False)
        self.k = nn.Linear(d_in,  d_out, bias=False)
        self.v = nn.Linear(d_in,  d_out, bias=False)
        self.o = nn.Linear(d_out, d_out, bias=False)
        self.edge_mlp = nn.Linear(edge_dim, heads) if edge_dim else None
        self.pos_mlp  = nn.Linear(pos_dim, heads)  if pos_dim  else None
        self.ln = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(drop)
        self.res_proj = nn.Linear(d_in, d_out, bias=False) if d_in != d_out else None
    def forward(self, x, edge_index, edge_attr=None, pos_ij=None):
        N = x.size(0)
        q = self.q(x).view(N, self.h, self.dk)
        k = self.k(x).view(N, self.h, self.dk)
        v = self.v(x).view(N, self.h, self.dk)
        src, dst = edge_index
        qi, kj, vj = q[dst], k[src], v[src]
        att = (qi * kj).sum(-1) / math.sqrt(self.dk)
        if self.edge_mlp is not None and edge_attr is not None:
            att = att + self.edge_mlp(edge_attr).to(att.dtype)
        if self.pos_mlp is not None and pos_ij is not None:
            att = att + self.pos_mlp(pos_ij).to(att.dtype)
        att = att.to(q.dtype)
        att = softmax(att, dst)
        att = self.drop(att)
        msg = vj * att.unsqueeze(-1)
        if msg.dtype != q.dtype:
            msg = msg.to(q.dtype)
        out = torch.zeros_like(q)
        out.index_add_(0, dst, msg)
        out = out.reshape(N, -1)
        proj = self.o(out)
        if proj.dtype != q.dtype:
            proj = proj.to(q.dtype)
        res = x if self.res_proj is None else self.res_proj(x)
        if res.dtype != proj.dtype:
            res = res.to(proj.dtype)
        out = self.ln(res + self.drop(F.relu(proj)))
        return out

class GraphTransformer(nn.Module):
    def __init__(self, in_dim, hid=256, out_dim=256, n_layers=4, heads=8, drop=0.1, edge_dim: Optional[int]=None, pos_dim: Optional[int]=None):
        super().__init__()
        dims = [in_dim] + [hid]*(n_layers-1) + [out_dim]
        self.layers = nn.ModuleList([
            GraphTransformerLayer(dims[i], dims[i+1], heads=heads, edge_dim=edge_dim, pos_dim=pos_dim, drop=drop)
            for i in range(n_layers)
        ])
        self.lns = nn.ModuleList([nn.LayerNorm(dims[i+1]) for i in range(n_layers)])
    def forward(self, x, edge_index, edge_attr=None, pos_ij=None):
        for layer, ln in zip(self.layers, self.lns):
            x = ln(layer(x, edge_index, edge_attr, pos_ij))
        return x

# ----------------------- Hetero Model -----------------------
class HeteroGATEncoder(nn.Module):
    """HeteroConv with GATv2 per relation; projects per-node-type dims to a shared out_dim."""
    def __init__(self, metadata, in_dims: Dict[str,int], hid=128, out_dim=256, heads=4, layers=2, drop=0.2):
        super().__init__()
        self.metadata = metadata
        self.proj_in = nn.ModuleDict({nt: nn.Linear(in_dims[nt], hid) for nt in metadata[0]})
        convs = []
        for _ in range(layers):
            convs.append(HeteroConv({
                et: GATv2Conv(hid, hid//heads, heads=heads, concat=False, dropout=drop)
                for et in metadata[1]
            }, aggr='mean'))
        self.convs = nn.ModuleList(convs)
        self.proj_out = nn.ModuleDict({nt: nn.Linear(hid, out_dim) for nt in metadata[0]})
        self.norms = nn.ModuleDict({nt: nn.LayerNorm(hid) for nt in metadata[0]})
        self.drop = nn.Dropout(drop)

    def forward(self, x_dict, edge_index_dict):
        h = {nt: F.relu(self.proj_in[nt](x)) for nt,x in x_dict.items()}
        for conv in self.convs:
            h_new = conv(h, edge_index_dict)
            for nt in h_new:
                h[nt] = self.norms[nt](h[nt] + self.drop(F.relu(h_new[nt])))
        h_out = {nt: self.proj_out[nt](h[nt]) for nt in h}
        return h_out

# ----------------------- Heads -----------------------
class NodeHead(nn.Module):
    def __init__(self, d, task_type, num_classes=None):
        super().__init__()
        self.task_type = task_type
        if task_type=="node_cls":
            assert num_classes is not None
            self.out = nn.Linear(d, num_classes)
        else:
            self.out = nn.Linear(d, 1)
    def forward(self, h): return self.out(h)

class GraphHead(nn.Module):
    def __init__(self, d, task_type, num_classes=None):
        super().__init__()
        self.task_type = task_type
        if task_type=="graph_cls":
            assert num_classes is not None
            self.out = nn.Linear(d, num_classes)
        else:
            self.out = nn.Linear(d, 1)
    def forward(self, hg): return self.out(hg)

class EdgeHead(nn.Module):
    """Bilinear or MLP edge scorer for edge cls/reg, homogeneous setting."""
    def __init__(self, d, task_type, num_classes=None, mode="bilinear"):
        super().__init__()
        self.task_type = task_type
        self.mode = mode
        outdim = num_classes if task_type=="edge_cls" else 1
        if mode=="bilinear":
            self.scorer = nn.Bilinear(d, d, outdim)
        else:
            self.scorer = nn.Sequential(nn.Linear(2*d, 2*d), nn.ReLU(), nn.Linear(2*d, outdim))
    def forward(self, hn, edge_index):
        src, dst = edge_index
        if self.mode=="bilinear":
            return self.scorer(hn[src], hn[dst])
        x = torch.cat([hn[src], hn[dst]], dim=-1)
        return self.scorer(x)

# ----------------------- Utils -----------------------
def make_masks(y: torch.Tensor, task_type: str, gid: Optional[torch.Tensor]=None, seed=0):
    rng = np.random.RandomState(seed)
    if task_type.startswith("graph_"):
        assert gid is not None, "graph task requires graph_id"
        gids = gid.unique().tolist()
        g2idx = {int(g):i for i,g in enumerate(gids)}
        grp = np.array([g2idx[int(v)] for v in gid.numpy()])
        idx = np.arange(len(gids)); rng.shuffle(idx)
        n = len(idx)
        trg, vag, teg = idx[:int(0.7*n)], idx[int(0.7*n):int(0.8*n)], idx[int(0.8*n):]
        tr = np.isin(grp, trg); va = np.isin(grp, vag); te = np.isin(grp, teg)
        return torch.from_numpy(tr), torch.from_numpy(va), torch.from_numpy(te)
    else:
        N = len(y)
        idx = np.arange(N); rng.shuffle(idx)
        if task_type.endswith("_cls"):
            try:
                import sklearn.model_selection as skms
                tr, te = skms.train_test_split(idx, test_size=0.2, stratify=y.numpy())
                tr, va = skms.train_test_split(tr, test_size=0.125, stratify=y[tr])
            except Exception:
                tr, va, te = idx[:int(0.7*N)], idx[int(0.7*N):int(0.8*N)], idx[int(0.8*N):]
        else:
            tr, va, te = idx[:int(0.7*N)], idx[int(0.7*N):int(0.8*N)], idx[int(0.8*N):]
        mtr = torch.zeros(N, dtype=torch.bool); mtr[tr]=True
        mva = torch.zeros(N, dtype=torch.bool); mva[va]=True
        mte = torch.zeros(N, dtype=torch.bool); mte[te]=True
        return mtr, mva, mte

def build_edgeattr_tensor(edge_attr: Optional[Dict], heads: int) -> Optional[torch.Tensor]:
    if edge_attr is None: return None
    parts = []
    if edge_attr.get("weight") is not None:
        w = edge_attr["weight"].astype(np.float32)
        mu, st = float(np.mean(w)), float(np.std(w) + 1e-8)
        w = (w - mu)/st
        parts.append(torch.from_numpy(w))
    if edge_attr.get("type") is not None:
        t = edge_attr["type"].astype(np.int64)
        num_types = int(t.max()+1) if t.size>0 and t.min()>=0 else 0
        if num_types>0 and num_types<=256:
            oh = torch.zeros(t.shape[0], num_types, dtype=torch.float32)
            oh[torch.arange(t.shape[0]), torch.from_numpy(t)] = 1.0
            parts.append(oh)
        else:
            parts.append(torch.from_numpy(t.astype(np.float32)).reshape(-1,1))
    if not parts: return None
    return torch.cat(parts, dim=1)

# ----------------------- Assembly (Homo + Hetero + Edge) -----------------------
def assemble(paths: List[str], label_override: Optional[str], edge_label_override: Optional[str], default_k: int, mutual: bool) -> Dict[str, Any]:
    dfs = [_load_one(p) for p in paths]
    
    # If multiple datasets loaded and all are node tables (no edges), combine them
    if len(dfs) > 1:
        all_node_tables = all(not is_edge_table(df) for df in dfs)
        if all_node_tables:
            # Combine multiple HuggingFace datasets
            num_datasets = len(dfs)
            combined_df = pd.concat(dfs, ignore_index=True)
            # Ensure unique IDs across combined datasets
            if "__id__" in combined_df.columns:
                combined_df["__id__"] = range(len(combined_df))
            elif "id" in combined_df.columns:
                combined_df["id"] = range(len(combined_df))
            dfs = [combined_df]
            print(f"[info] Combined {num_datasets} datasets into one with {len(combined_df)} total nodes")
    
    nodes, edges = split_node_edge_tables(dfs)

    edge_task = None
    edge_label_col = None
    hetero = False
    bipartite = False
    raw_edge_attr = None
    node_type_vec = None
    gid_nodes = None
    edge_index = None
    nid = None

    if edges is not None:
        ec = guess_cols(edges)
        uid, iid, src, dst = ec["uid"], ec["iid"], ec["src"], ec["dst"]
        weight_col, type_col = ec["weight"], ec["etype"]
        gid_edges = ec["gid"]
        eyl = edge_label_override if edge_label_override else ec["edge_label"]

        if uid and iid:
            # bipartite hetero
            bipartite = True
            hetero = True
            u_vals = edges[uid].dropna().unique(); i_vals = edges[iid].dropna().unique()
            u_map = {v:i for i,v in enumerate(u_vals)}
            i_map = {v:i for i,v in enumerate(i_vals)}
            if gid_edges and gid_edges in edges.columns:
                gid_users = np.zeros(len(u_vals), dtype=np.int64)
                gid_items = np.zeros(len(i_vals), dtype=np.int64)
            else:
                gid_users = np.zeros(len(u_vals), dtype=np.int64)
                gid_items = np.zeros(len(i_vals), dtype=np.int64)

            edge_label_col = eyl if (eyl and eyl in edges.columns) else None
            if edge_label_col:
                col = edges[edge_label_col]
                edge_task = "edge_reg" if (pd.api.types.is_float_dtype(col) or col.nunique()>100) else "edge_cls"

            EA = {}
            if weight_col and weight_col in edges.columns:
                EA["weight"] = edges.loc[:, weight_col].astype(float).to_numpy().reshape(-1,1)
            if type_col and type_col in edges.columns:
                cat = pd.Categorical(edges.loc[:, type_col].astype(str))
                EA["type"] = cat.codes.to_numpy()
            raw_edge_attr = EA if EA else None

            return {
                "mode": "hetero",
                "hetero_schema": {
                    "user": {"nodes": pd.DataFrame({"id": np.arange(len(u_vals))}), "nid": "id"},
                    "item": {"nodes": pd.DataFrame({"id": np.arange(len(i_vals))}), "nid": "id"},
                    "edges": edges, "maps": {"user": u_map, "item": i_map},
                    "relations": [("user", "interacts", "item")],
                    "edge_label_col": edge_label_col,
                    "edge_task": edge_task,
                    "gid": {"user": torch.from_numpy(gid_users), "item": torch.from_numpy(gid_items)}
                },
                "default_k": default_k,
                "mutual": mutual,
                "arch_hint": "hetero",
            }

        if type_col and type_col in edges.columns:
            # hetero relations (one node type "entity", multi-relations)
            hetero = True
            if nodes is None:
                uniq = pd.unique(pd.concat([edges[src], edges[dst]], ignore_index=True))
                nid = "__id__"
                nodes = pd.DataFrame({nid: uniq})
                id_map = {v:i for i,v in enumerate(uniq.tolist())}
            else:
                nc = guess_cols(nodes)
                nid = nc["nid"] or "id"
                if nid not in nodes.columns:
                    nodes = nodes.reset_index().rename(columns={"index": nid})
                nodes = nodes.drop_duplicates(subset=[nid]).reset_index(drop=True)
                nodes = nodes.sort_values(by=nid).reset_index(drop=True)
                id_map = {v:i for i,v in enumerate(nodes[nid].tolist())}
            cat = pd.Categorical(edges[type_col].astype(str))
            rels = cat.categories.tolist()
            edge_index_dict = {}
            for ridx, rel_name in enumerate(rels):
                mask = (cat.codes.values == ridx)
                s = edges.loc[mask, src].map(id_map)
                d = edges.loc[mask, dst].map(id_map)
                m = s.notna() & d.notna()
                ei = torch.stack([torch.from_numpy(s[m].astype(int).to_numpy()),
                                  torch.from_numpy(d[m].astype(int).to_numpy())], dim=0)
                edge_index_dict[("entity", rel_name, "entity")] = ei

            edge_label_col = edge_label_override if (edge_label_override and edge_label_override in edges.columns) else ec["edge_label"]
            if edge_label_col and edge_label_col in edges.columns:
                col = edges[edge_label_col]
                edge_task = "edge_reg" if (pd.api.types.is_float_dtype(col) or col.nunique()>100) else "edge_cls"

            return {
                "mode": "hetero_rel",
                "nodes": nodes, "nid": nid,
                "edge_index_dict": edge_index_dict,
                "edge_task": edge_task,
                "edge_label_col": edge_label_col,
                "edges_df": edges,
                "default_k": default_k,
                "mutual": mutual,
                "arch_hint": "hetero"
            }

    # homogeneous
    raw_edge_attr = None
    edge_label_col = edge_label_override
    if edges is not None:
        ec = guess_cols(edges)
        src, dst = ec["src"], ec["dst"]
        weight_col, type_col = ec["weight"], ec["etype"]
        if edge_label_col is None:
            edge_label_col = ec["edge_label"]
        if edge_label_col and edge_label_col in edges.columns:
            col = edges[edge_label_col]
            edge_task = "edge_reg" if (pd.api.types.is_float_dtype(col) or col.nunique()>100) else "edge_cls"

        if nodes is None:
            uniq = pd.unique(pd.concat([edges[src], edges[dst]], ignore_index=True))
            nid = "__id__"
            nodes = pd.DataFrame({nid: uniq})
            id_map = {v:i for i,v in enumerate(uniq.tolist())}
        else:
            nc = guess_cols(nodes)
            nid = nc["nid"] or "id"
            if nid not in nodes.columns:
                nodes = nodes.reset_index().rename(columns={"index": nid})
            nodes = nodes.drop_duplicates(subset=[nid]).reset_index(drop=True)
            nodes = nodes.sort_values(by=nid).reset_index(drop=True)
            id_map = {v:i for i,v in enumerate(nodes[nid].tolist())}
        s = edges[src].map(id_map); d = edges[dst].map(id_map)
        m = s.notna() & d.notna()
        edge_index = torch.stack([torch.from_numpy(s[m].astype(int).to_numpy()),
                                  torch.from_numpy(d[m].astype(int).to_numpy())], dim=0)
        EA = {}
        if weight_col and weight_col in edges.columns:
            EA["weight"] = edges.loc[m, weight_col].astype(float).to_numpy().reshape(-1,1)
        if type_col and type_col in edges.columns:
            cat = pd.Categorical(edges.loc[m, type_col].astype(str))
            EA["type"] = cat.codes.to_numpy()
        raw_edge_attr = EA if EA else None
    else:
        if nodes is None:
            nodes = dfs[0].copy()
            nodes["__row_id__"] = np.arange(len(nodes))
            nid = "__row_id__"
        else:
            nc = guess_cols(nodes)
            nid = nc["nid"] or "id"
            if nid not in nodes.columns:
                nodes = nodes.reset_index().rename(columns={"index": nid})

    label_cands = infer_label_candidates(nodes, prefer=label_override)

    return {
        "mode": "homo",
        "nodes": nodes,
        "nid": nid,
        "edge_index": edge_index,
        "raw_edge_attr": raw_edge_attr,
        "label_cands": label_cands,
        "edge_task": edge_task,
        "edge_label_col": edge_label_col,
        "default_k": default_k,
        "mutual": mutual,
        "arch_hint": "auto"
    }

# ----------------------- Eval / Train (Homogeneous, with PE & NeighborLoader) -----------------------
def eval_score_homo(enc, head, data, device, arch, task_type, pos_ij=None):
    enc.eval(); head.eval()
    if arch=="gatv2":
        h = enc(data.x.to(device), data.edge_index.to(device))
    else:
        h = enc(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device) if data.edge_attr is not None else None, pos_ij.to(device) if pos_ij is not None else None)
    if task_type.startswith("graph_"):
        hg = global_mean_pool(h, data.batch.to(device))
        out = head(hg)
        mask = data.val_mask.to(device)
        if task_type=="graph_cls":
            yb = data.y_graph[mask].to(device)
            return (out[mask].argmax(-1)==yb).float().mean().item()
        else:
            yb = data.y_graph[mask].float().to(device)
            return -torch.mean(torch.abs(out.squeeze(-1)[mask] - yb)).item()
    elif task_type.startswith("edge_"):
        mask = data.edge_val_mask.to(device)
        logits = head(h, data.edge_index.to(device))
        if task_type=="edge_cls":
            yb = data.edge_label[mask].to(device)
            return (logits[mask].argmax(-1)==yb).float().mean().item()
        else:
            yb = data.edge_label[mask].float().to(device)
            return -torch.mean(torch.abs(logits.squeeze(-1)[mask] - yb)).item()
    else:
        out = head(h)
        if task_type=="node_cls":
            logits = out[data.val_mask.to(device)]
            yb = data.y[data.val_mask].to(device)
            return (logits.argmax(-1)==yb).float().mean().item()
        else:
            pred = out.squeeze(-1)[data.val_mask.to(device)]
            yb = data.y[data.val_mask].float().to(device)
            return -torch.mean(torch.abs(pred - yb)).item()

def run_train_homo(data: Data, arch: str, hyper: dict, device, steps=300, lr=3e-4, wd=1e-4, drop=0.2, heads=4, layers=3, edge_dim=None, pos_dim=None, optim_name="adamw", sched_name="none", use_neighbor=False, fanout=(15,10,5), node_pe=None, build_pos_from="diff"):
    in_dim = data.x.size(1)
    if arch=="gatv2":
        enc = GATEncoder(in_dim, hid=hyper["hid"], out_dim=hyper["out"], n_layers=layers, heads=heads, drop=drop).to(device)
    else:
        enc = GraphTransformer(in_dim, hid=hyper["hid"], out_dim=hyper["out"], n_layers=layers, heads=heads, drop=drop, edge_dim=edge_dim, pos_dim=pos_dim).to(device)

    task = hyper["task"]
    if task.startswith("graph_"):
        head = GraphHead(d=hyper["out"], task_type=task, num_classes=hyper.get("num_classes")).to(device)
    elif task.startswith("edge_"):
        head = EdgeHead(d=hyper["out"], task_type=task, num_classes=hyper.get("num_classes"), mode="bilinear").to(device)
    else:
        head = NodeHead(d=hyper["out"], task_type=task, num_classes=hyper.get("num_classes")).to(device)

    params = list(enc.parameters()) + list(head.parameters())
    if optim_name=="adam":
        opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    else:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if sched_name=="cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=max(steps//4, 10), T_mult=1)
    else:
        sched = None

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))
    best = -1e9; best_state = None

    # Build per-edge positional bias once (full graph) if provided
    pos_ij_full = None
    if arch=="gtr" and node_pe is not None:
        pos_ij_full = build_edge_pos_bias_from_node_pe(node_pe, data.edge_index, method=build_pos_from)

    # Optional neighbor loader (node/edge/graph tasks: we do node mini-batching; edge tasks keep full for simplicity)
    loader = None
    loader_iter = None
    if use_neighbor and not task.startswith("edge_"):
        try:
            loader = NeighborLoader(data, num_neighbors=list(fanout), batch_size=1024, input_nodes=None, shuffle=True)
            loader_iter = iter(loader)
        except (ImportError, RuntimeError) as exc:
            print(f"[warn] NeighborLoader unavailable ({exc}); falling back to full-batch training.")
            loader = None
            loader_iter = None
            use_neighbor = False

    for step in range(steps):
        enc.train(); head.train()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=(device.type in ["cuda","cpu"])):
            neighbor_mode = use_neighbor and not task.startswith("edge_")
            batch = None
            if neighbor_mode:
                try:
                    try:
                        batch = next(loader_iter)
                    except StopIteration:
                        loader_iter = iter(loader)
                        batch = next(loader_iter)
                except ImportError as exc:
                    print(f"[warn] NeighborLoader sampling unavailable ({exc}); reverting to full-batch training.")
                    neighbor_mode = False
                    use_neighbor = False
                    loader = None
                    loader_iter = None
            if neighbor_mode and batch is not None:
                if arch=="gatv2":
                    h = enc(batch.x.to(device), batch.edge_index.to(device))
                else:
                    # rebuild pos_ij for subgraph if available
                    pos_ij_mb = None
                    if pos_ij_full is not None:
                        # subgraph mapping is complex; simple fallback: recompute from node_pe for subgraph edges
                        if node_pe is not None:
                            pos_ij_mb = build_edge_pos_bias_from_node_pe(node_pe[batch.n_id], batch.edge_index, method=build_pos_from)
                    h = enc(batch.x.to(device), batch.edge_index.to(device), None, pos_ij_mb.to(device) if pos_ij_mb is not None else None)
                if task.startswith("graph_"):
                    hg = global_mean_pool(h, batch.batch.to(device))
                    out = head(hg); mask = batch.train_mask.to(device)
                    if task=="graph_cls":
                        loss = F.cross_entropy(out[mask], batch.y_graph[mask].to(device))
                    else:
                        loss = F.smooth_l1_loss(out.squeeze(-1)[mask], batch.y_graph[mask].float().to(device))
                else:
                    out = head(h)
                    if task=="node_cls":
                        loss = F.cross_entropy(out[batch.train_mask.to(device)], batch.y[batch.train_mask].to(device))
                    else:
                        loss = F.smooth_l1_loss(out.squeeze(-1)[batch.train_mask.to(device)], batch.y[batch.train_mask].float().to(device))
            if not neighbor_mode:
                if arch=="gatv2":
                    h = enc(data.x.to(device), data.edge_index.to(device))
                else:
                    h = enc(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device) if data.edge_attr is not None else None,
                            pos_ij_full.to(device) if pos_ij_full is not None else None)
                if task.startswith("graph_"):
                    hg = global_mean_pool(h, data.batch.to(device))
                    out = head(hg); mask = data.train_mask.to(device)
                    if task=="graph_cls":
                        loss = F.cross_entropy(out[mask], data.y_graph[mask].to(device))
                    else:
                        loss = F.smooth_l1_loss(out.squeeze(-1)[mask], data.y_graph[mask].float().to(device))
                elif task.startswith("edge_"):
                    logits = head(h, data.edge_index.to(device)); mask = data.edge_train_mask.to(device)
                    if task=="edge_cls":
                        loss = F.cross_entropy(logits[mask], data.edge_label[mask].to(device))
                    else:
                        loss = F.smooth_l1_loss(logits.squeeze(-1)[mask], data.edge_label[mask].float().to(device))
                else:
                    out = head(h)
                    if task=="node_cls":
                        loss = F.cross_entropy(out[data.train_mask.to(device)], data.y[data.train_mask].to(device))
                    else:
                        loss = F.smooth_l1_loss(out.squeeze(-1)[data.train_mask.to(device)], data.y[data.train_mask].float().to(device))

        if device.type=="cuda":
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        if sched: sched.step(step)

        if (step+1) % max(1, hyper["eval_every"]) == 0:
            score = eval_score_homo(enc, head, data, device, arch, task, pos_ij_full)
            if score > best:
                best = score
                best_state = {"enc": {k:v.detach().cpu().clone() for k,v in enc.state_dict().items()},
                              "head": {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}}

    if best_state:
        enc.load_state_dict(best_state["enc"])
        head.load_state_dict(best_state["head"])
    test_metric = test_score_homo(enc, head, data, device, arch, task, pos_ij_full)
    return best, test_metric, enc, head

@torch.no_grad()
def test_score_homo(enc, head, data, device, arch, task_type, pos_ij=None):
    enc.eval(); head.eval()
    if arch=="gatv2":
        h = enc(data.x.to(device), data.edge_index.to(device))
    else:
        h = enc(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device) if data.edge_attr is not None else None, pos_ij.to(device) if pos_ij is not None else None)
    if task_type.startswith("graph_"):
        hg = global_mean_pool(h, data.batch.to(device))
        out = head(hg); mask = data.test_mask.to(device)
        if task_type=="graph_cls":
            yb = data.y_graph[mask].to(device)
            return (out[mask].argmax(-1)==yb).float().mean().item()
        else:
            yb = data.y_graph[mask].float().to(device)
            return -torch.mean(torch.abs(out.squeeze(-1)[mask] - yb)).item()
    elif task_type.startswith("edge_"):
        logits = head(h, data.edge_index.to(device)); mask = data.edge_test_mask.to(device)
        if task_type=="edge_cls":
            yb = data.edge_label[mask].to(device)
            return (logits[mask].argmax(-1)==yb).float().mean().item()
        else:
            yb = data.edge_label[mask].float().to(device)
            return -torch.mean(torch.abs(logits.squeeze(-1)[mask] - yb)).item()
    else:
        out = head(h)
        if task_type=="node_cls":
            logits = out[data.test_mask.to(device)]
            yb = data.y[data.test_mask].to(device)
            return (logits.argmax(-1)==yb).float().mean().item()
        else:
            pred = out.squeeze(-1)[data.test_mask.to(device)]
            yb = data.y[data.test_mask].float().to(device)
            return -torch.mean(torch.abs(pred - yb)).item()

# ----------------------- Hetero Training (unchanged heads) -----------------------
def run_train_hetero(data: HeteroData, hyper: dict, device, steps=300, lr=3e-4, wd=1e-4, heads=4, layers=2, optim_name="adamw", sched_name="none"):
    in_dims = {nt: data[nt].x.size(1) for nt in data.node_types}
    enc = HeteroGATEncoder(data.metadata(), in_dims, hid=128, out_dim=hyper["out"], heads=heads, layers=layers, drop=0.2).to(device)

    task = hyper["task"]
    if task.startswith("node_"):
        tgt = hyper["target_nt"]; num_classes = hyper.get("num_classes")
        head = NodeHead(hyper["out"], task, num_classes if task=="node_cls" else None).to(device)
    elif task.startswith("graph_"):
        head = GraphHead(hyper["out"], task, hyper.get("num_classes")).to(device)
    else:
        head = EdgeHead(hyper["out"], task, hyper.get("num_classes"), mode="bilinear").to(device)

    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=wd) if optim_name=="adam" else torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=max(steps//4, 10), T_mult=1) if sched_name=="cosine" else None
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))
    best = -1e9; best_state = None

    for step in range(steps):
        enc.train(); head.train()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=(device.type in ["cuda","cpu"])):
            h_dict = enc({nt: data[nt].x.to(device) for nt in data.node_types},
                         {et: data[et].edge_index.to(device) for et in data.edge_types})
            if task.startswith("node_"):
                tgt = hyper["target_nt"]
                h = h_dict[tgt]; out = head(h)
                if task=="node_cls":
                    loss = F.cross_entropy(out[data[tgt].train_mask.to(device)], data[tgt].y[data[tgt].train_mask].to(device))
                else:
                    loss = F.smooth_l1_loss(out.squeeze(-1)[data[tgt].train_mask.to(device)], data[tgt].y[data[tgt].train_mask].float().to(device))
            elif task.startswith("graph_"):
                nt0 = data.node_types[0]
                hg = global_mean_pool(h_dict[nt0], data[nt0].batch.to(device))
                out = head(hg)
                if task=="graph_cls":
                    loss = F.cross_entropy(out[data.graph_train_mask.to(device)], data.y_graph[data.graph_train_mask].to(device))
                else:
                    loss = F.smooth_l1_loss(out.squeeze(-1)[data.graph_train_mask.to(device)], data.y_graph[data.graph_train_mask].float().to(device))
            else:
                et = hyper["edge_type"]; ei = data[et].edge_index.to(device)
                src_nt,_,_ = et
                logits = head(h_dict[src_nt], ei)
                mask = data[et].edge_train_mask.to(device)
                if task=="edge_cls":
                    loss = F.cross_entropy(logits[mask], data[et].edge_label[mask].to(device))
                else:
                    loss = F.smooth_l1_loss(logits.squeeze(-1)[mask], data[et].edge_label[mask].float().to(device))
        if device.type=="cuda":
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        if sched: sched.step(step)

        if (step+1) % max(1, hyper["eval_every"]) == 0:
            score = eval_score_hetero(enc, head, data, device, hyper)
            if score > best:
                best = score
                best_state = {"enc": {k:v.detach().cpu().clone() for k,v in enc.state_dict().items()},
                              "head": {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}}

    if best_state:
        enc.load_state_dict(best_state["enc"])
        head.load_state_dict(best_state["head"])
    test_metric = test_score_hetero(enc, head, data, device, hyper)
    return best, test_metric, enc, head

@torch.no_grad()
def eval_score_hetero(enc, head, data, device, hyper):
    enc.eval(); head.eval()
    h_dict = enc({nt: data[nt].x.to(device) for nt in data.node_types},
                 {et: data[et].edge_index.to(device) for et in data.edge_types})
    task = hyper["task"]
    if task.startswith("node_"):
        tgt = hyper["target_nt"]
        out = head(h_dict[tgt]); mask = data[tgt].val_mask.to(device)
        if task=="node_cls":
            return (out[mask].argmax(-1)==data[tgt].y[mask].to(device)).float().mean().item()
        else:
            return -torch.mean(torch.abs(out.squeeze(-1)[mask] - data[tgt].y[mask].float().to(device))).item()
    if task.startswith("graph_"):
        nt0 = data.node_types[0]
        hg = global_mean_pool(h_dict[nt0], data[nt0].batch.to(device))
        out = head(hg); mask = data.graph_val_mask.to(device)
        if task=="graph_cls":
            return (out[mask].argmax(-1)==data.y_graph[mask].to(device)).float().mean().item()
        else:
            return -torch.mean(torch.abs(out.squeeze(-1)[mask] - data.y_graph[mask].float().to(device))).item()
    et = hyper["edge_type"]; ei = data[et].edge_index.to(device)
    src_nt,_,_ = et
    logits = head(h_dict[src_nt], ei); mask = data[et].edge_val_mask.to(device)
    if task=="edge_cls":
        return (logits[mask].argmax(-1)==data[et].edge_label[mask].to(device)).float().mean().item()
    return -torch.mean(torch.abs(logits.squeeze(-1)[mask] - data[et].edge_label[mask].float().to(device))).item()

@torch.no_grad()
def test_score_hetero(enc, head, data, device, hyper):
    enc.eval(); head.eval()
    h_dict = enc({nt: data[nt].x.to(device) for nt in data.node_types},
                 {et: data[et].edge_index.to(device) for et in data.edge_types})
    task = hyper["task"]
    if task.startswith("node_"):
        tgt = hyper["target_nt"]
        out = head(h_dict[tgt]); mask = data[tgt].test_mask.to(device)
        if task=="node_cls":
            return (out[mask].argmax(-1)==data[tgt].y[mask].to(device)).float().mean().item()
        else:
            return -torch.mean(torch.abs(out.squeeze(-1)[mask] - data[tgt].y[mask].float().to(device))).item()
    if task.startswith("graph_"):
        nt0 = data.node_types[0]
        hg = global_mean_pool(h_dict[nt0], data[nt0].batch.to(device))
        out = head(hg); mask = data.graph_test_mask.to(device)
        if task=="graph_cls":
            return (out[mask].argmax(-1)==data.y_graph[mask].to(device)).float().mean().item()
        else:
            return -torch.mean(torch.abs(out.squeeze(-1)[mask] - data.y_graph[mask].float().to(device))).item()
    et = hyper["edge_type"]; ei = data[et].edge_index.to(device)
    src_nt,_,_ = et
    logits = head(h_dict[src_nt], ei); mask = data[et].edge_test_mask.to(device)
    if task=="edge_cls":
        return (logits[mask].argmax(-1)==data[et].edge_label[mask].to(device)).float().mean().item()
    return -torch.mean(torch.abs(logits.squeeze(-1)[mask] - data[et].edge_label[mask].float().to(device))).item()

# ----------------------- Link Prediction (homogeneous) -----------------------
def prepare_link_pred_splits(edge_index: torch.Tensor, num_nodes: int, val_ratio=0.1, test_ratio=0.1, seed=0):
    rng = torch.Generator().manual_seed(seed)
    E = edge_index.size(1)
    perm = torch.randperm(E, generator=rng)
    a = int((1 - val_ratio - test_ratio) * E)
    b = int((1 - test_ratio) * E)
    train_pos = edge_index[:, perm[:a]]
    val_pos = edge_index[:, perm[a:b]]
    test_pos = edge_index[:, perm[b:]]

    # negative sampling per split
    num_neg_val = val_pos.size(1)
    num_neg_test = test_pos.size(1)
    val_neg = negative_sampling(edge_index, num_nodes=num_nodes, num_neg_samples=num_neg_val, method='sparse')
    test_neg = negative_sampling(edge_index, num_nodes=num_nodes, num_neg_samples=num_neg_test, method='sparse')
    return train_pos, val_pos, test_pos, val_neg, test_neg

def score_edges(h, edge_index):
    src, dst = edge_index
    return (h[src] * h[dst]).sum(dim=-1)  # dot product score

def train_link_pred(data: Data, arch: str, out_dim: int, device, steps=300, lr=1e-3, heads=4, layers=3, edge_dim=None, pos_dim=None, node_pe=None, build_pos_from="diff"):
    in_dim = data.x.size(1)
    if arch=="gatv2":
        enc = GATEncoder(in_dim, hid=out_dim, out_dim=out_dim, n_layers=layers, heads=heads, drop=0.2).to(device)
    else:
        enc = GraphTransformer(in_dim, hid=out_dim, out_dim=out_dim, n_layers=layers, heads=heads, drop=0.2, edge_dim=edge_dim, pos_dim=pos_dim).to(device)

    opt = torch.optim.AdamW(enc.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    # Splits
    train_pos, val_pos, test_pos, val_neg, test_neg = prepare_link_pred_splits(data.edge_index, data.x.size(0))
    pos_ij_full = None
    if arch=="gtr" and node_pe is not None:
        pos_ij_full = build_edge_pos_bias_from_node_pe(node_pe, data.edge_index, method=build_pos_from)

    best, best_state = -1e9, None
    for step in range(steps):
        enc.train()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=(device.type in ["cuda","cpu"])):
            if arch=="gatv2":
                h = enc(data.x.to(device), data.edge_index.to(device))
            else:
                h = enc(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device) if data.edge_attr is not None else None,
                        pos_ij_full.to(device) if pos_ij_full is not None else None)
            # Sample equal number of negatives for training each step
            num_train_neg = train_pos.size(1)
            train_neg = negative_sampling(data.edge_index, num_nodes=data.x.size(0), num_neg_samples=num_train_neg, method='sparse')

            pos_score = score_edges(h, train_pos.to(device))
            neg_score = score_edges(h, train_neg.to(device))
            loss = F.binary_cross_entropy_with_logits(torch.cat([pos_score, neg_score], dim=0),
                                                      torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0))
        if device.type=="cuda":
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            opt.step()

        if (step+1) % 20 == 0:
            enc.eval()
            with torch.no_grad():
                if arch=="gatv2":
                    h = enc(data.x.to(device), data.edge_index.to(device))
                else:
                    h = enc(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device) if data.edge_attr is not None else None,
                            pos_ij_full.to(device) if pos_ij_full is not None else None)
                val_pos_s = torch.sigmoid(score_edges(h, val_pos.to(device)))
                val_neg_s = torch.sigmoid(score_edges(h, val_neg.to(device)))
                val_auc = 0.5 * (val_pos_s.mean() + (1 - val_neg_s).mean())  # crude AUC proxy
                if val_auc.item() > best:
                    best = val_auc.item()
                    best_state = {k:v.detach().cpu().clone() for k,v in enc.state_dict().items()}
    if best_state:
        enc.load_state_dict(best_state)
    # Test proxy
    enc.eval()
    with torch.no_grad():
        if arch=="gatv2":
            h = enc(data.x.to(device), data.edge_index.to(device))
        else:
            h = enc(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device) if data.edge_attr is not None else None,
                    pos_ij_full.to(device) if pos_ij_full is not None else None)
        test_pos_s = torch.sigmoid(score_edges(h, test_pos.to(device)))
        test_neg_s = torch.sigmoid(score_edges(h, test_neg.to(device)))
        test_auc = 0.5 * (test_pos_s.mean() + (1 - test_neg_s).mean())
    return best, test_auc.item()

# ----------------------- Build Homo Data (with PE) -----------------------
def build_homo_data(cfg: Dict[str,Any], arch: str, device, label_override: Optional[str], add_lappe: bool, add_rwse: bool, pos_method: str):
    nodes = cfg["nodes"]; nid = cfg["nid"]
    edge_index = cfg.get("edge_index")
    raw_edge_attr = cfg.get("raw_edge_attr")
    k = cfg["default_k"]; mutual = cfg["mutual"]

    cands = cfg["label_cands"] if "label_cands" in cfg else infer_label_candidates(nodes, prefer=label_override)
    if not cands:
        raise RuntimeError("No plausible label columns found; set --label")

    # Build initial features & edges
    # We'll compute PE on the final chosen edge_index
    # OPTIMIZATION: Build graph ONCE and reuse for all candidates
    print(f"[info] Building kNN graph for {len(nodes)} nodes (this may take a moment)...")
    drop_cols_temp = [nid] + [c for c in nodes.columns if c in ["label", "complexity", "difficulty", "task_type", "y", "target"]]
    X_temp = build_node_features(nodes, drop_cols_temp)
    if edge_index is None:
        ei_shared = knn_edges(X_temp, k=k, mutual=mutual)
        print(f"[info] Built kNN graph with {ei_shared.size(1)} edges")
    else:
        ei_shared = edge_index
    
    # Try candidates quickly to pick best label - reuse graph!
    best_label = None; best_val = -1e9
    print(f"[info] Testing {min(len(cands), 4)} label candidates...")
    for idx, (col, ttype) in enumerate(cands[:4]):
        try:
            print(f"[info] Testing candidate {idx+1}/{min(len(cands), 4)}: {col} ({ttype})")
            drop_cols = [nid, col]
            X = build_node_features(nodes, drop_cols)
            X = torch.cat([X, degree_features(ei_shared, X.size(0))], dim=1)
            # Reuse shared graph!
            ei = ei_shared
            # labels + quick score
            if ttype=="node_cls":
                cat = pd.Categorical(nodes[col].astype(str))
                y = torch.from_numpy(cat.codes.astype(np.int64)); num_classes = int(y.max().item()+1)
                tr, va, te = make_masks(y, ttype)
                data = Data(x=X, edge_index=ei, y=y, train_mask=tr, val_mask=va, test_mask=te); data.edge_attr=None
                hyper = {"hid":256,"out":256,"task":ttype,"num_classes":num_classes,"eval_every":20}
                # Use fewer steps for large graphs to speed up candidate selection
                quick_steps = 30 if len(nodes) > 20000 else 60
                val = run_train_homo(data, arch, hyper, device, steps=quick_steps, heads=8 if arch=="gtr" else 4, layers=2)[0]
            elif ttype=="node_reg":
                y = torch.from_numpy(nodes[col].astype(float).to_numpy()).float()
                tr, va, te = make_masks(y, ttype)
                data = Data(x=X, edge_index=ei, y=y, train_mask=tr, val_mask=va, test_mask=te); data.edge_attr=None
                hyper = {"hid":256,"out":256,"task":ttype,"num_classes":None,"eval_every":20}
                val = run_train_homo(data, arch, hyper, device, steps=60, heads=8 if arch=="gtr" else 4, layers=2)[0]
            else:
                # graph tasks
                gc = guess_cols(nodes)["gid"]
                if not gc or gc not in nodes.columns:
                    continue
                gids = pd.Categorical(nodes[gc].astype(str))
                batch = torch.from_numpy(gids.codes.astype(np.int64))
                gvals = nodes[[col]].groupby(batch.numpy()).first()[col]
                if ttype=="graph_cls":
                    cat = pd.Categorical(gvals.astype(str)); y_graph = torch.from_numpy(cat.codes.astype(np.int64)); num_classes=int(y_graph.max().item()+1)
                else:
                    y_graph = torch.from_numpy(gvals.astype(float).to_numpy()).float(); num_classes=None
                tr, va, te = make_masks(y_graph if ttype=="graph_cls" else y_graph, ttype, gid=batch)
                data = Data(x=X, edge_index=ei, batch=batch, y_graph=y_graph, train_mask=tr, val_mask=va, test_mask=te); data.edge_attr=None
                hyper = {"hid":256,"out":256,"task":ttype,"num_classes":num_classes,"eval_every":20}
                quick_steps = 30 if len(nodes) > 20000 else 60
                val = run_train_homo(data, arch, hyper, device, steps=quick_steps, heads=8 if arch=="gtr" else 4, layers=3 if arch=="gtr" else 2)[0]
            print(f"[info] Candidate {col}: validation score = {val:.4f}")
            if val > best_val:
                best_val = val; best_label = (col, ttype)
        except Exception as e:
            print(f"[warn] Candidate {col} failed: {e}")
            continue
    if best_label is None:
        raise RuntimeError("Failed to score any label candidates; try --label")
    
    print(f"[info] Selected best label: {best_label[0]} ({best_label[1]}) with score {best_val:.4f}")
    
    # FINAL build - reuse graph if possible
    y_col, task_type = best_label
    drop_cols = [nid, y_col]
    X = build_node_features(nodes, drop_cols)
    if edge_index is None:
        # Reuse the graph we already built!
        ei = ei_shared
    else:
        ei = edge_index
    
    # Positional encodings (per-node) - optimized for large graphs
    node_pe_list = []
    num_nodes = X.size(0)
    
    if add_lappe:
        print(f"[info] Computing Laplacian PE for {num_nodes} nodes (may take a moment for large graphs)...")
        pe = laplacian_pe(ei, num_nodes, k=16, norm=True)
        if pe is not None:
            node_pe_list.append(pe)
            print(f"[info] Laplacian PE computed: {pe.shape}")
        else:
            print(f"[warn] Laplacian PE computation failed (graph may be too large or solver unavailable)")
    if add_rwse:
        print(f"[info] Computing RWSE for {num_nodes} nodes...")
        rw = rwse_return_probs(ei, num_nodes, k=8)
        if rw is not None:
            node_pe_list.append(rw)
            print(f"[info] RWSE computed: {rw.shape}")
        else:
            print(f"[warn] RWSE computation failed")
    node_pe = torch.cat(node_pe_list, dim=1) if node_pe_list else None

    X = torch.cat([X, degree_features(ei, X.size(0))], dim=1)
    EA_tensor = None; edge_dim = None; pos_dim = None; pos_ij = None
    if cfg.get("raw_edge_attr") is not None and edge_index is not None:
        EA_tensor = build_edgeattr_tensor(cfg["raw_edge_attr"], heads=8)
        if EA_tensor is not None and EA_tensor.size(0) != ei.size(1):
            EA_tensor = None
    if EA_tensor is not None: edge_dim = EA_tensor.size(1)
    if arch=="gtr" and node_pe is not None:
        pos_ij = build_edge_pos_bias_from_node_pe(node_pe, ei, method=pos_method)
        pos_dim = pos_ij.size(1)

    # Prepare Data & labels
    if task_type.startswith("graph_"):
        gc = guess_cols(nodes)["gid"]
        if not gc or gc not in nodes.columns:
            # fallback to node task
            task_type = "node_cls"
            cat = pd.Categorical(nodes[y_col].astype(str))
            y = torch.from_numpy(cat.codes.astype(np.int64)); num_classes = int(y.max().item()+1)
            tr, va, te = make_masks(y, task_type)
            data = Data(x=X, edge_index=ei, y=y, train_mask=tr, val_mask=va, test_mask=te)
        else:
            gids = pd.Categorical(nodes[gc].astype(str))
            batch = torch.from_numpy(gids.codes.astype(np.int64))
            gvals = nodes[[y_col]].groupby(batch.numpy()).first()[y_col]
            if task_type=="graph_cls":
                cat = pd.Categorical(gvals.astype(str)); y_graph = torch.from_numpy(cat.codes.astype(np.int64)); num_classes = int(y_graph.max().item()+1)
            else:
                y_graph = torch.from_numpy(gvals.astype(float).to_numpy()).float(); num_classes=None
            tr, va, te = make_masks(y_graph if task_type=="graph_cls" else y_graph, task_type, gid=batch)
            data = Data(x=X, edge_index=ei, batch=batch, y_graph=y_graph, train_mask=tr, val_mask=va, test_mask=te)
    else:
        if task_type=="node_cls":
            cat = pd.Categorical(nodes[y_col].astype(str))
            y = torch.from_numpy(cat.codes.astype(np.int64)); num_classes = int(y.max().item()+1)
        else:
            y = torch.from_numpy(nodes[y_col].astype(float).to_numpy()).float(); num_classes=None
        tr, va, te = make_masks(y if task_type=="node_cls" else y, task_type)
        data = Data(x=X, edge_index=ei, y=y, train_mask=tr, val_mask=va, test_mask=te)

    data.edge_attr = EA_tensor
    data._pos_ij_cache = pos_ij  # stash for caller if needed
    hyper = {"hid":256,"out":256,"task":task_type,"num_classes":num_classes, "eval_every":max(300//8,10)}
    return data, hyper, best_label, edge_dim, (node_pe, pos_ij, pos_dim)

# ----------------------- AutoML (tiny) -----------------------
def automl_homo(data: Data, hyper_base: dict, arch: str, device, edge_dim: Optional[int], pos_dim: Optional[int], use_neighbor, fanout, node_pe, pos_method):
    optims = ["adamw","adam"]; scheds = ["none","cosine"]; lrs=[3e-4,1e-3]
    results=[]
    for H in [4,8]:
        for L in [2,3]:
            for optn in optims:
                for scn in scheds:
                    for lr in lrs:
                        best_val, test_metric, _, _ = run_train_homo(
                            data, arch, {**hyper_base, "eval_every": max(200//8,10)}, device,
                            steps=200, lr=lr, wd=1e-4, drop=0.2, heads=H, layers=L,
                            edge_dim=edge_dim, pos_dim=pos_dim,
                            optim_name=optn, sched_name=scn,
                            use_neighbor=use_neighbor, fanout=fanout,
                            node_pe=node_pe, build_pos_from=pos_method
                        )
                        results.append({"heads":H,"layers":L,"optim":optn,"sched":scn,"lr":lr,"val":best_val,"test":test_metric})
                        print(f"[AutoML] H={H} L={L} opt={optn} sch={scn} lr={lr:.0e} | val={best_val:.4f} | test={test_metric:.4f}")
    return max(results, key=lambda r: r["val"])

def automl_hetero(data: HeteroData, hyper_base: dict, device):
    optims = ["adamw","adam"]; scheds=["none","cosine"]; lrs=[3e-4,1e-3]
    results=[]
    for heads in [2,4]:
        for layers in [2,3]:
            for optn in optims:
                for scn in scheds:
                    for lr in lrs:
                        best_val, test_metric, _, _ = run_train_hetero(
                            data, {**hyper_base, "eval_every": max(200//8,10)}, device,
                            steps=200, lr=lr, wd=1e-4, heads=heads, layers=layers,
                            optim_name=optn, sched_name=scn
                        )
                        results.append({"heads":heads,"layers":layers,"optim":optn,"sched":scn,"lr":lr,"val":best_val,"test":test_metric})
                        print(f"[AutoML][hetero] H={heads} L={layers} opt={optn} sch={scn} lr={lr:.0e} | val={best_val:.4f} | test={test_metric:.4f}")
    return max(results, key=lambda r: r["val"])

# ----------------------- Build HeteroData (bipartite / relations) -----------------------
def build_hetero_data_from_bipartite(schema: Dict[str,Any]):
    user_df = schema["user"]["nodes"]; item_df = schema["item"]["nodes"]
    uX = torch.zeros((len(user_df), 1), dtype=torch.float32)
    iX = torch.zeros((len(item_df), 1), dtype=torch.float32)
    data = HeteroData()
    data["user"].x = uX
    data["item"].x = iX

    u_map = schema["maps"]["user"]; i_map = schema["maps"]["item"]
    edges = schema["edges"]
    uid = guess_cols(edges)["uid"]; iid = guess_cols(edges)["iid"]
    su = edges[uid].map(u_map).astype(int).to_numpy()
    si = edges[iid].map(i_map).astype(int).to_numpy()
    ei = torch.tensor([su, si], dtype=torch.long)
    data[("user","interacts","item")].edge_index = ei

    # Edge labels?
    if schema["edge_task"]:
        col = schema["edge_label_col"]; lbl = edges[col]
        if schema["edge_task"]=="edge_cls":
            cat = pd.Categorical(lbl.astype(str)); y = torch.from_numpy(cat.codes.astype(np.int64)); num_classes=int(y.max().item()+1)
        else:
            y = torch.from_numpy(lbl.astype(float).to_numpy()).float(); num_classes=None
        M = ei.size(1)
        perm = torch.randperm(M); a,b = int(0.7*M), int(0.8*M)
        data[("user","interacts","item")].edge_label = y
        data[("user","interacts","item")].edge_train_mask = torch.zeros(M, dtype=torch.bool); data[("user","interacts","item")].edge_train_mask[perm[:a]]=True
        data[("user","interacts","item")].edge_val_mask = torch.zeros(M, dtype=torch.bool); data[("user","interacts","item")].edge_val_mask[perm[a:b]]=True
        data[("user","interacts","item")].edge_test_mask = torch.zeros(M, dtype=torch.bool); data[("user","interacts","item")].edge_test_mask[perm[b:]]=True
        hyper = {"out":256,"task":schema["edge_task"],"num_classes":num_classes,"edge_type":("user","interacts","item"),"eval_every":30}
        return data, hyper
    return data, None

def build_hetero_data_from_relations(nodes: pd.DataFrame, nid: str, edge_index_dict: Dict[tuple, torch.Tensor],
                                     edges_df: pd.DataFrame, edge_label_col: Optional[str], edge_task: Optional[str]):
    data = HeteroData()
    X = torch.zeros((nodes.shape[0], 1), dtype=torch.float32)
    data["entity"].x = X
    for et, ei in edge_index_dict.items():
        data[et].edge_index = ei
    if edge_task and edge_label_col:
        cat = pd.Categorical(edges_df[guess_cols(edges_df)["etype"]].astype(str))
        rels = cat.categories.tolist()
        idxs = []
        for ridx, rel in enumerate(rels):
            mask = (cat.codes.values == ridx)
            idxs.extend(np.where(mask)[0].tolist())
        lbl = edges_df.iloc[idxs][edge_label_col]
        if edge_task=="edge_cls":
            catl = pd.Categorical(lbl.astype(str)); y = torch.from_numpy(catl.codes.astype(np.int64)); num_classes=int(y.max().item()+1)
        else:
            y = torch.from_numpy(lbl.astype(float).to_numpy()).float(); num_classes=None
        et0 = list(edge_index_dict.keys())[0]
        M = data[et0].edge_index.size(1)
        perm = torch.randperm(M); a,b = int(0.7*M), int(0.8*M)
        data[et0].edge_label = y[:M]
        data[et0].edge_train_mask = torch.zeros(M, dtype=torch.bool); data[et0].edge_train_mask[perm[:a]]=True
        data[et0].edge_val_mask = torch.zeros(M, dtype=torch.bool); data[et0].edge_val_mask[perm[a:b]]=True
        data[et0].edge_test_mask = torch.zeros(M, dtype=torch.bool); data[et0].edge_test_mask[perm[b:]]=True
        hyper = {"out":256,"task":edge_task,"num_classes":num_classes,"edge_type":et0,"eval_every":30}
        return data, hyper
    return data, None

# ----------------------- Main -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, required=True, help="file, comma-separated files, or directory")
    ap.add_argument("--label", type=str, default=None, help="force node/graph label column")
    ap.add_argument("--edge_label", type=str, default=None, help="force edge label column")
    ap.add_argument("--arch", type=str, default="gtr", choices=["gatv2","gtr"], help="homogeneous encoder")
    ap.add_argument("--k", type=int, default=10, help="default k for kNN if no edges provided")
    ap.add_argument("--mutual", action="store_true", help="use mutual kNN when building edges")
    ap.add_argument("--search_heads", type=str, default="4,8")
    ap.add_argument("--search_layers", type=str, default="2,3")
    ap.add_argument("--use_neighbor_loader", action="store_true")
    ap.add_argument("--fanout", type=str, default="15,10,5")
    ap.add_argument("--pos_method", type=str, default="diff", choices=["diff","concat"], help="edge positional bias from node PE")
    ap.add_argument("--lappe", action="store_true", help="enable Laplacian PE")
    ap.add_argument("--rwse", action="store_true", help="enable RWSE PE")
    ap.add_argument("--link_pred", action="store_true", help="train link prediction instead of supervised label")
    ap.add_argument("--epochs", type=int, default=None, help="number of training epochs/steps (overrides AutoML)")
    ap.add_argument("--save_model", type=str, default=None, help="path to save best model checkpoint")
    ap.add_argument("--save_results", type=str, default=None, help="path to save training results JSON")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"[info] device={device}")

    paths = load_paths(args.path)
    print(f"[info] files: {paths}")

    cfg = assemble(paths, args.label, args.edge_label, args.k, args.mutual)

    # Heterogeneous
    if cfg["mode"] == "hetero":
        print("[mode] hetero (bipartite)")
        data, hyper_edge = build_hetero_data_from_bipartite(cfg["hetero_schema"])
        if hyper_edge is None:
            print("[warn] No edge label detected for hetero bipartite. For now, hetero auto-mode supports edge tasks. Provide --edge_label.")
            sys.exit(0)
        best_cfg = automl_hetero(data, hyper_edge, device)
        print(f"[choose] Best config: {best_cfg}")
        best_val, test_metric, _, _ = run_train_hetero(data, {**hyper_edge, "eval_every": max(400//8,10)}, device,
                                                       steps=400, lr=best_cfg["lr"], wd=1e-4,
                                                       heads=best_cfg["heads"], layers=best_cfg["layers"],
                                                       optim_name=best_cfg["optim"], sched_name=best_cfg["sched"])
        print("\n================ RESULT ================")
        print(f"Task: {hyper_edge['task']} (edge, hetero) | Best Val: {best_val:.4f} | Test: {test_metric:.4f}")
        print(f"Config: heads={best_cfg['heads']} layers={best_cfg['layers']} optim={best_cfg['optim']} sched={best_cfg['sched']} lr={best_cfg['lr']:.0e}")
        print("=======================================\n")
        return

    if cfg["mode"] == "hetero_rel":
        print("[mode] hetero (multi-relation)")
        data, hyper_edge = build_hetero_data_from_relations(cfg["nodes"], cfg["nid"], cfg["edge_index_dict"], cfg["edges_df"], cfg["edge_label_col"], cfg["edge_task"])
        if hyper_edge is None:
            print("[warn] No edge label detected for hetero relations. Currently only edge tasks are automated for hetero-rel mode.")
            sys.exit(0)
        best_cfg = automl_hetero(data, hyper_edge, device)
        print(f"[choose] Best config: {best_cfg}")
        best_val, test_metric, _, _ = run_train_hetero(data, {**hyper_edge, "eval_every": max(400//8,10)}, device,
                                                       steps=400, lr=best_cfg["lr"], wd=1e-4,
                                                       heads=best_cfg["heads"], layers=best_cfg["layers"],
                                                       optim_name=best_cfg["optim"], sched_name=best_cfg["sched"])
        print("\n================ RESULT ================")
        print(f"Task: {hyper_edge['task']} (edge, hetero-rel) | Best Val: {best_val:.4f} | Test: {test_metric:.4f}")
        print(f"Config: heads={best_cfg['heads']} layers={best_cfg['layers']} optim={best_cfg['optim']} sched={best_cfg['sched']} lr={best_cfg['lr']:.0e}")
        print("=======================================\n")
        return

    # Homogeneous
    print("[mode] homogeneous")

    # Link prediction branch
    if args.link_pred:
        if cfg.get("edge_index") is None:
            # need edges; build kNN
            nodes = cfg["nodes"]; nid = cfg["nid"]
            X = build_node_features(nodes, [nid])
            ei = knn_edges(X, k=cfg["default_k"], mutual=cfg["mutual"])
            data_lp = Data(x=torch.cat([X, degree_features(ei, X.size(0))], dim=1), edge_index=ei)
        else:
            nodes = cfg["nodes"]; nid = cfg["nid"]
            X = build_node_features(nodes, [nid])
            ei = cfg["edge_index"]
            data_lp = Data(x=torch.cat([X, degree_features(ei, X.size(0))], dim=1), edge_index=ei)

        # Build PE for GraphTransformer if selected
        node_pe_list = []
        if args.lappe:
            pe = laplacian_pe(data_lp.edge_index, data_lp.x.size(0), k=16)
            if pe is not None: node_pe_list.append(pe)
        if args.rwse:
            rw = rwse_return_probs(data_lp.edge_index, data_lp.x.size(0), k=8)
            if rw is not None: node_pe_list.append(rw)
        node_pe = torch.cat(node_pe_list, dim=1) if node_pe_list else None
        pos_dim = None
        if args.arch=="gtr" and node_pe is not None:
            pos_ij = build_edge_pos_bias_from_node_pe(node_pe, data_lp.edge_index, method=args.pos_method)
            pos_dim = pos_ij.size(1)

        best_val, test_auc = train_link_pred(
            data_lp, args.arch, out_dim=256, device=device, steps=400, lr=1e-3,
            heads=8 if args.arch=="gtr" else 4, layers=3 if args.arch=="gtr" else 2,
            edge_dim=None, pos_dim=pos_dim, node_pe=node_pe, build_pos_from=args.pos_method
        )
        print("\n================ RESULT (Link Prediction) ================")
        print(f"Val proxy AUC: {best_val:.4f} | Test proxy AUC: {test_auc:.4f} | arch={args.arch}")
        print("==========================================================\n")
        return

    # Build supervised (node/graph/edge) data with PE
    data, hyper, best_label, edge_dim, pos_pack = build_homo_data(cfg, args.arch, device, args.label, add_lappe=args.lappe, add_rwse=args.rwse, pos_method=args.pos_method)
    node_pe, pos_ij, pos_dim = pos_pack
    print(f"[choose] Best label = {best_label[0]} ({best_label[1]})")

    # Tiny AutoML
    fanout = tuple(int(x) for x in args.fanout.split(",")) if args.use_neighbor_loader else (15,10,5)
    best = automl_homo(
        data, hyper, args.arch, device, edge_dim, pos_dim,
        use_neighbor=args.use_neighbor_loader, fanout=fanout,
        node_pe=node_pe, pos_method=args.pos_method
    )
    print(f"[choose] Best config: {best}")

    # Final train with custom epochs if provided
    train_steps = args.epochs if args.epochs else 400
    best_val, test_metric, enc_final, head_final = run_train_homo(
        data, args.arch, {**hyper, "eval_every": max(train_steps//8,10)}, device,
        steps=train_steps, lr=best["lr"], wd=1e-4, heads=best["heads"], layers=best["layers"],
        edge_dim=edge_dim, pos_dim=pos_dim,
        optim_name=best["optim"], sched_name=best["sched"],
        use_neighbor=args.use_neighbor_loader, fanout=fanout,
        node_pe=node_pe, build_pos_from=args.pos_method
    )
    
    # Save model if requested
    if args.save_model:
        os.makedirs(os.path.dirname(args.save_model) if os.path.dirname(args.save_model) else ".", exist_ok=True)
        torch.save({
            "encoder": enc_final.state_dict(),
            "head": head_final.state_dict(),
            "arch": args.arch,
            "hyper": {**hyper, **best},
            "edge_dim": edge_dim,
            "pos_dim": pos_dim,
            "task": hyper["task"]
        }, args.save_model)
        print(f"[save] Model saved to {args.save_model}")
    
    # Save results if requested
    if args.save_results:
        import json
        results = {
            "task": hyper["task"],
            "best_val_metric": float(best_val),
            "test_metric": float(test_metric),
            "config": {
                "heads": best["heads"],
                "layers": best["layers"],
                "optimizer": best["optim"],
                "scheduler": best["sched"],
                "lr": float(best["lr"]),
                "weight_decay": 1e-4,
                "dropout": 0.2,
                "arch": args.arch,
                "edge_dim": edge_dim,
                "pos_dim": pos_dim,
                "train_steps": train_steps,
                "use_neighbor_loader": args.use_neighbor_loader,
                "lappe": args.lappe,
                "rwse": args.rwse
            },
            "dataset_info": {
                "num_nodes": int(data.x.size(0)),
                "num_edges": int(data.edge_index.size(1)),
                "feature_dim": int(data.x.size(1)),
                "label_column": best_label[0],
                "label_type": best_label[1]
            }
        }
        os.makedirs(os.path.dirname(args.save_results) if os.path.dirname(args.save_results) else ".", exist_ok=True)
        with open(args.save_results, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[save] Results saved to {args.save_results}")
    
    print("\n================ RESULT ================")
    kind = "acc" if hyper["task"] in ["node_cls","graph_cls","edge_cls"] else "-MAE (higher better)"
    print(f"Task: {hyper['task']} | Best Val: {best_val:.4f} | Test {kind}: {test_metric:.4f}")
    print(f"Config: heads={best['heads']} layers={best['layers']} optim={best['optim']} sched={best['sched']} lr={best['lr']:.0e} arch={args.arch} "
          f"edge_bias={'yes' if edge_dim else 'no'} pos_enc={'yes' if pos_dim else 'no'} neighbor_loader={'yes' if args.use_neighbor_loader else 'no'}")
    print("=======================================\n")

if __name__ == "__main__":
    main()
