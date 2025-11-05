"""
Sparse indices for fast entity retrieval in LOCAGENT.

Implements ID index, name index, and BM25 index for efficient searching.
"""

import re
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
from rank_bm25 import BM25Okapi

# Optional imports for GPU acceleration
try:
    import faiss
    FAISS_AVAILABLE = True
    FAISS_GPU_AVAILABLE = all(
        hasattr(faiss, attr) for attr in ("index_cpu_to_gpu", "StandardGpuResources")
    )
except ImportError:
    FAISS_AVAILABLE = False
    FAISS_GPU_AVAILABLE = False
    faiss = None

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    CUPY_AVAILABLE = False
    cp = None

# Initialize sentence transformers availability
SENTENCE_TRANSFORMERS_AVAILABLE = False
SentenceTransformer = None

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    pass

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    TORCH_AVAILABLE = False
    torch = None

from .entities import Entity, EntityType


class SparseIndex(ABC):
    """Abstract base class for sparse indices."""
    
    @abstractmethod
    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the index."""
        pass
    
    @abstractmethod
    def remove_entity(self, entity_id: str) -> None:
        """Remove an entity from the index."""
        pass
    
    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[Tuple[str, float]]:
        """Search the index and return (entity_id, score) pairs."""
        pass
    
    @abstractmethod
    def update_entity(self, entity: Entity) -> None:
        """Update an entity in the index."""
        pass


class IDIndex(SparseIndex):
    """Exact ID lookup index."""
    
    def __init__(self):
        self._entities: Dict[str, Entity] = {}
    
    def add_entity(self, entity: Entity) -> None:
        """Add entity to ID index."""
        self._entities[entity.entity_id] = entity
    
    def remove_entity(self, entity_id: str) -> None:
        """Remove entity from ID index."""
        self._entities.pop(entity_id, None)
    
    def search(self, query: str, limit: int = 10) -> List[Tuple[str, float]]:
        """Exact ID search."""
        if query in self._entities:
            return [(query, 1.0)]
        return []
    
    def update_entity(self, entity: Entity) -> None:
        """Update entity in ID index."""
        self._entities[entity.entity_id] = entity
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self._entities.get(entity_id)
    
    def get_all_entities(self) -> List[Entity]:
        """Get all entities in the index."""
        return list(self._entities.values())


class NameIndex(SparseIndex):
    """Name-based lookup index with fuzzy matching."""
    
    def __init__(self):
        self._name_to_entities: Dict[str, Set[str]] = {}
        self._entity_names: Dict[str, str] = {}
    
    def add_entity(self, entity: Entity) -> None:
        """Add entity to name index."""
        self._entity_names[entity.entity_id] = entity.name
        
        # Add to name mapping
        if entity.name not in self._name_to_entities:
            self._name_to_entities[entity.name] = set()
        self._name_to_entities[entity.name].add(entity.entity_id)
    
    def remove_entity(self, entity_id: str) -> None:
        """Remove entity from name index."""
        if entity_id in self._entity_names:
            name = self._entity_names[entity_id]
            self._name_to_entities[name].discard(entity_id)
            if not self._name_to_entities[name]:
                del self._name_to_entities[name]
            del self._entity_names[entity_id]
    
    def search(self, query: str, limit: int = 10) -> List[Tuple[str, float]]:
        """Search by name with fuzzy matching."""
        results = []
        query_lower = query.lower()
        
        for name, entity_ids in self._name_to_entities.items():
            if query_lower in name.lower():
                # Calculate similarity score
                score = self._calculate_similarity(query_lower, name.lower())
                for entity_id in entity_ids:
                    results.append((entity_id, score))
        
        # Sort by score and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def update_entity(self, entity: Entity) -> None:
        """Update entity in name index."""
        old_name = self._entity_names.get(entity.entity_id)
        if old_name and old_name != entity.name:
            # Remove from old name
            self._name_to_entities[old_name].discard(entity.entity_id)
            if not self._name_to_entities[old_name]:
                del self._name_to_entities[old_name]
        
        # Add with new name
        self.add_entity(entity)
    
    def _calculate_similarity(self, query: str, name: str) -> float:
        """Calculate similarity score between query and name."""
        if query == name:
            return 1.0
        
        # Simple substring matching
        if query in name:
            return 0.8
        
        # Levenshtein distance-based similarity
        distance = self._levenshtein_distance(query, name)
        max_len = max(len(query), len(name))
        if max_len == 0:
            return 1.0
        
        return 1.0 - (distance / max_len)
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]


class BM25Index(SparseIndex):
    """BM25-based content search index with GPU acceleration."""
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self._corpus: List[str] = []
        self._entity_ids: List[str] = []
        self._bm25: Optional[BM25Okapi] = None
        self._faiss_index: Optional[faiss.Index] = None
        self._embeddings: Optional[np.ndarray] = None
    
    def add_entity(self, entity: Entity) -> None:
        """Add entity to BM25 index."""
        # Extract searchable content
        content = self._extract_searchable_content(entity)
        if content:
            self._corpus.append(content)
            self._entity_ids.append(entity.entity_id)
            self._rebuild_index()
    
    def remove_entity(self, entity_id: str) -> None:
        """Remove entity from BM25 index."""
        if entity_id in self._entity_ids:
            idx = self._entity_ids.index(entity_id)
            self._corpus.pop(idx)
            self._entity_ids.pop(idx)
            self._rebuild_index()
    
    def search(self, query: str, limit: int = 10) -> List[Tuple[str, float]]:
        """Search using BM25 scoring."""
        if not self._bm25 or not self._corpus:
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        
        # Get BM25 scores
        scores = self._bm25.get_scores(query_tokens)
        
        # Create results with entity IDs and scores
        results = [(self._entity_ids[i], float(scores[i])) 
                  for i in range(len(scores))]
        
        # Sort by score and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def update_entity(self, entity: Entity) -> None:
        """Update entity in BM25 index."""
        if entity.entity_id in self._entity_ids:
            self.remove_entity(entity.entity_id)
        self.add_entity(entity)
    
    def _extract_searchable_content(self, entity: Entity) -> str:
        """Extract searchable content from entity."""
        content_parts = []
        
        # Add entity name
        content_parts.append(entity.name)
        
        # Add content if available
        if entity.content:
            content_parts.append(entity.content)
        
        # Add docstring if available
        if entity.docstring:
            content_parts.append(entity.docstring)
        
        # Add metadata keywords
        for key, value in entity.metadata.items():
            if isinstance(value, str):
                content_parts.append(f"{key}: {value}")
        
        return " ".join(content_parts)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        # Simple tokenization - can be enhanced with proper tokenizers
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        return [token for token in tokens if len(token) > 1]
    
    def _rebuild_index(self) -> None:
        """Rebuild the BM25 index."""
        if not self._corpus:
            self._bm25 = None
            return
        
        # Tokenize corpus
        tokenized_corpus = [self._tokenize(doc) for doc in self._corpus]
        
        # Build BM25 index
        self._bm25 = BM25Okapi(tokenized_corpus)
        
        # Build FAISS index for GPU acceleration if enabled
        if self.use_gpu and len(self._corpus) > 100 and FAISS_AVAILABLE:
            self._build_faiss_index()
    
    def _build_faiss_index(self) -> None:
        """Build FAISS index for GPU-accelerated similarity search."""
        if not self._bm25 or not FAISS_AVAILABLE:
            return
        
        # Get document embeddings using BM25
        doc_embeddings = []
        for i, doc_tokens in enumerate([self._tokenize(doc) for doc in self._corpus]):
            scores = self._bm25.get_scores(doc_tokens)
            doc_embeddings.append(scores)
        
        # Convert to numpy array
        self._embeddings = np.array(doc_embeddings, dtype=np.float32)
        
        # Create FAISS index
        dimension = self._embeddings.shape[1]
        if self.use_gpu and CUPY_AVAILABLE and cp.cuda.is_available() and FAISS_GPU_AVAILABLE:
            # Use GPU for FAISS
            self._faiss_index = faiss.IndexFlatIP(dimension)
            self._faiss_index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, self._faiss_index
            )
        else:
            # Use CPU for FAISS
            self._faiss_index = faiss.IndexFlatIP(dimension)
        
        # Add embeddings to index
        self._faiss_index.add(self._embeddings)
    
    def search_with_faiss(self, query: str, limit: int = 10) -> List[Tuple[str, float]]:
        """Search using FAISS index for GPU acceleration."""
        if self._faiss_index is None or self._embeddings is None:
            return self.search(query, limit)

        # Get query embedding
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        query_scores = self._bm25.get_scores(query_tokens)
        query_embedding = np.array([query_scores], dtype=np.float32)

        # Search using FAISS
        scores, indices = self._faiss_index.search(query_embedding, limit)

        # Return results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self._entity_ids):
                results.append((self._entity_ids[idx], float(score)))

        return results


class DenseIndex(SparseIndex):
    """Semantic embedding index backed by FAISS."""

    def __init__(self, use_gpu: bool = True, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.use_gpu = use_gpu and CUPY_AVAILABLE and FAISS_AVAILABLE and FAISS_GPU_AVAILABLE
        self.model_name = model_name
        self._encoder: Optional[SentenceTransformer] = None
        self._entity_ids: List[str] = []
        self._embeddings: Optional[np.ndarray] = None
        self._faiss_index: Optional[faiss.Index] = None
        self._embedding_cache: Dict[str, np.ndarray] = {}

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                device = "cuda" if (torch and torch.cuda.is_available() and use_gpu) else "cpu"
                self._encoder = SentenceTransformer(self.model_name, device=device)
            except Exception as exc:  # pragma: no cover - environment specific
                print(f"Warning: Failed to initialize SentenceTransformer ({exc}); dense retrieval disabled.")
                self._encoder = None
        else:
            print("Warning: sentence-transformers not installed; dense retrieval disabled.")

    def _ensure_index(self) -> None:
        if not FAISS_AVAILABLE or self._embeddings is None or len(self._embeddings) == 0:
            self._faiss_index = None
            return

        dimension = self._embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        if self.use_gpu and FAISS_GPU_AVAILABLE and getattr(faiss, "get_num_gpus", lambda: 0)() > 0:
            resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(resources, 0, index)
        index.add(self._embeddings)
        self._faiss_index = index

    def add_entity(self, entity: Entity) -> None:
        if self._encoder is None:
            return

        content = entity.content or entity.docstring
        if not content:
            return

        embedding = self._encoder.encode(content, convert_to_numpy=True)
        self._embedding_cache[entity.entity_id] = embedding
        self._entity_ids.append(entity.entity_id)

        if self._embeddings is None:
            self._embeddings = embedding[np.newaxis, :]
        else:
            self._embeddings = np.vstack([self._embeddings, embedding])

        self._ensure_index()

    def remove_entity(self, entity_id: str) -> None:
        if entity_id not in self._entity_ids:
            return

        idx = self._entity_ids.index(entity_id)
        self._entity_ids.pop(idx)
        self._embedding_cache.pop(entity_id, None)

        if self._embeddings is not None:
            self._embeddings = np.delete(self._embeddings, idx, axis=0)
            if self._embeddings.size == 0:
                self._embeddings = None

        self._ensure_index()

    def update_entity(self, entity: Entity) -> None:
        self.remove_entity(entity.entity_id)
        self.add_entity(entity)

    def search(self, query: str, limit: int = 10) -> List[Tuple[str, float]]:
        if self._encoder is None or not self._entity_ids:
            return []

        query_embedding = self._encoder.encode(query, convert_to_numpy=True)
        query_embedding = query_embedding[np.newaxis, :]

        if self._faiss_index is None:
            # Fall back to cosine similarity in numpy
            embeddings = self._embeddings
            if embeddings is None:
                return []
            scores = embeddings @ query_embedding.T
            scores = scores.squeeze(-1)
            top_idx = np.argsort(scores)[::-1][:limit]
            return [(self._entity_ids[i], float(scores[i])) for i in top_idx]

        scores, indices = self._faiss_index.search(query_embedding, limit)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self._entity_ids):
                results.append((self._entity_ids[idx], float(score)))
        return results


class HierarchicalIndex:
    """Hierarchical index combining all sparse indices."""
    
    def __init__(self, use_gpu: bool = True):
        self.id_index = IDIndex()
        self.name_index = NameIndex()
        self.bm25_index = BM25Index(use_gpu=use_gpu)
        self.dense_index = DenseIndex(use_gpu=use_gpu)
        self.use_gpu = use_gpu
        self._query_cache: Dict[str, List[Tuple[str, float]]] = {}
    
    def add_entity(self, entity: Entity) -> None:
        """Add entity to all indices."""
        self.id_index.add_entity(entity)
        self.name_index.add_entity(entity)
        self.bm25_index.add_entity(entity)
        self.dense_index.add_entity(entity)
        self._query_cache.clear()
    
    def remove_entity(self, entity_id: str) -> None:
        """Remove entity from all indices."""
        self.id_index.remove_entity(entity_id)
        self.name_index.remove_entity(entity_id)
        self.bm25_index.remove_entity(entity_id)
        self.dense_index.remove_entity(entity_id)
        self._query_cache.clear()
    
    def update_entity(self, entity: Entity) -> None:
        """Update entity in all indices."""
        self.id_index.update_entity(entity)
        self.name_index.update_entity(entity)
        self.bm25_index.update_entity(entity)
        self.dense_index.update_entity(entity)
        self._query_cache.clear()
    
    def search(self, query: str, limit: int = 10, 
               search_types: List[str] = None) -> List[Tuple[str, float]]:
        """Search across all indices."""
        if search_types is None:
            search_types = ["id", "name", "content"]
        
        all_results = []
        
        if "id" in search_types:
            id_results = self.id_index.search(query, limit)
            all_results.extend(id_results)
        
        if "name" in search_types:
            name_results = self.name_index.search(query, limit)
            all_results.extend(name_results)
        
        if "content" in search_types:
            if self.bm25_index._faiss_index is not None:
                content_results = self.bm25_index.search_with_faiss(query, limit)
            else:
                content_results = self.bm25_index.search(query, limit)
            all_results.extend(content_results)

        if "semantic" in search_types:
            dense_results = self.dense_index.search(query, limit)
            all_results.extend(dense_results)
        
        # Deduplicate and sort by score
        seen = set()
        unique_results = []
        for entity_id, score in all_results:
            if entity_id not in seen:
                seen.add(entity_id)
                unique_results.append((entity_id, score))
        
        unique_results.sort(key=lambda x: x[1], reverse=True)
        return unique_results[:limit]

    def search_hybrid(
        self,
        query: str,
        limit: int = 10,
        alpha: float = 0.6,
        candidates: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> List[Tuple[str, float]]:
        """Hybrid semantic + sparse retrieval with optional reranking."""

        cache_key = None
        if use_cache and candidates is None:
            cache_key = f"{query}|{limit}|{alpha:.3f}"
            if cache_key in self._query_cache:
                return self._query_cache[cache_key][:limit]

        # Retrieve more candidates for fusion
        sparse_limit = max(limit * 3, 50)
        sparse_results = self.bm25_index.search(query, sparse_limit)
        dense_results = self.dense_index.search(query, sparse_limit)

        sparse_dict = {eid: score for eid, score in sparse_results}
        dense_dict = {eid: score for eid, score in dense_results}

        all_entity_ids = set(sparse_dict.keys()) | set(dense_dict.keys())
        if candidates:
            allowed = set(candidates)
            all_entity_ids &= allowed

        if not all_entity_ids:
            return []

        # Normalise scores
        def normalise(score_dict: Dict[str, float]) -> Dict[str, float]:
            if not score_dict:
                return {}
            values = np.array(list(score_dict.values()))
            max_abs = np.max(np.abs(values))
            if max_abs == 0:
                return {k: 0.0 for k in score_dict}
            return {k: v / max_abs for k, v in score_dict.items()}

        sparse_norm = normalise(sparse_dict)
        dense_norm = normalise(dense_dict)

        fused_scores: Dict[str, float] = {}
        for entity_id in all_entity_ids:
            sparse_score = sparse_norm.get(entity_id, 0.0)
            dense_score = dense_norm.get(entity_id, 0.0)
            fused_scores[entity_id] = alpha * dense_score + (1 - alpha) * sparse_score

        # Final rerank by BM25 score as a tie-breaker
        results = [
            (
                entity_id,
                fused_scores[entity_id],
                sparse_dict.get(entity_id, 0.0),
                dense_dict.get(entity_id, 0.0),
            )
            for entity_id in fused_scores
        ]
        results.sort(key=lambda x: (x[1], x[2]), reverse=True)
        final_records = results[:limit]

        if cache_key:
            self._query_cache[cache_key] = final_records

        return final_records
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self.id_index.get_entity(entity_id)
    
    def get_all_entities(self) -> List[Entity]:
        """Get all entities."""
        return self.id_index.get_all_entities()

    def clear_cache(self) -> None:
        self._query_cache.clear()
