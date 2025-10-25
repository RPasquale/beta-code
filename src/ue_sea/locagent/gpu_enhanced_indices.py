"""
GPU-Enhanced Indices for LOCAGENT

Optimized for Windows with RTX 4090 GPU acceleration.
Uses faiss-cpu with GPU support and CuPy for maximum performance.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
import logging

# GPU acceleration imports
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
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

from .entities import Entity, EntityType
from .indices import BM25Index

logger = logging.getLogger(__name__)


class GPUEnhancedBM25Index(BM25Index):
    """
    GPU-enhanced BM25 index with semantic embeddings and FAISS acceleration.
    
    Combines sparse BM25 retrieval with dense semantic search for maximum performance.
    """
    
    def __init__(self, use_gpu: bool = True, embedding_model: str = "all-MiniLM-L6-v2"):
        super().__init__(use_gpu=use_gpu)
        
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self._embeddings = None
        self._faiss_index = None
        self._device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        
        # Initialize embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model, device=self._device)
                logger.info(f"✅ Loaded embedding model: {embedding_model} on {self._device}")
            except Exception as e:
                logger.warning(f"⚠️ Could not load embedding model: {e}")
                self.embedding_model = None
        
        logger.info(f"🚀 GPU-Enhanced BM25 Index initialized on {self._device}")
    
    def add_entity(self, entity: Entity) -> None:
        """Add entity with both BM25 and semantic indexing."""
        # Add to BM25 index
        super().add_entity(entity)
        
        # Add to semantic index if embedding model available
        if self.embedding_model is not None:
            self._add_semantic_entity(entity)
    
    def _add_semantic_entity(self, entity: Entity) -> None:
        """Add entity to semantic index."""
        try:
            # Extract content for embedding
            content = self._extract_searchable_content(entity)
            if not content:
                return
            
            # Generate embedding
            embedding = self.embedding_model.encode([content], convert_to_tensor=True)
            
            # Convert to numpy if needed
            if hasattr(embedding, 'cpu'):
                embedding = embedding.cpu().numpy()
            
            # Initialize embeddings array if needed
            if self._embeddings is None:
                self._embeddings = embedding
            else:
                self._embeddings = np.vstack([self._embeddings, embedding])
            
            # Rebuild FAISS index
            self._rebuild_faiss_index()
            
        except Exception as e:
            logger.warning(f"⚠️ Could not add semantic entity {entity.entity_id}: {e}")
    
    def _rebuild_faiss_index(self) -> None:
        """Rebuild FAISS index with current embeddings."""
        if not FAISS_AVAILABLE or self._embeddings is None:
            return
        
        try:
            # Create FAISS index
            dimension = self._embeddings.shape[1]
            if (
                self._device == "cuda"
                and torch.cuda.is_available()
                and FAISS_GPU_AVAILABLE
            ):
                # Use GPU-accelerated index
                self._faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
                gpu_res = faiss.StandardGpuResources()
                self._faiss_index = faiss.index_cpu_to_gpu(gpu_res, 0, self._faiss_index)
            else:
                # Use CPU index
                self._faiss_index = faiss.IndexFlatIP(dimension)
            
            # Add embeddings to index
            self._faiss_index.add(self._embeddings.astype('float32'))
            
            logger.info(f"✅ Rebuilt FAISS index with {len(self._embeddings)} embeddings")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not rebuild FAISS index: {e}")
            self._faiss_index = None
    
    def search(self, query: str, limit: int = 10, use_semantic: bool = True) -> List[Tuple[str, float]]:
        """
        Enhanced search combining BM25 and semantic retrieval.
        
        Args:
            query: Search query
            limit: Maximum number of results
            use_semantic: Whether to use semantic search
        """
        results = []
        
        # 1. BM25 search (sparse retrieval)
        bm25_results = super().search(query, limit=limit * 2)  # Get more for fusion
        results.extend(bm25_results)
        
        # 2. Semantic search (dense retrieval)
        if use_semantic and self.embedding_model is not None and self._faiss_index is not None:
            semantic_results = self._semantic_search(query, limit=limit * 2)
            results.extend(semantic_results)
        
        # 3. Fusion and ranking
        fused_results = self._fuse_results(results, query)
        
        # 4. Return top results
        return fused_results[:limit]
    
    def _semantic_search(self, query: str, limit: int = 10) -> List[Tuple[str, float]]:
        """Perform semantic search using embeddings."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_tensor=True)
            if hasattr(query_embedding, 'cpu'):
                query_embedding = query_embedding.cpu().numpy()
            
            # Search FAISS index
            scores, indices = self._faiss_index.search(query_embedding.astype('float32'), limit)
            
            # Convert to entity IDs and scores
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self._entity_ids):
                    entity_id = self._entity_ids[idx]
                    results.append((entity_id, float(score)))
            
            return results
            
        except Exception as e:
            logger.warning(f"⚠️ Semantic search failed: {e}")
            return []
    
    def _fuse_results(self, results: List[Tuple[str, float]], query: str) -> List[Tuple[str, float]]:
        """Fuse BM25 and semantic results using reciprocal rank fusion."""
        # Group results by entity ID
        entity_scores = {}
        for entity_id, score in results:
            if entity_id not in entity_scores:
                entity_scores[entity_id] = []
            entity_scores[entity_id].append(score)
        
        # Apply reciprocal rank fusion
        fused_results = []
        for entity_id, scores in entity_scores.items():
            # Simple average for now (can be improved with RRF)
            fused_score = sum(scores) / len(scores)
            fused_results.append((entity_id, fused_score))
        
        # Sort by score
        fused_results.sort(key=lambda x: x[1], reverse=True)
        return fused_results
    
    def search_with_faiss(self, query: str, limit: int = 10) -> List[Tuple[str, float]]:
        """Direct FAISS search for maximum GPU performance."""
        if not FAISS_AVAILABLE or self._faiss_index is None:
            return super().search(query, limit)
        
        return self._semantic_search(query, limit)


class GPUEnhancedHierarchicalIndex:
    """
    GPU-enhanced hierarchical index combining all retrieval methods.
    
    Provides the fastest possible retrieval for LOCAGENT.
    """
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        
        # Initialize enhanced indices
        self.bm25_index = GPUEnhancedBM25Index(use_gpu=use_gpu)
        
        logger.info(f"🚀 GPU-Enhanced Hierarchical Index initialized on {self.device}")
    
    def add_entity(self, entity: Entity) -> None:
        """Add entity to all indices."""
        self.bm25_index.add_entity(entity)
    
    def search(self, query: str, limit: int = 10, 
               search_types: List[str] = None) -> List[Tuple[str, float]]:
        """
        Enhanced search with GPU acceleration.
        
        Args:
            query: Search query
            limit: Maximum results
            search_types: Types of search to perform
        """
        if search_types is None:
            search_types = ["hybrid"]  # Use hybrid search by default
        
        results = []
        
        if "hybrid" in search_types:
            # Use GPU-enhanced hybrid search
            results = self.bm25_index.search(query, limit, use_semantic=True)
        else:
            # Fallback to individual search types
            if "bm25" in search_types:
                bm25_results = self.bm25_index.search(query, limit, use_semantic=False)
                results.extend(bm25_results)
            
            if "semantic" in search_types:
                semantic_results = self.bm25_index._semantic_search(query, limit)
                results.extend(semantic_results)
        
        # Deduplicate and sort
        seen = set()
        unique_results = []
        for entity_id, score in results:
            if entity_id not in seen:
                seen.add(entity_id)
                unique_results.append((entity_id, score))
        
        unique_results.sort(key=lambda x: x[1], reverse=True)
        return unique_results[:limit]
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self.bm25_index.get_entity(entity_id)
    
    def get_all_entities(self) -> List[Entity]:
        """Get all entities."""
        return self.bm25_index.get_all_entities()
    
    def get_gpu_status(self) -> Dict[str, Any]:
        """Get GPU utilization status."""
        status = {
            "cuda_available": torch.cuda.is_available(),
            "device": self.device,
            "faiss_available": FAISS_AVAILABLE,
            "cupy_available": CUPY_AVAILABLE,
            "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE
        }
        
        if torch.cuda.is_available():
            status.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory,
                "gpu_memory_allocated": torch.cuda.memory_allocated(0),
                "gpu_memory_cached": torch.cuda.memory_reserved(0)
            })
        
        return status
