"""
Test LOCAGENT indexing functionality.
"""

import pytest
import numpy as np
from src.ue_sea.locagent.entities import Entity, EntityType
from src.ue_sea.locagent.indices import IDIndex, NameIndex, BM25Index, HierarchicalIndex


class TestIDIndex:
    """Test IDIndex functionality."""
    
    def test_id_index_creation(self):
        """Test ID index creation."""
        index = IDIndex()
        assert len(index._entities) == 0
    
    def test_add_entity(self):
        """Test adding entity to ID index."""
        index = IDIndex()
        entity = Entity(
            entity_id="test:entity",
            entity_type=EntityType.FUNCTION,
            name="test_function",
            location="test.py"
        )
        
        index.add_entity(entity)
        assert "test:entity" in index._entities
        assert index._entities["test:entity"] == entity
    
    def test_search_exact_id(self):
        """Test exact ID search."""
        index = IDIndex()
        entity = Entity(
            entity_id="test:entity",
            entity_type=EntityType.FUNCTION,
            name="test_function",
            location="test.py"
        )
        index.add_entity(entity)
        
        results = index.search("test:entity")
        assert len(results) == 1
        assert results[0][0] == "test:entity"
        assert results[0][1] == 1.0
    
    def test_search_nonexistent_id(self):
        """Test search for nonexistent ID."""
        index = IDIndex()
        results = index.search("nonexistent")
        assert len(results) == 0


class TestNameIndex:
    """Test NameIndex functionality."""
    
    def test_name_index_creation(self):
        """Test name index creation."""
        index = NameIndex()
        assert len(index._name_to_entities) == 0
        assert len(index._entity_names) == 0
    
    def test_add_entity(self):
        """Test adding entity to name index."""
        index = NameIndex()
        entity = Entity(
            entity_id="test:entity",
            entity_type=EntityType.FUNCTION,
            name="test_function",
            location="test.py"
        )
        
        index.add_entity(entity)
        assert "test_function" in index._name_to_entities
        assert "test:entity" in index._entity_names
    
    def test_search_by_name(self):
        """Test searching by name."""
        index = NameIndex()
        entity = Entity(
            entity_id="test:entity",
            entity_type=EntityType.FUNCTION,
            name="test_function",
            location="test.py"
        )
        index.add_entity(entity)
        
        results = index.search("test_function")
        assert len(results) == 1
        assert results[0][0] == "test:entity"
        assert results[0][1] == 1.0
    
    def test_search_partial_name(self):
        """Test searching by partial name."""
        index = NameIndex()
        entity = Entity(
            entity_id="test:entity",
            entity_type=EntityType.FUNCTION,
            name="test_function",
            location="test.py"
        )
        index.add_entity(entity)
        
        results = index.search("test")
        assert len(results) == 1
        assert results[0][0] == "test:entity"
        assert results[0][1] > 0.0


class TestBM25Index:
    """Test BM25Index functionality."""
    
    def test_bm25_index_creation(self):
        """Test BM25 index creation."""
        index = BM25Index(use_gpu=False)
        assert len(index._corpus) == 0
        assert len(index._entity_ids) == 0
        assert index._bm25 is None
    
    def test_add_entity(self):
        """Test adding entity to BM25 index."""
        index = BM25Index(use_gpu=False)
        entity = Entity(
            entity_id="test:entity",
            entity_type=EntityType.FUNCTION,
            name="test_function",
            location="test.py",
            content="def test_function(): pass"
        )
        
        index.add_entity(entity)
        assert len(index._corpus) == 1
        assert len(index._entity_ids) == 1
        assert "test:entity" in index._entity_ids
    
    def test_search_content(self):
        """Test searching content."""
        index = BM25Index(use_gpu=False)
        entity = Entity(
            entity_id="test:entity",
            entity_type=EntityType.FUNCTION,
            name="test_function",
            location="test.py",
            content="def test_function(): pass"
        )
        index.add_entity(entity)
        
        results = index.search("test_function")
        assert len(results) == 1
        assert results[0][0] == "test:entity"
        assert results[0][1] > 0.0
    
    def test_search_empty_query(self):
        """Test searching with empty query."""
        index = BM25Index(use_gpu=False)
        entity = Entity(
            entity_id="test:entity",
            entity_type=EntityType.FUNCTION,
            name="test_function",
            location="test.py",
            content="def test_function(): pass"
        )
        index.add_entity(entity)
        
        results = index.search("")
        assert len(results) == 0


class TestHierarchicalIndex:
    """Test HierarchicalIndex functionality."""
    
    def test_hierarchical_index_creation(self):
        """Test hierarchical index creation."""
        index = HierarchicalIndex(use_gpu=False)
        assert index.id_index is not None
        assert index.name_index is not None
        assert index.bm25_index is not None
    
    def test_add_entity(self):
        """Test adding entity to hierarchical index."""
        index = HierarchicalIndex(use_gpu=False)
        entity = Entity(
            entity_id="test:entity",
            entity_type=EntityType.FUNCTION,
            name="test_function",
            location="test.py",
            content="def test_function(): pass"
        )
        
        index.add_entity(entity)
        
        # Check that entity was added to all sub-indices
        assert index.id_index.get_entity("test:entity") == entity
        assert "test:entity" in index.name_index._entity_names
        assert "test:entity" in index.bm25_index._entity_ids
    
    def test_search_all_types(self):
        """Test searching across all index types."""
        index = HierarchicalIndex(use_gpu=False)
        entity = Entity(
            entity_id="test:entity",
            entity_type=EntityType.FUNCTION,
            name="test_function",
            location="test.py",
            content="def test_function(): pass"
        )
        index.add_entity(entity)
        
        # Search with all types
        results = index.search("test_function", search_types=["id", "name", "content"])
        assert len(results) >= 1
        
        # Search with specific types
        id_results = index.search("test:entity", search_types=["id"])
        assert len(id_results) == 1
        
        name_results = index.search("test_function", search_types=["name"])
        assert len(name_results) == 1
        
        content_results = index.search("test_function", search_types=["content"])
        assert len(content_results) == 1
    
    def test_get_entity(self):
        """Test getting entity by ID."""
        index = HierarchicalIndex(use_gpu=False)
        entity = Entity(
            entity_id="test:entity",
            entity_type=EntityType.FUNCTION,
            name="test_function",
            location="test.py"
        )
        index.add_entity(entity)
        
        retrieved_entity = index.get_entity("test:entity")
        assert retrieved_entity == entity
        
        # Test nonexistent entity
        assert index.get_entity("nonexistent") is None
