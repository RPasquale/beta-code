"""
Test LOCAGENT entities and relations.
"""

import pytest
from src.ue_sea.locagent.entities import Entity, EntityType, Relation, RelationType


class TestEntity:
    """Test Entity class."""
    
    def test_entity_creation(self):
        """Test basic entity creation."""
        entity = Entity(
            entity_id="test:entity",
            entity_type=EntityType.FUNCTION,
            name="test_function",
            location="test.py"
        )
        
        assert entity.entity_id == "test:entity"
        assert entity.entity_type == EntityType.FUNCTION
        assert entity.name == "test_function"
        assert entity.location == "test.py"
    
    def test_entity_with_content(self):
        """Test entity with content and metadata."""
        entity = Entity(
            entity_id="test:entity",
            entity_type=EntityType.FUNCTION,
            name="test_function",
            location="test.py",
            content="def test_function(): pass",
            start_line=1,
            end_line=1,
            docstring="Test function"
        )
        
        assert entity.content == "def test_function(): pass"
        assert entity.start_line == 1
        assert entity.end_line == 1
        assert entity.docstring == "Test function"
    
    def test_entity_properties(self):
        """Test entity properties."""
        entity = Entity(
            entity_id="src/test.py:MyClass.method",
            entity_type=EntityType.FUNCTION,
            name="method",
            location="src/test.py"
        )
        
        assert entity.file_path == "src/test.py"
        assert entity.symbol_name == "method"
        assert entity.qualified_name == "src/test.py:MyClass.method"
    
    def test_entity_serialization(self):
        """Test entity serialization."""
        entity = Entity(
            entity_id="test:entity",
            entity_type=EntityType.FUNCTION,
            name="test_function",
            location="test.py",
            content="def test_function(): pass"
        )
        
        # Test to_dict
        entity_dict = entity.to_dict()
        assert entity_dict["entity_id"] == "test:entity"
        assert entity_dict["entity_type"] == "function"
        assert entity_dict["name"] == "test_function"
        
        # Test from_dict
        restored_entity = Entity.from_dict(entity_dict)
        assert restored_entity.entity_id == entity.entity_id
        assert restored_entity.entity_type == entity.entity_type
        assert restored_entity.name == entity.name


class TestRelation:
    """Test Relation class."""
    
    def test_relation_creation(self):
        """Test basic relation creation."""
        relation = Relation(
            source_id="test:source",
            target_id="test:target",
            relation_type=RelationType.INVOKES
        )
        
        assert relation.source_id == "test:source"
        assert relation.target_id == "test:target"
        assert relation.relation_type == RelationType.INVOKES
        assert relation.weight == 1.0
    
    def test_relation_with_metadata(self):
        """Test relation with metadata."""
        relation = Relation(
            source_id="test:source",
            target_id="test:target",
            relation_type=RelationType.INVOKES,
            weight=0.8,
            metadata={"confidence": 0.9, "context": "test"}
        )
        
        assert relation.weight == 0.8
        assert relation.metadata["confidence"] == 0.9
        assert relation.metadata["context"] == "test"
    
    def test_relation_serialization(self):
        """Test relation serialization."""
        relation = Relation(
            source_id="test:source",
            target_id="test:target",
            relation_type=RelationType.INVOKES,
            weight=0.8,
            metadata={"confidence": 0.9}
        )
        
        # Test to_dict
        relation_dict = relation.to_dict()
        assert relation_dict["source_id"] == "test:source"
        assert relation_dict["target_id"] == "test:target"
        assert relation_dict["relation_type"] == "invokes"
        assert relation_dict["weight"] == 0.8
        
        # Test from_dict
        restored_relation = Relation.from_dict(relation_dict)
        assert restored_relation.source_id == relation.source_id
        assert restored_relation.target_id == relation.target_id
        assert restored_relation.relation_type == relation.relation_type


class TestEntityTypes:
    """Test EntityType enum."""
    
    def test_entity_types(self):
        """Test all entity types."""
        assert EntityType.DIRECTORY.value == "directory"
        assert EntityType.FILE.value == "file"
        assert EntityType.CLASS.value == "class"
        assert EntityType.FUNCTION.value == "function"
        assert EntityType.VARIABLE.value == "variable"
        assert EntityType.IMPORT.value == "import"
        assert EntityType.MODULE.value == "module"


class TestRelationTypes:
    """Test RelationType enum."""
    
    def test_relation_types(self):
        """Test all relation types."""
        assert RelationType.CONTAINS.value == "contains"
        assert RelationType.IMPORTS.value == "imports"
        assert RelationType.INHERITS.value == "inherits"
        assert RelationType.INVOKES.value == "invokes"
        assert RelationType.REFERENCES.value == "references"
        assert RelationType.DEFINES.value == "defines"
