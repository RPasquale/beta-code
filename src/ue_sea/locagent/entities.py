"""
Entity and relation definitions for the LOCAGENT graph.

Defines the core data structures for representing code entities and their relationships.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Any
import hashlib


class EntityType(Enum):
    """Types of code entities in the graph."""
    DIRECTORY = "directory"
    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"
    VARIABLE = "variable"
    IMPORT = "import"
    MODULE = "module"


class RelationType(Enum):
    """Types of relationships between entities."""
    CONTAINS = "contains"  # directory contains file, class contains function
    IMPORTS = "imports"    # file imports module, function imports class
    INHERITS = "inherits"  # class inherits from another class
    INVOKES = "invokes"    # function calls another function
    REFERENCES = "references"  # variable references another entity
    DEFINES = "defines"    # file defines class/function


@dataclass
class Entity:
    """Represents a code entity in the graph."""
    
    # Core identification
    entity_id: str  # Fully-qualified path (e.g., "src/utils.py:MathUtils.calculate_sum")
    entity_type: EntityType
    name: str
    location: str  # File path
    
    # Content and metadata
    content: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    docstring: Optional[str] = None
    
    # Graph properties
    in_relations: Set[str] = field(default_factory=set)  # Entity IDs that point to this
    out_relations: Set[str] = field(default_factory=set)  # Entity IDs this points to
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate entity after initialization."""
        if not self.entity_id:
            raise ValueError("Entity ID cannot be empty")
        if not self.name:
            raise ValueError("Entity name cannot be empty")
    
    @property
    def qualified_name(self) -> str:
        """Get the fully qualified name of the entity."""
        return self.entity_id
    
    @property
    def file_path(self) -> str:
        """Get the file path from the entity ID."""
        if ":" in self.entity_id:
            return self.entity_id.split(":")[0]
        return self.entity_id
    
    @property
    def symbol_name(self) -> str:
        """Get the symbol name from the entity ID."""
        if ":" in self.entity_id:
            return self.entity_id.split(":")[1]
        return self.name
    
    def add_relation(self, target_id: str, relation_type: RelationType, direction: str = "out") -> None:
        """Add a relation to another entity."""
        if direction == "out":
            self.out_relations.add(target_id)
        else:
            self.in_relations.add(target_id)
    
    def remove_relation(self, target_id: str, direction: str = "out") -> None:
        """Remove a relation to another entity."""
        if direction == "out":
            self.out_relations.discard(target_id)
        else:
            self.in_relations.discard(target_id)
    
    def get_content_hash(self) -> str:
        """Get a hash of the entity content for change detection."""
        if self.content:
            return hashlib.md5(self.content.encode()).hexdigest()
        return ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary for serialization."""
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type.value,
            "name": self.name,
            "location": self.location,
            "content": self.content,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "docstring": self.docstring,
            "in_relations": list(self.in_relations),
            "out_relations": list(self.out_relations),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Entity:
        """Create entity from dictionary."""
        return cls(
            entity_id=data["entity_id"],
            entity_type=EntityType(data["entity_type"]),
            name=data["name"],
            location=data["location"],
            content=data.get("content"),
            start_line=data.get("start_line"),
            end_line=data.get("end_line"),
            docstring=data.get("docstring"),
            in_relations=set(data.get("in_relations", [])),
            out_relations=set(data.get("out_relations", [])),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Relation:
    """Represents a relationship between two entities."""
    
    source_id: str
    target_id: str
    relation_type: RelationType
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate relation after initialization."""
        if not self.source_id or not self.target_id:
            raise ValueError("Source and target IDs cannot be empty")
        if self.source_id == self.target_id:
            raise ValueError("Source and target cannot be the same")
    
    @property
    def relation_id(self) -> str:
        """Get unique identifier for this relation."""
        return f"{self.source_id}->{self.target_id}:{self.relation_type.value}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relation to dictionary for serialization."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "weight": self.weight,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Relation:
        """Create relation from dictionary."""
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=RelationType(data["relation_type"]),
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {}),
        )
