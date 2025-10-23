"""
LOCAGENT Tools: SearchEntity, TraverseGraph, RetrieveEntity

Implements the three core tools for code understanding and navigation.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import networkx as nx
from dataclasses import dataclass
from enum import Enum

from .entities import Entity, EntityType, RelationType


class SnippetLevel(Enum):
    """Level of code snippet detail."""
    FOLD = "fold"        # Just the signature/declaration
    PREVIEW = "preview"   # First few lines
    FULL = "full"        # Complete content


@dataclass
class SearchResult:
    """Result from entity search."""
    entity_id: str
    entity_type: EntityType
    location: str
    name: str
    snippet: str
    snippet_level: SnippetLevel
    score: float
    metadata: Dict[str, Any]


@dataclass
class TraverseResult:
    """Result from graph traversal."""
    subgraph: Dict[str, Any]  # Tree-expanded subgraph
    entities: List[Entity]
    relations: List[Tuple[str, str, str]]  # (source, target, relation_type)
    metadata: Dict[str, Any]


@dataclass
class RetrieveResult:
    """Result from entity retrieval."""
    entity_id: str
    path: str
    span: Tuple[int, int]  # (start_line, end_line)
    full_code: str
    metadata: Dict[str, Any]


class SearchEntity:
    """Tool for searching entities by keywords."""
    
    def __init__(self, graph: 'LOCAGENT_Graph'):
        self.graph = graph
    
    async def search(self, keywords: List[str], 
                    entity_types: List[EntityType] = None,
                    snippet_level: SnippetLevel = SnippetLevel.PREVIEW,
                    limit: int = 10) -> List[SearchResult]:
        """
        Search for entities by keywords.
        
        Args:
            keywords: List of keywords to search for
            entity_types: Filter by entity types (None for all)
            snippet_level: Level of code snippet detail
            limit: Maximum number of results
        
        Returns:
            List of SearchResult objects
        """
        # Combine keywords into search query
        query = " ".join(keywords)
        
        # Search using hierarchical index
        search_results = self.graph.search_entities(query, limit * 2)  # Get more for filtering
        
        results = []
        for entity_id, score in search_results:
            entity = self.graph.get_entity(entity_id)
            if not entity:
                continue
            
            # Filter by entity type if specified
            if entity_types and entity.entity_type not in entity_types:
                continue
            
            # Generate snippet based on level
            snippet = self._generate_snippet(entity, snippet_level)
            
            result = SearchResult(
                entity_id=entity.entity_id,
                entity_type=entity.entity_type,
                location=entity.location,
                name=entity.name,
                snippet=snippet,
                snippet_level=snippet_level,
                score=score,
                metadata=entity.metadata
            )
            results.append(result)
            
            if len(results) >= limit:
                break
        
        return results
    
    def _generate_snippet(self, entity: Entity, level: SnippetLevel) -> str:
        """Generate code snippet based on level."""
        if not entity.content:
            return ""
        
        lines = entity.content.split('\n')
        
        if level == SnippetLevel.FOLD:
            # Just the first line (signature/declaration)
            return lines[0] if lines else ""
        
        elif level == SnippetLevel.PREVIEW:
            # First 5 lines
            return '\n'.join(lines[:5])
        
        elif level == SnippetLevel.FULL:
            # Complete content
            return entity.content
        
        return ""


class TraverseGraph:
    """Tool for traversing the code graph."""
    
    def __init__(self, graph: 'LOCAGENT_Graph'):
        self.graph = graph
    
    async def traverse(self, start_ids: List[str],
                      direction: str = "both",
                      hops: int = 2,
                      entity_types: List[EntityType] = None,
                      relation_types: List[RelationType] = None) -> TraverseResult:
        """
        Traverse the graph from starting entities.
        
        Args:
            start_ids: List of starting entity IDs
            direction: "in", "out", or "both"
            hops: Maximum number of hops
            entity_types: Filter by entity types
            relation_types: Filter by relation types
        
        Returns:
            TraverseResult with subgraph and metadata
        """
        # Get subgraph
        subgraph = self.graph.get_subgraph(start_ids, hops)
        
        # Filter by entity types if specified
        if entity_types:
            nodes_to_remove = []
            for node in subgraph.nodes():
                entity = self.graph.get_entity(node)
                if entity and entity.entity_type not in entity_types:
                    nodes_to_remove.append(node)
            for node in nodes_to_remove:
                subgraph.remove_node(node)
        
        # Filter by relation types if specified
        if relation_types:
            edges_to_remove = []
            for source, target, data in subgraph.edges(data=True):
                relation_type = data.get("relation_type", "references")
                if RelationType(relation_type) not in relation_types:
                    edges_to_remove.append((source, target))
            for source, target in edges_to_remove:
                subgraph.remove_edge(source, target)
        
        # Convert to tree-expanded format
        tree_subgraph = self._convert_to_tree_format(subgraph, start_ids, direction)
        
        # Get entities and relations
        entities = []
        relations = []
        
        for node in subgraph.nodes():
            entity = self.graph.get_entity(node)
            if entity:
                entities.append(entity)
        
        for source, target, data in subgraph.edges(data=True):
            relation_type = data.get("relation_type", "references")
            relations.append((source, target, relation_type))
        
        return TraverseResult(
            subgraph=tree_subgraph,
            entities=entities,
            relations=relations,
            metadata={
                "start_ids": start_ids,
                "direction": direction,
                "hops": hops,
                "entity_types": [et.value for et in (entity_types or [])],
                "relation_types": [rt.value for rt in (relation_types or [])],
            }
        )
    
    def _convert_to_tree_format(self, subgraph: nx.DiGraph, 
                               start_ids: List[str], 
                               direction: str) -> Dict[str, Any]:
        """Convert subgraph to tree-expanded format."""
        tree = {}
        
        # Start from each starting ID
        for start_id in start_ids:
            if start_id in subgraph:
                tree[start_id] = self._build_tree_node(subgraph, start_id, direction, set())
        
        return tree
    
    def _build_tree_node(self, subgraph: nx.DiGraph, 
                        node_id: str, 
                        direction: str, 
                        visited: Set[str]) -> Dict[str, Any]:
        """Build tree node recursively."""
        if node_id in visited:
            return {"entity_id": node_id, "children": {}}
        
        visited.add(node_id)
        entity = self.graph.get_entity(node_id)
        
        node_data = {
            "entity_id": node_id,
            "entity_type": entity.entity_type.value if entity else "unknown",
            "name": entity.name if entity else node_id,
            "children": {}
        }
        
        # Add children based on direction
        if direction in ["out", "both"]:
            for successor in subgraph.successors(node_id):
                if successor not in visited:
                    node_data["children"][successor] = self._build_tree_node(
                        subgraph, successor, direction, visited.copy()
                    )
        
        if direction in ["in", "both"]:
            for predecessor in subgraph.predecessors(node_id):
                if predecessor not in visited:
                    node_data["children"][predecessor] = self._build_tree_node(
                        subgraph, predecessor, direction, visited.copy()
                    )
        
        return node_data


class RetrieveEntity:
    """Tool for retrieving full entity details."""
    
    def __init__(self, graph: 'LOCAGENT_Graph'):
        self.graph = graph
    
    async def retrieve(self, entity_ids: List[str]) -> List[RetrieveResult]:
        """
        Retrieve full details for entities.
        
        Args:
            entity_ids: List of entity IDs to retrieve
        
        Returns:
            List of RetrieveResult objects
        """
        results = []
        
        for entity_id in entity_ids:
            entity = self.graph.get_entity(entity_id)
            if not entity:
                continue
            
            # Get file path and span
            path = entity.location
            span = (entity.start_line or 0, entity.end_line or 0)
            
            # Get full code
            full_code = entity.content or ""
            
            # Get metadata
            metadata = {
                "entity_type": entity.entity_type.value,
                "name": entity.name,
                "docstring": entity.docstring,
                "in_relations": list(entity.in_relations),
                "out_relations": list(entity.out_relations),
                **entity.metadata
            }
            
            result = RetrieveResult(
                entity_id=entity.entity_id,
                path=path,
                span=span,
                full_code=full_code,
                metadata=metadata
            )
            results.append(result)
        
        return results
    
    async def retrieve_by_name(self, names: List[str]) -> List[RetrieveResult]:
        """
        Retrieve entities by name.
        
        Args:
            names: List of entity names to retrieve
        
        Returns:
            List of RetrieveResult objects
        """
        # Search for entities by name
        search_results = []
        for name in names:
            results = self.graph.search_entities(name, limit=10)
            search_results.extend(results)
        
        # Get unique entity IDs
        entity_ids = list(set(entity_id for entity_id, _ in search_results))
        
        return await self.retrieve(entity_ids)
