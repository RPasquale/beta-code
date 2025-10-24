"""
LOCAGENT Graph: Core graph data structure for code understanding.

Implements the main graph with entities, relations, and hierarchical indices.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

from .entities import Entity, EntityType, Relation, RelationType
from .indices import HierarchicalIndex
from .parser import PythonParser, JavaScriptParser, TypeScriptParser
from .tools import SearchEntity, TraverseGraph, RetrieveEntity


class LOCAGENT_Graph:
    """
    Main graph data structure for code understanding.
    
    Maintains entities, relations, and provides tools for searching,
    traversing, and retrieving code entities.
    """
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self._entities: Dict[str, Entity] = {}
        self._relations: Dict[str, Relation] = {}
        self._index = HierarchicalIndex(use_gpu=use_gpu)
        self._graph = nx.DiGraph()
        
        # Initialize parsers
        self._parsers = self._initialize_parsers()
        
        # Initialize tools
        self._search_tool = SearchEntity(self)
        self._traverse_tool = TraverseGraph(self)
        self._retrieve_tool = RetrieveEntity(self)
    
    def _initialize_parsers(self) -> Dict[str, Any]:
        """Initialize language parsers."""
        parsers = {}
        
        try:
            from tree_sitter import Language
            
            # Try to build language libraries
            import tempfile
            import os
            
            # Create temporary directory for compiled languages
            temp_dir = tempfile.mkdtemp()
            
            # Build Python language
            try:
                python_lib = os.path.join(temp_dir, "python.so")
                Language.build_library(python_lib, ["tree-sitter-python"])
                python_lang = Language(python_lib, "python")
                from .parser import PythonParser
                parsers['.py'] = PythonParser(python_lang)
            except Exception as e:
                print(f"Warning: Could not build Python parser: {e}")
            
            # Build JavaScript language
            try:
                js_lib = os.path.join(temp_dir, "javascript.so")
                Language.build_library(js_lib, ["tree-sitter-javascript"])
                js_lang = Language(js_lib, "javascript")
                from .parser import JavaScriptParser
                parsers['.js'] = JavaScriptParser(js_lang)
            except Exception as e:
                print(f"Warning: Could not build JavaScript parser: {e}")
            
            # Build TypeScript language
            try:
                ts_lib = os.path.join(temp_dir, "typescript.so")
                Language.build_library(ts_lib, ["tree-sitter-typescript", "tree-sitter-typescript"])
                ts_lang = Language(ts_lib, "typescript")
                from .parser import TypeScriptParser
                parsers['.ts'] = TypeScriptParser(ts_lang)
                parsers['.tsx'] = TypeScriptParser(ts_lang)
            except Exception as e:
                print(f"Warning: Could not build TypeScript parser: {e}")
                
        except ImportError as e:
            print(f"Warning: Could not initialize tree-sitter: {e}")
            print("Falling back to basic parsing without AST support.")
        
        return parsers
    
    async def build_from_repository(self, repo_path: str, 
                                  file_patterns: List[str] = None) -> None:
        """
        Build graph from a repository.
        
        Args:
            repo_path: Path to the repository
            file_patterns: List of file patterns to include (e.g., ['*.py', '*.js'])
        """
        if file_patterns is None:
            file_patterns = ['*.py', '*.js', '*.ts', '*.tsx']
        
        repo_path = Path(repo_path)
        if not repo_path.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")
        
        # Find all relevant files
        files_to_parse = []
        for pattern in file_patterns:
            files_to_parse.extend(repo_path.rglob(pattern))
        
        print(f"Found {len(files_to_parse)} files to parse")
        
        # Parse files in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for file_path in files_to_parse:
                future = executor.submit(self._parse_file, file_path)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    entities, relations = future.result()
                    await self._add_entities_and_relations(entities, relations)
                except Exception as e:
                    print(f"Error parsing file: {e}")
        
        print(f"Graph built with {len(self._entities)} entities and {len(self._relations)} relations")
    
    def _parse_file(self, file_path: Path) -> Tuple[List[Entity], List[Relation]]:
        """Parse a single file and return entities and relations."""
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Get file extension
            ext = file_path.suffix.lower()
            
            # Use appropriate parser
            if ext in self._parsers:
                parser = self._parsers[ext]
                return parser.parse_file(str(file_path), content)
            else:
                # Fallback: create basic file entity
                file_entity = Entity(
                    entity_id=str(file_path),
                    entity_type=EntityType.FILE,
                    name=file_path.name,
                    location=str(file_path),
                    content=content
                )
                return [file_entity], []
        
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return [], []
    
    async def _add_entities_and_relations(self, entities: List[Entity], 
                                         relations: List[Relation]) -> None:
        """Add entities and relations to the graph."""
        # Add entities
        for entity in entities:
            self._entities[entity.entity_id] = entity
            self._index.add_entity(entity)
            self._graph.add_node(entity.entity_id, **entity.to_dict())
        
        # Add relations
        for relation in relations:
            relation_id = relation.relation_id
            self._relations[relation_id] = relation
            self._graph.add_edge(
                relation.source_id,
                relation.target_id,
                relation_type=relation.relation_type.value,
                weight=relation.weight,
                **relation.metadata
            )
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self._entities.get(entity_id)
    
    def get_relations(self, entity_id: str, direction: str = "out") -> List[Relation]:
        """Get relations for an entity."""
        relations = []
        
        if direction == "out":
            for target_id in self._graph.successors(entity_id):
                edge_data = self._graph.get_edge_data(entity_id, target_id)
                if edge_data:
                    relation = Relation(
                        source_id=entity_id,
                        target_id=target_id,
                        relation_type=RelationType(edge_data.get("relation_type", "references")),
                        weight=edge_data.get("weight", 1.0),
                        metadata={k: v for k, v in edge_data.items() 
                                if k not in ["relation_type", "weight"]}
                    )
                    relations.append(relation)
        else:
            for source_id in self._graph.predecessors(entity_id):
                edge_data = self._graph.get_edge_data(source_id, entity_id)
                if edge_data:
                    relation = Relation(
                        source_id=source_id,
                        target_id=entity_id,
                        relation_type=RelationType(edge_data.get("relation_type", "references")),
                        weight=edge_data.get("weight", 1.0),
                        metadata={k: v for k, v in edge_data.items() 
                                if k not in ["relation_type", "weight"]}
                    )
                    relations.append(relation)
        
        return relations
    
    def get_subgraph(self, entity_ids: List[str], 
                    max_hops: int = 2) -> nx.DiGraph:
        """Get subgraph around specified entities."""
        subgraph_nodes = set(entity_ids)
        
        # Expand by hops
        for _ in range(max_hops):
            new_nodes = set()
            for node in subgraph_nodes:
                new_nodes.update(self._graph.successors(node))
                new_nodes.update(self._graph.predecessors(node))
            subgraph_nodes.update(new_nodes)
        
        return self._graph.subgraph(subgraph_nodes)
    
    def search_entities(self, query: str, limit: int = 10,
                       search_types: List[str] = None) -> List[Tuple[str, float]]:
        """Search for entities using the hierarchical index."""
        return self._index.search(query, limit, search_types)
    
    def get_entity_statistics(self) -> Dict[str, Any]:
        """Get statistics about the graph."""
        entity_types = {}
        relation_types = {}
        
        for entity in self._entities.values():
            entity_types[entity.entity_type.value] = entity_types.get(entity.entity_type.value, 0) + 1
        
        for relation in self._relations.values():
            relation_types[relation.relation_type.value] = relation_types.get(relation.relation_type.value, 0) + 1
        
        return {
            "total_entities": len(self._entities),
            "total_relations": len(self._relations),
            "entity_types": entity_types,
            "relation_types": relation_types,
            "graph_density": nx.density(self._graph),
            "connected_components": nx.number_weakly_connected_components(self._graph.to_undirected()),
        }
    
    def save_to_file(self, file_path: str) -> None:
        """Save graph to file."""
        data = {
            "entities": {eid: entity.to_dict() for eid, entity in self._entities.items()},
            "relations": {rid: relation.to_dict() for rid, relation in self._relations.items()},
            "graph_edges": list(self._graph.edges(data=True)),
            "statistics": self.get_entity_statistics()
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, file_path: str) -> None:
        """Load graph from file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Load entities
        for eid, entity_data in data["entities"].items():
            entity = Entity.from_dict(entity_data)
            self._entities[eid] = entity
            self._index.add_entity(entity)
            self._graph.add_node(eid, **entity.to_dict())
        
        # Load relations
        for rid, relation_data in data["relations"].items():
            relation = Relation.from_dict(relation_data)
            self._relations[rid] = relation
            self._graph.add_edge(
                relation.source_id,
                relation.target_id,
                relation_type=relation.relation_type.value,
                weight=relation.weight,
                **relation.metadata
            )
    
    # Tool access methods
    @property
    def search_entity(self) -> SearchEntity:
        """Get SearchEntity tool."""
        return self._search_tool
    
    @property
    def traverse_graph(self) -> TraverseGraph:
        """Get TraverseGraph tool."""
        return self._traverse_tool
    
    @property
    def retrieve_entity(self) -> RetrieveEntity:
        """Get RetrieveEntity tool."""
        return self._retrieve_tool
