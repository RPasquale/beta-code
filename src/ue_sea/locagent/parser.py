"""
Code parsers for extracting entities and relations from source code.

Supports multiple languages with tree-sitter for accurate AST parsing.
"""

import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import tree_sitter
from tree_sitter import Language, Parser, Node

from .entities import Entity, EntityType, Relation, RelationType


class CodeParser(ABC):
    """Abstract base class for code parsers."""
    
    def __init__(self, language: Language):
        self.language = language
        self.parser = Parser()
        self.parser.set_language(language)
    
    @abstractmethod
    def parse_file(self, file_path: str, content: str) -> Tuple[List[Entity], List[Relation]]:
        """Parse a file and extract entities and relations."""
        pass
    
    def _extract_docstring(self, node: Node, content: str) -> Optional[str]:
        """Extract docstring from a node."""
        # Look for docstring in the first few children
        for child in node.children:
            if child.type == "expression_statement":
                expr = child.child_by_field_name("expression")
                if expr and expr.type == "string":
                    docstring = self._get_node_text(expr, content)
                    # Remove quotes and clean up
                    docstring = docstring.strip('"""\'"').strip()
                    if docstring:
                        return docstring
        return None
    
    def _get_node_text(self, node: Node, content: str) -> str:
        """Get the text content of a node."""
        return content[node.start_byte:node.end_byte]
    
    def _get_node_lines(self, node: Node, content: str) -> Tuple[int, int]:
        """Get start and end line numbers for a node."""
        lines = content[:node.start_byte].count('\n')
        start_line = lines + 1
        end_line = start_line + content[node.start_byte:node.end_byte].count('\n')
        return start_line, end_line


class PythonParser(CodeParser):
    """Parser for Python code using tree-sitter."""
    
    def parse_file(self, file_path: str, content: str) -> Tuple[List[Entity], List[Relation]]:
        """Parse Python file and extract entities and relations."""
        tree = self.parser.parse(bytes(content, "utf8"))
        entities = []
        relations = []
        
        # Create file entity
        file_entity = Entity(
            entity_id=file_path,
            entity_type=EntityType.FILE,
            name=Path(file_path).name,
            location=file_path,
            content=content
        )
        entities.append(file_entity)
        
        # Walk the AST
        self._walk_tree(tree.root_node, content, file_path, entities, relations)
        
        return entities, relations
    
    def _walk_tree(self, node: Node, content: str, file_path: str, 
                   entities: List[Entity], relations: List[Relation]) -> None:
        """Recursively walk the AST and extract entities."""
        
        if node.type == "class_definition":
            self._extract_class(node, content, file_path, entities, relations)
        elif node.type == "function_definition":
            self._extract_function(node, content, file_path, entities, relations)
        elif node.type == "import_statement" or node.type == "import_from_statement":
            self._extract_import(node, content, file_path, entities, relations)
        
        # Recursively process children
        for child in node.children:
            self._walk_tree(child, content, file_path, entities, relations)
    
    def _extract_class(self, node: Node, content: str, file_path: str,
                      entities: List[Entity], relations: List[Relation]) -> None:
        """Extract class definition."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        
        class_name = self._get_node_text(name_node, content)
        class_id = f"{file_path}:{class_name}"
        start_line, end_line = self._get_node_lines(node, content)
        
        # Extract docstring
        docstring = self._extract_docstring(node, content)
        
        class_entity = Entity(
            entity_id=class_id,
            entity_type=EntityType.CLASS,
            name=class_name,
            location=file_path,
            content=self._get_node_text(node, content),
            start_line=start_line,
            end_line=end_line,
            docstring=docstring
        )
        entities.append(class_entity)
        
        # Add contains relation from file to class
        file_entity = next(e for e in entities if e.entity_id == file_path)
        relations.append(Relation(
            source_id=file_path,
            target_id=class_id,
            relation_type=RelationType.CONTAINS
        ))
        
        # Extract inheritance
        inheritance = node.child_by_field_name("superclasses")
        if inheritance:
            for child in inheritance.children:
                if child.type == "identifier":
                    parent_name = self._get_node_text(child, content)
                    parent_id = f"{file_path}:{parent_name}"
                    relations.append(Relation(
                        source_id=class_id,
                        target_id=parent_id,
                        relation_type=RelationType.INHERITS
                    ))
    
    def _extract_function(self, node: Node, content: str, file_path: str,
                         entities: List[Entity], relations: List[Relation]) -> None:
        """Extract function definition."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        
        func_name = self._get_node_text(name_node, content)
        func_id = f"{file_path}:{func_name}"
        start_line, end_line = self._get_node_lines(node, content)
        
        # Extract docstring
        docstring = self._extract_docstring(node, content)
        
        func_entity = Entity(
            entity_id=func_id,
            entity_type=EntityType.FUNCTION,
            name=func_name,
            location=file_path,
            content=self._get_node_text(node, content),
            start_line=start_line,
            end_line=end_line,
            docstring=docstring
        )
        entities.append(func_entity)
        
        # Add contains relation from file to function
        file_entity = next(e for e in entities if e.entity_id == file_path)
        relations.append(Relation(
            source_id=file_path,
            target_id=func_id,
            relation_type=RelationType.CONTAINS
        ))
        
        # Extract function calls within the function
        self._extract_function_calls(node, content, func_id, relations)
    
    def _extract_function_calls(self, node: Node, content: str, 
                               caller_id: str, relations: List[Relation]) -> None:
        """Extract function calls within a function."""
        for child in node.children:
            if child.type == "call":
                # Extract the function being called
                func_node = child.child_by_field_name("function")
                if func_node and func_node.type == "identifier":
                    callee_name = self._get_node_text(func_node, content)
                    # Try to find the callee entity
                    callee_id = f"{caller_id.split(':')[0]}:{callee_name}"
                    relations.append(Relation(
                        source_id=caller_id,
                        target_id=callee_id,
                        relation_type=RelationType.INVOKES
                    ))
            
            # Recursively check children
            self._extract_function_calls(child, content, caller_id, relations)
    
    def _extract_import(self, node: Node, content: str, file_path: str,
                       entities: List[Entity], relations: List[Relation]) -> None:
        """Extract import statements."""
        if node.type == "import_statement":
            # Handle: import module
            module_list = node.child_by_field_name("module")
            if module_list:
                for child in module_list.children:
                    if child.type == "dotted_name":
                        module_name = self._get_node_text(child, content)
                        import_id = f"import:{module_name}"
                        
                        import_entity = Entity(
                            entity_id=import_id,
                            entity_type=EntityType.IMPORT,
                            name=module_name,
                            location=file_path
                        )
                        entities.append(import_entity)
                        
                        relations.append(Relation(
                            source_id=file_path,
                            target_id=import_id,
                            relation_type=RelationType.IMPORTS
                        ))
        
        elif node.type == "import_from_statement":
            # Handle: from module import name
            module_node = node.child_by_field_name("module")
            if module_node:
                module_name = self._get_node_text(module_node, content)
                import_id = f"import:{module_name}"
                
                import_entity = Entity(
                    entity_id=import_id,
                    entity_type=EntityType.IMPORT,
                    name=module_name,
                    location=file_path
                )
                entities.append(import_entity)
                
                relations.append(Relation(
                    source_id=file_path,
                    target_id=import_id,
                    relation_type=RelationType.IMPORTS
                ))


class JavaScriptParser(CodeParser):
    """Parser for JavaScript code using tree-sitter."""
    
    def parse_file(self, file_path: str, content: str) -> Tuple[List[Entity], List[Relation]]:
        """Parse JavaScript file and extract entities and relations."""
        tree = self.parser.parse(bytes(content, "utf8"))
        entities = []
        relations = []
        
        # Create file entity
        file_entity = Entity(
            entity_id=file_path,
            entity_type=EntityType.FILE,
            name=Path(file_path).name,
            location=file_path,
            content=content
        )
        entities.append(file_entity)
        
        # Walk the AST
        self._walk_tree(tree.root_node, content, file_path, entities, relations)
        
        return entities, relations
    
    def _walk_tree(self, node: Node, content: str, file_path: str,
                   entities: List[Entity], relations: List[Relation]) -> None:
        """Recursively walk the AST and extract entities."""
        
        if node.type == "class_declaration":
            self._extract_class(node, content, file_path, entities, relations)
        elif node.type == "function_declaration":
            self._extract_function(node, content, file_path, entities, relations)
        elif node.type == "import_statement":
            self._extract_import(node, content, file_path, entities, relations)
        
        # Recursively process children
        for child in node.children:
            self._walk_tree(child, content, file_path, entities, relations)
    
    def _extract_class(self, node: Node, content: str, file_path: str,
                      entities: List[Entity], relations: List[Relation]) -> None:
        """Extract class declaration."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        
        class_name = self._get_node_text(name_node, content)
        class_id = f"{file_path}:{class_name}"
        start_line, end_line = self._get_node_lines(node, content)
        
        class_entity = Entity(
            entity_id=class_id,
            entity_type=EntityType.CLASS,
            name=class_name,
            location=file_path,
            content=self._get_node_text(node, content),
            start_line=start_line,
            end_line=end_line
        )
        entities.append(class_entity)
        
        # Add contains relation from file to class
        file_entity = next(e for e in entities if e.entity_id == file_path)
        relations.append(Relation(
            source_id=file_path,
            target_id=class_id,
            relation_type=RelationType.CONTAINS
        ))
    
    def _extract_function(self, node: Node, content: str, file_path: str,
                         entities: List[Entity], relations: List[Relation]) -> None:
        """Extract function declaration."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        
        func_name = self._get_node_text(name_node, content)
        func_id = f"{file_path}:{func_name}"
        start_line, end_line = self._get_node_lines(node, content)
        
        func_entity = Entity(
            entity_id=func_id,
            entity_type=EntityType.FUNCTION,
            name=func_name,
            location=file_path,
            content=self._get_node_text(node, content),
            start_line=start_line,
            end_line=end_line
        )
        entities.append(func_entity)
        
        # Add contains relation from file to function
        file_entity = next(e for e in entities if e.entity_id == file_path)
        relations.append(Relation(
            source_id=file_path,
            target_id=func_id,
            relation_type=RelationType.CONTAINS
        ))
    
    def _extract_import(self, node: Node, content: str, file_path: str,
                       entities: List[Entity], relations: List[Relation]) -> None:
        """Extract import statements."""
        source_node = node.child_by_field_name("source")
        if source_node:
            module_name = self._get_node_text(source_node, content)
            import_id = f"import:{module_name}"
            
            import_entity = Entity(
                entity_id=import_id,
                entity_type=EntityType.IMPORT,
                name=module_name,
                location=file_path
            )
            entities.append(import_entity)
            
            relations.append(Relation(
                source_id=file_path,
                target_id=import_id,
                relation_type=RelationType.IMPORTS
            ))


class TypeScriptParser(JavaScriptParser):
    """Parser for TypeScript code (extends JavaScript parser)."""
    
    def _walk_tree(self, node: Node, content: str, file_path: str,
                   entities: List[Entity], relations: List[Relation]) -> None:
        """Recursively walk the AST and extract entities."""
        
        if node.type == "class_declaration":
            self._extract_class(node, content, file_path, entities, relations)
        elif node.type == "function_declaration":
            self._extract_function(node, content, file_path, entities, relations)
        elif node.type == "import_statement":
            self._extract_import(node, content, file_path, entities, relations)
        elif node.type == "interface_declaration":
            self._extract_interface(node, content, file_path, entities, relations)
        
        # Recursively process children
        for child in node.children:
            self._walk_tree(child, content, file_path, entities, relations)
    
    def _extract_interface(self, node: Node, content: str, file_path: str,
                          entities: List[Entity], relations: List[Relation]) -> None:
        """Extract TypeScript interface declaration."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        
        interface_name = self._get_node_text(name_node, content)
        interface_id = f"{file_path}:{interface_name}"
        start_line, end_line = self._get_node_lines(node, content)
        
        interface_entity = Entity(
            entity_id=interface_id,
            entity_type=EntityType.CLASS,  # Treat interfaces as classes
            name=interface_name,
            location=file_path,
            content=self._get_node_text(node, content),
            start_line=start_line,
            end_line=end_line
        )
        entities.append(interface_entity)
        
        # Add contains relation from file to interface
        file_entity = next(e for e in entities if e.entity_id == file_path)
        relations.append(Relation(
            source_id=file_path,
            target_id=interface_id,
            relation_type=RelationType.CONTAINS
        ))
