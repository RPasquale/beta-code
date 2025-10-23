"""
Diff Generator: Handles SEARCH/REPLACE diff format for code modifications.

Implements the strict diff format as specified in AlphaEvolve.
"""

import re
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class DiffType(Enum):
    """Types of diffs."""
    SEARCH_REPLACE = "search_replace"
    INSERT = "insert"
    DELETE = "delete"


@dataclass
class DiffBlock:
    """Represents a single diff block."""
    diff_type: DiffType
    search_content: str
    replace_content: str
    line_number: Optional[int] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Diff:
    """Represents a complete diff with multiple blocks."""
    blocks: List[DiffBlock]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DiffGenerator:
    """
    Generates and applies SEARCH/REPLACE diffs for code modifications.
    
    Implements the strict diff format as specified in AlphaEvolve:
    <<<<<<< SEARCH
    <original code>
    =======
    <new code>
    >>>>>>> REPLACE
    """
    
    def __init__(self):
        self.diff_pattern = re.compile(
            r'<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE',
            re.DOTALL
        )
    
    def parse_diff(self, diff_text: str) -> Diff:
        """
        Parse a diff text into Diff object.
        
        Args:
            diff_text: Raw diff text with SEARCH/REPLACE blocks
        
        Returns:
            Diff object with parsed blocks
        """
        blocks = []
        matches = self.diff_pattern.findall(diff_text)
        
        for i, (search_content, replace_content) in enumerate(matches):
            block = DiffBlock(
                diff_type=DiffType.SEARCH_REPLACE,
                search_content=search_content.strip(),
                replace_content=replace_content.strip(),
                line_number=None,
                metadata={"block_index": i}
            )
            blocks.append(block)
        
        return Diff(blocks=blocks)
    
    async def apply_diff(self, original_code: str, diff_text: str) -> str:
        """
        Apply a diff to original code.
        
        Args:
            original_code: Original code to modify
            replace_content: Diff text to apply
        
        Returns:
            Modified code
        """
        diff = self.parse_diff(diff_text)
        
        if not diff.blocks:
            return original_code
        
        modified_code = original_code
        
        # Apply blocks in order
        for block in diff.blocks:
            if block.diff_type == DiffType.SEARCH_REPLACE:
                modified_code = self._apply_search_replace(
                    modified_code, block.search_content, block.replace_content
                )
        
        return modified_code
    
    def _apply_search_replace(self, code: str, search_content: str, 
                            replace_content: str) -> str:
        """
        Apply a single search/replace operation.
        
        Args:
            code: Code to modify
            search_content: Content to search for
            replace_content: Content to replace with
        
        Returns:
            Modified code
        """
        # Handle exact match first
        if search_content in code:
            return code.replace(search_content, replace_content, 1)
        
        # Handle fuzzy matching for slight variations
        lines = code.split('\n')
        search_lines = search_content.split('\n')
        
        # Find best match
        best_match_start = -1
        best_match_score = 0
        
        for i in range(len(lines) - len(search_lines) + 1):
            score = self._calculate_match_score(
                lines[i:i+len(search_lines)], search_lines
            )
            if score > best_match_score:
                best_match_score = score
                best_match_start = i
        
        # Apply replacement if good match found
        if best_match_score > 0.8:  # Threshold for fuzzy matching
            new_lines = lines[:best_match_start]
            new_lines.extend(replace_content.split('\n'))
            new_lines.extend(lines[best_match_start + len(search_lines):])
            return '\n'.join(new_lines)
        
        # If no good match, return original code
        return code
    
    def _calculate_match_score(self, code_lines: List[str], 
                              search_lines: List[str]) -> float:
        """Calculate match score between code lines and search lines."""
        if len(code_lines) != len(search_lines):
            return 0.0
        
        matches = 0
        for code_line, search_line in zip(code_lines, search_lines):
            # Strip whitespace for comparison
            if code_line.strip() == search_line.strip():
                matches += 1
        
        return matches / len(search_lines)
    
    def generate_diff(self, original_code: str, modified_code: str,
                     context: Dict[str, Any] = None) -> str:
        """
        Generate a diff between original and modified code.
        
        Args:
            original_code: Original code
            modified_code: Modified code
            context: Additional context
        
        Returns:
            Diff text in SEARCH/REPLACE format
        """
        if original_code == modified_code:
            return ""
        
        # Simple diff generation - can be enhanced with more sophisticated algorithms
        diff_blocks = []
        
        # For now, generate a single block with the entire change
        # This can be enhanced to generate more granular diffs
        diff_text = f"""<<<<<<< SEARCH
{original_code}
=======
{modified_code}
>>>>>>> REPLACE"""
        
        return diff_text
    
    def validate_diff(self, diff_text: str) -> Tuple[bool, List[str]]:
        """
        Validate a diff text for correctness.
        
        Args:
            diff_text: Diff text to validate
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check for proper SEARCH/REPLACE format
        matches = self.diff_pattern.findall(diff_text)
        if not matches:
            errors.append("No valid SEARCH/REPLACE blocks found")
        
        # Validate each block
        for i, (search_content, replace_content) in enumerate(matches):
            if not search_content.strip():
                errors.append(f"Block {i}: Empty search content")
            
            # Check for nested diff markers
            if '<<<<<<< SEARCH' in search_content or '>>>>>>> REPLACE' in search_content:
                errors.append(f"Block {i}: Nested diff markers in search content")
            
            if '<<<<<<< SEARCH' in replace_content or '>>>>>>> REPLACE' in replace_content:
                errors.append(f"Block {i}: Nested diff markers in replace content")
        
        return len(errors) == 0, errors
    
    def extract_changes(self, diff_text: str) -> List[Dict[str, Any]]:
        """
        Extract structured information about changes in a diff.
        
        Args:
            diff_text: Diff text to analyze
        
        Returns:
            List of change dictionaries
        """
        changes = []
        diff = self.parse_diff(diff_text)
        
        for block in diff.blocks:
            change = {
                "type": block.diff_type.value,
                "search_content": block.search_content,
                "replace_content": block.replace_content,
                "search_lines": len(block.search_content.split('\n')),
                "replace_lines": len(block.replace_content.split('\n')),
                "line_change": len(block.replace_content.split('\n')) - len(block.search_content.split('\n')),
                "metadata": block.metadata
            }
            changes.append(change)
        
        return changes
    
    def merge_diffs(self, diff1: str, diff2: str) -> str:
        """
        Merge two diffs into a single diff.
        
        Args:
            diff1: First diff
            diff2: Second diff
        
        Returns:
            Merged diff
        """
        # Parse both diffs
        parsed_diff1 = self.parse_diff(diff1)
        parsed_diff2 = self.parse_diff(diff2)
        
        # Combine blocks
        merged_blocks = parsed_diff1.blocks + parsed_diff2.blocks
        
        # Generate merged diff text
        diff_text = ""
        for block in merged_blocks:
            diff_text += f"""<<<<<<< SEARCH
{block.search_content}
=======
{block.replace_content}
>>>>>>> REPLACE

"""
        
        return diff_text.strip()
    
    def get_diff_statistics(self, diff_text: str) -> Dict[str, Any]:
        """
        Get statistics about a diff.
        
        Args:
            diff_text: Diff text to analyze
        
        Returns:
            Dictionary with diff statistics
        """
        changes = self.extract_changes(diff_text)
        
        total_additions = sum(change["replace_lines"] for change in changes)
        total_deletions = sum(change["search_lines"] for change in changes)
        net_change = total_additions - total_deletions
        
        return {
            "num_blocks": len(changes),
            "total_additions": total_additions,
            "total_deletions": total_deletions,
            "net_change": net_change,
            "changes": changes
        }
