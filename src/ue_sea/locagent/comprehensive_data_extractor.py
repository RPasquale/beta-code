"""
Comprehensive Data Extractor for LOCAGENT Training

Extracts MAXIMUM training data from codebases using ALL available sources:
- GitHub issues, PRs, commits, discussions
- Code comments, docstrings, TODOs
- Test files and their descriptions
- Documentation and README files
- Code review comments
- Stack Overflow questions (if linked)
- And much more!
"""

import os
import re
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
import subprocess
import ast
import difflib

try:
    from github import Github
    from github.GithubException import GithubException
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class TrainingExample:
    """Comprehensive training example from any source."""
    issue_description: str
    repository_path: str
    target_entities: List[str]
    source_type: str  # "issue", "pr", "commit", "comment", "test", "doc", etc.
    source_id: str
    context: Dict[str, Any] = field(default_factory=dict)
    difficulty: str = "medium"
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ComprehensiveDataExtractor:
    """Extracts training data from ALL possible sources in a codebase."""
    
    def __init__(self, repo_path: str, github_token: Optional[str] = None):
        self.repo_path = Path(repo_path)
        self.github_token = github_token
        self.github = None
        self.repo = None
        self.logger = logging.getLogger(__name__)
        
        if github_token and GITHUB_AVAILABLE:
            self.github = Github(github_token)
        
        # Track what we've already processed
        self.processed_entities: Set[str] = set()
        self.training_examples: List[TrainingExample] = []
    
    async def extract_all_training_data(self, 
                                      max_examples: int = 1000,
                                      include_synthetic: bool = True) -> List[TrainingExample]:
        """Extract training data from ALL possible sources."""
        self.logger.info("🚀 Starting comprehensive data extraction...")
        
        # 1. GitHub sources (if available)
        if self.github:
            await self._extract_github_sources()
        
        # 2. Code-based sources
        await self._extract_code_sources()
        
        # 3. Documentation sources
        await self._extract_documentation_sources()
        
        # 4. Test-based sources
        await self._extract_test_sources()
        
        # 5. Comment-based sources
        await self._extract_comment_sources()
        
        # 6. Git history sources
        await self._extract_git_history_sources()
        
        # 7. Synthetic sources (if requested)
        if include_synthetic:
            await self._generate_synthetic_sources(max_examples - len(self.training_examples))
        
        self.logger.info(f"✅ Extracted {len(self.training_examples)} training examples")
        return self.training_examples
    
    async def _extract_github_sources(self):
        """Extract from GitHub issues, PRs, discussions, etc."""
        if not self.github:
            return
        
        try:
            # Get repository
            repo_name = self._get_repo_name_from_path()
            if not repo_name:
                return
            
            self.repo = self.github.get_repo(repo_name)
            
            # Extract issues
            await self._extract_issues()
            
            # Extract pull requests
            await self._extract_pull_requests()
            
            # Extract discussions (if available)
            await self._extract_discussions()
            
            # Extract releases and tags
            await self._extract_releases()
            
        except Exception as e:
            self.logger.error(f"Error extracting GitHub sources: {e}")
    
    async def _extract_issues(self):
        """Extract training examples from GitHub issues."""
        if not self.repo:
            return
        
        try:
            # Get all issues (open and closed)
            issues = self.repo.get_issues(state="all", sort="created", direction="desc")
            
            for issue in issues:
                if len(self.training_examples) >= 1000:  # Limit for performance
                    break
                
                # Extract issue content
                issue_text = f"{issue.title}\n{issue.body or ''}"
                
                # Find related code entities
                target_entities = await self._find_related_entities(issue_text)
                
                if target_entities:
                    example = TrainingExample(
                        issue_description=issue_text,
                        repository_path=str(self.repo_path),
                        target_entities=target_entities,
                        source_type="issue",
                        source_id=str(issue.number),
                        context={
                            "labels": [label.name for label in issue.labels],
                            "comments": issue.comments,
                            "created_at": issue.created_at.isoformat(),
                            "url": issue.html_url
                        },
                        difficulty=self._classify_difficulty(issue_text, target_entities),
                        confidence=0.9
                    )
                    self.training_examples.append(example)
        
        except Exception as e:
            self.logger.error(f"Error extracting issues: {e}")
    
    async def _extract_pull_requests(self):
        """Extract training examples from pull requests."""
        if not self.repo:
            return
        
        try:
            prs = self.repo.get_pulls(state="all", sort="created", direction="desc")
            
            for pr in prs:
                if len(self.training_examples) >= 1000:
                    break
                
                # Extract PR content
                pr_text = f"{pr.title}\n{pr.body or ''}"
                
                # Find related code entities
                target_entities = await self._find_related_entities(pr_text)
                
                if target_entities:
                    example = TrainingExample(
                        issue_description=pr_text,
                        repository_path=str(self.repo_path),
                        target_entities=target_entities,
                        source_type="pull_request",
                        source_id=str(pr.number),
                        context={
                            "labels": [label.name for label in pr.labels],
                            "comments": pr.comments,
                            "created_at": pr.created_at.isoformat(),
                            "url": pr.html_url,
                            "files_changed": pr.changed_files
                        },
                        difficulty=self._classify_difficulty(pr_text, target_entities),
                        confidence=0.8
                    )
                    self.training_examples.append(example)
        
        except Exception as e:
            self.logger.error(f"Error extracting PRs: {e}")
    
    async def _extract_code_sources(self):
        """Extract training examples from code itself."""
        self.logger.info("🔍 Extracting from code sources...")
        
        # Extract from function docstrings
        await self._extract_docstring_examples()
        
        # Extract from class docstrings
        await self._extract_class_examples()
        
        # Extract from module docstrings
        await self._extract_module_examples()
        
        # Extract from TODO/FIXME comments
        await self._extract_todo_examples()
    
    async def _extract_docstring_examples(self):
        """Extract examples from function docstrings."""
        for py_file in self.repo_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.docstring:
                        # Create training example from docstring
                        docstring = node.docstring
                        
                        # Find the function in the file
                        target_entities = [f"{py_file.relative_to(self.repo_path)}:{node.name}"]
                        
                        example = TrainingExample(
                            issue_description=f"Understand function: {node.name}\n{docstring}",
                            repository_path=str(self.repo_path),
                            target_entities=target_entities,
                            source_type="docstring",
                            source_id=f"{py_file.name}:{node.name}",
                            context={
                                "file": str(py_file.relative_to(self.repo_path)),
                                "function": node.name,
                                "line": node.lineno
                            },
                            difficulty="easy",
                            confidence=0.7
                        )
                        self.training_examples.append(example)
            
            except Exception as e:
                self.logger.debug(f"Error parsing {py_file}: {e}")
    
    async def _extract_test_sources(self):
        """Extract training examples from test files."""
        self.logger.info("🧪 Extracting from test sources...")
        
        # Find test files
        test_files = []
        for pattern in ["test_*.py", "*_test.py", "tests/*.py"]:
            test_files.extend(self.repo_path.rglob(pattern))
        
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract test class and method docstrings
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if node.name.startswith('test_') and node.docstring:
                            # Create training example from test
                            test_description = f"Test: {node.name}\n{node.docstring}"
                            
                            # Find the code being tested
                            target_entities = await self._find_tested_entities(test_file, node.name)
                            
                            example = TrainingExample(
                                issue_description=test_description,
                                repository_path=str(self.repo_path),
                                target_entities=target_entities,
                                source_type="test",
                                source_id=f"{test_file.name}:{node.name}",
                                context={
                                    "test_file": str(test_file.relative_to(self.repo_path)),
                                    "test_method": node.name,
                                    "line": node.lineno
                                },
                                difficulty="medium",
                                confidence=0.8
                            )
                            self.training_examples.append(example)
            
            except Exception as e:
                self.logger.debug(f"Error parsing test file {test_file}: {e}")
    
    async def _extract_comment_sources(self):
        """Extract training examples from code comments."""
        self.logger.info("💬 Extracting from comment sources...")
        
        comment_patterns = [
            r'# TODO: (.+)',
            r'# FIXME: (.+)',
            r'# BUG: (.+)',
            r'# HACK: (.+)',
            r'# NOTE: (.+)',
            r'# WARNING: (.+)',
            r'# XXX: (.+)',
        ]
        
        for py_file in self.repo_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for i, line in enumerate(lines):
                    for pattern in comment_patterns:
                        match = re.search(pattern, line, re.IGNORECASE)
                        if match:
                            comment_text = match.group(1).strip()
                            
                            # Find related entities in the same function/class
                            target_entities = await self._find_nearby_entities(py_file, i)
                            
                            if target_entities:
                                example = TrainingExample(
                                    issue_description=f"Code comment: {comment_text}",
                                    repository_path=str(self.repo_path),
                                    target_entities=target_entities,
                                    source_type="comment",
                                    source_id=f"{py_file.name}:{i}",
                                    context={
                                        "file": str(py_file.relative_to(self.repo_path)),
                                        "line": i + 1,
                                        "comment_type": pattern.split(':')[0].replace('# ', '')
                                    },
                                    difficulty="easy",
                                    confidence=0.6
                                )
                                self.training_examples.append(example)
            
            except Exception as e:
                self.logger.debug(f"Error processing comments in {py_file}: {e}")
    
    async def _extract_git_history_sources(self):
        """Extract training examples from git history."""
        self.logger.info("📜 Extracting from git history...")
        
        try:
            # Get recent commits
            result = subprocess.run(
                ["git", "log", "--oneline", "-n", "100"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                commits = result.stdout.strip().split('\n')
                
                for commit_line in commits:
                    if not commit_line:
                        continue
                    
                    commit_hash = commit_line.split()[0]
                    commit_message = ' '.join(commit_line.split()[1:])
                    
                    # Find related entities
                    target_entities = await self._find_related_entities(commit_message)
                    
                    if target_entities:
                        example = TrainingExample(
                            issue_description=f"Commit: {commit_message}",
                            repository_path=str(self.repo_path),
                            target_entities=target_entities,
                            source_type="commit",
                            source_id=commit_hash,
                            context={
                                "commit_hash": commit_hash,
                                "message": commit_message
                            },
                            difficulty="medium",
                            confidence=0.5
                        )
                        self.training_examples.append(example)
        
        except Exception as e:
            self.logger.error(f"Error extracting git history: {e}")
    
    async def _generate_synthetic_sources(self, count: int):
        """Generate synthetic training examples."""
        self.logger.info(f"🎭 Generating {count} synthetic examples...")
        
        # Get all Python files
        py_files = list(self.repo_path.rglob("*.py"))
        
        for i in range(min(count, len(py_files) * 2)):
            py_file = py_files[i % len(py_files)]
            
            # Generate synthetic issue
            synthetic_issue = self._generate_synthetic_issue(py_file)
            
            # Find related entities
            target_entities = await self._find_related_entities(synthetic_issue)
            
            example = TrainingExample(
                issue_description=synthetic_issue,
                repository_path=str(self.repo_path),
                target_entities=target_entities,
                source_type="synthetic",
                source_id=f"synthetic_{i}",
                context={"synthetic": True},
                difficulty="medium",
                confidence=0.3
            )
            self.training_examples.append(example)
    
    def _generate_synthetic_issue(self, py_file: Path) -> str:
        """Generate a synthetic issue based on file content."""
        synthetic_issues = [
            f"Improve performance in {py_file.name}",
            f"Add error handling to {py_file.name}",
            f"Refactor code in {py_file.name}",
            f"Add documentation to {py_file.name}",
            f"Fix potential bugs in {py_file.name}",
            f"Optimize {py_file.name}",
            f"Add tests for {py_file.name}",
            f"Update {py_file.name} to use new API",
            f"Make {py_file.name} more maintainable",
            f"Add logging to {py_file.name}"
        ]
        
        import random
        return random.choice(synthetic_issues)
    
    async def _find_related_entities(self, text: str) -> List[str]:
        """Find code entities related to the given text."""
        entities = []
        
        # Simple keyword matching for now
        # In a real implementation, this would use the graph
        keywords = re.findall(r'\b[A-Z][a-zA-Z0-9_]*\b', text)
        
        for keyword in keywords:
            # Find files that might contain this keyword
            for py_file in self.repo_path.rglob("*.py"):
                if keyword.lower() in py_file.name.lower():
                    entities.append(str(py_file.relative_to(self.repo_path)))
        
        return entities[:5]  # Limit to 5 entities
    
    async def _find_tested_entities(self, test_file: Path, test_name: str) -> List[str]:
        """Find entities being tested by a test method."""
        entities = []
        
        # Simple heuristic: look for imports and function calls
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find import statements
            import_matches = re.findall(r'from\s+(\S+)\s+import', content)
            entities.extend(import_matches)
            
            # Find function calls
            func_matches = re.findall(r'(\w+)\(', content)
            entities.extend(func_matches)
        
        except Exception as e:
            self.logger.debug(f"Error finding tested entities: {e}")
        
        return entities[:3]  # Limit to 3 entities
    
    async def _find_nearby_entities(self, py_file: Path, line_num: int) -> List[str]:
        """Find entities near a specific line in a file."""
        entities = []
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Look for function/class definitions near the line
            start_line = max(0, line_num - 10)
            end_line = min(len(lines), line_num + 10)
            
            for i in range(start_line, end_line):
                line = lines[i]
                if re.match(r'^\s*(def|class)\s+\w+', line):
                    entities.append(f"{py_file.name}:{i+1}")
        
        except Exception as e:
            self.logger.debug(f"Error finding nearby entities: {e}")
        
        return entities[:2]  # Limit to 2 entities
    
    def _classify_difficulty(self, text: str, entities: List[str]) -> str:
        """Classify the difficulty of a training example."""
        if len(entities) <= 1:
            return "easy"
        elif len(entities) <= 3:
            return "medium"
        else:
            return "hard"
    
    def _get_repo_name_from_path(self) -> Optional[str]:
        """Extract repository name from local path."""
        # This is a simplified version - in practice, you'd need to
        # check git remote or other methods
        return None  # Placeholder
    
    async def _extract_documentation_sources(self):
        """Extract from documentation files."""
        self.logger.info("📚 Extracting from documentation...")
        
        doc_files = []
        for pattern in ["README.md", "*.md", "docs/*.md", "*.rst", "*.txt"]:
            doc_files.extend(self.repo_path.rglob(pattern))
        
        for doc_file in doc_files:
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract code blocks and examples
                code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', content, re.DOTALL)
                
                for i, code_block in enumerate(code_blocks):
                    if code_block.strip():
                        example = TrainingExample(
                            issue_description=f"Documentation example from {doc_file.name}",
                            repository_path=str(self.repo_path),
                            target_entities=[],  # Would need to parse code block
                            source_type="documentation",
                            source_id=f"{doc_file.name}:{i}",
                            context={
                                "doc_file": str(doc_file.relative_to(self.repo_path)),
                                "code_block": code_block[:200]  # Truncate
                            },
                            difficulty="easy",
                            confidence=0.4
                        )
                        self.training_examples.append(example)
            
            except Exception as e:
                self.logger.debug(f"Error processing doc file {doc_file}: {e}")
    
    async def _extract_discussions(self):
        """Extract from GitHub discussions (if available)."""
        # Placeholder for discussions extraction
        pass
    
    async def _extract_releases(self):
        """Extract from GitHub releases."""
        # Placeholder for releases extraction
        pass


async def extract_comprehensive_training_data(repo_path: str, 
                                           output_file: str,
                                           github_token: Optional[str] = None,
                                           max_examples: int = 1000) -> List[TrainingExample]:
    """
    Extract comprehensive training data from a repository.
    
    Args:
        repo_path: Path to the repository
        output_file: Path to save the training data
        github_token: GitHub token for API access
        max_examples: Maximum number of examples to extract
    
    Returns:
        List of training examples
    """
    extractor = ComprehensiveDataExtractor(repo_path, github_token)
    examples = await extractor.extract_all_training_data(max_examples)
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps({
                "issue_description": example.issue_description,
                "repository_path": example.repository_path,
                "target_entities": example.target_entities,
                "source_type": example.source_type,
                "source_id": example.source_id,
                "context": example.context,
                "difficulty": example.difficulty,
                "confidence": example.confidence,
                "metadata": example.metadata
            }) + '\n')
    
    return examples
