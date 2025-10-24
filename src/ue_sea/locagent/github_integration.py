"""
LOCAGENT GitHub Integration: Clone repositories and generate training data from GitHub.

Provides functionality to clone GitHub repositories, fetch issues, and generate
training data for LOCAGENT training.
"""

import os
import json
import tempfile
import subprocess
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import re
from urllib.parse import urlparse

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from github import Github
    PYTHON_GITHUB_AVAILABLE = True
except ImportError:
    PYTHON_GITHUB_AVAILABLE = False

from .entities import Entity, EntityType
from .training import LocalizationExample
from .graph import LOCAGENT_Graph
from .agent import LOCAGENT_Agent


@dataclass
class GitHubConfig:
    """Configuration for GitHub integration."""
    repo_url: str
    github_token: Optional[str] = None
    issues_limit: int = 100
    issues_labels: Optional[List[str]] = None
    issues_state: str = "open"  # "open", "closed", "all"
    include_prs: bool = False
    min_issue_comments: int = 1
    temp_dir: Optional[str] = None


class GitHubRepository:
    """GitHub repository handler."""
    
    def __init__(self, config: GitHubConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.temp_repo_path: Optional[Path] = None
        
        # Parse repository URL
        self.repo_owner, self.repo_name = self._parse_repo_url(config.repo_url)
        
        # Initialize GitHub API if available
        self.github_api = None
        if PYTHON_GITHUB_AVAILABLE and config.github_token:
            self.github_api = Github(config.github_token)
    
    def _parse_repo_url(self, repo_url: str) -> Tuple[str, str]:
        """Parse GitHub repository URL to extract owner and name."""
        # Handle different URL formats
        if repo_url.startswith("https://github.com/"):
            path = repo_url.replace("https://github.com/", "").rstrip("/")
        elif repo_url.startswith("git@github.com:"):
            path = repo_url.replace("git@github.com:", "").replace(".git", "")
        else:
            # Assume it's already in owner/repo format
            path = repo_url
        
        parts = path.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid repository URL format: {repo_url}")
        
        return parts[0], parts[1]
    
    async def clone_repository(self) -> Path:
        """Clone the GitHub repository to a temporary directory."""
        self.logger.info(f"Cloning repository: {self.repo_owner}/{self.repo_name}")
        
        # Create temporary directory
        if self.config.temp_dir:
            temp_dir = Path(self.config.temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
        else:
            temp_dir = Path(tempfile.mkdtemp())
        
        self.temp_repo_path = temp_dir / f"{self.repo_owner}_{self.repo_name}"
        
        # Clone repository
        clone_url = f"https://github.com/{self.repo_owner}/{self.repo_name}.git"
        try:
            result = subprocess.run([
                "git", "clone", clone_url, str(self.temp_repo_path)
            ], capture_output=True, text=True, check=True)
            
            self.logger.info(f"Repository cloned to: {self.temp_repo_path}")
            return self.temp_repo_path
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to clone repository: {e}")
            self.logger.error(f"Git output: {e.stderr}")
            raise RuntimeError(f"Failed to clone repository: {e}")
    
    def fetch_issues(self) -> List[Dict[str, Any]]:
        """Fetch issues from GitHub repository."""
        if not self.github_api:
            self.logger.warning("GitHub API not available, using fallback method")
            return self._fetch_issues_fallback()
        
        self.logger.info(f"Fetching issues from {self.repo_owner}/{self.repo_name}")
        
        try:
            repo = self.github_api.get_repo(f"{self.repo_owner}/{self.repo_name}")
            
            # Fetch issues
            issues = []
            for issue in repo.get_issues(
                state=self.config.issues_state,
                labels=self.config.issues_labels
            ):
                if len(issues) >= self.config.issues_limit:
                    break
                
                # Skip pull requests if not requested
                if not self.config.include_prs and issue.pull_request:
                    continue
                
                # Filter by minimum comments
                if issue.comments < self.config.min_issue_comments:
                    continue
                
                issue_data = {
                    "number": issue.number,
                    "title": issue.title,
                    "body": issue.body or "",
                    "state": issue.state,
                    "labels": [label.name for label in issue.labels],
                    "comments": issue.comments,
                    "created_at": issue.created_at.isoformat(),
                    "updated_at": issue.updated_at.isoformat(),
                    "user": issue.user.login if issue.user else None,
                    "assignee": issue.assignee.login if issue.assignee else None,
                    "url": issue.html_url
                }
                
                # Fetch issue comments
                comments = []
                for comment in issue.get_comments():
                    comments.append({
                        "user": comment.user.login if comment.user else None,
                        "body": comment.body,
                        "created_at": comment.created_at.isoformat()
                    })
                
                issue_data["comments_data"] = comments
                issues.append(issue_data)
            
            self.logger.info(f"Fetched {len(issues)} issues")
            return issues
            
        except Exception as e:
            self.logger.error(f"Error fetching issues: {e}")
            return self._fetch_issues_fallback()
    
    def _fetch_issues_fallback(self) -> List[Dict[str, Any]]:
        """Fallback method to fetch issues using GitHub API without authentication."""
        if not REQUESTS_AVAILABLE:
            self.logger.warning("No GitHub API access available")
            return []
        
        self.logger.info("Using fallback method to fetch issues")
        
        # Use GitHub REST API
        api_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/issues"
        params = {
            "state": self.config.issues_state,
            "per_page": min(self.config.issues_limit, 100),
            "page": 1
        }
        
        if self.config.issues_labels:
            params["labels"] = ",".join(self.config.issues_labels)
        
        try:
            response = requests.get(api_url, params=params)
            response.raise_for_status()
            
            issues = response.json()
            
            # Filter out pull requests if not requested
            if not self.config.include_prs:
                issues = [issue for issue in issues if "pull_request" not in issue]
            
            # Filter by minimum comments
            issues = [issue for issue in issues if issue.get("comments", 0) >= self.config.min_issue_comments]
            
            self.logger.info(f"Fetched {len(issues)} issues via REST API")
            return issues
            
        except Exception as e:
            self.logger.error(f"Error fetching issues via REST API: {e}")
            return []
    
    def extract_file_references(self, text: str) -> List[str]:
        """Extract file references from issue text."""
        # Common patterns for file references
        patterns = [
            r'`([^`]+\.(py|js|ts|tsx|java|cpp|c|h|go|rs|php|rb|swift|kt|scala|r|m|pl|sh|bash|zsh|fish|ps1|bat|cmd))`',  # Backticks
            r'([a-zA-Z0-9_/.-]+\.(py|js|ts|tsx|java|cpp|c|h|go|rs|php|rb|swift|kt|scala|r|m|pl|sh|bash|zsh|fish|ps1|bat|cmd))',  # Direct file names
            r'([a-zA-Z0-9_/.-]+/[a-zA-Z0-9_/.-]+\.(py|js|ts|tsx|java|cpp|c|h|go|rs|php|rb|swift|kt|scala|r|m|pl|sh|bash|zsh|fish|ps1|bat|cmd))',  # Paths with files
        ]
        
        file_refs = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    file_refs.add(match[0])
                else:
                    file_refs.add(match)
        
        return list(file_refs)
    
    def generate_training_examples(self, issues: List[Dict[str, Any]]) -> List[LocalizationExample]:
        """Generate training examples from GitHub issues."""
        self.logger.info(f"Generating training examples from {len(issues)} issues")
        
        examples = []
        
        for issue in issues:
            # Combine issue title and body
            issue_text = f"{issue.get('title', '')}\n{issue.get('body', '')}"
            
            # Add comments
            for comment in issue.get('comments_data', []):
                issue_text += f"\n{comment.get('body', '')}"
            
            # Extract file references
            file_refs = self.extract_file_references(issue_text)
            
            if file_refs:
                # Create training example
                example = LocalizationExample(
                    issue_description=issue.get('title', ''),
                    repository_path=str(self.temp_repo_path) if self.temp_repo_path else "",
                    target_entities=file_refs,
                    context={
                        "issue_number": issue.get('number'),
                        "issue_url": issue.get('url'),
                        "labels": issue.get('labels', []),
                        "state": issue.get('state'),
                        "user": issue.get('user'),
                        "created_at": issue.get('created_at'),
                        "full_text": issue_text
                    },
                    difficulty="medium"  # Default difficulty
                )
                examples.append(example)
        
        self.logger.info(f"Generated {len(examples)} training examples")
        return examples
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_repo_path and self.temp_repo_path.exists():
            import shutil
            shutil.rmtree(self.temp_repo_path)
            self.logger.info(f"Cleaned up temporary repository: {self.temp_repo_path}")


async def github_train(
    repo_url: str,
    output_dir: str,
    github_token: Optional[str] = None,
    issues_limit: int = 100,
    issues_labels: Optional[List[str]] = None,
    issues_state: str = "open",
    include_prs: bool = False,
    min_issue_comments: int = 1,
    model_name: str = "microsoft/DialoGPT-medium",
    num_epochs: int = 3,
    batch_size: int = 4,
    use_gpu: bool = True,
    temp_dir: Optional[str] = None
) -> None:
    """
    Train LOCAGENT on a GitHub repository.
    
    Args:
        repo_url: GitHub repository URL
        output_dir: Directory to save trained model
        github_token: GitHub API token (optional)
        issues_limit: Maximum number of issues to fetch
        issues_labels: Filter issues by labels
        issues_state: Issue state filter ("open", "closed", "all")
        include_prs: Include pull requests
        min_issue_comments: Minimum number of comments per issue
        model_name: Base model name for fine-tuning
        num_epochs: Number of training epochs
        batch_size: Training batch size
        use_gpu: Enable GPU acceleration
        temp_dir: Temporary directory for cloning
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting GitHub training for: {repo_url}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Configure GitHub integration
    config = GitHubConfig(
        repo_url=repo_url,
        github_token=github_token,
        issues_limit=issues_limit,
        issues_labels=issues_labels,
        issues_state=issues_state,
        include_prs=include_prs,
        min_issue_comments=min_issue_comments,
        temp_dir=temp_dir
    )
    
    github_repo = GitHubRepository(config)
    
    try:
        # Step 1: Clone repository
        logger.info("Step 1: Cloning repository...")
        repo_path = await github_repo.clone_repository()
        
        # Step 2: Fetch issues
        logger.info("Step 2: Fetching issues...")
        issues = github_repo.fetch_issues()
        
        if not issues:
            logger.warning("No issues found, creating sample training data")
            from .data_loader import create_sample_dataset
            sample_data_path = output_path / "sample_training_data.jsonl"
            create_sample_dataset(str(sample_data_path), num_examples=50)
            training_data_path = str(sample_data_path)
        else:
            # Step 3: Generate training examples
            logger.info("Step 3: Generating training examples...")
            examples = github_repo.generate_training_examples(issues)
            
            # Save training data
            training_data_path = output_path / "github_training_data.jsonl"
            with open(training_data_path, 'w') as f:
                for example in examples:
                    f.write(json.dumps(example.__dict__) + '\n')
            
            logger.info(f"Saved {len(examples)} training examples to {training_data_path}")
        
        # Step 4: Build graph
        logger.info("Step 4: Building graph...")
        graph = LOCAGENT_Graph(use_gpu=use_gpu)
        await graph.build_from_repository(str(repo_path))
        
        # Step 5: Train agent
        logger.info("Step 5: Training agent...")
        from .training import train_locagent_agent
        
        await train_locagent_agent(
            graph=graph,
            training_data_path=str(training_data_path),
            output_dir=output_dir,
            model_name=model_name,
            num_epochs=num_epochs,
            batch_size=batch_size
        )
        
        logger.info("GitHub training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during GitHub training: {e}")
        raise
    
    finally:
        # Cleanup
        github_repo.cleanup()


def check_github_dependencies() -> Tuple[bool, List[str]]:
    """Check if GitHub integration dependencies are available."""
    missing = []
    
    if not REQUESTS_AVAILABLE:
        missing.append("requests")
    
    if not PYTHON_GITHUB_AVAILABLE:
        missing.append("PyGithub")
    
    return len(missing) == 0, missing
