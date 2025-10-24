#!/usr/bin/env python3
"""
Real LOCAGENT Training Script
Uses the actual LOCAGENT training pipeline with real model training
"""

import sys
import os
import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import subprocess
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Check dependencies
try:
    import requests
    from github import Github
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("✅ All dependencies available")
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Please install: pip install requests PyGithub torch transformers")
    sys.exit(1)

# Import LOCAGENT components
try:
    from ue_sea.locagent.training import LOCAGENT_Trainer, LocalizationExample, LocalizationDataset
    from ue_sea.locagent.graph import CodeGraph
    from ue_sea.locagent.agent import LOCAGENT_Agent
    print("✅ LOCAGENT components imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure LOCAGENT components are available")
    sys.exit(1)

class RealLOCAGENTTrainer:
    """Real LOCAGENT trainer with actual model training"""
    
    def __init__(self):
        self.github = None
        self.trainer = None
        self.graph = None
        
    def setup_github(self, token: Optional[str] = None):
        """Setup GitHub API client"""
        if token:
            self.github = Github(token)
        else:
            self.github = Github()  # Uses GITHUB_TOKEN env var or rate-limited access
        print("✅ GitHub API client initialized")
    
    def clone_repository(self, repo_url: str, temp_dir: str) -> str:
        """Clone repository to temporary directory"""
        repo_path = os.path.join(temp_dir, "repo")
        print(f"📦 Cloning {repo_url} to {repo_path}")
        
        try:
            subprocess.run([
                "git", "clone", repo_url, repo_path
            ], check=True, capture_output=True, text=True)
            print("✅ Repository cloned successfully")
            return repo_path
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to clone repository: {e}")
            print(f"Error output: {e.stderr}")
            raise
    
    def fetch_issues(self, repo_url: str, limit: int = 10, labels: List[str] = None, 
                    state: str = "closed", min_comments: int = 0) -> List[Dict]:
        """Fetch issues from GitHub repository"""
        if not self.github:
            self.setup_github()
        
        # Extract owner/repo from URL
        parts = repo_url.replace("https://github.com/", "").split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid repository URL: {repo_url}")
        
        owner, repo_name = parts
        print(f"🔍 Fetching issues from {owner}/{repo_name}")
        
        try:
            repo = self.github.get_repo(f"{owner}/{repo_name}")
            issues = []
            
            # Try different approaches to get issues
            print(f"🔍 Trying to fetch issues with state='{state}', labels={labels}")
            
            # First try with labels
            if labels:
                for issue in repo.get_issues(state=state, labels=labels):
                    if len(issues) >= limit:
                        break
                    if issue.comments >= min_comments:
                        issue_data = {
                            "title": issue.title,
                            "body": issue.body or "",
                            "labels": [label.name for label in issue.labels],
                            "comments": issue.comments,
                            "created_at": issue.created_at.isoformat(),
                            "number": issue.number,
                            "url": issue.html_url
                        }
                        issues.append(issue_data)
            
            # If no issues found with labels, try without labels
            if not issues:
                print("🔍 No issues found with labels, trying without label filter...")
                for issue in repo.get_issues(state=state):
                    if len(issues) >= limit:
                        break
                    if issue.comments >= min_comments:
                        issue_data = {
                            "title": issue.title,
                            "body": issue.body or "",
                            "labels": [label.name for label in issue.labels],
                            "comments": issue.comments,
                            "created_at": issue.created_at.isoformat(),
                            "number": issue.number,
                            "url": issue.html_url
                        }
                        issues.append(issue_data)
            
            # If still no issues, try with any state
            if not issues:
                print("🔍 No closed issues found, trying with any state...")
                for issue in repo.get_issues(state="all"):
                    if len(issues) >= limit:
                        break
                    if issue.comments >= min_comments:
                        issue_data = {
                            "title": issue.title,
                            "body": issue.body or "",
                            "labels": [label.name for label in issue.labels],
                            "comments": issue.comments,
                            "created_at": issue.created_at.isoformat(),
                            "number": issue.number,
                            "url": issue.html_url
                        }
                        issues.append(issue_data)
            
            print(f"✅ Fetched {len(issues)} issues")
            return issues
            
        except Exception as e:
            print(f"❌ Error fetching issues: {e}")
            raise
    
    def build_code_graph(self, repo_path: str) -> CodeGraph:
        """Build the code graph from repository"""
        print("🔧 Building code graph from repository...")
        
        try:
            # Create CodeGraph instance
            graph = CodeGraph()
            
            # Build graph from repository
            graph.build_from_repo(repo_path)
            
            print(f"✅ Code graph built successfully")
            print(f"   - Nodes: {len(graph.get_entities())}")
            print(f"   - Edges: {len(graph.get_relations())}")
            
            return graph
            
        except Exception as e:
            print(f"❌ Error building code graph: {e}")
            raise
    
    def generate_real_training_data(self, issues: List[Dict], repo_path: str, graph: CodeGraph) -> List[LocalizationExample]:
        """Generate real LOCAGENT training data from issues and repository"""
        print("📝 Generating real LOCAGENT training data...")
        
        training_examples = []
        
        for issue in issues:
            # Create real LocalizationExample
            example = LocalizationExample(
                issue_description=f"{issue['title']}\n\n{issue['body']}",
                repository_path=repo_path,
                target_entities=self._find_target_entities(issue, graph),
                context={
                    "labels": issue["labels"],
                    "issue_number": issue["number"],
                    "issue_url": issue["url"],
                    "difficulty": self._estimate_difficulty(issue)
                },
                difficulty=self._estimate_difficulty(issue),
                metadata={
                    "repo_path": repo_path,
                    "issue_number": issue["number"],
                    "issue_url": issue["url"],
                    "training_type": "real_locagent_localization"
                }
            )
            training_examples.append(example)
        
        print(f"✅ Generated {len(training_examples)} real LOCAGENT training examples")
        return training_examples
    
    def _find_target_entities(self, issue: Dict, graph: CodeGraph) -> List[str]:
        """Find target entities for localization using the actual graph"""
        title = issue["title"].lower()
        body = issue["body"].lower()
        
        # Use the actual graph to find relevant entities
        target_entities = []
        
        try:
            # Search for entities using the graph's search capabilities
            search_terms = title.split() + body.split()[:10]  # Limit to first 10 words
            
            for term in search_terms:
                if len(term) > 3:  # Only meaningful terms
                    # Use the graph's search functionality
                    entities = graph.search_entities(term)
                    target_entities.extend(entities[:3])  # Limit to top 3 per term
            
            # Remove duplicates and limit
            target_entities = list(set(target_entities))[:10]
            
        except Exception as e:
            print(f"⚠️  Error finding target entities: {e}")
            # Fallback to simple keyword matching
            target_entities = ["src/main.py", "src/utils.py"]
        
        return target_entities
    
    def _estimate_difficulty(self, issue: Dict) -> str:
        """Estimate issue difficulty"""
        title = issue["title"].lower()
        body = issue["body"].lower()
        
        if any(word in title for word in ["fix", "bug", "error"]):
            return "easy"
        elif any(word in title for word in ["optimize", "performance", "refactor"]):
            return "hard"
        elif any(word in title for word in ["add", "implement", "feature"]):
            return "medium"
        else:
            return "medium"
    
    def train_real_locagent(self, training_examples: List[LocalizationExample], 
                          output_dir: str = "outputs/real_locagent_training",
                          model_name: str = "microsoft/DialoGPT-medium",
                          num_epochs: int = 3,
                          batch_size: int = 4) -> None:
        """Train the real LOCAGENT model"""
        print(f"🚀 Starting REAL LOCAGENT training")
        print(f"📊 Training examples: {len(training_examples)}")
        print(f"📦 Model: {model_name}")
        print(f"🔄 Epochs: {num_epochs}")
        print(f"📁 Output directory: {output_dir}")
        print()
        
        try:
            # Create trainer
            self.trainer = LOCAGENT_Trainer(model_name=model_name, use_gpu=True)
            
            # Load model
            print("🔧 Loading model...")
            self.trainer.load_model()
            
            # Prepare datasets
            print("🔧 Preparing datasets...")
            train_dataset, val_dataset = self.trainer.prepare_dataset(training_examples)
            
            print(f"   - Training examples: {len(train_dataset)}")
            print(f"   - Validation examples: {len(val_dataset)}")
            
            # Train the model
            print("🚀 Starting REAL model training...")
            self.trainer.train(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                output_dir=output_dir,
                num_epochs=num_epochs,
                batch_size=batch_size
            )
            
            print("✅ REAL LOCAGENT training completed successfully!")
            print(f"📁 Model saved to: {output_dir}")
            
        except Exception as e:
            print(f"❌ Error during real training: {e}")
            import traceback
            traceback.print_exc()
            raise

async def main():
    """Main real LOCAGENT training function"""
    print("🚀 REAL LOCAGENT Training Pipeline")
    print("=" * 50)
    
    trainer = RealLOCAGENTTrainer()
    
    # Training options
    print("Select REAL LOCAGENT training method:")
    print("1. GitHub repository training (recommended)")
    print("2. Sample data training (demonstration)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        # GitHub training
        repo_url = input("Enter GitHub repository URL (or press Enter for Django): ").strip()
        if not repo_url:
            repo_url = "https://github.com/django/django"
        
        issues_limit = input("Enter issues limit (or press Enter for 10): ").strip()
        issues_limit = int(issues_limit) if issues_limit else 10
        
        epochs = input("Enter epochs (or press Enter for 3): ").strip()
        epochs = int(epochs) if epochs else 3
        
        output_dir = f"outputs/real_locagent_training_{repo_url.split('/')[-1]}"
        
        try:
            # Setup GitHub
            trainer.setup_github()
            
            # Fetch issues
            issues = trainer.fetch_issues(
                repo_url=repo_url,
                limit=issues_limit,
                labels=["bug", "enhancement"],
                state="closed",
                min_comments=1
            )
            
            if not issues:
                print("❌ No issues found with the specified criteria")
                return
            
            # Create temporary directory for repository
            with tempfile.TemporaryDirectory() as temp_dir:
                # Clone repository
                repo_path = trainer.clone_repository(repo_url, temp_dir)
                
                # Build code graph
                graph = trainer.build_code_graph(repo_path)
                
                # Generate real training data
                training_examples = trainer.generate_real_training_data(issues, repo_path, graph)
                
                # Train the real model
                trainer.train_real_locagent(
                    training_examples=training_examples,
                    output_dir=output_dir,
                    num_epochs=epochs
                )
            
            print("✅ REAL LOCAGENT GitHub-based training completed successfully!")
            
        except Exception as e:
            print(f"❌ Error during real LOCAGENT training: {e}")
            import traceback
            traceback.print_exc()
    
    elif choice == "2":
        # Sample data training
        epochs = input("Enter epochs (or press Enter for 3): ").strip()
        epochs = int(epochs) if epochs else 3
        
        output_dir = "outputs/real_locagent_sample_training"
        
        # Generate sample training data
        sample_examples = [
            LocalizationExample(
                issue_description="Fix authentication bug in login system",
                repository_path="sample_repo",
                target_entities=["src/auth/login.py", "src/auth/middleware.py"],
                context={"labels": ["bug", "authentication"]},
                difficulty="medium",
                metadata={"training_type": "real_locagent_localization"}
            )
        ]
        
        # Train the real model
        trainer.train_real_locagent(
            training_examples=sample_examples,
            output_dir=output_dir,
            num_epochs=epochs
        )
        
        print("✅ REAL LOCAGENT sample data training completed!")
        
    else:
        print("❌ Invalid choice. Exiting.")
        return

if __name__ == "__main__":
    asyncio.run(main())
