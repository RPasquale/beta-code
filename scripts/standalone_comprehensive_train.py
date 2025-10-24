"""
Standalone Comprehensive LOCAGENT Training Script

Uses MAXIMUM training data from codebases by extracting from ALL possible sources.
This version directly imports locagent components without going through the main ue_sea package.
"""

import sys
import os
import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import ast
import subprocess
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    # Direct imports from locagent
    from ue_sea.locagent.comprehensive_data_extractor import (
        ComprehensiveDataExtractor, 
        TrainingExample
    )
    from ue_sea.locagent.training import LOCAGENT_Trainer, LocalizationExample, LocalizationDataset
    from ue_sea.locagent.graph import LOCAGENT_Graph
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
    import torch
    import numpy as np
    
    print("✅ All dependencies available")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure LOCAGENT components are available")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class StandaloneComprehensiveExtractor:
    """Standalone version of comprehensive data extractor."""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.training_examples: List[TrainingExample] = []
        self.logger = logging.getLogger(__name__)
    
    async def extract_all_training_data(self, max_examples: int = 1000) -> List[TrainingExample]:
        """Extract training data from ALL possible sources."""
        self.logger.info("🚀 Starting comprehensive data extraction...")
        
        # 1. Extract from code sources
        await self._extract_code_sources()
        
        # 2. Extract from test sources
        await self._extract_test_sources()
        
        # 3. Extract from comment sources
        await self._extract_comment_sources()
        
        # 4. Extract from git history
        await self._extract_git_history_sources()
        
        # 5. Generate synthetic sources
        await self._generate_synthetic_sources(max_examples - len(self.training_examples))
        
        self.logger.info(f"✅ Extracted {len(self.training_examples)} comprehensive training examples")
        return self.training_examples
    
    async def _extract_code_sources(self):
        """Extract from code docstrings and comments."""
        self.logger.info("🔍 Extracting from code sources...")
        
        for py_file in self.repo_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract function docstrings
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.docstring:
                        docstring = node.docstring
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
        """Extract from test files."""
        self.logger.info("🧪 Extracting from test sources...")
        
        test_files = []
        for pattern in ["test_*.py", "*_test.py", "tests/*.py"]:
            test_files.extend(self.repo_path.rglob(pattern))
        
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if node.name.startswith('test_') and node.docstring:
                            test_description = f"Test: {node.name}\n{node.docstring}"
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
        """Extract from code comments."""
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
        """Extract from git history."""
        self.logger.info("📜 Extracting from git history...")
        
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "-n", "50"],
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
        
        py_files = list(self.repo_path.rglob("*.py"))
        
        for i in range(min(count, len(py_files) * 2)):
            py_file = py_files[i % len(py_files)]
            synthetic_issue = self._generate_synthetic_issue(py_file)
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
        return random.choice(synthetic_issues)
    
    async def _find_related_entities(self, text: str) -> List[str]:
        """Find code entities related to the given text."""
        entities = []
        keywords = re.findall(r'\b[A-Z][a-zA-Z0-9_]*\b', text)
        
        for keyword in keywords:
            for py_file in self.repo_path.rglob("*.py"):
                if keyword.lower() in py_file.name.lower():
                    entities.append(str(py_file.relative_to(self.repo_path)))
        
        return entities[:5]
    
    async def _find_tested_entities(self, test_file: Path, test_name: str) -> List[str]:
        """Find entities being tested by a test method."""
        entities = []
        
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            import_matches = re.findall(r'from\s+(\S+)\s+import', content)
            entities.extend(import_matches)
            
            func_matches = re.findall(r'(\w+)\(', content)
            entities.extend(func_matches)
        
        except Exception as e:
            self.logger.debug(f"Error finding tested entities: {e}")
        
        return entities[:3]
    
    async def _find_nearby_entities(self, py_file: Path, line_num: int) -> List[str]:
        """Find entities near a specific line in a file."""
        entities = []
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            start_line = max(0, line_num - 10)
            end_line = min(len(lines), line_num + 10)
            
            for i in range(start_line, end_line):
                line = lines[i]
                if re.match(r'^\s*(def|class)\s+\w+', line):
                    entities.append(f"{py_file.name}:{i+1}")
        
        except Exception as e:
            self.logger.debug(f"Error finding nearby entities: {e}")
        
        return entities[:2]


async def main():
    print("🚀 STANDALONE COMPREHENSIVE LOCAGENT Training Pipeline")
    print("==================================================")
    print("This will extract training data from ALL possible sources!")
    print()
    
    # Configuration
    repo_url = input("Enter GitHub repository URL (or press Enter for Django): ").strip()
    if not repo_url:
        repo_url = "https://github.com/django/django"
    
    max_examples = int(input("Enter max examples to extract (default 1000): ") or "1000")
    num_epochs = int(input("Enter training epochs (default 3): ") or "3")
    
    output_dir = f"outputs/standalone_comprehensive_locagent_{Path(repo_url.split('/')[-1]).stem}"
    
    print(f"\n📊 Configuration:")
    print(f"   Repository: {repo_url}")
    print(f"   Max examples: {max_examples}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Output: {output_dir}")
    print()
    
    # Step 1: Clone repository
    print("📥 Step 1: Cloning repository...")
    temp_repo_path = Path(os.environ.get("TEMP", "/tmp")) / f"tmp_standalone_repo_{os.getpid()}"
    
    try:
        subprocess.run(["git", "clone", repo_url, str(temp_repo_path)], check=True)
        print(f"✅ Repository cloned to {temp_repo_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to clone repository: {e}")
        return
    
    # Step 2: Extract comprehensive training data
    print("\n🔍 Step 2: Extracting comprehensive training data...")
    print("   This will extract from ALL possible sources:")
    print("   - Code docstrings and comments")
    print("   - Test files and descriptions")
    print("   - Git history and commit messages")
    print("   - Synthetic examples")
    print("   - And much more!")
    
    try:
        extractor = StandaloneComprehensiveExtractor(str(temp_repo_path))
        training_examples = await extractor.extract_all_training_data(max_examples)
        
        print(f"✅ Extracted {len(training_examples)} comprehensive training examples")
        
        # Show breakdown by source type
        source_counts = {}
        for example in training_examples:
            source_type = example.source_type
            source_counts[source_type] = source_counts.get(source_type, 0) + 1
        
        print("\n📊 Training data breakdown:")
        for source_type, count in sorted(source_counts.items()):
            print(f"   {source_type}: {count} examples")
        
    except Exception as e:
        print(f"❌ Error extracting training data: {e}")
        return
    
    # Step 3: Convert to LocalizationExample format
    print("\n🔄 Step 3: Converting to training format...")
    
    localization_examples = []
    for example in training_examples:
        loc_example = LocalizationExample(
            issue_description=example.issue_description,
            repository_path=example.repository_path,
            target_entities=example.target_entities,
            context=example.context,
            difficulty=example.difficulty,
            metadata={
                "source_type": example.source_type,
                "source_id": example.source_id,
                "confidence": example.confidence,
                **example.metadata
            }
        )
        localization_examples.append(loc_example)
    
    print(f"✅ Converted {len(localization_examples)} examples to training format")
    
    # Step 4: Build code graph
    print("\n🏗️ Step 4: Building code graph...")
    
    try:
        graph = LOCAGENT_Graph(use_gpu=True)
        await graph.build_from_repository(str(temp_repo_path))
        print(f"✅ Built graph with {len(graph.get_all_entities())} entities")
    except Exception as e:
        print(f"❌ Error building graph: {e}")
        return
    
    # Step 5: Train the model
    print("\n🚀 Step 5: Training LOCAGENT model...")
    
    try:
        # Initialize trainer
        trainer = LOCAGENT_Trainer(model_name="microsoft/DialoGPT-medium")
        trainer.load_model()
        
        # Prepare dataset
        train_dataset, val_dataset = trainer.prepare_dataset(localization_examples)
        
        print(f"📊 Training examples: {len(train_dataset)}")
        print(f"📊 Validation examples: {len(val_dataset)}")
        print(f"🔄 Epochs: {num_epochs}")
        print(f"📦 Batch size: {trainer.batch_size}")
        print(f"📁 Output directory: {output_dir}")
        
        # Train the model
        trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_dir=output_dir,
            num_epochs=num_epochs,
            batch_size=trainer.batch_size
        )
        
        print("✅ Model training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return
    
    # Step 6: Save comprehensive training data
    print("\n💾 Step 6: Saving comprehensive training data...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training examples
    with open(Path(output_dir) / "standalone_training_data.jsonl", "w") as f:
        for example in training_examples:
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
            }) + "\n")
    
    # Save training summary
    training_summary = {
        "total_examples": len(training_examples),
        "source_breakdown": source_counts,
        "repository": repo_url,
        "training_epochs": num_epochs,
        "model_name": "microsoft/DialoGPT-medium",
        "output_directory": output_dir
    }
    
    with open(Path(output_dir) / "training_summary.json", "w") as f:
        json.dump(training_summary, f, indent=2)
    
    print(f"✅ Comprehensive training data saved to {output_dir}")
    
    # Step 7: Cleanup
    print("\n🧹 Step 7: Cleaning up...")
    
    try:
        import shutil
        shutil.rmtree(temp_repo_path)
        print(f"✅ Cleaned up temporary repository: {temp_repo_path}")
    except Exception as e:
        print(f"⚠️  Warning: Could not clean up {temp_repo_path}: {e}")
    
    print("\n🎉 STANDALONE COMPREHENSIVE LOCAGENT TRAINING COMPLETED!")
    print("==================================================")
    print(f"📁 Model saved to: {output_dir}")
    print(f"📊 Total examples: {len(training_examples)}")
    print(f"🔍 Sources used: {len(source_counts)} different types")
    print(f"🎯 Training epochs: {num_epochs}")
    print()
    print("The model has been trained on the MAXIMUM amount of data possible!")
    print("This includes docstrings, comments, tests, git history, and more!")


if __name__ == "__main__":
    asyncio.run(main())
