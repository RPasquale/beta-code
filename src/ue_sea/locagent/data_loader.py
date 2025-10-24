"""
LOCAGENT Data Loader: Load and prepare training data for the agent.

Supports SWE-Bench, LocBench, and custom datasets.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from .entities import Entity, EntityType
from .training import LocalizationExample


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    name: str
    data_path: str
    format: str  # "swe_bench", "locbench", "custom"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    max_examples: Optional[int] = None


class LOCAGENT_DataLoader:
    """Data loader for LOCAGENT training."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def load_data(self) -> Tuple[List[LocalizationExample], List[LocalizationExample], List[LocalizationExample]]:
        """Load and split data into train/val/test sets."""
        self.logger.info(f"Loading dataset: {self.config.name}")
        
        # Load examples based on format
        if self.config.format == "swe_bench":
            examples = self._load_swe_bench()
        elif self.config.format == "locbench":
            examples = self._load_locbench()
        elif self.config.format == "custom":
            examples = self._load_custom()
        else:
            raise ValueError(f"Unknown dataset format: {self.config.format}")
        
        # Limit examples if specified
        if self.config.max_examples:
            examples = examples[:self.config.max_examples]
        
        self.logger.info(f"Loaded {len(examples)} examples")
        
        # Split data
        train_examples, val_examples, test_examples = self._split_data(examples)
        
        self.logger.info(f"Split: {len(train_examples)} train, {len(val_examples)} val, {len(test_examples)} test")
        
        return train_examples, val_examples, test_examples
    
    def _load_swe_bench(self) -> List[LocalizationExample]:
        """Load SWE-Bench dataset."""
        examples = []
        
        if not os.path.exists(self.config.data_path):
            self.logger.error(f"SWE-Bench data not found at {self.config.data_path}")
            return examples
        
        with open(self.config.data_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    
                    # Extract issue description
                    issue_description = data.get('problem_statement', '')
                    if not issue_description:
                        continue
                    
                    # Extract target files from patch
                    target_files = []
                    patch = data.get('patch', '')
                    if patch:
                        # Simple extraction of file paths from patch
                        lines = patch.split('\n')
                        for line in lines:
                            if line.startswith('+++') or line.startswith('---'):
                                file_path = line[4:].strip()
                                if file_path and not file_path.startswith('/dev/null'):
                                    target_files.append(file_path)
                    
                    if target_files:
                        example = LocalizationExample(
                            issue_description=issue_description,
                            repository_path=data.get('repo', ''),
                            target_entities=target_files,
                            context={
                                'instance_id': data.get('instance_id', ''),
                                'base_commit': data.get('base_commit', ''),
                                'patch': patch,
                                'test_patch': data.get('test_patch', ''),
                                'problem_type': data.get('problem_type', '')
                            },
                            difficulty='medium'
                        )
                        examples.append(example)
                
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Error parsing line: {e}")
                    continue
        
        return examples
    
    def _load_locbench(self) -> List[LocalizationExample]:
        """Load LocBench dataset."""
        examples = []
        
        if not os.path.exists(self.config.data_path):
            self.logger.error(f"LocBench data not found at {self.config.data_path}")
            return examples
        
        with open(self.config.data_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    
                    issue_description = data.get('issue_description', '')
                    target_entities = data.get('target_entities', [])
                    
                    if issue_description and target_entities:
                        example = LocalizationExample(
                            issue_description=issue_description,
                            repository_path=data.get('repository_path', ''),
                            target_entities=target_entities,
                            context=data.get('context', {}),
                            difficulty=data.get('difficulty', 'medium')
                        )
                        examples.append(example)
                
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Error parsing line: {e}")
                    continue
        
        return examples
    
    def _load_custom(self) -> List[LocalizationExample]:
        """Load custom dataset."""
        examples = []
        
        if not os.path.exists(self.config.data_path):
            self.logger.error(f"Custom data not found at {self.config.data_path}")
            return examples
        
        with open(self.config.data_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    
                    # Expected format: {"issue_description": str, "target_entities": List[str], ...}
                    issue_description = data.get('issue_description', '')
                    target_entities = data.get('target_entities', [])
                    
                    if issue_description and target_entities:
                        example = LocalizationExample(
                            issue_description=issue_description,
                            repository_path=data.get('repository_path', ''),
                            target_entities=target_entities,
                            context=data.get('context', {}),
                            difficulty=data.get('difficulty', 'medium')
                        )
                        examples.append(example)
                
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Error parsing line: {e}")
                    continue
        
        return examples
    
    def _split_data(self, examples: List[LocalizationExample]) -> Tuple[List[LocalizationExample], List[LocalizationExample], List[LocalizationExample]]:
        """Split data into train/val/test sets."""
        import random
        random.shuffle(examples)
        
        n = len(examples)
        train_end = int(n * self.config.train_ratio)
        val_end = train_end + int(n * self.config.val_ratio)
        
        train_examples = examples[:train_end]
        val_examples = examples[train_end:val_end]
        test_examples = examples[val_end:]
        
        return train_examples, val_examples, test_examples


def create_sample_dataset(output_path: str, num_examples: int = 100) -> None:
    """Create a sample dataset for testing."""
    import random
    
    # Sample issues and target entities
    sample_issues = [
        "Fix authentication bug in login system",
        "Add error handling for database connection",
        "Optimize performance of data processing function",
        "Fix memory leak in image processing",
        "Add validation for user input",
        "Fix race condition in concurrent processing",
        "Add logging for debugging purposes",
        "Fix null pointer exception in data access",
        "Optimize query performance in database",
        "Add unit tests for utility functions"
    ]
    
    sample_entities = [
        "src/auth/login.py:authenticate_user",
        "src/database/connection.py:get_connection",
        "src/processing/data.py:process_data",
        "src/image/processor.py:process_image",
        "src/validation/input.py:validate_input",
        "src/concurrent/processor.py:process_concurrent",
        "src/logging/logger.py:log_debug",
        "src/data/access.py:get_data",
        "src/database/query.py:execute_query",
        "src/tests/utils.py:test_utility"
    ]
    
    examples = []
    for i in range(num_examples):
        issue = random.choice(sample_issues)
        entities = random.sample(sample_entities, random.randint(1, 3))
        
        example = {
            "issue_description": issue,
            "repository_path": f"sample_repo_{i % 10}",
            "target_entities": entities,
            "context": {
                "difficulty": random.choice(["easy", "medium", "hard"]),
                "category": random.choice(["bug", "feature", "optimization"])
            },
            "difficulty": random.choice(["easy", "medium", "hard"])
        }
        examples.append(example)
    
    # Write to file
    with open(output_path, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Created sample dataset with {len(examples)} examples at {output_path}")


def load_training_data(data_path: str, format: str = "auto") -> List[LocalizationExample]:
    """Load training data from file."""
    if format == "auto":
        # Auto-detect format based on filename
        if "swe_bench" in data_path.lower():
            format = "swe_bench"
        elif "locbench" in data_path.lower():
            format = "locbench"
        else:
            format = "custom"
    
    config = DatasetConfig(
        name="training_data",
        data_path=data_path,
        format=format
    )
    
    loader = LOCAGENT_DataLoader(config)
    train_examples, _, _ = loader.load_data()
    
    return train_examples
