"""
Test LOCAGENT training functionality.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock
from src.ue_sea.locagent.training import (
    LocalizationExample, AgentTrajectory, LocalizationDataset,
    LOCAGENT_Trainer, TrajectoryGenerator
)
from src.ue_sea.locagent.agent import LocalizationResult
from src.ue_sea.locagent.entities import Entity, EntityType


class TestLocalizationExample:
    """Test LocalizationExample class."""
    
    def test_example_creation(self):
        """Test basic example creation."""
        example = LocalizationExample(
            issue_description="Fix authentication bug",
            repository_path="/path/to/repo",
            target_entities=["auth.py:login", "auth.py:authenticate"]
        )
        
        assert example.issue_description == "Fix authentication bug"
        assert example.repository_path == "/path/to/repo"
        assert example.target_entities == ["auth.py:login", "auth.py:authenticate"]
        assert example.difficulty == "medium"
        assert example.context is None
    
    def test_example_with_context(self):
        """Test example with context."""
        context = {"commit": "abc123", "author": "developer"}
        example = LocalizationExample(
            issue_description="Fix authentication bug",
            repository_path="/path/to/repo",
            target_entities=["auth.py:login"],
            context=context,
            difficulty="hard"
        )
        
        assert example.context == context
        assert example.difficulty == "hard"


class TestAgentTrajectory:
    """Test AgentTrajectory class."""
    
    def test_trajectory_creation(self):
        """Test trajectory creation."""
        example = LocalizationExample(
            issue_description="Test issue",
            repository_path="/path/to/repo",
            target_entities=["entity1"]
        )
        
        result = LocalizationResult(
            issue_description="Test issue",
            relevant_entities=[],
            confidence_scores={},
            reasoning_trace=[],
            final_ranking=[]
        )
        
        trajectory = AgentTrajectory(
            example=example,
            actions=[],
            observations=[],
            final_result=result,
            success=True,
            accuracy_score=0.8
        )
        
        assert trajectory.example == example
        assert trajectory.success == True
        assert trajectory.accuracy_score == 0.8


class TestLocalizationDataset:
    """Test LocalizationDataset class."""
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        examples = [
            LocalizationExample(
                issue_description="Test issue 1",
                repository_path="/path/to/repo",
                target_entities=["entity1"]
            ),
            LocalizationExample(
                issue_description="Test issue 2",
                repository_path="/path/to/repo",
                target_entities=["entity2"]
            )
        ]
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1]
        }
        
        dataset = LocalizationDataset(examples, mock_tokenizer, max_length=512)
        
        assert len(dataset) == 2
        assert dataset.examples == examples
        assert dataset.max_length == 512
    
    def test_dataset_getitem(self):
        """Test dataset item access."""
        examples = [
            LocalizationExample(
                issue_description="Test issue",
                repository_path="/path/to/repo",
                target_entities=["entity1"]
            )
        ]
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1]
        }
        
        dataset = LocalizationDataset(examples, mock_tokenizer, max_length=512)
        
        # Test getting item
        item = dataset[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item


class TestTrajectoryGenerator:
    """Test TrajectoryGenerator class."""
    
    def test_generator_creation(self):
        """Test trajectory generator creation."""
        mock_graph = Mock()
        mock_agent = Mock()
        
        generator = TrajectoryGenerator(mock_graph, mock_agent)
        
        assert generator.graph == mock_graph
        assert generator.agent == mock_agent
    
    def test_calculate_success_metrics(self):
        """Test success metrics calculation."""
        mock_graph = Mock()
        mock_agent = Mock()
        generator = TrajectoryGenerator(mock_graph, mock_agent)
        
        # Test with matching entities
        target_entities = ["entity1", "entity2"]
        final_ranking = [("entity1", 0.9), ("entity3", 0.7), ("entity2", 0.6)]
        
        success, accuracy = generator._calculate_success_metrics(target_entities, final_ranking)
        
        assert success == True  # entity1 is in top-5
        assert accuracy > 0.0
    
    def test_calculate_success_metrics_no_match(self):
        """Test success metrics with no matches."""
        mock_graph = Mock()
        mock_agent = Mock()
        generator = TrajectoryGenerator(mock_graph, mock_agent)
        
        target_entities = ["entity1"]
        final_ranking = [("entity2", 0.9), ("entity3", 0.7)]
        
        success, accuracy = generator._calculate_success_metrics(target_entities, final_ranking)
        
        assert success == False
        assert accuracy == 0.0
    
    def test_calculate_success_metrics_empty(self):
        """Test success metrics with empty inputs."""
        mock_graph = Mock()
        mock_agent = Mock()
        generator = TrajectoryGenerator(mock_graph, mock_agent)
        
        success, accuracy = generator._calculate_success_metrics([], [])
        
        assert success == False
        assert accuracy == 0.0


class TestLOCAGENTTrainer:
    """Test LOCAGENT_Trainer class."""
    
    def test_trainer_creation(self):
        """Test trainer creation."""
        trainer = LOCAGENT_Trainer(
            model_name="microsoft/DialoGPT-medium",
            use_gpu=False
        )
        
        assert trainer.model_name == "microsoft/DialoGPT-medium"
        assert trainer.use_gpu == False
        assert trainer.tokenizer is None
        assert trainer.model is None
    
    def test_prepare_dataset(self):
        """Test dataset preparation."""
        trainer = LOCAGENT_Trainer(use_gpu=False)
        
        examples = [
            LocalizationExample(
                issue_description="Test issue 1",
                repository_path="/path/to/repo",
                target_entities=["entity1"]
            ),
            LocalizationExample(
                issue_description="Test issue 2",
                repository_path="/path/to/repo",
                target_entities=["entity2"]
            ),
            LocalizationExample(
                issue_description="Test issue 3",
                repository_path="/path/to/repo",
                target_entities=["entity3"]
            )
        ]
        
        train_dataset, val_dataset = trainer.prepare_dataset(examples, train_ratio=0.6)
        
        assert len(train_dataset.examples) == 1  # 60% of 3 = 1.8 -> 1
        assert len(val_dataset.examples) == 2  # Remaining examples


class TestDataLoading:
    """Test data loading functionality."""
    
    def test_load_swe_bench_data(self):
        """Test loading SWE-Bench data."""
        from src.ue_sea.locagent.training import load_swe_bench_data
        
        # Create temporary file with SWE-Bench format
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            data = {
                "problem_statement": "Fix authentication bug",
                "repo": "test/repo",
                "instance_id": "test-1",
                "base_commit": "abc123",
                "patch": "+++ auth.py\n@@ -1,1 +1,1 @@\n-def login():\n+def login():\n     pass",
                "test_patch": "+++ test_auth.py\n@@ -0,0 +1,1 @@\n+def test_login(): pass"
            }
            f.write(json.dumps(data) + '\n')
            temp_path = f.name
        
        try:
            examples = load_swe_bench_data(temp_path)
            assert len(examples) == 1
            assert examples[0].issue_description == "Fix authentication bug"
            assert examples[0].repository_path == "test/repo"
        finally:
            Path(temp_path).unlink()
    
    def test_load_locbench_data(self):
        """Test loading LocBench data."""
        from src.ue_sea.locagent.training import load_locbench_data
        
        # Create temporary file with LocBench format
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            data = {
                "issue_description": "Fix authentication bug",
                "repository_path": "/path/to/repo",
                "target_entities": ["auth.py:login", "auth.py:authenticate"],
                "context": {"commit": "abc123"},
                "difficulty": "medium"
            }
            f.write(json.dumps(data) + '\n')
            temp_path = f.name
        
        try:
            examples = load_locbench_data(temp_path)
            assert len(examples) == 1
            assert examples[0].issue_description == "Fix authentication bug"
            assert examples[0].target_entities == ["auth.py:login", "auth.py:authenticate"]
        finally:
            Path(temp_path).unlink()
