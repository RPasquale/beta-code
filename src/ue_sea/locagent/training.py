"""
LOCAGENT Training: Training pipeline for the localization agent.

Implements data loading, trajectory generation, and fine-tuning for the agent.
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)

from .agent import LOCAGENT_Agent, AgentAction, AgentObservation, LocalizationResult
from .entities import Entity, EntityType


@dataclass
class LocalizationExample:
    """Training example for localization."""
    issue_description: str
    repository_path: str
    target_entities: List[str]  # Entity IDs that should be localized
    context: Optional[Dict[str, Any]] = None
    difficulty: str = "medium"  # easy, medium, hard
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTrajectory:
    """Trajectory of agent actions and observations."""
    example: LocalizationExample
    actions: List[AgentAction]
    observations: List[AgentObservation]
    final_result: LocalizationResult
    success: bool
    accuracy_score: float


class LocalizationDataset(Dataset):
    """Dataset for localization training."""
    
    def __init__(self, examples: List[LocalizationExample], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Create input text
        input_text = f"Issue: {example.issue_description}\nRepository: {example.repository_path}\n"
        
        # Add context if available
        if example.context:
            input_text += f"Context: {json.dumps(example.context)}\n"
        
        # Create target text (simplified for now)
        target_text = f"Target entities: {', '.join(example.target_entities)}"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        targets = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets["input_ids"].squeeze()
        }


class LOCAGENT_Trainer:
    """Trainer for the LOCAGENT agent."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", 
                 use_gpu: bool = True):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.tokenizer = None
        self.model = None
        self.logger = logging.getLogger(__name__)
    
    def load_model(self):
        """Load the base model and tokenizer."""
        self.logger.info(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.logger.info("Model loaded successfully")
    
    def prepare_dataset(self, examples: List[LocalizationExample], 
                       train_ratio: float = 0.8) -> Tuple[LocalizationDataset, LocalizationDataset]:
        """Prepare training and validation datasets."""
        # Split examples
        split_idx = int(len(examples) * train_ratio)
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]
        
        # Create datasets
        train_dataset = LocalizationDataset(train_examples, self.tokenizer)
        val_dataset = LocalizationDataset(val_examples, self.tokenizer)
        
        return train_dataset, val_dataset
    
    def train(self, train_dataset: LocalizationDataset, 
              val_dataset: LocalizationDataset,
              output_dir: str = "outputs/locagent_training",
              num_epochs: int = 3,
              batch_size: int = 4,
              learning_rate: float = 5e-5) -> None:
        """Train the model."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        # Train
        self.logger.info("Starting training...")
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        self.logger.info(f"Training completed. Model saved to {output_dir}")
    
    def evaluate(self, test_dataset: LocalizationDataset) -> Dict[str, float]:
        """Evaluate the model on test data."""
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        # Create trainer for evaluation
        trainer = Trainer(
            model=self.model,
            eval_dataset=test_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        )
        
        # Evaluate
        eval_results = trainer.evaluate()
        
        return eval_results


class TrajectoryGenerator:
    """Generates training trajectories for the agent."""
    
    def __init__(self, graph, agent: LOCAGENT_Agent):
        self.graph = graph
        self.agent = agent
        self.logger = logging.getLogger(__name__)
    
    async def generate_trajectory(self, example: LocalizationExample) -> AgentTrajectory:
        """Generate a trajectory for a training example."""
        # Run the agent
        result = await self.agent.localize(
            example.issue_description,
            example.context
        )
        
        # Calculate success metrics
        success, accuracy = self._calculate_success_metrics(
            example.target_entities,
            result.final_ranking
        )
        
        return AgentTrajectory(
            example=example,
            actions=[obs.action for obs in result.reasoning_trace],
            observations=result.reasoning_trace,
            final_result=result,
            success=success,
            accuracy_score=accuracy
        )
    
    def _calculate_success_metrics(self, target_entities: List[str], 
                                  final_ranking: List[Tuple[str, float]]) -> Tuple[bool, float]:
        """Calculate success metrics for a trajectory."""
        if not target_entities or not final_ranking:
            return False, 0.0
        
        # Check if any target entity is in top-k results
        ranked_entities = [entity_id for entity_id, _ in final_ranking]
        
        # Calculate accuracy@k for different k values
        accuracy_at_1 = 1.0 if any(target in ranked_entities[:1] for target in target_entities) else 0.0
        accuracy_at_5 = 1.0 if any(target in ranked_entities[:5] for target in target_entities) else 0.0
        accuracy_at_10 = 1.0 if any(target in ranked_entities[:10] for target in target_entities) else 0.0
        
        # Overall accuracy (weighted average)
        overall_accuracy = (accuracy_at_1 * 0.5 + accuracy_at_5 * 0.3 + accuracy_at_10 * 0.2)
        
        success = accuracy_at_5 > 0.0  # Success if target is in top-5
        
        return success, overall_accuracy
    
    async def generate_trajectories(self, examples: List[LocalizationExample]) -> List[AgentTrajectory]:
        """Generate trajectories for multiple examples."""
        trajectories = []
        
        for i, example in enumerate(examples):
            self.logger.info(f"Generating trajectory {i+1}/{len(examples)}")
            
            try:
                trajectory = await self.generate_trajectory(example)
                trajectories.append(trajectory)
            except Exception as e:
                self.logger.error(f"Error generating trajectory for example {i}: {e}")
                continue
        
        return trajectories


def load_swe_bench_data(data_path: str) -> List[LocalizationExample]:
    """Load SWE-Bench dataset for training."""
    examples = []
    
    with open(data_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            # Extract issue description and target files
            issue_description = data.get('problem_statement', '')
            target_files = data.get('test_patch', [])
            
            if issue_description and target_files:
                example = LocalizationExample(
                    issue_description=issue_description,
                    repository_path=data.get('repo', ''),
                    target_entities=target_files,
                    context={
                        'instance_id': data.get('instance_id', ''),
                        'base_commit': data.get('base_commit', ''),
                        'patch': data.get('patch', '')
                    },
                    difficulty='medium'  # Default difficulty
                )
                examples.append(example)
    
    return examples


def load_locbench_data(data_path: str) -> List[LocalizationExample]:
    """Load LocBench dataset for training."""
    examples = []
    
    with open(data_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            # Extract issue and target entities
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
    
    return examples


async def train_locagent_agent(graph, 
                             training_data_path: str,
                             output_dir: str = "outputs/locagent_training",
                             model_name: str = "microsoft/DialoGPT-medium",
                             num_epochs: int = 3,
                             batch_size: int = 4) -> None:
    """
    Main training function for LOCAGENT.
    
    Args:
        graph: LOCAGENT_Graph instance
        training_data_path: Path to training data (SWE-Bench or LocBench format)
        output_dir: Directory to save trained model
        model_name: Base model name for fine-tuning
        num_epochs: Number of training epochs
        batch_size: Training batch size
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting LOCAGENT training...")
    
    # Load training data
    logger.info(f"Loading training data from {training_data_path}")
    if "swe_bench" in training_data_path.lower():
        examples = load_swe_bench_data(training_data_path)
    else:
        examples = load_locbench_data(training_data_path)
    
    logger.info(f"Loaded {len(examples)} training examples")
    
    # Create agent and trajectory generator
    agent = LOCAGENT_Agent(graph)
    trajectory_generator = TrajectoryGenerator(graph, agent)
    
    # Generate trajectories
    logger.info("Generating training trajectories...")
    trajectories = await trajectory_generator.generate_trajectories(examples)
    
    logger.info(f"Generated {len(trajectories)} trajectories")
    
    # Create trainer
    trainer = LOCAGENT_Trainer(model_name=model_name)
    trainer.load_model()
    
    # Prepare datasets
    train_dataset, val_dataset = trainer.prepare_dataset(examples)
    
    # Train
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size
    )
    
    logger.info("Training completed successfully!")
