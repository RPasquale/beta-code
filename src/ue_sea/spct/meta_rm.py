"""
Meta-RM (Meta Reward Model) for SPCT.

Filters judge samples based on quality of principles and critique.
Implements the Meta-RM component for inference-time scaling.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Any, Tuple
from torch.optim import AdamW
import numpy as np
from dataclasses import dataclass


@dataclass
class MetaRMConfig:
    """Configuration for Meta-RM."""
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B"
    hidden_size: int = 768
    num_classes: int = 2  # Good/Bad judge sample
    learning_rate: float = 1e-5
    batch_size: int = 8
    num_epochs: int = 3


class MetaRM(nn.Module):
    """
    Meta Reward Model for filtering judge samples.
    
    Scores the quality of judge samples (principles + critique + scores)
    to filter out low-quality samples before voting.
    """
    
    def __init__(self, config: MetaRMConfig):
        super().__init__()
        self.config = config
        
        # Load base model
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.base_model = AutoModel.from_pretrained(config.model_name)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.num_classes)
        )
        
    def forward(self, input_ids, attention_mask=None):
        """Forward pass for Meta-RM."""
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)  # Pool over sequence
        logits = self.classifier(pooled)
        return logits
    
    def predict_sample_quality(self, query: str, responses: List[str], 
                             judgement: Dict[str, Any]) -> float:
        """Predict the quality of a judge sample."""
        # Build input text
        input_text = self._build_meta_rm_input(query, responses, judgement)
        
        # Tokenize and predict
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048)
        model_device = next(self.base_model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self.forward(**inputs)
            probabilities = F.softmax(logits, dim=-1)
            quality_score = probabilities[0][1].item()  # Probability of "good" class
        
        return quality_score
    
    def _build_meta_rm_input(self, query: str, responses: List[str], 
                           judgement: Dict[str, Any]) -> str:
        """Build input text for Meta-RM evaluation."""
        input_text = f"Query: {query}\n\n"
        
        input_text += "Responses:\n"
        for i, response in enumerate(responses, 1):
            input_text += f"Response {i}: {response}\n"
        
        input_text += f"\nJudge Sample:\n"
        input_text += f"Criteria: {judgement.get('self_criteria', [])}\n"
        input_text += f"Critique: {judgement.get('critique', '')}\n"
        input_text += f"Scores: {judgement.get('scores', [])}\n"
        input_text += f"Principles: {judgement.get('principles', [])}\n"
        
        return input_text
    
    def filter_samples(self, judgements: List[Dict[str, Any]], 
                      query: str, responses: List[str],
                      filter_ratio: float = 0.5) -> List[Dict[str, Any]]:
        """Filter judge samples based on Meta-RM quality scores."""
        if not judgements:
            return judgements
        
        # Score all samples
        quality_scores = []
        for judgement in judgements:
            score = self.predict_sample_quality(query, responses, judgement)
            quality_scores.append(score)
        
        # Sort by quality and keep top samples
        num_keep = max(1, int(len(judgements) * filter_ratio))
        sorted_indices = np.argsort(quality_scores)[::-1]  # Descending order
        
        filtered_judgements = [judgements[i] for i in sorted_indices[:num_keep]]
        
        return filtered_judgements


class MetaRMTrainer:
    """Trainer for Meta-RM using RFT and online samples."""
    
    def __init__(self, config: MetaRMConfig):
        self.config = config
        self.meta_rm = MetaRM(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.meta_rm.to(self.device)
        
    async def train_on_rft_samples(self, rft_traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train Meta-RM on RFT samples."""
        print("🎯 Training Meta-RM on RFT samples...")
        
        # Prepare training data
        training_data = []
        for trace in rft_traces:
            query = trace["query"]
            responses = trace["responses"]
            judgement = trace["judgement"]
            
            # RFT samples are all "good" (correct)
            # Convert JudgeOutput to dict
            judgement_dict = {
                "self_criteria": judgement.self_criteria,
                "critique": judgement.critique,
                "scores": judgement.scores,
                "principles": judgement.principles
            }
            
            input_text = self.meta_rm._build_meta_rm_input(query, responses, judgement_dict)
            
            training_data.append({
                "input_text": input_text,
                "label": 1,  # Good sample
                "quality_score": 1.0
            })
        
        # Train model
        training_metrics = await self._train_model(training_data)
        
        return {
            "training_samples": len(training_data),
            "training_metrics": training_metrics
        }
    
    async def train_on_online_samples(self, online_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train Meta-RM on online RL samples."""
        print("🔄 Training Meta-RM on online samples...")
        
        # Prepare training data
        training_data = []
        for sample in online_samples:
            query = sample["query"]
            responses = sample["responses"]
            judgement = sample["judgement"]
            is_correct = sample.get("is_correct", False)
            
            # Label based on correctness
            label = 1 if is_correct else 0
            
            # Convert JudgeOutput to dict if needed
            if hasattr(judgement, 'self_criteria'):
                judgement_dict = {
                    "self_criteria": judgement.self_criteria,
                    "critique": judgement.critique,
                    "scores": judgement.scores,
                    "principles": judgement.principles
                }
            else:
                judgement_dict = judgement
            
            input_text = self.meta_rm._build_meta_rm_input(query, responses, judgement_dict)
            
            training_data.append({
                "input_text": input_text,
                "label": label,
                "quality_score": 1.0 if is_correct else 0.0
            })
        
        # Train model
        training_metrics = await self._train_model(training_data)
        
        return {
            "training_samples": len(training_data),
            "training_metrics": training_metrics
        }
    
    async def _train_model(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train the Meta-RM model."""
        if not training_data:
            return {"error": "No training data"}
        
        self.meta_rm.train()
        optimizer = AdamW(self.meta_rm.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_steps = 0

        batch_size = max(1, min(self.config.batch_size, len(training_data)))
        for epoch in range(self.config.num_epochs):
            random.shuffle(training_data)
            for batch_start in range(0, len(training_data), batch_size):
                batch = training_data[batch_start: batch_start + batch_size]
                texts = [item["input_text"] for item in batch]
                labels = torch.tensor([item["label"] for item in batch], dtype=torch.long, device=self.device)
                inputs = self.meta_rm.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=2048,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                logits = self.meta_rm(**inputs)
                loss = criterion(logits, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.meta_rm.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                total_steps += 1

        avg_loss = total_loss / total_steps if total_steps else 0.0
        self.meta_rm.eval()

        return {
            "training_loss": avg_loss,
            "samples_processed": len(training_data),
            "epochs": self.config.num_epochs,
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get Meta-RM metrics."""
        return {
            "config": self.config,
            "model_loaded": self.meta_rm is not None
        }
