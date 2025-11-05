"""
SPCT (Self-Principled Critique Tuning) Orchestrator.

Implements the complete SPCT pipeline:
1. Pointwise Generative RM with principles + critique + scoring
2. RFT cold start training
3. GRPO online RL training
4. Meta-RM for sample filtering
5. Inference-time scaling with voting
"""

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from .pointwise_rm import PointwiseGRM, JudgeOutput, SPCTConfig
from .meta_rm import MetaRM, MetaRMTrainer, MetaRMConfig


@dataclass
class SPCTTrainingData:
    """Training data for SPCT."""
    query: str
    responses: List[str]
    best_index: int
    ground_truth_scores: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SPCTResults:
    """Results from SPCT training and inference."""
    rft_results: Dict[str, Any]
    drgrpo_results: Dict[str, Any]
    meta_rm_results: Dict[str, Any]
    inference_results: Dict[str, Any]
    scaling_curves: Dict[str, List[float]]


class SPCTOrchestrator:
    """
    Complete SPCT orchestrator implementing the full pipeline.
    
    Features:
    - Pointwise Generative RM with self-principles
    - RFT cold start with trajectory filtering
    - GRPO online RL with rule-based rewards
    - Meta-RM for sample quality filtering
    - Inference-time scaling with voting
    """
    
    def __init__(self, config: SPCTConfig):
        self.config = config
        
        # Initialize components
        self.pointwise_rm = PointwiseGRM(config)
        self.meta_rm_trainer = MetaRMTrainer(MetaRMConfig())
        
        # Training state
        self.rft_traces = []
        self.online_samples = []
        self.training_metrics = []
        
        # Inference state
        self.scaling_curves = {}
        
    async def train_full_pipeline(self, training_data: List[SPCTTrainingData]) -> SPCTResults:
        """Run the complete SPCT training pipeline."""
        print("🚀 Starting Full SPCT Training Pipeline...")
        
        results = SPCTResults(
            rft_results={},
            drgrpo_results={},
            meta_rm_results={},
            inference_results={},
            scaling_curves={}
        )
        
        # Stage 1: RFT Cold Start
        print("\n=== Stage 1: RFT Cold Start ===")
        rft_results = await self._run_rft_cold_start(training_data)
        results.rft_results = rft_results
        
        # Stage 2: Dr.GRPO Online RL
        print("\n=== Stage 2: Dr.GRPO Online RL ===")
        drgrpo_results = await self._run_drgrpo_online_rl(training_data)
        results.drgrpo_results = drgrpo_results
        
        # Stage 3: Meta-RM Training
        print("\n=== Stage 3: Meta-RM Training ===")
        meta_rm_results = await self._train_meta_rm()
        results.meta_rm_results = meta_rm_results
        
        # Stage 4: Inference-time Scaling
        print("\n=== Stage 4: Inference-time Scaling ===")
        inference_results = await self._test_inference_scaling(training_data[:10])
        results.inference_results = inference_results
        
        print("✅ SPCT Training Pipeline Complete!")
        return results
    
    async def _run_rft_cold_start(self, training_data: List[SPCTTrainingData]) -> Dict[str, Any]:
        """Run RFT cold start training."""
        print("🔄 Running RFT cold start...")
        
        # Convert training data to format expected by PointwiseGRM
        rm_training_data = []
        for item in training_data:
            rm_training_data.append({
                "query": item.query,
                "responses": item.responses,
                "best_index": item.best_index
            })
        
        # Run RFT training
        rft_results = await self.pointwise_rm.train_rft_cold_start(rm_training_data)
        
        self.rft_traces = getattr(self.pointwise_rm, "last_rft_traces", [])
        
        print(f"📊 RFT Results: {rft_results['accepted_traces']} traces accepted")
        return rft_results
    
    async def _run_drgrpo_online_rl(self, training_data: List[SPCTTrainingData]) -> Dict[str, Any]:
        """Run Dr.GRPO online RL training."""
        print("🚀 Running Dr.GRPO online RL...")
        
        # Convert training data
        rm_training_data = []
        for item in training_data:
            rm_training_data.append({
                "query": item.query,
                "responses": item.responses,
                "best_index": item.best_index
            })
        
        # Run Dr.GRPO training
        drgrpo_results = await self.pointwise_rm.train_drgrpo_online_rl(rm_training_data)
        
        self.online_samples = getattr(self.pointwise_rm, "last_drgrpo_samples", [])
        print(f"📊 Dr.GRPO Results: {drgrpo_results['accuracy']:.3f} accuracy")
        return drgrpo_results
    
    async def _train_meta_rm(self) -> Dict[str, Any]:
        """Train Meta-RM on RFT and online samples."""
        print("🎯 Training Meta-RM...")
        
        meta_rm_results: Dict[str, Any] = {}

        rft_traces = getattr(self.pointwise_rm, "last_rft_traces", [])
        if rft_traces:
            rft_metrics = await self.meta_rm_trainer.train_on_rft_samples(rft_traces)
            meta_rm_results["rft_training"] = rft_metrics
        else:
            meta_rm_results["rft_training"] = {"training_samples": 0}

        online_samples = getattr(self.pointwise_rm, "last_drgrpo_samples", [])
        if online_samples:
            online_metrics = await self.meta_rm_trainer.train_on_online_samples(online_samples)
            meta_rm_results["online_training"] = online_metrics
        else:
            meta_rm_results["online_training"] = {"training_samples": 0}

        meta_rm_results["samples_processed"] = (
            meta_rm_results["rft_training"].get("training_samples", 0)
            + meta_rm_results["online_training"].get("training_samples", 0)
        )
        print("📊 Meta-RM training complete")

        return meta_rm_results
    
    async def _test_inference_scaling(self, test_data: List[SPCTTrainingData]) -> Dict[str, Any]:
        """Test inference-time scaling with different sample counts."""
        print("🔍 Testing inference-time scaling...")
        
        scaling_results = {}
        sample_counts = [1, 2, 4, 8, 16]
        
        for k in sample_counts:
            print(f"  Testing with {k} samples...")
            
            k_results = []
            for item in test_data:
                # Generate k independent judgements
                judgements = await self.pointwise_rm.generate_multiple_judgements(
                    item.query, item.responses, num_samples=k
                )
                
                # Convert to dict format for Meta-RM
                judgement_dicts = []
                for judgement in judgements:
                    judgement_dicts.append({
                        "self_criteria": judgement.self_criteria,
                        "critique": judgement.critique,
                        "scores": judgement.scores,
                        "principles": judgement.principles,
                        "confidence": judgement.confidence
                    })
                
                # Apply Meta-RM filtering if enabled
                if self.config.use_meta_rm_filter:
                    filtered_judgements = self.meta_rm_trainer.meta_rm.filter_samples(
                        judgement_dicts, item.query, item.responses,
                        filter_ratio=self.config.meta_rm_filter_ratio
                    )
                else:
                    filtered_judgements = judgement_dicts
                
                # Aggregate judgements
                aggregated = self.pointwise_rm.aggregate_judgements(
                    [JudgeOutput(**j) for j in filtered_judgements],
                    method=self.config.voting_method
                )
                
                # Check correctness
                is_correct = self.pointwise_rm.calculate_correctness(
                    aggregated["scores"], item.best_index
                )
                
                k_results.append({
                    "correct": is_correct,
                    "confidence": aggregated["confidence"],
                    "num_judgements": aggregated["num_judgements"]
                })
            
            # Calculate metrics for this k
            accuracy = np.mean([r["correct"] for r in k_results])
            avg_confidence = np.mean([r["confidence"] for r in k_results])
            
            scaling_results[f"k_{k}"] = {
                "accuracy": accuracy,
                "avg_confidence": avg_confidence,
                "samples_tested": len(k_results)
            }
            
            print(f"    k={k}: {accuracy:.3f} accuracy, {avg_confidence:.3f} confidence")
        
        return scaling_results
    
    async def inference_with_scaling(self, query: str, responses: List[str], 
                                   num_samples: int = 8) -> Dict[str, Any]:
        """Run inference with scaling for a single query."""
        
        # Generate multiple judgements
        judgements = await self.pointwise_rm.generate_multiple_judgements(
            query, responses, num_samples=num_samples
        )
        
        # Convert to dict format
        judgement_dicts = []
        for judgement in judgements:
            judgement_dicts.append({
                "self_criteria": judgement.self_criteria,
                "critique": judgement.critique,
                "scores": judgement.scores,
                "principles": judgement.principles,
                "confidence": judgement.confidence
            })
        
        # Apply Meta-RM filtering if enabled
        if self.config.use_meta_rm_filter:
            filtered_judgements = self.meta_rm_trainer.meta_rm.filter_samples(
                judgement_dicts, query, responses,
                filter_ratio=self.config.meta_rm_filter_ratio
            )
        else:
            filtered_judgements = judgement_dicts
        
        # Aggregate judgements
        aggregated = self.pointwise_rm.aggregate_judgements(
            [JudgeOutput(**j) for j in filtered_judgements],
            method=self.config.voting_method
        )
        
        return {
            "query": query,
            "responses": responses,
            "aggregated_scores": aggregated["scores"],
            "best_index": aggregated["best_index"],
            "confidence": aggregated["confidence"],
            "num_judgements": aggregated["num_judgements"],
            "raw_judgements": judgement_dicts,
            "filtered_judgements": filtered_judgements
        }
    
    def get_scaling_curves(self) -> Dict[str, List[float]]:
        """Get scaling curves for different metrics."""
        return self.scaling_curves
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current orchestrator metrics."""
        return {
            "rft_traces": len(self.rft_traces),
            "online_samples": len(self.online_samples),
            "training_metrics": len(self.training_metrics),
            "scaling_curves": len(self.scaling_curves),
            "config": self.config
        }
