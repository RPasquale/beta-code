"""
Evaluation Metrics: Defines fitness scoring and evaluation metrics.

Implements the scoring object contract and fitness calculation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import time
import json


class MetricType(Enum):
    """Types of evaluation metrics."""
    FUNCTIONAL = "functional"      # Correctness, test pass rate
    PERFORMANCE = "performance"     # Latency, memory usage
    QUALITY = "quality"            # Code quality, readability
    COST = "cost"                  # Token cost, resource usage
    SECURITY = "security"          # Security vulnerabilities
    HACKING = "hacking"            # Reward hacking detection


@dataclass
class MetricValue:
    """Represents a single metric value."""
    name: str
    value: float
    metric_type: MetricType
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class FitnessScore:
    """
    Fitness score object as specified in the contract.
    
    Contains all metrics and the composite score used for selection.
    """
    # Core metrics
    pass_rate: float = 0.0
    latency_ms: float = 0.0
    token_cost: float = 0.0
    code_quality: float = 0.0
    hacking_flags: float = 0.0
    
    # Composite score
    score: float = 0.0
    
    # Additional metrics
    memory_usage: float = 0.0
    test_coverage: float = 0.0
    complexity_score: float = 0.0
    maintainability: float = 0.0
    
    # Metadata
    evaluation_time: float = 0.0
    stage: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        
        # Calculate composite score if not provided
        if self.score == 0.0:
            self.score = self._calculate_composite_score()
    
    def _calculate_composite_score(self) -> float:
        """Calculate composite score from individual metrics."""
        # Weighted combination of metrics
        weights = {
            "pass_rate": 0.4,      # Most important - correctness
            "code_quality": 0.2,   # Code quality
            "latency_ms": 0.15,    # Performance (inverted)
            "test_coverage": 0.1,  # Test coverage
            "maintainability": 0.1, # Maintainability
            "complexity_score": 0.05, # Complexity (inverted)
        }
        
        # Normalize latency (lower is better)
        latency_score = max(0, 1.0 - (self.latency_ms / 1000.0))  # Normalize to 0-1
        
        # Normalize complexity (lower is better)
        complexity_score = max(0, 1.0 - (self.complexity_score / 10.0))  # Normalize to 0-1
        
        # Calculate weighted score
        score = (
            weights["pass_rate"] * self.pass_rate +
            weights["code_quality"] * self.code_quality +
            weights["latency_ms"] * latency_score +
            weights["test_coverage"] * self.test_coverage +
            weights["maintainability"] * self.maintainability +
            weights["complexity_score"] * complexity_score
        )
        
        # Apply penalty for hacking flags
        if self.hacking_flags > 0:
            score *= 0.1  # Severe penalty for hacking
        
        # Apply penalty for high token cost
        if self.token_cost > 1000:  # Threshold for expensive
            score *= 0.8
        
        return max(0.0, min(1.0, score))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pass_rate": self.pass_rate,
            "latency_ms": self.latency_ms,
            "token_cost": self.token_cost,
            "code_quality": self.code_quality,
            "hacking_flags": self.hacking_flags,
            "score": self.score,
            "memory_usage": self.memory_usage,
            "test_coverage": self.test_coverage,
            "complexity_score": self.complexity_score,
            "maintainability": self.maintainability,
            "evaluation_time": self.evaluation_time,
            "stage": self.stage,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FitnessScore:
        """Create from dictionary."""
        return cls(
            pass_rate=data.get("pass_rate", 0.0),
            latency_ms=data.get("latency_ms", 0.0),
            token_cost=data.get("token_cost", 0.0),
            code_quality=data.get("code_quality", 0.0),
            hacking_flags=data.get("hacking_flags", 0.0),
            score=data.get("score", 0.0),
            memory_usage=data.get("memory_usage", 0.0),
            test_coverage=data.get("test_coverage", 0.0),
            complexity_score=data.get("complexity_score", 0.0),
            maintainability=data.get("maintainability", 0.0),
            evaluation_time=data.get("evaluation_time", 0.0),
            stage=data.get("stage", 1),
            metadata=data.get("metadata", {})
        )


class EvaluationMetrics:
    """
    Manages evaluation metrics and fitness calculation.
    
    Provides utilities for metric collection, normalization,
    and composite score calculation.
    """
    
    def __init__(self):
        self.metrics_history = []
        self.baseline_metrics = {}
    
    def calculate_fitness(self, metrics: Dict[str, float], 
                         stage: int = 1) -> FitnessScore:
        """
        Calculate fitness score from metrics.
        
        Args:
            metrics: Dictionary of metric values
            stage: Evaluation stage (1=fast, 2=medium, 3=full)
        
        Returns:
            FitnessScore object
        """
        # Extract core metrics
        pass_rate = metrics.get("pass_rate", 0.0)
        latency_ms = metrics.get("latency_ms", 0.0)
        token_cost = metrics.get("token_cost", 0.0)
        code_quality = metrics.get("code_quality", 0.0)
        hacking_flags = metrics.get("hacking_flags", 0.0)
        
        # Extract additional metrics
        memory_usage = metrics.get("memory_usage", 0.0)
        test_coverage = metrics.get("test_coverage", 0.0)
        complexity_score = metrics.get("complexity_score", 0.0)
        maintainability = metrics.get("maintainability", 0.0)
        
        # Create fitness score
        fitness = FitnessScore(
            pass_rate=pass_rate,
            latency_ms=latency_ms,
            token_cost=token_cost,
            code_quality=code_quality,
            hacking_flags=hacking_flags,
            memory_usage=memory_usage,
            test_coverage=test_coverage,
            complexity_score=complexity_score,
            maintainability=maintainability,
            stage=stage,
            metadata=metrics.get("metadata", {})
        )
        
        # Store in history
        self.metrics_history.append(fitness)
        
        return fitness
    
    def normalize_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize metrics using baseline values.
        
        Args:
            metrics: Raw metrics to normalize
        
        Returns:
            Normalized metrics
        """
        normalized = {}
        
        for metric_name, value in metrics.items():
            if metric_name in self.baseline_metrics:
                baseline = self.baseline_metrics[metric_name]
                
                # Normalize based on metric type
                if metric_name in ["latency_ms", "token_cost", "complexity_score"]:
                    # Lower is better
                    normalized[metric_name] = max(0, 1.0 - (value / baseline))
                else:
                    # Higher is better
                    normalized[metric_name] = min(1.0, value / baseline)
            else:
                # No baseline, use raw value
                normalized[metric_name] = value
        
        return normalized
    
    def update_baseline(self, metrics: Dict[str, float]) -> None:
        """Update baseline metrics."""
        for metric_name, value in metrics.items():
            if metric_name not in self.baseline_metrics:
                self.baseline_metrics[metric_name] = value
            else:
                # Update baseline with exponential moving average
                self.baseline_metrics[metric_name] = (
                    0.9 * self.baseline_metrics[metric_name] + 0.1 * value
                )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about evaluation metrics."""
        if not self.metrics_history:
            return {"total_evaluations": 0}
        
        scores = [fitness.score for fitness in self.metrics_history]
        
        return {
            "total_evaluations": len(self.metrics_history),
            "average_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "score_std": self._calculate_std(scores),
            "hacking_detections": sum(1 for f in self.metrics_history if f.hacking_flags > 0),
            "high_quality_count": sum(1 for f in self.metrics_history if f.code_quality > 0.8),
        }
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def export_metrics(self, file_path: str) -> None:
        """Export metrics to file."""
        data = {
            "baseline_metrics": self.baseline_metrics,
            "metrics_history": [fitness.to_dict() for fitness in self.metrics_history],
            "statistics": self.get_statistics()
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def import_metrics(self, file_path: str) -> None:
        """Import metrics from file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        self.baseline_metrics = data.get("baseline_metrics", {})
        self.metrics_history = [
            FitnessScore.from_dict(fitness_data)
            for fitness_data in data.get("metrics_history", [])
        ]
