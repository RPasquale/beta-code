"""
SPCT (Self-Principled Critique Tuning) Package.

Implements the complete SPCT pipeline for pointwise generative reward models.
"""

from .pointwise_rm import PointwiseGRM, JudgeOutput, SPCTConfig
from .meta_rm import MetaRM, MetaRMTrainer, MetaRMConfig
from .spct_orchestrator import SPCTOrchestrator, SPCTTrainingData, SPCTResults

__all__ = [
    "PointwiseGRM",
    "JudgeOutput", 
    "SPCTConfig",
    "MetaRM",
    "MetaRMTrainer",
    "MetaRMConfig",
    "SPCTOrchestrator",
    "SPCTTrainingData",
    "SPCTResults"
]
