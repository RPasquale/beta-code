"""
Unified Evolutionary Software Engineering Agent (UE-SEA)

A self-improving software-engineering agent that understands large codebases,
evolves code using evolutionary algorithms, and trains modular reasoning skills at scale.
"""

__version__ = "0.1.0"
__author__ = "UE-SEA Team"

from .orchestrator import UE_SEA_Orchestrator
from .locagent import LOCAGENT_Graph, SearchEntity, TraverseGraph, RetrieveEntity
from .alphaevolve import AlphaEvolve_Controller
from .reasoning import SkillPath, ReasoningCurriculum
from .distributed import StreamingDiLoCo, DiPaCo_Router
from .optimizer import EvolutionStrategies
from .evaluation import EvaluatorPool, GuardrailSystem

__all__ = [
    "UE_SEA_Orchestrator",
    "LOCAGENT_Graph",
    "SearchEntity", 
    "TraverseGraph",
    "RetrieveEntity",
    "AlphaEvolve_Controller",
    "SkillPath",
    "ReasoningCurriculum", 
    "StreamingDiLoCo",
    "DiPaCo_Router",
    "EvolutionStrategies",
    "EvaluatorPool",
    "GuardrailSystem",
]
