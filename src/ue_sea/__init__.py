"""
Unified Evolutionary Software Engineering Agent (UE-SEA)

A self-improving software-engineering agent that understands large codebases,
evolves code using evolutionary algorithms, and trains modular reasoning skills at scale.
"""

__version__ = "0.1.0"
__author__ = "UE-SEA Team"

from .orchestrator import UE_SEA_Orchestrator
from .locagent.graph import LOCAGENT_Graph
from .locagent.tools import SearchEntity, TraverseGraph, RetrieveEntity
from .alphaevolve import AlphaEvolve_Controller
from .reasoning import SkillPath, ReasoningCurriculum
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
    "EvolutionStrategies",
    "EvaluatorPool",
    "GuardrailSystem",
]
