"""
Evolution Strategies Optimizer: Global parameter optimization.

The optimizer currently exposes the core EvolutionStrategies implementation.
Optional helpers (parameter perturbation, reward normalization, gradient
estimation) will be reintroduced once their modules are added back to the
package.
"""

from .evolution_strategies import EvolutionStrategies, ESConfig, ESResult

__all__ = ["EvolutionStrategies", "ESConfig", "ESResult"]
