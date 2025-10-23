"""
AlphaEvolve Controller: Main evolutionary loop for code improvement.

Implements the core evolutionary algorithm with prompt sampling, LLM ensemble,
diff generation, and selection mechanisms.
"""

import asyncio
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
import time

from .prompt_sampler import PromptSampler
from .llm_ensemble import LLM_Ensemble
from .diff_generator import DiffGenerator
from .evaluators import EvaluatorPool
from .evolutionary_db import EvolutionaryDB, MAP_Elites_Selector


class EvolutionStage(Enum):
    """Stages of the evolutionary process."""
    INITIALIZATION = "initialization"
    MUTATION = "mutation"
    CROSSOVER = "crossover"
    EVALUATION = "evaluation"
    SELECTION = "selection"
    TERMINATION = "termination"


@dataclass
class EvolutionConfig:
    """Configuration for the evolutionary process."""
    max_generations: int = 100
    population_size: int = 50
    elite_size: int = 10
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    evaluation_budget: int = 1000
    early_stopping_patience: int = 10
    temperature: float = 0.3  # For LLM sampling
    max_diff_size: int = 1000  # Maximum lines in a diff


@dataclass
class EvolutionResult:
    """Result of an evolutionary run."""
    best_program: str
    best_score: float
    generation: int
    total_evaluations: int
    convergence_history: List[float]
    elite_programs: List[Tuple[str, float]]


class AlphaEvolve_Controller:
    """
    Main controller for the AlphaEvolve evolutionary algorithm.
    
    Orchestrates the evolutionary process including prompt sampling,
    LLM ensemble generation, evaluation, and selection.
    """
    
    def __init__(self, config: EvolutionConfig = None):
        self.config = config or EvolutionConfig()
        
        # Initialize components
        self.prompt_sampler = PromptSampler()
        self.llm_ensemble = LLM_Ensemble()
        self.diff_generator = DiffGenerator()
        self.evaluator_pool = EvaluatorPool()
        self.evolutionary_db = EvolutionaryDB()
        self.selector = MAP_Elites_Selector()
        
        # Evolution state
        self.current_generation = 0
        self.total_evaluations = 0
        self.convergence_history = []
        self.best_score = float('-inf')
        self.best_program = None
        
        # Statistics
        self.generation_stats = []
    
    async def evolve(self, initial_program: str, task_description: str,
                   context: Dict[str, Any] = None) -> EvolutionResult:
        """
        Run the evolutionary algorithm to improve a program.
        
        Args:
            initial_program: Starting program to improve
            task_description: Description of the improvement task
            context: Additional context for the task
        
        Returns:
            EvolutionResult with best program and statistics
        """
        print(f"Starting evolution with {self.config.max_generations} generations")
        
        # Initialize population
        population = await self._initialize_population(initial_program, task_description, context)
        
        # Evolution loop
        for generation in range(self.config.max_generations):
            print(f"Generation {generation + 1}/{self.config.max_generations}")
            
            # Evaluate current population
            evaluated_population = await self._evaluate_population(population)
            
            # Update statistics
            self._update_statistics(evaluated_population, generation)
            
            # Check for convergence
            if self._check_convergence():
                print(f"Converged at generation {generation + 1}")
                break
            
            # Selection and reproduction
            if generation < self.config.max_generations - 1:
                population = await self._reproduce_population(evaluated_population, task_description, context)
            
            self.current_generation = generation + 1
        
        # Get final results
        final_population = await self._evaluate_population(population)
        best_program, best_score = self._get_best_program(final_population)
        
        return EvolutionResult(
            best_program=best_program,
            best_score=best_score,
            generation=self.current_generation,
            total_evaluations=self.total_evaluations,
            convergence_history=self.convergence_history,
            elite_programs=self.selector.get_elites()
        )
    
    async def _initialize_population(self, initial_program: str, 
                                  task_description: str,
                                  context: Dict[str, Any]) -> List[str]:
        """Initialize the population with the initial program and variations."""
        population = [initial_program]
        
        # Generate initial variations using prompt sampling
        for _ in range(self.config.population_size - 1):
            prompt = await self.prompt_sampler.sample_initial_variation(
                initial_program, task_description, context
            )
            
            # Generate variation using LLM ensemble
            variation = await self.llm_ensemble.generate_variation(prompt)
            if variation:
                population.append(variation)
        
        return population
    
    async def _evaluate_population(self, population: List[str]) -> List[Tuple[str, float]]:
        """Evaluate all programs in the population."""
        evaluated = []
        
        for program in population:
            # Use staged evaluation
            score = await self.evaluator_pool.evaluate_staged(program)
            evaluated.append((program, score))
            self.total_evaluations += 1
            
            # Update evolutionary database
            self.evolutionary_db.add_program(program, score)
        
        return evaluated
    
    async def _reproduce_population(self, evaluated_population: List[Tuple[str, float]],
                                  task_description: str,
                                  context: Dict[str, Any]) -> List[str]:
        """Generate new population through selection and reproduction."""
        new_population = []
        
        # Select parents using MAP-elites
        parents = self.selector.select_parents(evaluated_population, self.config.elite_size)
        
        # Generate offspring
        for _ in range(self.config.population_size):
            # Choose reproduction strategy
            if random.random() < self.config.mutation_rate:
                # Mutation
                parent = random.choice(parents)
                offspring = await self._mutate_program(parent[0], task_description, context)
            elif random.random() < self.config.crossover_rate:
                # Crossover
                parent1, parent2 = random.sample(parents, 2)
                offspring = await self._crossover_programs(
                    parent1[0], parent2[0], task_description, context
                )
            else:
                # Direct selection
                offspring = random.choice(parents)[0]
            
            if offspring:
                new_population.append(offspring)
        
        return new_population
    
    async def _mutate_program(self, program: str, task_description: str,
                            context: Dict[str, Any]) -> Optional[str]:
        """Mutate a program using LLM ensemble."""
        # Sample mutation prompt
        prompt = await self.prompt_sampler.sample_mutation_prompt(
            program, task_description, context
        )
        
        # Generate mutation using LLM ensemble
        mutation = await self.llm_ensemble.generate_mutation(prompt)
        
        if mutation:
            # Apply mutation as diff
            return await self.diff_generator.apply_diff(program, mutation)
        
        return None
    
    async def _crossover_programs(self, program1: str, program2: str,
                                task_description: str,
                                context: Dict[str, Any]) -> Optional[str]:
        """Create crossover between two programs."""
        # Sample crossover prompt
        prompt = await self.prompt_sampler.sample_crossover_prompt(
            program1, program2, task_description, context
        )
        
        # Generate crossover using LLM ensemble
        crossover = await self.llm_ensemble.generate_crossover(prompt)
        
        if crossover:
            # Apply crossover as diff
            return await self.diff_generator.apply_diff(program1, crossover)
        
        return None
    
    def _update_statistics(self, evaluated_population: List[Tuple[str, float]], 
                          generation: int) -> None:
        """Update evolution statistics."""
        scores = [score for _, score in evaluated_population]
        
        # Update best score
        max_score = max(scores)
        if max_score > self.best_score:
            self.best_score = max_score
            self.best_program = max(evaluated_population, key=lambda x: x[1])[0]
        
        # Update convergence history
        self.convergence_history.append(max_score)
        
        # Update generation stats
        stats = {
            "generation": generation,
            "max_score": max_score,
            "mean_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "std_score": (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5,
            "total_evaluations": self.total_evaluations
        }
        self.generation_stats.append(stats)
        
        print(f"Generation {generation}: max={max_score:.3f}, mean={stats['mean_score']:.3f}")
    
    def _check_convergence(self) -> bool:
        """Check if the evolution has converged."""
        if len(self.convergence_history) < self.config.early_stopping_patience:
            return False
        
        # Check if best score has improved in the last N generations
        recent_scores = self.convergence_history[-self.config.early_stopping_patience:]
        return all(score == recent_scores[0] for score in recent_scores)
    
    def _get_best_program(self, evaluated_population: List[Tuple[str, float]]) -> Tuple[str, float]:
        """Get the best program from the population."""
        return max(evaluated_population, key=lambda x: x[1])
    
    async def single_improvement(self, program: str, task_description: str,
                               context: Dict[str, Any] = None) -> Tuple[str, float]:
        """
        Perform a single improvement step (useful for quick iterations).
        
        Args:
            program: Program to improve
            task_description: Description of improvement task
            context: Additional context
        
        Returns:
            Tuple of (improved_program, score)
        """
        # Generate improvement prompt
        prompt = await self.prompt_sampler.sample_improvement_prompt(
            program, task_description, context
        )
        
        # Generate improvement using LLM ensemble
        improvement = await self.llm_ensemble.generate_improvement(prompt)
        
        if improvement:
            # Apply improvement as diff
            improved_program = await self.diff_generator.apply_diff(program, improvement)
            
            # Evaluate improvement
            score = await self.evaluator_pool.evaluate_staged(improved_program)
            
            return improved_program, score
        
        return program, 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current evolution statistics."""
        return {
            "current_generation": self.current_generation,
            "total_evaluations": self.total_evaluations,
            "best_score": self.best_score,
            "convergence_history": self.convergence_history,
            "generation_stats": self.generation_stats,
            "elite_count": len(self.selector.get_elites())
        }
    
    def reset(self) -> None:
        """Reset the controller state."""
        self.current_generation = 0
        self.total_evaluations = 0
        self.convergence_history = []
        self.best_score = float('-inf')
        self.best_program = None
        self.generation_stats = []
        self.evolutionary_db.reset()
        self.selector.reset()
