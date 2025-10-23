"""
UE-SEA Orchestrator: Main coordinator for the unified system.

Schedules cycles: Index → Evolve → Evaluate → Select → Train → Deploy
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from .locagent import LOCAGENT_Graph
from .alphaevolve import AlphaEvolve_Controller
from .evaluation import EvaluatorPool, GuardrailSystem
from .reasoning import SkillPath, ReasoningCurriculum
from .distributed import StreamingDiLoCo, DiPaCo_Router
from .optimizer import EvolutionStrategies


class CycleStage(Enum):
    """Stages of the UE-SEA cycle."""
    INDEX = "index"           # Build code graph
    EVOLVE = "evolve"         # Evolutionary improvement
    EVALUATE = "evaluate"    # Staged evaluation
    SELECT = "select"        # Elite selection
    TRAIN = "train"          # Skill training
    DEPLOY = "deploy"        # Deployment


@dataclass
class CycleConfig:
    """Configuration for UE-SEA cycles."""
    max_cycles: int = 100
    cycle_timeout: int = 3600  # seconds
    index_frequency: int = 10  # Re-index every N cycles
    evolution_budget: int = 1000  # Max evaluations per cycle
    training_budget: int = 100  # Max training steps per cycle
    deployment_threshold: float = 0.8  # Min score for deployment
    parallel_workers: int = 4
    gpu_enabled: bool = True


@dataclass
class CycleResult:
    """Result of a UE-SEA cycle."""
    cycle_id: int
    stage: CycleStage
    success: bool
    duration: float
    metrics: Dict[str, Any]
    artifacts: List[str]  # Generated artifacts
    errors: List[str]


class UE_SEA_Orchestrator:
    """
    Main orchestrator for the Unified Evolutionary Software Engineering Agent.
    
    Coordinates all components and manages the evolution cycles.
    """
    
    def __init__(self, config: CycleConfig = None):
        self.config = config or CycleConfig()
        
        # Initialize components
        self.locagent = LOCAGENT_Graph(use_gpu=self.config.gpu_enabled)
        self.alphaevolve = AlphaEvolve_Controller()
        self.evaluator_pool = EvaluatorPool()
        self.guardrails = GuardrailSystem()
        self.reasoning_curriculum = ReasoningCurriculum()
        self.distributed_training = StreamingDiLoCo()
        self.dipaco_router = DiPaCo_Router()
        self.es_optimizer = EvolutionStrategies()
        
        # Orchestration state
        self.current_cycle = 0
        self.cycle_results = []
        self.active_tasks = {}
        self.telemetry = {}
        
        # Registries
        self.program_registry = {}
        self.skill_registry = {}
        self.model_registry = {}
    
    async def start(self, repo_path: str, task_description: str,
                   context: Dict[str, Any] = None) -> None:
        """
        Start the UE-SEA system.
        
        Args:
            repo_path: Path to the repository to analyze
            task_description: Description of the improvement task
            context: Additional context
        """
        print("Starting UE-SEA Orchestrator...")
        
        if context is None:
            context = {}
        
        # Initialize system
        await self._initialize_system(repo_path, task_description, context)
        
        # Start evolution cycles
        await self._run_evolution_cycles(task_description, context)
    
    async def _initialize_system(self, repo_path: str, task_description: str,
                              context: Dict[str, Any]) -> None:
        """Initialize the system components."""
        print("Initializing system components...")
        
        # Build LOCAGENT graph
        print("Building code graph...")
        await self.locagent.build_from_repository(repo_path)
        
        # Initialize skill paths
        print("Initializing skill paths...")
        await self._initialize_skill_paths()
        
        # Initialize distributed training
        print("Initializing distributed training...")
        await self.distributed_training.initialize()
        
        # Initialize ES optimizer
        print("Initializing Evolution Strategies...")
        await self.es_optimizer.initialize()
        
        print("System initialization complete!")
    
    async def _initialize_skill_paths(self) -> None:
        """Initialize reasoning skill paths."""
        # Localization skill
        localize_skill = SkillPath(
            id="localize",
            role="bug_localization",
            description="Localize bugs and issues in code"
        )
        self.skill_registry["localize"] = localize_skill
        
        # Verification skill
        verify_skill = SkillPath(
            id="verify",
            role="code_verification",
            description="Verify code correctness and properties"
        )
        self.skill_registry["verify"] = verify_skill
        
        # Refactoring skill
        refactor_skill = SkillPath(
            id="refactor",
            role="code_refactoring",
            description="Refactor code for better structure"
        )
        self.skill_registry["refactor"] = refactor_skill
        
        # Documentation skill
        doc_skill = SkillPath(
            id="document",
            role="documentation_generation",
            description="Generate code documentation"
        )
        self.skill_registry["document"] = doc_skill
    
    async def _run_evolution_cycles(self, task_description: str,
                                   context: Dict[str, Any]) -> None:
        """Run the main evolution cycles."""
        print(f"Starting evolution cycles (max: {self.config.max_cycles})")
        
        for cycle_id in range(self.config.max_cycles):
            print(f"\n=== Cycle {cycle_id + 1}/{self.config.max_cycles} ===")
            
            try:
                result = await self._run_single_cycle(cycle_id, task_description, context)
                self.cycle_results.append(result)
                
                if not result.success:
                    print(f"Cycle {cycle_id + 1} failed: {result.errors}")
                    break
                
                # Check for convergence
                if self._check_convergence():
                    print(f"Converged at cycle {cycle_id + 1}")
                    break
                
            except Exception as e:
                print(f"Cycle {cycle_id + 1} error: {e}")
                break
        
        print(f"\nEvolution complete! Ran {len(self.cycle_results)} cycles")
        await self._finalize_system()
    
    async def _run_single_cycle(self, cycle_id: int, task_description: str,
                               context: Dict[str, Any]) -> CycleResult:
        """Run a single evolution cycle."""
        start_time = time.time()
        cycle_metrics = {}
        artifacts = []
        errors = []
        
        try:
            # Stage 1: Index (if needed)
            if cycle_id % self.config.index_frequency == 0:
                print("  Indexing code graph...")
                await self._index_stage(cycle_metrics)
            
            # Stage 2: Evolve
            print("  Running evolution...")
            evolution_result = await self._evolve_stage(task_description, context)
            cycle_metrics["evolution"] = evolution_result
            artifacts.extend(evolution_result.get("artifacts", []))
            
            # Stage 3: Evaluate
            print("  Evaluating candidates...")
            evaluation_result = await self._evaluate_stage(evolution_result)
            cycle_metrics["evaluation"] = evaluation_result
            
            # Stage 4: Select
            print("  Selecting elites...")
            selection_result = await self._select_stage(evaluation_result)
            cycle_metrics["selection"] = selection_result
            
            # Stage 5: Train (if needed)
            if cycle_id % 5 == 0:  # Train every 5 cycles
                print("  Training skills...")
                training_result = await self._train_stage()
                cycle_metrics["training"] = training_result
            
            # Stage 6: Deploy (if threshold met)
            if selection_result.get("best_score", 0) >= self.config.deployment_threshold:
                print("  Deploying elite...")
                deployment_result = await self._deploy_stage(selection_result)
                cycle_metrics["deployment"] = deployment_result
                artifacts.extend(deployment_result.get("artifacts", []))
            
            duration = time.time() - start_time
            
            return CycleResult(
                cycle_id=cycle_id,
                stage=CycleStage.EVOLVE,  # Main stage
                success=True,
                duration=duration,
                metrics=cycle_metrics,
                artifacts=artifacts,
                errors=errors
            )
            
        except Exception as e:
            duration = time.time() - start_time
            errors.append(str(e))
            
            return CycleResult(
                cycle_id=cycle_id,
                stage=CycleStage.EVOLVE,
                success=False,
                duration=duration,
                metrics=cycle_metrics,
                artifacts=artifacts,
                errors=errors
            )
    
    async def _index_stage(self, metrics: Dict[str, Any]) -> None:
        """Index stage: Update code graph."""
        # Re-index repository if needed
        # This would involve re-parsing changed files
        metrics["index_updated"] = True
        metrics["entities_count"] = len(self.locagent._entities)
        metrics["relations_count"] = len(self.locagent._relations)
    
    async def _evolve_stage(self, task_description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evolve stage: Run evolutionary improvement."""
        # Get current best program
        best_program = self._get_best_program()
        
        # Run evolution
        evolution_result = await self.alphaevolve.evolve(
            best_program, task_description, context
        )
        
        return {
            "best_program": evolution_result.best_program,
            "best_score": evolution_result.best_score,
            "generation": evolution_result.generation,
            "total_evaluations": evolution_result.total_evaluations,
            "artifacts": [evolution_result.best_program]
        }
    
    async def _evaluate_stage(self, evolution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate stage: Staged evaluation of candidates."""
        program = evolution_result["best_program"]
        
        # Run staged evaluation
        score = await self.evaluator_pool.evaluate_staged(program)
        
        # Check guardrails
        guardrail_result = await self.guardrails.evaluate_guardrails(program, {"score": score})
        
        return {
            "score": score,
            "guardrail_passed": guardrail_result.passed,
            "hacking_detections": len(guardrail_result.hacking_detections),
            "safety_score": guardrail_result.safety_score,
            "warnings": guardrail_result.warnings
        }
    
    async def _select_stage(self, evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Select stage: Elite selection and program registry update."""
        score = evaluation_result["score"]
        
        # Update program registry
        program_id = f"program_{int(time.time())}"
        self.program_registry[program_id] = {
            "score": score,
            "timestamp": time.time(),
            "evaluation_result": evaluation_result
        }
        
        return {
            "program_id": program_id,
            "best_score": score,
            "registry_size": len(self.program_registry)
        }
    
    async def _train_stage(self) -> Dict[str, Any]:
        """Train stage: Skill training and model updates."""
        training_results = {}
        
        # Train each skill path
        for skill_id, skill in self.skill_registry.items():
            print(f"    Training {skill_id}...")
            
            # Run training for this skill
            training_result = await self.reasoning_curriculum.train_skill(
                skill, max_steps=self.config.training_budget
            )
            
            training_results[skill_id] = training_result
        
        return training_results
    
    async def _deploy_stage(self, selection_result: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy stage: Deploy elite program."""
        program_id = selection_result["program_id"]
        
        # Deploy the elite program
        deployment_result = {
            "deployed_program": program_id,
            "deployment_time": time.time(),
            "artifacts": [f"deployed_{program_id}"]
        }
        
        return deployment_result
    
    def _get_best_program(self) -> str:
        """Get the current best program."""
        if not self.program_registry:
            return "# Default starting program\nprint('Hello, UE-SEA!')"
        
        # Find program with highest score
        best_program_id = max(
            self.program_registry.keys(),
            key=lambda pid: self.program_registry[pid]["score"]
        )
        
        return f"# Best program (score: {self.program_registry[best_program_id]['score']})\nprint('Elite program')"
    
    def _check_convergence(self) -> bool:
        """Check if the system has converged."""
        if len(self.cycle_results) < 10:
            return False
        
        # Check if scores have plateaued
        recent_scores = [
            result.metrics.get("selection", {}).get("best_score", 0)
            for result in self.cycle_results[-10:]
        ]
        
        if not recent_scores:
            return False
        
        # Check if variance is low
        mean_score = sum(recent_scores) / len(recent_scores)
        variance = sum((s - mean_score) ** 2 for s in recent_scores) / len(recent_scores)
        
        return variance < 0.01  # Low variance indicates convergence
    
    async def _finalize_system(self) -> None:
        """Finalize the system and save results."""
        print("Finalizing system...")
        
        # Save telemetry
        await self._save_telemetry()
        
        # Export results
        await self._export_results()
        
        print("System finalization complete!")
    
    async def _save_telemetry(self) -> None:
        """Save system telemetry."""
        telemetry = {
            "total_cycles": len(self.cycle_results),
            "successful_cycles": sum(1 for r in self.cycle_results if r.success),
            "program_registry_size": len(self.program_registry),
            "skill_registry_size": len(self.skill_registry),
            "cycle_results": [
                {
                    "cycle_id": r.cycle_id,
                    "success": r.success,
                    "duration": r.duration,
                    "metrics": r.metrics,
                    "errors": r.errors
                }
                for r in self.cycle_results
            ]
        }
        
        with open("ue_sea_telemetry.json", "w") as f:
            json.dump(telemetry, f, indent=2)
    
    async def _export_results(self) -> None:
        """Export final results."""
        results = {
            "best_programs": self.program_registry,
            "skill_paths": {k: v.to_dict() for k, v in self.skill_registry.items()},
            "final_metrics": self._get_final_metrics()
        }
        
        with open("ue_sea_results.json", "w") as f:
            json.dump(results, f, indent=2)
    
    def _get_final_metrics(self) -> Dict[str, Any]:
        """Get final system metrics."""
        if not self.cycle_results:
            return {}
        
        successful_cycles = [r for r in self.cycle_results if r.success]
        
        return {
            "total_cycles": len(self.cycle_results),
            "successful_cycles": len(successful_cycles),
            "success_rate": len(successful_cycles) / len(self.cycle_results),
            "average_cycle_duration": sum(r.duration for r in successful_cycles) / len(successful_cycles),
            "best_score": max(
                r.metrics.get("selection", {}).get("best_score", 0)
                for r in successful_cycles
            ),
            "total_artifacts": sum(len(r.artifacts) for r in successful_cycles)
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "current_cycle": self.current_cycle,
            "total_cycles": len(self.cycle_results),
            "program_registry_size": len(self.program_registry),
            "skill_registry_size": len(self.skill_registry),
            "active_tasks": len(self.active_tasks),
            "config": self.config.__dict__
        }
    
    async def stop(self) -> None:
        """Stop the orchestrator."""
        print("Stopping UE-SEA Orchestrator...")
        
        # Cancel active tasks
        for task in self.active_tasks.values():
            task.cancel()
        
        # Finalize system
        await self._finalize_system()
        
        print("Orchestrator stopped.")
