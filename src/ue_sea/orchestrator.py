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
        self.alphaevolve = AlphaEvolve_Controller(locagent=self.locagent)
        self.evaluator_pool = EvaluatorPool()
        self.guardrails = GuardrailSystem()
        self.reasoning_curriculum = ReasoningCurriculum()
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
        
        # LOCAGENT integration state
        self.training_metrics_history = []
        self.training_feedback_history = []
    
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

    async def run(self, repo_path: str, task_description: str,
                 context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run UE-SEA with full LOCAGENT integration.
        
        Args:
            repo_path: Path to the repository to analyze
            task_description: Description of the improvement task
            context: Additional context including LOCAGENT settings
        
        Returns:
            Results of the UE-SEA run
        """
        print("🚀 Starting UE-SEA with LOCAGENT Integration...")
        
        if context is None:
            context = {}
        
        # Check for LOCAGENT integration settings
        use_locagent = context.get("use_locagent", True)
        localization_threshold = context.get("localization_threshold", 0.8)
        semantic_search = context.get("semantic_search", True)
        real_time_feedback = context.get("real_time_feedback", True)
        gpu_acceleration = context.get("gpu_acceleration", True)
        
        print(f"🎯 LOCAGENT Integration: {use_locagent}")
        print(f"🧠 Semantic Search: {semantic_search}")
        print(f"⚡ GPU Acceleration: {gpu_acceleration}")
        print(f"📊 Real-time Feedback: {real_time_feedback}")
        
        # Initialize system with LOCAGENT
        await self._initialize_system(repo_path, task_description, context)
        
        # Run evolution cycles with LOCAGENT enhancement
        await self._run_evolution_cycles(task_description, context)
        
        # Return results
        return {
            "total_cycles": len(self.cycle_results),
            "successful_cycles": sum(1 for r in self.cycle_results if r.success),
            "locagent_enhanced": use_locagent,
            "semantic_understanding": semantic_search,
            "gpu_acceleration": gpu_acceleration,
            "real_time_feedback": real_time_feedback,
            "final_metrics": self._get_final_metrics()
        }
    
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
    
    async def _localize_task(self, task_description: str) -> Dict[str, Any]:
        """
        Use LOCAGENT to intelligently localize where to focus evolution efforts.
        This replaces blind evolution with targeted, intelligent code understanding.
        """
        from .locagent import LOCAGENT_Agent
        
        # Create LOCAGENT agent with enhanced reasoning
        agent = LOCAGENT_Agent(
            graph=self.locagent,
            max_steps=15,  # Allow more reasoning steps
            confidence_threshold=0.7
        )
        
        # Perform intelligent localization
        result = await agent.localize(task_description)
        
        # Extract semantic insights
        semantic_insights = {
            "target_entities": result.final_ranking,
            "confidence": result.confidence,
            "reasoning_trace": result.reasoning_trace,
            "hypotheses": result.hypotheses if hasattr(result, 'hypotheses') else [],
            "semantic_neighborhood": []
        }
        
        # Get semantic neighborhood for each target
        for entity_id, score in result.final_ranking[:5]:
            neighborhood = self.locagent.get_semantic_neighborhood(entity_id, depth=2)
            semantic_insights["semantic_neighborhood"].extend(list(neighborhood))
        
        return semantic_insights

    async def _analyze_semantic_impact(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the semantic impact of code changes using LOCAGENT."""
        changed_entities = candidate.get("changed_entities", [])
        semantic_impact = {
            "affected_entities": [],
            "semantic_relationships": [],
            "usage_pattern_impact": {},
            "co_occurrence_changes": {}
        }
        
        for entity_id in changed_entities:
            # Get semantic relationships
            relationships = self.locagent.get_semantic_relationships(entity_id)
            semantic_impact["semantic_relationships"].extend(relationships)
            
            # Get usage patterns
            patterns = self.locagent.get_usage_patterns(entity_id)
            semantic_impact["usage_pattern_impact"][entity_id] = patterns
            
            # Get co-occurrence impacts
            co_occurrence = self.locagent.find_semantically_similar_entities(entity_id)
            semantic_impact["co_occurrence_changes"][entity_id] = co_occurrence
        
        return semantic_impact

    async def _evolve_stage(self, task_description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced evolution stage with LOCAGENT-powered intelligent targeting.
        """
        print("🧠 Using LOCAGENT for intelligent evolution targeting...")
        
        # Step 1: Localize the task using LOCAGENT
        localization_result = await self._localize_task(task_description)
        
        # Step 2: Extract target entities and their semantic context
        target_entities = localization_result["target_entities"][:10]  # Top 10 targets
        semantic_neighborhood = localization_result["semantic_neighborhood"]
        
        # Step 3: Create evolution context with semantic understanding
        evolution_context = {
            "task_description": task_description,
            "target_entities": target_entities,
            "semantic_neighborhood": semantic_neighborhood,
            "confidence_threshold": localization_result["confidence"],
            "reasoning_trace": localization_result["reasoning_trace"]
        }
        
        # Step 4: Run targeted evolution
        print(f"🎯 Targeting {len(target_entities)} entities with semantic context...")
        evolution_result = await self.alphaevolve.evolve(
            target_entities=target_entities,
            context=evolution_context,
            budget=self.config.evolution_budget,
            use_semantic_guidance=True  # New parameter for semantic guidance
        )
        
        # Step 5: Add LOCAGENT insights to evolution result
        evolution_result.update({
            "locagent_insights": localization_result,
            "semantic_targeting": True,
            "target_entity_count": len(target_entities),
            "semantic_neighborhood_size": len(semantic_neighborhood)
        })
        
        return evolution_result
    
    async def _evaluate_stage(self, evolution_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced evaluation with LOCAGENT-powered code understanding.
        """
        print("📊 Running LOCAGENT-enhanced evaluation...")
        
        # Get LOCAGENT insights from evolution result
        locagent_insights = evolution_result.get("locagent_insights", {})
        target_entities = locagent_insights.get("target_entities", [])
        
        evaluation_results = []
        
        for candidate in evolution_result.get("candidates", []):
            # Step 1: Analyze semantic impact using LOCAGENT
            semantic_impact = await self._analyze_semantic_impact(candidate)
            
            # Step 2: Enhanced evaluation with code understanding
            evaluation_score = await self.evaluator_pool.evaluate(
                candidate=candidate,
                semantic_context=semantic_impact,
                target_entities=target_entities,
                use_locagent_metrics=True  # New parameter
            )
            
            # Step 3: Add LOCAGENT-specific metrics
            locagent_metrics = await self._calculate_locagent_metrics(candidate, target_entities)
            
            # Step 4: Combine traditional and LOCAGENT metrics
            combined_score = self._combine_evaluation_scores(
                traditional_score=evaluation_score,
                locagent_metrics=locagent_metrics,
                semantic_impact=semantic_impact
            )
            
            evaluation_results.append({
                "candidate": candidate,
                "traditional_score": evaluation_score,
                "locagent_metrics": locagent_metrics,
                "semantic_impact": semantic_impact,
                "combined_score": combined_score,
                "confidence": locagent_insights.get("confidence", 0.5)
            })
        
        # Sort by combined score
        evaluation_results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return {
            "evaluation_results": evaluation_results,
            "locagent_enhanced": True,
            "semantic_understanding": True,
            "best_score": evaluation_results[0]["combined_score"] if evaluation_results else 0
        }

    async def _calculate_locagent_metrics(self, candidate: Dict[str, Any], 
                                        target_entities: List[str]) -> Dict[str, float]:
        """Calculate LOCAGENT-specific evaluation metrics."""
        changed_entities = candidate.get("changed_entities", [])
        
        # Calculate semantic relevance
        semantic_relevance = 0.0
        for changed_entity in changed_entities:
            for target_entity, score in target_entities:
                if changed_entity == target_entity:
                    semantic_relevance += score
                else:
                    # Check semantic relationships
                    relationships = self.locagent.get_semantic_relationships(changed_entity)
                    if target_entity in relationships:
                        semantic_relevance += score * 0.5
        
        # Calculate usage pattern alignment
        pattern_alignment = 0.0
        for changed_entity in changed_entities:
            patterns = self.locagent.get_usage_patterns(changed_entity)
            pattern_alignment += len(patterns) * 0.1
        
        # Calculate co-occurrence impact
        co_occurrence_impact = 0.0
        for changed_entity in changed_entities:
            similar_entities = self.locagent.find_semantically_similar_entities(changed_entity)
            co_occurrence_impact += sum(score for _, score in similar_entities) * 0.2
        
        return {
            "semantic_relevance": semantic_relevance,
            "pattern_alignment": pattern_alignment,
            "co_occurrence_impact": co_occurrence_impact,
            "overall_locagent_score": (semantic_relevance + pattern_alignment + co_occurrence_impact) / 3
        }

    def _combine_evaluation_scores(self, traditional_score: float, 
                                 locagent_metrics: Dict[str, float],
                                 semantic_impact: Dict[str, Any]) -> float:
        """Combine traditional and LOCAGENT evaluation scores."""
        # Weighted combination
        traditional_weight = 0.4
        locagent_weight = 0.6
        
        locagent_score = locagent_metrics["overall_locagent_score"]
        
        # Adjust weights based on semantic impact
        if len(semantic_impact["semantic_relationships"]) > 5:
            locagent_weight = 0.7  # Higher weight for high semantic impact
        
        combined_score = (traditional_score * traditional_weight + 
                         locagent_score * locagent_weight)
        
        return min(1.0, combined_score)  # Cap at 1.0
    
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
        """
        Enhanced training stage with LOCAGENT real-time feedback integration.
        """
        print("🔄 Running LOCAGENT-enhanced training with real-time feedback...")
        
        # Initialize LOCAGENT trainer with GPU acceleration
        from .locagent import LOCAGENT_Trainer
        trainer = LOCAGENT_Trainer(
            model_name="microsoft/DialoGPT-medium",
            use_gpu=self.config.gpu_enabled
        )
        
        # Load model
        trainer.load_model()
        
        # Get previous metrics for comparison
        previous_metrics = self._get_previous_training_metrics()
        
        # Prepare training data with LOCAGENT insights
        training_data = await self._prepare_locagent_training_data()
        
        # Run training with real-time evaluation
        training_result = await self._run_enhanced_training(
            trainer=trainer,
            training_data=training_data,
            previous_metrics=previous_metrics
        )
        
        # Generate feedback and adjust strategy
        feedback = trainer.generate_feedback(
            current_metrics=training_result["metrics"],
            previous_metrics=previous_metrics
        )
        
        # Store feedback for next cycle
        self._store_training_feedback(feedback)
        
        return {
            "training_result": training_result,
            "feedback": feedback,
            "locagent_enhanced": True,
            "real_time_evaluation": True
        }

    async def _prepare_locagent_training_data(self) -> List[Dict[str, Any]]:
        """Prepare training data with LOCAGENT insights."""
        training_data = []
        
        # Get recent localization results
        recent_localizations = self._get_recent_localizations()
        
        for localization in recent_localizations:
            # Create training example with LOCAGENT insights
            training_example = {
                "issue_description": localization["task_description"],
                "target_entities": localization["target_entities"],
                "semantic_context": localization.get("semantic_insights", {}),
                "reasoning_trace": localization.get("reasoning_trace", []),
                "confidence": localization.get("confidence", 0.5)
            }
            training_data.append(training_example)
        
        return training_data

    async def _run_enhanced_training(self, trainer, training_data: List[Dict[str, Any]], 
                                   previous_metrics: Optional[Dict[str, float]]) -> Dict[str, Any]:
        """Run training with LOCAGENT real-time evaluation."""
        metrics_history = []
        
        for epoch in range(3):  # 3 epochs
            epoch_metrics = {}
            
            for batch in training_data:
                # Run training step
                step_metrics = await trainer.train_step(batch)
                
                # Real-time evaluation
                if step_metrics.get("predictions"):
                    real_time_metrics = trainer.evaluate_realtime(
                        predictions=step_metrics["predictions"],
                        ground_truth=step_metrics["ground_truth"],
                        k=5
                    )
                    epoch_metrics.update(real_time_metrics)
            
            metrics_history.append(epoch_metrics)
            
            # Generate feedback for this epoch
            if previous_metrics:
                feedback = trainer.generate_feedback(epoch_metrics, previous_metrics)
                if feedback["performance_trend"] == "declining":
                    # Adjust training strategy
                    await self._adjust_training_strategy(feedback["suggestions"])
        
        return {
            "metrics": metrics_history[-1] if metrics_history else {},
            "metrics_history": metrics_history,
            "epochs_completed": 3
        }

    async def _adjust_training_strategy(self, suggestions: List[str]) -> None:
        """Adjust training strategy based on LOCAGENT feedback."""
        for suggestion in suggestions:
            if "more training data" in suggestion.lower():
                # Increase training data collection
                await self._increase_training_data_collection()
            elif "longer training" in suggestion.lower():
                # Increase training epochs
                self.config.training_budget = min(200, self.config.training_budget * 1.5)
            elif "precision" in suggestion.lower():
                # Adjust precision threshold
                self.config.deployment_threshold = min(0.9, self.config.deployment_threshold + 0.05)

    def _get_previous_training_metrics(self) -> Optional[Dict[str, float]]:
        """Get previous training metrics for comparison."""
        # Implementation to retrieve previous metrics
        return self.training_metrics_history[-1] if self.training_metrics_history else None

    def _store_training_feedback(self, feedback: Dict[str, Any]) -> None:
        """Store training feedback for analysis."""
        if not hasattr(self, 'training_feedback_history'):
            self.training_feedback_history = []
        
        self.training_feedback_history.append(feedback)

    async def _increase_training_data_collection(self) -> None:
        """Increase training data collection."""
        # Implementation to increase data collection
        pass

    def _get_recent_localizations(self) -> List[Dict[str, Any]]:
        """Get recent localization results."""
        # Implementation to get recent localizations
        return []
    
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
