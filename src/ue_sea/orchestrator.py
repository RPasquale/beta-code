"""
UE-SEA Orchestrator: Main coordinator for the unified system.

Schedules cycles: Index → Evolve → Evaluate → Select → Train → Deploy
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple, Callable, Awaitable
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import numpy as np
from pathlib import Path

from .locagent import LOCAGENT_Graph
from .alphaevolve import AlphaEvolve_Controller
from .evaluation import EvaluatorPool, GuardrailSystem
from .reasoning import SkillPath, ReasoningCurriculum
from .optimizer import EvolutionStrategies
from .rl.drgrpo_orchestrator import DrGRPOOrchestrator, DrGRPOConfig
from .spct import SPCTOrchestrator, SPCTConfig, SPCTTrainingData


class CycleStage(Enum):
    """Stages of the UE-SEA cycle."""
    INDEX = "index"           # Build code graph
    EVOLVE = "evolve"         # Evolutionary improvement
    EVALUATE = "evaluate"    # Staged evaluation
    SELECT = "select"        # Elite selection
    TRAIN = "train"          # Skill training
    SPCT = "spct"           # Self-Programming Code Training
    DEPLOY = "deploy"        # Deployment


@dataclass
class CycleConfig:
    """Configuration for UE-SEA cycles."""
    max_cycles: int = 100
    cycle_timeout: int = 3600  # seconds
    index_frequency: int = 10  # Re-index every N cycles
    evolution_budget: int = 1000  # Max evaluations per cycle
    training_budget: int = 100  # Max training steps per cycle
    spct_budget: int = 200  # Max SPCT training steps per cycle
    deployment_threshold: float = 0.8  # Min score for deployment
    parallel_workers: int = 4
    gpu_enabled: bool = True
    spct_enabled: bool = True  # Enable SPCT training
    spct_frequency: int = 3  # Run SPCT every N cycles


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
        
        # Dr.GRPO RL integration
        self.drgrpo_orchestrator = None
        self.rl_enabled = False
        
        # SPCT integration
        self.spct_orchestrator = None
        self.spct_enabled = False
    
    async def start(self, repo_path: str, task_description: str = None,
                   context: Dict[str, Any] = None) -> None:
        """
        Start the UE-SEA system.
        
        Args:
            repo_path: Path to the repository to analyze
            task_description: Description of the improvement task (optional for end-to-end training)
            context: Additional context
        """
        print("Starting UE-SEA Orchestrator...")
        
        if context is None:
            context = {}
        
        # For end-to-end training, task is derived from training data
        if task_description is None:
            print("🧠 End-to-end training mode: Tasks will be derived from training data")
            task_description = "End-to-end agent training from diverse tasks"
        
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

    async def train_end_to_end(self, training_data_path: str, 
                              context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Train the agent end-to-end using training data that specifies tasks.
        
        Args:
            training_data_path: Path to training data (JSONL format)
            context: Additional context for training
        
        Returns:
            Training results and metrics
        """
        print("🚀 Starting End-to-End Agent Training...")
        
        if context is None:
            context = {}
        
        # Load training data
        training_tasks = await self._load_training_tasks(training_data_path)
        print(f"📚 Loaded {len(training_tasks)} training tasks")
        
        # Initialize system for end-to-end training
        await self._initialize_end_to_end_training(training_tasks, context)
        
        # Run training cycles using tasks from training data
        training_results = await self._run_end_to_end_training_cycles(training_tasks, context)
        
        return {
            "training_completed": True,
            "total_tasks": len(training_tasks),
            "training_cycles": len(training_results),
            "successful_cycles": sum(1 for r in training_results if r.get("success", False)),
            "final_metrics": self._get_final_metrics(),
            "agent_capabilities": self._get_agent_capabilities()
        }
    
    async def train_end_to_end_rl(self, training_data_path: str, 
                                 rl_config: Dict[str, Any] = None,
                                 reward_fn: Optional[Callable[[str, str, Any], Awaitable[float]]] = None) -> Dict[str, Any]:
        """
        Run full end-to-end RL training with Dr.GRPO orchestrator.
        
        This includes:
        1. LOCAGENT as training tool
        2. SFT with reasoning traces
        3. Reward model training from reasoning traces
        4. Dr.GRPO RL training
        5. AlphaCode evolution
        6. End-to-end learning loop
        """
        print("🚀 Starting Full End-to-End RL Training with Dr.GRPO...")
        
        # Initialize Dr.GRPO orchestrator
        drgrpo_config = DrGRPOConfig()
        if rl_config:
            # Update config with provided parameters
            for key, value in rl_config.items():
                if hasattr(drgrpo_config, key):
                    setattr(drgrpo_config, key, value)
        
        self.drgrpo_orchestrator = DrGRPOOrchestrator(drgrpo_config, reward_fn=reward_fn)
        self.rl_enabled = True
        
        # Initialize with training data
        await self.drgrpo_orchestrator.initialize(training_data_path)
        
        # Run full end-to-end training pipeline
        results = await self.drgrpo_orchestrator.run_end_to_end_training()
        
        # Update orchestrator state with RL results
        self._update_orchestrator_from_rl_results(results)
        
        print("✅ Full End-to-End RL Training Complete!")
        
        return {
            "rl_results": results,
            "orchestrator_metrics": self.get_metrics(),
            "agent_capabilities": len(self.skill_registry)
        }

    async def train_full_agent(self, spct_training_data_path: str, rl_training_data_path: str,
                               spct_config: Optional[Dict[str, Any]] = None,
                               rl_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run SPCT followed by Dr.GRPO RL using the SPCT judge as reward."""

        spct_results = await self.train_spct(spct_training_data_path, spct_config)

        if not self.spct_orchestrator:
            raise RuntimeError("SPCT orchestration failed to initialize")

        pointwise_rm = self.spct_orchestrator.pointwise_rm
        score_min, score_max = pointwise_rm.config.score_range
        score_span = max(1e-5, score_max - score_min)

        async def spct_reward_fn(question: str, response: str, trace: Any) -> float:
            num_samples = max(1, pointwise_rm.config.inference_samples)
            judgements = await pointwise_rm.generate_multiple_judgements(question, [response], num_samples=num_samples)
            if not judgements:
                return 0.0
            scores = [j.scores[0] for j in judgements]
            confidences = [j.confidence for j in judgements]
            avg_score = float(np.mean(scores))
            avg_conf = float(np.mean(confidences))
            normalized = (avg_score - score_min) / score_span
            reward = (normalized * 2.0) - 1.0
            reward = reward * avg_conf
            return float(np.clip(reward, -1.0, 1.0))

        rl_results = await self.train_end_to_end_rl(
            training_data_path=rl_training_data_path,
            rl_config=rl_config,
            reward_fn=spct_reward_fn,
        )

        return {
            "spct_results": spct_results,
            "rl_results": rl_results,
        }
    
    def _update_orchestrator_from_rl_results(self, rl_results: Dict[str, Any]) -> None:
        """Update orchestrator state from RL training results."""
        # Update skill registry with learned capabilities
        if "drgrpo_results" in rl_results:
            drgrpo_metrics = rl_results["drgrpo_results"]
            if "avg_reward" in drgrpo_metrics:
                # Create or update reasoning skill based on RL performance
                from .reasoning.skill_path import SkillType
                reasoning_skill = SkillPath(
                    skill_id="rl_reasoning",
                    skill_type=SkillType.LOCALIZE,  # Use available skill type
                    description="RL-trained reasoning capability"
                )
                reasoning_skill.metrics.accuracy = drgrpo_metrics.get("avg_reward", 0.0)
                self.skill_registry["rl_reasoning"] = reasoning_skill
        
        # Update LOCAGENT with learned code patterns
        if "evolution_results" in rl_results:
            evolution_metrics = rl_results["evolution_results"]
            if "avg_fitness" in evolution_metrics:
                # Update LOCAGENT with evolved code patterns
                self._update_locagent_with_evolution(evolution_metrics)
        
        # Update training metrics history
        import time
        self.training_metrics_history.append({
            "timestamp": time.time(),
            "rl_results": rl_results,
            "agent_capabilities": len(self.skill_registry)
        })
    
    def _update_locagent_with_evolution(self, evolution_metrics: Dict[str, Any]) -> None:
        """Update LOCAGENT with evolved code patterns."""
        try:
            # Add evolved code patterns to LOCAGENT
            if hasattr(self.locagent, 'add_evolved_patterns'):
                self.locagent.add_evolved_patterns(evolution_metrics)
            else:
                # Update LOCAGENT graph with new patterns
                print("🧬 Updating LOCAGENT with evolved code patterns...")
        except Exception as e:
            print(f"Warning: Failed to update LOCAGENT with evolution: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current orchestrator metrics."""
        return {
            "current_cycle": self.current_cycle,
            "total_cycles": len(self.cycle_results),
            "programs_registered": len(self.program_registry),
            "skills_registered": len(self.skill_registry),
            "rl_enabled": self.rl_enabled,
            "spct_enabled": self.spct_enabled,
            "training_metrics_history": len(self.training_metrics_history)
        }
    
    async def train_spct(self, training_data_path: str, 
                        spct_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run SPCT (Self-Principled Critique Tuning) training.
        
        This includes:
        1. Pointwise Generative RM with self-principles
        2. RFT cold start with trajectory filtering
        3. GRPO online RL with rule-based rewards
        4. Meta-RM for sample quality filtering
        5. Inference-time scaling with voting
        """
        print("🚀 Starting SPCT (Self-Principled Critique Tuning)...")
        
        # Initialize SPCT orchestrator
        config = SPCTConfig()
        if spct_config:
            # Update config with provided parameters
            for key, value in spct_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        self.spct_orchestrator = SPCTOrchestrator(config)
        self.spct_enabled = True
        
        # Load and prepare training data
        training_data = await self._load_spct_training_data(training_data_path)
        
        # Run full SPCT pipeline
        results = await self.spct_orchestrator.train_full_pipeline(training_data)
        
        # Update orchestrator state with SPCT results
        self._update_orchestrator_from_spct_results(results)
        
        print("✅ SPCT Training Complete!")
        
        return {
            "spct_results": results,
            "orchestrator_metrics": self.get_metrics(),
            "agent_capabilities": len(self.skill_registry)
        }
    
    async def _load_spct_training_data(self, data_path: str) -> List[SPCTTrainingData]:
        """Load training data for SPCT."""
        training_data = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    
                    # Extract query and responses
                    query = item.get("instruction", item.get("prompt", ""))
                    responses = []
                    
                    # Get current code as one response
                    current_code = item.get("current_program", item.get("code", ""))
                    if current_code:
                        responses.append(current_code)
                    
                    # Get target code as another response
                    target_code = item.get("target_output", item.get("target", ""))
                    if target_code:
                        responses.append(target_code)
                    
                    # If we have both, target is usually better (index 1)
                    # If only one, it's the best (index 0)
                    best_index = 1 if len(responses) == 2 else 0
                    
                    if query and responses:
                        training_data.append(SPCTTrainingData(
                            query=query,
                            responses=responses,
                            best_index=best_index,
                            metadata=item
                        ))
        
        print(f"📚 Loaded {len(training_data)} SPCT training examples")
        return training_data
    
    def _update_orchestrator_from_spct_results(self, spct_results: Any) -> None:
        """Update orchestrator state from SPCT results."""
        # Accept dataclass or mapping
        if hasattr(spct_results, "rft_results"):
            rft_metrics = spct_results.rft_results
            serializable = asdict(spct_results)
        else:
            rft_metrics = spct_results.get("rft_results", {}) if isinstance(spct_results, dict) else {}
            serializable = spct_results

        if rft_metrics:
            if "accepted_traces" in rft_metrics:
                from .reasoning.skill_path import SkillType
                spct_skill = SkillPath(
                    skill_id="spct_reasoning",
                    skill_type=SkillType.LOCALIZE,
                    description="SPCT-trained reasoning capability"
                )
                spct_skill.metrics.accuracy = rft_metrics.get("acceptance_rate", 0.0)
                self.skill_registry["spct_reasoning"] = spct_skill

        import time
        self.training_metrics_history.append({
            "timestamp": time.time(),
            "spct_results": serializable,
            "agent_capabilities": len(self.skill_registry)
        })
    
    async def inference_with_spct(self, query: str, responses: List[str], 
                                 num_samples: int = 8) -> Dict[str, Any]:
        """Run inference using SPCT with scaling."""
        if not self.spct_enabled or not self.spct_orchestrator:
            raise ValueError("SPCT not initialized. Run train_spct() first.")
        
        return await self.spct_orchestrator.inference_with_scaling(
            query, responses, num_samples
        )
    
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
            
            # Stage 6: SPCT (if enabled and needed)
            if self.config.spct_enabled and cycle_id % self.config.spct_frequency == 0:
                print("  Running SPCT training...")
                spct_result = await self._spct_stage()
                cycle_metrics["spct"] = spct_result
            
            # Stage 7: Deploy (if threshold met)
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

    async def _spct_stage(self) -> Dict[str, Any]:
        """
        SPCT (Self-Programming Code Training) stage.
        
        This stage uses the agent's own generated code and evolution results
        to create training data and improve its own capabilities.
        """
        print("🧠 Running SPCT (Self-Programming Code Training)...")
        
        # Step 1: Collect self-generated code from recent cycles
        self_generated_code = await self._collect_self_generated_code()
        
        # Step 2: Create SPCT training dataset
        spct_dataset = await self._create_spct_dataset(self_generated_code)
        
        # Step 3: Run SPCT training using the existing training pipeline
        spct_result = await self._run_spct_training(spct_dataset)
        
        # Step 4: Update agent capabilities based on SPCT results
        await self._update_agent_capabilities(spct_result)
        
        return {
            "spct_training_completed": True,
            "dataset_size": len(spct_dataset),
            "training_metrics": spct_result.get("metrics", {}),
            "capability_updates": spct_result.get("capability_updates", {}),
            "self_improvement_score": spct_result.get("improvement_score", 0.0)
        }

    async def _collect_self_generated_code(self) -> List[Dict[str, Any]]:
        """Collect code generated by the agent in recent cycles."""
        self_generated = []
        
        # Get recent evolution results
        for result in self.cycle_results[-5:]:  # Last 5 cycles
            if result.success and "evolution" in result.metrics:
                evolution_data = result.metrics["evolution"]
                candidates = evolution_data.get("candidates", [])
                
                for candidate in candidates:
                    if candidate.get("generated_by_agent", False):
                        self_generated.append({
                            "code": candidate.get("code", ""),
                            "task": candidate.get("task", ""),
                            "quality_score": candidate.get("quality_score", 0.0),
                            "cycle_id": result.cycle_id,
                            "generation_method": candidate.get("method", "evolution")
                        })
        
        return self_generated

    async def _create_spct_dataset(self, self_generated_code: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create SPCT training dataset from self-generated code."""
        spct_dataset = []
        
        for item in self_generated_code:
            # Create training examples for different skills
            code = item["code"]
            task = item["task"]
            quality_score = item["quality_score"]
            
            # Skill 1: Code Generation (from task to code)
            spct_dataset.append({
                "prompt": f"Generate code for: {task}",
                "response": code,
                "skill": "code_generation",
                "quality": quality_score,
                "source": "self_generated"
            })
            
            # Skill 2: Code Improvement (from code to better code)
            if quality_score > 0.7:  # Only use high-quality code for improvement
                improved_prompt = f"Improve this code:\n{code}"
                # Use LOCAGENT to suggest improvements
                improvements = await self._suggest_code_improvements(code)
                if improvements:
                    spct_dataset.append({
                        "prompt": improved_prompt,
                        "response": improvements,
                        "skill": "code_improvement",
                        "quality": quality_score,
                        "source": "self_generated"
                    })
            
            # Skill 3: Code Understanding (from code to explanation)
            explanation = await self._generate_code_explanation(code)
            spct_dataset.append({
                "prompt": f"Explain this code:\n{code}",
                "response": explanation,
                "skill": "code_understanding",
                "quality": quality_score,
                "source": "self_generated"
            })
        
        return spct_dataset

    async def _suggest_code_improvements(self, code: str) -> str:
        """Use LOCAGENT to suggest code improvements."""
        # This would use your existing LOCAGENT system to analyze code
        # and suggest improvements
        try:
            # Simple code analysis without LOCAGENT for now
            improvements = []
            
            # Basic code quality checks
            if "def " in code:
                if "return" not in code:
                    improvements.append("Performance: Add return statement to function")
                if len(code.split('\n')) > 20:
                    improvements.append("Readability: Consider breaking down large function")
                if "try:" not in code and "except" not in code:
                    improvements.append("Security: Add error handling")
            
            return "\n".join(improvements) if improvements else "No improvements suggested."
        except Exception:
            return "Unable to analyze code for improvements."

    async def _generate_code_explanation(self, code: str) -> str:
        """Generate explanation for code using LOCAGENT."""
        try:
            # Simple code explanation without LOCAGENT for now
            explanation = "Code Analysis:\n"
            
            if "def " in code:
                explanation += "Purpose: Function definition\n"
                explanation += "Key Functions: 1 function found\n"
            else:
                explanation += "Purpose: Code snippet\n"
                explanation += "Key Functions: No functions found\n"
            
            # Basic complexity analysis
            lines = len(code.split('\n'))
            if lines < 5:
                explanation += "Complexity: Low\n"
            elif lines < 15:
                explanation += "Complexity: Medium\n"
            else:
                explanation += "Complexity: High\n"
            
            return explanation
        except Exception:
            return "Unable to generate code explanation."

    async def _run_spct_training(self, spct_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run SPCT training using the existing training pipeline."""
        if not spct_dataset:
            return {"error": "No SPCT dataset available"}
        
        # Save SPCT dataset to temporary file
        import tempfile
        import json
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in spct_dataset:
                f.write(json.dumps(item) + '\n')
            temp_file = f.name
        
        try:
            # Run training using your existing training pipeline
            # This integrates with your training/train.py
            import subprocess
            import os
            
            # Change to training directory
            training_dir = Path(__file__).parent.parent.parent / "training"
            
            # Run SPCT training
            cmd = [
                "python", "train.py",
                "--stage", "rft",  # Use RFT for SPCT
                "--output-dir", f"outputs/spct_cycle_{self.current_cycle}",
                "--max-samples-per-objective", str(len(spct_dataset)),
                "--validation-ratio", "0.1",
                "--learning-rate", "1e-5",  # Lower LR for SPCT
                "--num-train-epochs", "1",
                "--wandb-project", "ue-sea-spct"
            ]
            
            result = subprocess.run(
                cmd,
                cwd=training_dir,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "metrics": {"training_completed": True},
                    "improvement_score": 0.8,  # Placeholder
                    "capability_updates": {
                        "code_generation": 0.1,
                        "code_improvement": 0.1,
                        "code_understanding": 0.1
                    }
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr,
                    "metrics": {}
                }
                
        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    async def _update_agent_capabilities(self, spct_result: Dict[str, Any]) -> None:
        """Update agent capabilities based on SPCT results."""
        if not spct_result.get("success", False):
            return
        
        capability_updates = spct_result.get("capability_updates", {})
        
        # Update skill registry with improved capabilities
        for skill_name, improvement in capability_updates.items():
            if skill_name in self.skill_registry:
                skill = self.skill_registry[skill_name]
                # Update skill metrics
                skill.metrics.accuracy = min(1.0, skill.metrics.accuracy + improvement)
                skill.metrics.f1_score = min(1.0, skill.metrics.f1_score + improvement)
        
        # Update evolution strategies based on SPCT learning
        if hasattr(self.es_optimizer, 'update_from_spct'):
            await self.es_optimizer.update_from_spct(spct_result)
        
        # Update LOCAGENT with new understanding
        if hasattr(self.locagent, 'update_from_spct'):
            await self.locagent.update_from_spct(spct_result)

    async def _increase_training_data_collection(self) -> None:
        """Increase training data collection."""
        # Implementation to increase data collection
        pass

    def _get_recent_localizations(self) -> List[Dict[str, Any]]:
        """Get recent localization results."""
        # Implementation to get recent localizations
        return []

    async def _load_training_tasks(self, training_data_path: str) -> List[Dict[str, Any]]:
        """Load training tasks from JSONL file."""
        import json
        from pathlib import Path
        
        training_tasks = []
        data_path = Path(training_data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Training data file not found: {training_data_path}")
        
        with data_path.open('r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        task = json.loads(line)
                        training_tasks.append(task)
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON line: {line[:100]}...")
                        continue
        
        return training_tasks

    async def _initialize_end_to_end_training(self, training_tasks: List[Dict[str, Any]], 
                                            context: Dict[str, Any]) -> None:
        """Initialize system for end-to-end training."""
        print("🔧 Initializing system for end-to-end training...")
        
        # Initialize LOCAGENT with training data context
        await self._initialize_locagent_for_training(training_tasks)
        
        # Initialize skill paths for diverse tasks
        await self._initialize_diverse_skill_paths(training_tasks)
        
        # Initialize training-specific components
        await self._initialize_training_components(context)
        
        print("✅ End-to-end training initialization complete!")

    async def _initialize_locagent_for_training(self, training_tasks: List[Dict[str, Any]]) -> None:
        """Initialize LOCAGENT for training with task context."""
        print("🧠 Initializing LOCAGENT for training...")
        
        # Extract code samples from training tasks for LOCAGENT context
        code_samples = []
        for task in training_tasks:
            # Check both current_program and code fields
            current_code = task.get("current_program", task.get("code", ""))
            if current_code:
                code_samples.append(current_code)
            
            # Check both target_output and target fields
            target_code = task.get("target_output", task.get("target", ""))
            if target_code:
                if isinstance(target_code, str):
                    code_samples.append(target_code)
                elif isinstance(target_code, list):
                    code_samples.extend([str(item) for item in target_code])
        
        # Build a temporary repository structure for LOCAGENT
        if code_samples:
            await self._build_training_repository(code_samples)
        
        print(f"📊 LOCAGENT initialized with {len(code_samples)} code samples")

    async def _build_training_repository(self, code_samples: List[str]) -> None:
        """Build a temporary repository structure for LOCAGENT from training code samples."""
        import tempfile
        import os
        
        # Create temporary directory structure
        temp_dir = tempfile.mkdtemp(prefix="ue_sea_training_")
        
        # Create a simple Python file with all code samples
        training_file = os.path.join(temp_dir, "training_samples.py")
        with open(training_file, 'w', encoding='utf-8') as f:
            f.write("# Training code samples\n")
            for i, code in enumerate(code_samples[:10]):  # Limit to first 10 samples
                f.write(f"\n# Sample {i+1}\n")
                f.write(code)
                f.write("\n")
        
        # Build LOCAGENT graph from the temporary repository
        try:
            await self.locagent.build_from_repository(temp_dir)
            print(f"🏗️ Built LOCAGENT graph from training samples")
        except Exception as e:
            print(f"Warning: Could not build LOCAGENT graph from training samples: {e}")
        finally:
            # Clean up temporary directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def _initialize_diverse_skill_paths(self, training_tasks: List[Dict[str, Any]]) -> None:
        """Initialize skill paths based on diverse training tasks."""
        # Analyze training tasks to determine required skills
        task_types = set()
        for task in training_tasks:
            # Check both instruction and prompt fields
            instruction = task.get("instruction", task.get("prompt", "")).lower()
            if instruction:
                if "fix" in instruction or "bug" in instruction:
                    task_types.add("localize")
                if "optimize" in instruction or "performance" in instruction:
                    task_types.add("optimize")
                if "refactor" in instruction or "clean" in instruction or "readable" in instruction:
                    task_types.add("refactor")
                if "test" in instruction:
                    task_types.add("test_gen")
                if "document" in instruction:
                    task_types.add("document")
                if "error" in instruction or "handling" in instruction:
                    task_types.add("error_handling")
        
        # Initialize skill paths for detected task types
        for skill_type in task_types:
            # Map skill type to SkillType enum
            from .reasoning.skill_path import SkillType
            skill_type_enum = SkillType.LOCALIZE  # Default
            if skill_type == "localize":
                skill_type_enum = SkillType.LOCALIZE
            elif skill_type == "optimize":
                skill_type_enum = SkillType.OPTIMIZE
            elif skill_type == "refactor":
                skill_type_enum = SkillType.REFACTOR
            elif skill_type == "test_gen":
                skill_type_enum = SkillType.TEST_GEN
            elif skill_type == "document":
                skill_type_enum = SkillType.DOCUMENT
            elif skill_type == "error_handling":
                skill_type_enum = SkillType.VERIFY  # Use VERIFY for error handling
            
            skill = SkillPath(
                skill_id=skill_type,
                skill_type=skill_type_enum,
                description=f"Specialized in {skill_type} tasks"
            )
            self.skill_registry[skill_type] = skill
        
        print(f"🎯 Initialized {len(task_types)} skill paths: {list(task_types)}")

    async def _initialize_training_components(self, context: Dict[str, Any]) -> None:
        """Initialize training-specific components."""
        print("🔧 Initializing training components...")
        
        # Initialize ES optimizer (use existing initialize method)
        if hasattr(self.es_optimizer, 'initialize'):
            await self.es_optimizer.initialize()
        
        # Initialize evaluator pool (use existing methods)
        if hasattr(self.evaluator_pool, 'initialize'):
            await self.evaluator_pool.initialize()
        
        # Initialize reasoning curriculum (use existing methods)
        if hasattr(self.reasoning_curriculum, 'initialize'):
            await self.reasoning_curriculum.initialize()
        
        print("✅ Training components initialized")

    async def _run_end_to_end_training_cycles(self, training_tasks: List[Dict[str, Any]], 
                                            context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run end-to-end training cycles using tasks from training data."""
        print(f"🔄 Starting end-to-end training with {len(training_tasks)} tasks...")
        
        training_results = []
        batch_size = context.get("batch_size", 10)  # Process tasks in batches
        
        for i in range(0, len(training_tasks), batch_size):
            batch = training_tasks[i:i + batch_size]
            batch_id = i // batch_size + 1
            
            print(f"\n=== Training Batch {batch_id}/{len(training_tasks)//batch_size + 1} ===")
            
            # Process each task in the batch
            batch_results = []
            for task_idx, task in enumerate(batch):
                # Get task description from either instruction or prompt field
                task_desc = task.get('instruction', task.get('prompt', 'Unknown'))
                print(f"  📝 Processing task {task_idx + 1}/{len(batch)}: {task_desc[:50]}...")
                
                # Run training cycle for this task
                task_result = await self._run_training_cycle_for_task(task, context)
                batch_results.append(task_result)
            
            # Aggregate batch results
            batch_metrics = self._aggregate_batch_metrics(batch_results)
            training_results.append({
                "batch_id": batch_id,
                "tasks_processed": len(batch),
                "success": all(r.get("success", False) for r in batch_results),
                "metrics": batch_metrics,
                "task_results": batch_results
            })
            
            # Run SPCT if enabled and batch is complete
            if self.config.spct_enabled and batch_id % self.config.spct_frequency == 0:
                print("  🧠 Running SPCT on completed batch...")
                spct_result = await self._spct_stage()
                training_results[-1]["spct"] = spct_result
        
        return training_results

    async def _run_training_cycle_for_task(self, task: Dict[str, Any], 
                                         context: Dict[str, Any]) -> Dict[str, Any]:
        """Run a training cycle for a specific task."""
        try:
            # Extract task information - handle different data formats
            instruction = task.get("instruction", task.get("prompt", ""))
            current_code = task.get("current_program", task.get("code", ""))
            target_output = task.get("target_output", task.get("target", ""))
            
            # Step 1: Localize the task using LOCAGENT
            localization_result = await self._localize_task(instruction)
            
            # Step 2: Generate code using AlphaEvolve
            evolution_result = await self._evolve_for_task(
                task_description=instruction,
                current_code=current_code,
                target_output=target_output,
                context=localization_result
            )
            
            # Step 3: Evaluate the generated code
            evaluation_result = await self._evaluate_task(
                task=task,
                generated_code=evolution_result.get("generated_code", ""),
                context=context
            )
            
            # Step 4: Update agent capabilities based on results
            capability_update = await self._update_capabilities_from_task_result(
                task, evolution_result, evaluation_result
            )
            
            return {
                "success": True,
                "task_id": task.get("task_id", "unknown"),
                "instruction": instruction,
                "localization": localization_result,
                "evolution": evolution_result,
                "evaluation": evaluation_result,
                "capability_update": capability_update
            }
            
        except Exception as e:
            return {
                "success": False,
                "task_id": task.get("task_id", "unknown"),
                "error": str(e),
                "instruction": task.get("instruction", "")
            }

    async def _update_capabilities_from_task_result(self, task: Dict[str, Any], 
                                                  evolution_result: Dict[str, Any],
                                                  evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Update agent capabilities based on task result."""
        capability_updates = {}
        
        # Update based on task type
        instruction = task.get("instruction", "").lower()
        if "fix" in instruction or "bug" in instruction:
            capability_updates["localization"] = 0.01
        if "optimize" in instruction:
            capability_updates["optimization"] = 0.01
        if "refactor" in instruction:
            capability_updates["refactoring"] = 0.01
        
        # Update based on evaluation score
        score = evaluation_result.get("score", 0.0)
        if score > 0.8:
            # High score - increase capabilities
            for skill in capability_updates:
                capability_updates[skill] *= 2
        elif score < 0.3:
            # Low score - decrease capabilities slightly
            for skill in capability_updates:
                capability_updates[skill] *= 0.5
        
        return capability_updates

    def _aggregate_batch_metrics(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics from a batch of task results."""
        successful_tasks = [r for r in batch_results if r.get("success", False)]
        
        if not successful_tasks:
            return {"success_rate": 0.0, "avg_score": 0.0}
        
        # Calculate success rate
        success_rate = len(successful_tasks) / len(batch_results)
        
        # Calculate average evaluation score
        scores = []
        for result in successful_tasks:
            if "evaluation" in result and "score" in result["evaluation"]:
                scores.append(result["evaluation"]["score"])
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        return {
            "success_rate": success_rate,
            "avg_score": avg_score,
            "total_tasks": len(batch_results),
            "successful_tasks": len(successful_tasks)
        }

    def _get_agent_capabilities(self) -> Dict[str, Any]:
        """Get current agent capabilities."""
        capabilities = {}
        
        for skill_id, skill in self.skill_registry.items():
            capabilities[skill_id] = {
                "accuracy": skill.metrics.accuracy,
                "precision": skill.metrics.precision,
                "recall": skill.metrics.recall,
                "f1_score": skill.metrics.f1_score
            }
        
        return capabilities

    async def _evolve_for_task(self, task_description: str, current_code: str, 
                              target_output: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evolve code for a specific task using AlphaEvolve."""
        try:
            # Use existing AlphaEvolve controller
            evolution_result = await self.alphaevolve.evolve(
                target_entities=[],  # Empty for training tasks
                context={
                    "task_description": task_description,
                    "current_code": current_code,
                    "target_output": target_output,
                    "training_mode": True
                },
                budget=50  # Smaller budget for training
            )
            
            return {
                "generated_code": evolution_result.get("best_candidate", {}).get("code", ""),
                "evolution_metrics": evolution_result.get("metrics", {}),
                "success": True
            }
        except Exception as e:
            return {
                "generated_code": "",
                "evolution_metrics": {},
                "success": False,
                "error": str(e)
            }

    async def _evaluate_task(self, task: Dict[str, Any], generated_code: str, 
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate generated code for a task."""
        try:
            # Simple evaluation based on task type
            instruction = task.get("instruction", "").lower()
            target_output = task.get("target_output", "")
            
            # Basic scoring
            score = 0.0
            
            if generated_code:
                score += 0.3  # Base score for generating code
                
                # Check for task-specific improvements
                if "fix" in instruction and "bug" in instruction:
                    if "def " in generated_code and "return" in generated_code:
                        score += 0.3
                
                if "optimize" in instruction:
                    if "for " in generated_code or "while " in generated_code:
                        score += 0.2
                
                if "refactor" in instruction:
                    if len(generated_code.split('\n')) > 3:
                        score += 0.2
                
                # Check similarity to target (simplified)
                if target_output and isinstance(target_output, str):
                    common_words = set(generated_code.lower().split()) & set(target_output.lower().split())
                    if common_words:
                        score += min(0.2, len(common_words) * 0.05)
            
            return {
                "score": min(1.0, score),
                "metrics": {
                    "code_length": len(generated_code),
                    "has_functions": "def " in generated_code,
                    "has_returns": "return" in generated_code
                },
                "success": True
            }
        except Exception as e:
            return {
                "score": 0.0,
                "metrics": {},
                "success": False,
                "error": str(e)
            }
    
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
