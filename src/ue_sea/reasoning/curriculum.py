"""
Reasoning Curriculum: SFT→RL curriculum for skill training.

Implements staged training with temperature-entropy targeting and overlong handling.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .skill_path import SkillPath, SkillType, SkillConfig


class TrainingStage(Enum):
    """Stages of the training curriculum."""
    SFT_STAGE_1 = "sft_stage_1"    # Math 8K warmup
    SFT_STAGE_2 = "sft_stage_2"    # Math 16K core
    SFT_STAGE_3 = "sft_stage_3"    # Math 24K hard
    RL_STAGE_1 = "rl_stage_1"      # Code RL 24K→32K
    RL_STAGE_2 = "rl_stage_2"      # Math Stage-4 32K
    FINAL = "final"                # Final evaluation


@dataclass
class StageConfig:
    """Configuration for a training stage."""
    stage: TrainingStage
    max_length: int
    temperature: float = 0.3
    entropy_target: float = 0.3
    max_epochs: int = 5
    learning_rate: float = 1e-5
    batch_size: int = 4
    overlong_penalty: float = 0.1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TrainingResult:
    """Result of training a skill."""
    skill_id: str
    stage: TrainingStage
    success: bool
    duration: float
    metrics: Dict[str, Any]
    artifacts: List[str]
    errors: List[str]


class ReasoningCurriculum:
    """
    Manages the SFT→RL curriculum for reasoning skills.
    
    Implements staged training with temperature-entropy targeting
    and overlong handling policies.
    """
    
    def __init__(self):
        self.stages = self._initialize_stages()
        self.training_history = []
        self.skill_registry = {}
    
    def _initialize_stages(self) -> Dict[TrainingStage, StageConfig]:
        """Initialize training stages according to AceReason-Nemotron."""
        return {
            TrainingStage.SFT_STAGE_1: StageConfig(
                stage=TrainingStage.SFT_STAGE_1,
                max_length=8192,  # Math 8K warmup
                temperature=0.3,
                entropy_target=0.3,
                max_epochs=3,
                learning_rate=1e-5,
                batch_size=4,
                overlong_penalty=0.1
            ),
            TrainingStage.SFT_STAGE_2: StageConfig(
                stage=TrainingStage.SFT_STAGE_2,
                max_length=16384,  # Math 16K core
                temperature=0.3,
                entropy_target=0.3,
                max_epochs=5,
                learning_rate=1e-5,
                batch_size=4,
                overlong_penalty=0.1
            ),
            TrainingStage.SFT_STAGE_3: StageConfig(
                stage=TrainingStage.SFT_STAGE_3,
                max_length=24576,  # Math 24K hard
                temperature=0.3,
                entropy_target=0.3,
                max_epochs=5,
                learning_rate=1e-5,
                batch_size=4,
                overlong_penalty=0.1
            ),
            TrainingStage.RL_STAGE_1: StageConfig(
                stage=TrainingStage.RL_STAGE_1,
                max_length=32768,  # Code RL 24K→32K
                temperature=0.3,
                entropy_target=0.3,
                max_epochs=3,
                learning_rate=5e-6,
                batch_size=2,
                overlong_penalty=0.2
            ),
            TrainingStage.RL_STAGE_2: StageConfig(
                stage=TrainingStage.RL_STAGE_2,
                max_length=32768,  # Math Stage-4 32K
                temperature=0.3,
                entropy_target=0.3,
                max_epochs=3,
                learning_rate=5e-6,
                batch_size=2,
                overlong_penalty=0.2
            ),
            TrainingStage.FINAL: StageConfig(
                stage=TrainingStage.FINAL,
                max_length=32768,
                temperature=0.3,
                entropy_target=0.3,
                max_epochs=1,
                learning_rate=1e-6,
                batch_size=1,
                overlong_penalty=0.3
            )
        }
    
    async def train_skill(self, skill: SkillPath, max_steps: int = 1000) -> TrainingResult:
        """
        Train a skill through the complete curriculum.
        
        Args:
            skill: Skill to train
            max_steps: Maximum training steps
        
        Returns:
            Training result
        """
        print(f"Training skill {skill.skill_id} through curriculum...")
        
        start_time = time.time()
        training_artifacts = []
        training_errors = []
        
        try:
            # Stage 1: SFT Stage 1 (8K warmup)
            print("  Stage 1: SFT 8K warmup...")
            sft1_result = await self._train_sft_stage(skill, TrainingStage.SFT_STAGE_1)
            training_artifacts.extend(sft1_result.get("artifacts", []))
            
            # Stage 2: SFT Stage 2 (16K core)
            print("  Stage 2: SFT 16K core...")
            sft2_result = await self._train_sft_stage(skill, TrainingStage.SFT_STAGE_2)
            training_artifacts.extend(sft2_result.get("artifacts", []))
            
            # Stage 3: SFT Stage 3 (24K hard)
            print("  Stage 3: SFT 24K hard...")
            sft3_result = await self._train_sft_stage(skill, TrainingStage.SFT_STAGE_3)
            training_artifacts.extend(sft3_result.get("artifacts", []))
            
            # Stage 4: RL Stage 1 (Code RL 24K→32K)
            print("  Stage 4: RL Code 24K→32K...")
            rl1_result = await self._train_rl_stage(skill, TrainingStage.RL_STAGE_1)
            training_artifacts.extend(rl1_result.get("artifacts", []))
            
            # Stage 5: RL Stage 2 (Math Stage-4 32K)
            print("  Stage 5: RL Math 32K...")
            rl2_result = await self._train_rl_stage(skill, TrainingStage.RL_STAGE_2)
            training_artifacts.extend(rl2_result.get("artifacts", []))
            
            # Final evaluation
            print("  Final evaluation...")
            final_result = await self._final_evaluation(skill)
            training_artifacts.extend(final_result.get("artifacts", []))
            
            duration = time.time() - start_time
            
            # Create training result
            result = TrainingResult(
                skill_id=skill.skill_id,
                stage=TrainingStage.FINAL,
                success=True,
                duration=duration,
                metrics={
                    "sft_stage_1": sft1_result,
                    "sft_stage_2": sft2_result,
                    "sft_stage_3": sft3_result,
                    "rl_stage_1": rl1_result,
                    "rl_stage_2": rl2_result,
                    "final_evaluation": final_result
                },
                artifacts=training_artifacts,
                errors=training_errors
            )
            
            # Store in history
            self.training_history.append(result)
            self.skill_registry[skill.skill_id] = skill
            
            print(f"  Skill {skill.skill_id} training complete!")
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            training_errors.append(str(e))
            
            return TrainingResult(
                skill_id=skill.skill_id,
                stage=TrainingStage.FINAL,
                success=False,
                duration=duration,
                metrics={},
                artifacts=training_artifacts,
                errors=training_errors
            )
    
    async def _train_sft_stage(self, skill: SkillPath, stage: TrainingStage) -> Dict[str, Any]:
        """Train a skill using SFT for a specific stage."""
        stage_config = self.stages[stage]
        
        # Generate SFT corpus for this stage
        corpus = await self._generate_sft_corpus(skill, stage_config)
        
        # Train with SFT
        sft_result = await skill.train_sft(corpus, stage_config.max_epochs)
        
        # Apply overlong handling
        if stage_config.max_length < 24576:  # For shorter stages
            sft_result = await self._apply_overlong_handling(sft_result, stage_config)
        
        return {
            "stage": stage.value,
            "corpus_size": len(corpus),
            "sft_result": sft_result,
            "artifacts": [f"sft_{stage.value}_{skill.skill_id}"]
        }
    
    async def _train_rl_stage(self, skill: SkillPath, stage: TrainingStage) -> Dict[str, Any]:
        """Train a skill using RL for a specific stage."""
        stage_config = self.stages[stage]
        
        # Generate RL tasks for this stage
        tasks = await self._generate_rl_tasks(skill, stage_config)
        
        # Train with RL using temperature-entropy targeting
        rl_result = await skill.train_rl(tasks, max_steps=1000)
        
        # Apply entropy targeting
        rl_result = await self._apply_entropy_targeting(rl_result, stage_config)
        
        return {
            "stage": stage.value,
            "tasks": len(tasks),
            "rl_result": rl_result,
            "artifacts": [f"rl_{stage.value}_{skill.skill_id}"]
        }
    
    async def _final_evaluation(self, skill: SkillPath) -> Dict[str, Any]:
        """Perform final evaluation of the skill."""
        # Generate test data
        test_data = await self._generate_test_data(skill)
        
        # Evaluate skill
        metrics = await skill.evaluate(test_data)
        
        return {
            "test_data_size": len(test_data),
            "metrics": metrics.__dict__,
            "artifacts": [f"final_eval_{skill.skill_id}"]
        }
    
    async def _generate_sft_corpus(self, skill: SkillPath, stage_config: StageConfig) -> List[Dict[str, Any]]:
        """Generate SFT corpus for a skill and stage."""
        corpus = []
        
        # Generate corpus based on skill type and stage
        if skill.skill_type == SkillType.LOCALIZE:
            corpus = await self._generate_localization_corpus(stage_config)
        elif skill.skill_type == SkillType.VERIFY:
            corpus = await self._generate_verification_corpus(stage_config)
        elif skill.skill_type == SkillType.REFACTOR:
            corpus = await self._generate_refactoring_corpus(stage_config)
        elif skill.skill_type == SkillType.DOCUMENT:
            corpus = await self._generate_documentation_corpus(stage_config)
        else:
            corpus = await self._generate_generic_corpus(stage_config)
        
        return corpus
    
    async def _generate_rl_tasks(self, skill: SkillPath, stage_config: StageConfig) -> List[Dict[str, Any]]:
        """Generate RL tasks for a skill and stage."""
        tasks = []
        
        # Generate tasks based on skill type
        if skill.skill_type == SkillType.LOCALIZE:
            tasks = await self._generate_localization_rl_tasks(stage_config)
        elif skill.skill_type == SkillType.VERIFY:
            tasks = await self._generate_verification_rl_tasks(stage_config)
        elif skill.skill_type == SkillType.REFACTOR:
            tasks = await self._generate_refactoring_rl_tasks(stage_config)
        elif skill.skill_type == SkillType.DOCUMENT:
            tasks = await self._generate_documentation_rl_tasks(stage_config)
        else:
            tasks = await self._generate_generic_rl_tasks(stage_config)
        
        return tasks
    
    async def _generate_test_data(self, skill: SkillPath) -> List[Dict[str, Any]]:
        """Generate test data for skill evaluation."""
        test_data = []
        
        # Generate test cases based on skill type
        for i in range(100):  # 100 test cases
            if skill.skill_type == SkillType.LOCALIZE:
                test_data.append({
                    "code": f"def test_function_{i}():\n    # Buggy code\n    return None",
                    "error": f"Test error {i}",
                    "expected": f"Expected localization {i}"
                })
            elif skill.skill_type == SkillType.VERIFY:
                test_data.append({
                    "code": f"def verify_function_{i}():\n    # Code to verify\n    return True",
                    "properties": ["correctness", "safety"],
                    "expected": f"Expected verification {i}"
                })
            else:
                test_data.append({
                    "input": f"Test input {i}",
                    "expected": f"Expected output {i}"
                })
        
        return test_data
    
    async def _apply_overlong_handling(self, result: Dict[str, Any], 
                                     stage_config: StageConfig) -> Dict[str, Any]:
        """Apply overlong handling policy."""
        # Simulate overlong handling
        if "overlong_penalty" in result:
            result["overlong_penalty"] *= stage_config.overlong_penalty
        
        return result
    
    async def _apply_entropy_targeting(self, result: Dict[str, Any], 
                                     stage_config: StageConfig) -> Dict[str, Any]:
        """Apply temperature-entropy targeting."""
        # Simulate entropy targeting
        result["entropy_target"] = stage_config.entropy_target
        result["temperature"] = stage_config.temperature
        
        return result
    
    # Corpus generation methods for different skills
    async def _generate_localization_corpus(self, stage_config: StageConfig) -> List[Dict[str, Any]]:
        """Generate localization corpus."""
        corpus = []
        for i in range(1000):
            corpus.append({
                "input": f"Bug in function_{i}: {i} error",
                "output": f"Localized to line {i*10} in file_{i}.py",
                "metadata": {"stage": stage_config.stage.value}
            })
        return corpus
    
    async def _generate_verification_corpus(self, stage_config: StageConfig) -> List[Dict[str, Any]]:
        """Generate verification corpus."""
        corpus = []
        for i in range(1000):
            corpus.append({
                "input": f"Verify function_{i}",
                "output": f"Verified: correctness={0.8+i*0.001}, safety={0.9+i*0.001}",
                "metadata": {"stage": stage_config.stage.value}
            })
        return corpus
    
    async def _generate_refactoring_corpus(self, stage_config: StageConfig) -> List[Dict[str, Any]]:
        """Generate refactoring corpus."""
        corpus = []
        for i in range(1000):
            corpus.append({
                "input": f"Refactor function_{i}",
                "output": f"Refactored function_{i} with improved structure",
                "metadata": {"stage": stage_config.stage.value}
            })
        return corpus
    
    async def _generate_documentation_corpus(self, stage_config: StageConfig) -> List[Dict[str, Any]]:
        """Generate documentation corpus."""
        corpus = []
        for i in range(1000):
            corpus.append({
                "input": f"Document function_{i}",
                "output": f"Documentation for function_{i} with examples",
                "metadata": {"stage": stage_config.stage.value}
            })
        return corpus
    
    async def _generate_generic_corpus(self, stage_config: StageConfig) -> List[Dict[str, Any]]:
        """Generate generic corpus."""
        corpus = []
        for i in range(1000):
            corpus.append({
                "input": f"Generic input {i}",
                "output": f"Generic output {i}",
                "metadata": {"stage": stage_config.stage.value}
            })
        return corpus
    
    # RL task generation methods
    async def _generate_localization_rl_tasks(self, stage_config: StageConfig) -> List[Dict[str, Any]]:
        """Generate localization RL tasks."""
        tasks = []
        for i in range(500):
            tasks.append({
                "task": f"Localize bug {i}",
                "reward_function": "accuracy_based",
                "max_length": stage_config.max_length
            })
        return tasks
    
    async def _generate_verification_rl_tasks(self, stage_config: StageConfig) -> List[Dict[str, Any]]:
        """Generate verification RL tasks."""
        tasks = []
        for i in range(500):
            tasks.append({
                "task": f"Verify code {i}",
                "reward_function": "correctness_based",
                "max_length": stage_config.max_length
            })
        return tasks
    
    async def _generate_refactoring_rl_tasks(self, stage_config: StageConfig) -> List[Dict[str, Any]]:
        """Generate refactoring RL tasks."""
        tasks = []
        for i in range(500):
            tasks.append({
                "task": f"Refactor code {i}",
                "reward_function": "quality_based",
                "max_length": stage_config.max_length
            })
        return tasks
    
    async def _generate_documentation_rl_tasks(self, stage_config: StageConfig) -> List[Dict[str, Any]]:
        """Generate documentation RL tasks."""
        tasks = []
        for i in range(500):
            tasks.append({
                "task": f"Document code {i}",
                "reward_function": "clarity_based",
                "max_length": stage_config.max_length
            })
        return tasks
    
    async def _generate_generic_rl_tasks(self, stage_config: StageConfig) -> List[Dict[str, Any]]:
        """Generate generic RL tasks."""
        tasks = []
        for i in range(500):
            tasks.append({
                "task": f"Generic task {i}",
                "reward_function": "performance_based",
                "max_length": stage_config.max_length
            })
        return tasks
    
    def get_curriculum_statistics(self) -> Dict[str, Any]:
        """Get curriculum statistics."""
        return {
            "total_skills_trained": len(self.skill_registry),
            "total_training_sessions": len(self.training_history),
            "successful_sessions": sum(1 for h in self.training_history if h.success),
            "average_training_time": sum(h.duration for h in self.training_history) / max(1, len(self.training_history)),
            "stages": {stage.value: config.__dict__ for stage, config in self.stages.items()}
        }
