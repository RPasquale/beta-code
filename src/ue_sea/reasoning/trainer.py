"""Skill trainer coordinating the reasoning curriculum."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from .curriculum import ReasoningCurriculum, TrainingStage
from .skill_path import SkillPath

logger = logging.getLogger(__name__)


@dataclass
class SkillTrainingSummary:
    skill_id: str
    completed_stages: List[TrainingStage]
    metrics: Dict[str, float]
    artifacts: List[str]


class SkillTrainer:
    def __init__(self, curriculum: Optional[ReasoningCurriculum] = None):
        self.curriculum = curriculum or ReasoningCurriculum()
        self.registry: Dict[str, SkillPath] = {}

    def register_skill(self, skill: SkillPath) -> None:
        self.registry[skill.id] = skill
        logger.debug("Registered skill %s", skill.id)

    async def _train_stage(self, skill: SkillPath, stage: TrainingStage) -> Dict[str, float]:
        # Placeholder training loop; in a full implementation we would launch SFT/RL jobs here.
        await asyncio.sleep(0.1)
        logger.debug("Stage %s for skill %s completed", stage, skill.id)
        return {"loss": 0.01, "accuracy": 0.99}

    async def train_skill(self, skill_id: str) -> SkillTrainingSummary:
        skill = self.registry[skill_id]
        completed: List[TrainingStage] = []
        metrics: Dict[str, float] = {}
        artifacts: List[str] = []

        for stage, config in self.curriculum.stages.items():
            start = time.time()
            stage_metrics = await self._train_stage(skill, stage)
            completed.append(stage)
            metrics.update({f"{stage.value}_{k}": v for k, v in stage_metrics.items()})
            artifacts.append(f"{skill_id}_{stage.value}")
            logger.info(
                "Finished %s for %s in %.2fs",
                stage.value,
                skill.name,
                time.time() - start,
            )

        return SkillTrainingSummary(skill_id=skill_id, completed_stages=completed, metrics=metrics, artifacts=artifacts)

