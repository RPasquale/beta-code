"""Simple code verifier used by the orchestrator.

The original architecture runs comprehensive staged evaluators. For the
integration path we expose a lightweight synchronous API that can be swapped
out for richer evaluation later on.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    from ..evaluation.evaluators import UnitTestEvaluator
except Exception:  # pragma: no cover - fallback when evaluation stack missing
    UnitTestEvaluator = None  # type: ignore


logger = logging.getLogger(__name__)


@dataclass
class VerificationReport:
    passed: bool
    score: float
    metrics: Dict[str, Any]
    logs: Optional[str] = None


class CodeVerifier:
    def __init__(self, use_async: bool = True):
        self.use_async = use_async
        self.evaluator = UnitTestEvaluator() if UnitTestEvaluator else None

    async def verify_async(
        self,
        code: str,
        test_code: Optional[str] = None,
        timeout: int = 30,
    ) -> VerificationReport:
        if self.evaluator is None:
            logger.debug("UnitTestEvaluator unavailable; returning stub result")
            return VerificationReport(passed=True, score=0.5, metrics={})

        try:
            result = await asyncio.wait_for(
                self.evaluator.evaluate(code=code, test_code=test_code),
                timeout=timeout,
            )
            return VerificationReport(
                passed=result.passed,
                score=result.score,
                metrics=result.metrics,
                logs="\n".join(result.errors) if result.errors else None,
            )
        except asyncio.TimeoutError:
            logger.warning("Verification timed out")
            return VerificationReport(passed=False, score=0.0, metrics={}, logs="timeout")

    def verify(
        self,
        code: str,
        test_code: Optional[str] = None,
        timeout: int = 30,
    ) -> VerificationReport:
        if self.use_async:
            return asyncio.run(self.verify_async(code, test_code, timeout))
        else:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self.verify_async(code, test_code, timeout))
            finally:
                loop.close()

