"""Minimal refactoring helper.

Provides a simple API to clean up code snippets. We keep the implementation
lightweight so the orchestrator can always call it even if optional formatters
are missing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


try:  # Optional prettier formatting libraries
    import autopep8
    AUTOPEP8_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    autopep8 = None  # type: ignore
    AUTOPEP8_AVAILABLE = False


@dataclass
class RefactorReport:
    original: str
    refactored: str
    transformations: Dict[str, Optional[float]]


class CodeRefactorer:
    def __init__(self, aggressive: bool = False):
        self.aggressive = aggressive

    def refactor(self, code: str) -> RefactorReport:
        if AUTOPEP8_AVAILABLE:
            options = {"aggressive": 2 if self.aggressive else 0}
            refactored = autopep8.fix_code(code, options=options)
            return RefactorReport(
                original=code,
                refactored=refactored,
                transformations={"autopep8": None},
            )

        # Fallback: return the code unchanged but ensure trailing spaces removed
        cleaned = "\n".join(line.rstrip() for line in code.splitlines())
        if cleaned and not cleaned.endswith("\n"):
            cleaned += "\n"
        return RefactorReport(
            original=code,
            refactored=cleaned,
            transformations={"autopep8": None, "fallback": 1.0},
        )

