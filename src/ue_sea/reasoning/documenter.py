"""Simple documentation generator used by the reasoning pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import textwrap


logger = logging.getLogger(__name__)


@dataclass
class DocumentationSnippet:
    signature: str
    docstring: str


class CodeDocumenter:
    def generate_docstring(self, signature: str, description: str) -> DocumentationSnippet:
        template = textwrap.dedent(
            f"""
            {description}

            Args:
                *args: see original function.
                **kwargs: see original function.

            Returns:
                Result of the underlying implementation.
            """
        )
        return DocumentationSnippet(signature=signature, docstring=template)

    def document(self, code: str, description: str) -> str:
        snippet = self.generate_docstring("function", description)
        logger.debug("Generated docstring: %s", snippet.docstring)
        return snippet.docstring
