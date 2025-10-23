"""
Prompt Sampler: Builds rich prompts for LLM ensemble generation.

Implements prompt construction with parent solutions, context, and meta-prompt evolution.
"""

import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class PromptType(Enum):
    """Types of prompts for different generation tasks."""
    INITIAL_VARIATION = "initial_variation"
    MUTATION = "mutation"
    CROSSOVER = "crossover"
    IMPROVEMENT = "improvement"


@dataclass
class PromptTemplate:
    """Template for generating prompts."""
    template: str
    prompt_type: PromptType
    variables: List[str]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PromptSampler:
    """
    Samples and constructs prompts for LLM ensemble generation.
    
    Builds rich prompts from parent solutions, inspirations, context,
    and supports meta-prompt evolution.
    """
    
    def __init__(self):
        self.templates = self._initialize_templates()
        self.meta_prompts = self._initialize_meta_prompts()
        self.prompt_history = []
    
    def _initialize_templates(self) -> Dict[PromptType, PromptTemplate]:
        """Initialize prompt templates for different generation tasks."""
        templates = {}
        
        # Initial variation template
        templates[PromptType.INITIAL_VARIATION] = PromptTemplate(
            template="""You are an expert software engineer. Create a variation of the following program that addresses the task requirements.

TASK: {task_description}

CONTEXT: {context}

ORIGINAL PROGRAM:
```{language}
{original_program}
```

REQUIREMENTS:
- Maintain the core functionality
- Improve code quality, performance, or maintainability
- Follow best practices for {language}
- Ensure the code is well-documented and readable

Generate a complete, improved version of the program:""",
            prompt_type=PromptType.INITIAL_VARIATION,
            variables=["task_description", "context", "language", "original_program"]
        )
        
        # Mutation template
        templates[PromptType.MUTATION] = PromptTemplate(
            template="""You are an expert software engineer. Apply a targeted mutation to improve the following program.

TASK: {task_description}

CONTEXT: {context}

CURRENT PROGRAM:
```{language}
{current_program}
```

INSPIRATION FROM PREVIOUS SOLUTIONS:
{inspirations}

MUTATION GUIDELINES:
- Make a focused, specific improvement
- Preserve the overall structure and functionality
- Focus on one aspect: performance, readability, maintainability, or correctness
- Keep changes minimal but impactful

Generate the mutated program:""",
            prompt_type=PromptType.MUTATION,
            variables=["task_description", "context", "language", "current_program", "inspirations"]
        )
        
        # Crossover template
        templates[PromptType.CROSSOVER] = PromptTemplate(
            template="""You are an expert software engineer. Create a crossover between two programs, combining their best features.

TASK: {task_description}

CONTEXT: {context}

PARENT PROGRAM 1:
```{language}
{parent1_program}
```

PARENT PROGRAM 2:
```{language}
{parent2_program}
```

CROSSOVER GUIDELINES:
- Combine the best features from both programs
- Maintain functionality and correctness
- Create a coherent, well-structured result
- Preserve the strengths of each parent

Generate the crossover program:""",
            prompt_type=PromptType.CROSSOVER,
            variables=["task_description", "context", "language", "parent1_program", "parent2_program"]
        )
        
        # Improvement template
        templates[PromptType.IMPROVEMENT] = PromptTemplate(
            template="""You are an expert software engineer. Improve the following program to better meet the task requirements.

TASK: {task_description}

CONTEXT: {context}

CURRENT PROGRAM:
```{language}
{current_program}
```

IMPROVEMENT GUIDELINES:
- Address the specific task requirements
- Improve code quality, performance, or maintainability
- Follow best practices for {language}
- Ensure the code is well-documented and readable
- Make meaningful improvements without over-engineering

Generate the improved program:""",
            prompt_type=PromptType.IMPROVEMENT,
            variables=["task_description", "context", "language", "current_program"]
        )
        
        return templates
    
    def _initialize_meta_prompts(self) -> Dict[str, str]:
        """Initialize meta-prompts for different scenarios."""
        return {
            "performance_focus": "Focus on performance optimization, algorithmic efficiency, and resource usage.",
            "readability_focus": "Focus on code readability, clear naming, and comprehensive documentation.",
            "maintainability_focus": "Focus on maintainability, modular design, and extensibility.",
            "correctness_focus": "Focus on correctness, error handling, and robustness.",
            "security_focus": "Focus on security best practices and vulnerability prevention."
        }
    
    async def sample_initial_variation(self, program: str, task_description: str,
                                    context: Dict[str, Any] = None) -> str:
        """Sample prompt for initial variation generation."""
        if context is None:
            context = {}
        
        # Determine language from context or program
        language = context.get("language", "python")
        
        # Select meta-prompt based on context
        meta_prompt = self._select_meta_prompt(context)
        
        # Build prompt
        prompt = self._build_prompt(
            PromptType.INITIAL_VARIATION,
            {
                "task_description": task_description,
                "context": self._format_context(context),
                "language": language,
                "original_program": program
            }
        )
        
        # Add meta-prompt
        if meta_prompt:
            prompt += f"\n\nFOCUS: {meta_prompt}"
        
        return prompt
    
    async def sample_mutation_prompt(self, program: str, task_description: str,
                                   context: Dict[str, Any] = None) -> str:
        """Sample prompt for mutation generation."""
        if context is None:
            context = {}
        
        language = context.get("language", "python")
        
        # Get inspirations from previous solutions
        inspirations = self._get_inspirations(context)
        
        # Select meta-prompt
        meta_prompt = self._select_meta_prompt(context)
        
        # Build prompt
        prompt = self._build_prompt(
            PromptType.MUTATION,
            {
                "task_description": task_description,
                "context": self._format_context(context),
                "language": language,
                "current_program": program,
                "inspirations": inspirations
            }
        )
        
        # Add meta-prompt
        if meta_prompt:
            prompt += f"\n\nFOCUS: {meta_prompt}"
        
        return prompt
    
    async def sample_crossover_prompt(self, program1: str, program2: str,
                                    task_description: str,
                                    context: Dict[str, Any] = None) -> str:
        """Sample prompt for crossover generation."""
        if context is None:
            context = {}
        
        language = context.get("language", "python")
        
        # Build prompt
        prompt = self._build_prompt(
            PromptType.CROSSOVER,
            {
                "task_description": task_description,
                "context": self._format_context(context),
                "language": language,
                "parent1_program": program1,
                "parent2_program": program2
            }
        )
        
        return prompt
    
    async def sample_improvement_prompt(self, program: str, task_description: str,
                                      context: Dict[str, Any] = None) -> str:
        """Sample prompt for improvement generation."""
        if context is None:
            context = {}
        
        language = context.get("language", "python")
        
        # Select meta-prompt
        meta_prompt = self._select_meta_prompt(context)
        
        # Build prompt
        prompt = self._build_prompt(
            PromptType.IMPROVEMENT,
            {
                "task_description": task_description,
                "context": self._format_context(context),
                "language": language,
                "current_program": program
            }
        )
        
        # Add meta-prompt
        if meta_prompt:
            prompt += f"\n\nFOCUS: {meta_prompt}"
        
        return prompt
    
    def _build_prompt(self, prompt_type: PromptType, variables: Dict[str, Any]) -> str:
        """Build a prompt from template and variables."""
        template = self.templates[prompt_type]
        
        # Format the template with variables
        try:
            prompt = template.template.format(**variables)
        except KeyError as e:
            raise ValueError(f"Missing variable {e} for prompt type {prompt_type}")
        
        return prompt
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context dictionary into readable text."""
        if not context:
            return "No additional context provided."
        
        context_parts = []
        for key, value in context.items():
            if isinstance(value, str):
                context_parts.append(f"{key}: {value}")
            elif isinstance(value, (list, tuple)):
                context_parts.append(f"{key}: {', '.join(map(str, value))}")
            else:
                context_parts.append(f"{key}: {value}")
        
        return "\n".join(context_parts)
    
    def _get_inspirations(self, context: Dict[str, Any]) -> str:
        """Get inspirations from previous solutions."""
        inspirations = context.get("inspirations", [])
        
        if not inspirations:
            return "No previous solutions available for inspiration."
        
        inspiration_text = "Previous successful solutions:\n"
        for i, inspiration in enumerate(inspirations[:3]):  # Limit to top 3
            inspiration_text += f"\nSolution {i+1}:\n{inspiration}\n"
        
        return inspiration_text
    
    def _select_meta_prompt(self, context: Dict[str, Any]) -> Optional[str]:
        """Select appropriate meta-prompt based on context."""
        focus = context.get("focus", "general")
        
        if focus in self.meta_prompts:
            return self.meta_prompts[focus]
        
        # Random selection for diversity
        return random.choice(list(self.meta_prompts.values()))
    
    def add_prompt_feedback(self, prompt: str, result: str, score: float) -> None:
        """Add feedback about prompt effectiveness."""
        self.prompt_history.append({
            "prompt": prompt,
            "result": result,
            "score": score,
            "timestamp": time.time()
        })
    
    def get_effective_prompts(self, min_score: float = 0.7) -> List[Dict[str, Any]]:
        """Get prompts that have been effective (high scores)."""
        return [
            entry for entry in self.prompt_history
            if entry["score"] >= min_score
        ]
    
    def evolve_meta_prompts(self) -> None:
        """Evolve meta-prompts based on feedback."""
        effective_prompts = self.get_effective_prompts()
        
        if not effective_prompts:
            return
        
        # Analyze patterns in effective prompts
        focus_patterns = {}
        for entry in effective_prompts:
            # Extract focus from prompt (simplified)
            if "performance" in entry["prompt"].lower():
                focus_patterns["performance"] = focus_patterns.get("performance", 0) + 1
            elif "readability" in entry["prompt"].lower():
                focus_patterns["readability"] = focus_patterns.get("readability", 0) + 1
            # Add more pattern analysis
        
        # Update meta-prompts based on patterns
        for focus, count in focus_patterns.items():
            if count > len(effective_prompts) * 0.3:  # If focus appears in >30% of effective prompts
                # Enhance the corresponding meta-prompt
                if focus in self.meta_prompts:
                    self.meta_prompts[focus] += f" (Enhanced based on {count} successful applications)"
    
    def get_prompt_statistics(self) -> Dict[str, Any]:
        """Get statistics about prompt usage and effectiveness."""
        if not self.prompt_history:
            return {"total_prompts": 0, "average_score": 0.0}
        
        scores = [entry["score"] for entry in self.prompt_history]
        
        return {
            "total_prompts": len(self.prompt_history),
            "average_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "effective_prompts": len(self.get_effective_prompts())
        }
