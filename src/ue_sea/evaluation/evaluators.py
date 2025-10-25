"""
Evaluators: Staged evaluation pipeline with machine-grade tests.

Implements cascaded evaluation, parallel testing, and comprehensive metrics.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import subprocess
import tempfile
import os
from pathlib import Path


class EvaluationStage(Enum):
    """Stages of evaluation cascade."""
    FAST = "fast"        # Syntax, linting, basic tests
    MEDIUM = "medium"    # Performance, coverage, security
    FULL = "full"        # Complete CI, integration tests


@dataclass
class EvaluationResult:
    """Result of program evaluation."""
    stage: EvaluationStage
    score: float
    passed: bool
    metrics: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TestResult:
    """Result of a single test."""
    test_name: str
    passed: bool
    score: float
    duration: float
    output: str
    error: Optional[str] = None


class UnitTestEvaluator:
    """Evaluator for unit tests."""
    
    def __init__(self):
        self.test_runner = "pytest"
        self.timeout = 30  # seconds
    
    async def evaluate(self, code: str, test_code: str = None) -> EvaluationResult:
        """
        Evaluate code using unit tests.
        
        Args:
            code: Code to evaluate
            test_code: Test code (if None, will try to find tests)
        
        Returns:
            Evaluation result
        """
        start_time = time.time()
        
        try:
            # Create temporary files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Write code to file
                code_file = temp_path / "main.py"
                code_file.write_text(code)
                
                # Write test code if provided
                if test_code:
                    test_file = temp_path / "test_main.py"
                    test_file.write_text(test_code)
                
                # Run tests
                test_results = await self._run_tests(temp_path)
                
                # Calculate score
                score = self._calculate_score(test_results)
                
                duration = time.time() - start_time
                
                return EvaluationResult(
                    stage=EvaluationStage.FAST,
                    score=score,
                    passed=score > 0.7,
                    metrics={
                        "test_count": len(test_results),
                        "passed_tests": sum(1 for t in test_results if t.passed),
                        "total_duration": sum(t.duration for t in test_results)
                    },
                    errors=[t.error for t in test_results if t.error],
                    warnings=[],
                    duration=duration
                )
                
        except Exception as e:
            duration = time.time() - start_time
            return EvaluationResult(
                stage=EvaluationStage.FAST,
                score=0.0,
                passed=False,
                metrics={},
                errors=[str(e)],
                warnings=[],
                duration=duration
            )
    
    async def _run_tests(self, test_dir: Path) -> List[TestResult]:
        """Run tests in the directory."""
        test_results = []
        
        try:
            # Run pytest
            process = await asyncio.create_subprocess_exec(
                self.test_runner, str(test_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(test_dir)
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout
            )
            
            # Parse results
            output = stdout.decode()
            error_output = stderr.decode()
            
            # Simple test result parsing
            if "failed" in output.lower():
                test_results.append(TestResult(
                    test_name="pytest_run",
                    passed=False,
                    score=0.0,
                    duration=0.0,
                    output=output,
                    error=error_output
                ))
            else:
                test_results.append(TestResult(
                    test_name="pytest_run",
                    passed=True,
                    score=1.0,
                    duration=0.0,
                    output=output
                ))
            
        except asyncio.TimeoutError:
            test_results.append(TestResult(
                test_name="pytest_run",
                passed=False,
                score=0.0,
                duration=self.timeout,
                output="",
                error="Test timeout"
            ))
        except Exception as e:
            test_results.append(TestResult(
                test_name="pytest_run",
                passed=False,
                score=0.0,
                duration=0.0,
                output="",
                error=str(e)
            ))
        
        return test_results
    
    def _calculate_score(self, test_results: List[TestResult]) -> float:
        """Calculate score from test results."""
        if not test_results:
            return 0.0
        
        passed_tests = sum(1 for t in test_results if t.passed)
        total_tests = len(test_results)
        
        return passed_tests / total_tests if total_tests > 0 else 0.0


class PerformanceEvaluator:
    """Evaluator for performance metrics."""
    
    def __init__(self):
        self.benchmark_timeout = 60  # seconds
    
    async def evaluate(self, code: str) -> EvaluationResult:
        """
        Evaluate code performance.
        
        Args:
            code: Code to evaluate
        
        Returns:
            Performance evaluation result
        """
        start_time = time.time()
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Run performance benchmarks
            performance_metrics = await self._run_performance_benchmarks(temp_file)
            
            # Calculate score
            score = self._calculate_performance_score(performance_metrics)
            
            duration = time.time() - start_time
            
            return EvaluationResult(
                stage=EvaluationStage.MEDIUM,
                score=score,
                passed=score > 0.6,
                metrics=performance_metrics,
                errors=[],
                warnings=[],
                duration=duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return EvaluationResult(
                stage=EvaluationStage.MEDIUM,
                score=0.0,
                passed=False,
                metrics={},
                errors=[str(e)],
                warnings=[],
                duration=duration
            )
        finally:
            # Cleanup
            if 'temp_file' in locals():
                os.unlink(temp_file)
    
    async def _run_performance_benchmarks(self, code_file: str) -> Dict[str, Any]:
        """Run performance benchmarks."""
        metrics = {}
        
        try:
            # Time execution
            start_time = time.time()
            
            process = await asyncio.create_subprocess_exec(
                "python", code_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.benchmark_timeout
            )
            
            execution_time = time.time() - start_time
            
            # Memory usage (simplified)
            memory_usage = self._estimate_memory_usage(code_file)
            
            metrics = {
                "execution_time": execution_time,
                "memory_usage": memory_usage,
                "cpu_usage": 0.5,  # Simulated
                "throughput": 1.0 / execution_time if execution_time > 0 else 0
            }
            
        except asyncio.TimeoutError:
            metrics = {
                "execution_time": self.benchmark_timeout,
                "memory_usage": 0,
                "cpu_usage": 0,
                "throughput": 0,
                "timeout": True
            }
        except Exception as e:
            metrics = {
                "execution_time": 0,
                "memory_usage": 0,
                "cpu_usage": 0,
                "throughput": 0,
                "error": str(e)
            }
        
        return metrics
    
    def _estimate_memory_usage(self, code_file: str) -> float:
        """Estimate memory usage (simplified)."""
        # Simple estimation based on file size
        file_size = os.path.getsize(code_file)
        return file_size / 1024 / 1024  # MB
    
    def _calculate_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score."""
        if metrics.get("timeout") or metrics.get("error"):
            return 0.0
        
        execution_time = metrics.get("execution_time", 1.0)
        memory_usage = metrics.get("memory_usage", 1.0)
        
        # Score based on execution time and memory usage
        time_score = max(0, 1.0 - execution_time / 10.0)  # Penalize slow execution
        memory_score = max(0, 1.0 - memory_usage / 100.0)  # Penalize high memory usage
        
        return (time_score + memory_score) / 2.0


class StagedEvaluator:
    """Staged evaluator implementing the cascade."""
    
    def __init__(self):
        self.unit_evaluator = UnitTestEvaluator()
        self.performance_evaluator = PerformanceEvaluator()
        self.stage_configs = {
            EvaluationStage.FAST: {"timeout": 30, "max_tests": 10},
            EvaluationStage.MEDIUM: {"timeout": 120, "max_tests": 50},
            EvaluationStage.FULL: {"timeout": 300, "max_tests": 200}
        }
    
    async def evaluate_staged(self, code: str, test_code: str = None) -> float:
        """
        Run staged evaluation cascade.
        
        Args:
            code: Code to evaluate
            test_code: Test code
        
        Returns:
            Final score
        """
        # Stage 1: Fast evaluation
        print("  Stage 1: Fast evaluation...")
        fast_result = await self.unit_evaluator.evaluate(code, test_code)
        
        if not fast_result.passed:
            print(f"    Fast stage failed: {fast_result.score:.3f}")
            return fast_result.score
        
        # Stage 2: Medium evaluation
        print("  Stage 2: Medium evaluation...")
        medium_result = await self.performance_evaluator.evaluate(code)
        
        if not medium_result.passed:
            print(f"    Medium stage failed: {medium_result.score:.3f}")
            return medium_result.score
        
        # Stage 3: Full evaluation (simplified for demo)
        print("  Stage 3: Full evaluation...")
        full_result = await self._full_evaluation(code)
        
        # Calculate composite score
        composite_score = (
            fast_result.score * 0.4 +
            medium_result.score * 0.4 +
            full_result.score * 0.2
        )
        
        print(f"    Final score: {composite_score:.3f}")
        return composite_score

    async def evaluate(self, candidate: Dict[str, Any], semantic_context: Dict[str, Any] = None,
                      target_entities: List[str] = None, use_locagent_metrics: bool = False) -> float:
        """
        Enhanced evaluation with LOCAGENT metrics support.
        
        Args:
            candidate: Evolution candidate to evaluate
            semantic_context: LOCAGENT semantic context
            target_entities: Target entities from LOCAGENT
            use_locagent_metrics: Whether to use LOCAGENT-specific metrics
        
        Returns:
            Evaluation score
        """
        if use_locagent_metrics and semantic_context:
            # Use LOCAGENT-enhanced evaluation
            return await self._evaluate_with_locagent_metrics(
                candidate, semantic_context, target_entities
            )
        else:
            # Standard evaluation
            code = candidate.get("changes", "")
            return await self.evaluate_staged(code)

    async def _evaluate_with_locagent_metrics(self, candidate: Dict[str, Any],
                                            semantic_context: Dict[str, Any],
                                            target_entities: List[str]) -> float:
        """Evaluate with LOCAGENT-specific metrics."""
        # Get base evaluation score
        code = candidate.get("changes", "")
        base_score = await self.evaluate_staged(code)
        
        # Calculate LOCAGENT enhancement
        locagent_enhancement = 0.0
        
        if semantic_context:
            # Semantic relationship bonus
            relationships = semantic_context.get("semantic_relationships", [])
            if relationships:
                locagent_enhancement += min(0.2, len(relationships) * 0.05)
            
            # Usage pattern bonus
            usage_patterns = semantic_context.get("usage_pattern_impact", {})
            if usage_patterns:
                pattern_count = sum(len(patterns) for patterns in usage_patterns.values())
                locagent_enhancement += min(0.2, pattern_count * 0.02)
            
            # Co-occurrence bonus
            co_occurrence = semantic_context.get("co_occurrence_changes", {})
            if co_occurrence:
                co_occurrence_count = sum(len(entities) for entities in co_occurrence.values())
                locagent_enhancement += min(0.1, co_occurrence_count * 0.01)
        
        # Combine scores
        enhanced_score = base_score + locagent_enhancement
        return min(1.0, enhanced_score)  # Cap at 1.0
    
    async def _full_evaluation(self, code: str) -> EvaluationResult:
        """Full evaluation stage."""
        start_time = time.time()
        
        # Simulate comprehensive evaluation
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Check code quality metrics
        quality_score = self._calculate_quality_score(code)
        
        duration = time.time() - start_time
        
        return EvaluationResult(
            stage=EvaluationStage.FULL,
            score=quality_score,
            passed=quality_score > 0.5,
            metrics={"quality_score": quality_score},
            errors=[],
            warnings=[],
            duration=duration
        )
    
    def _calculate_quality_score(self, code: str) -> float:
        """Calculate code quality score."""
        lines = code.split('\n')
        
        # Simple quality metrics
        comment_ratio = sum(1 for line in lines if line.strip().startswith('#')) / max(1, len(lines))
        function_count = sum(1 for line in lines if line.strip().startswith('def '))
        class_count = sum(1 for line in lines if line.strip().startswith('class '))
        
        # Calculate score
        score = 0.0
        score += min(0.3, comment_ratio * 0.3)  # Comment ratio
        score += min(0.3, function_count * 0.1)  # Function count
        score += min(0.4, class_count * 0.2)  # Class count
        
        return min(1.0, score)


class EvaluatorPool:
    """Pool of evaluators for parallel evaluation."""
    
    def __init__(self):
        self.staged_evaluator = StagedEvaluator()
        self.evaluators = {
            "unit": UnitTestEvaluator(),
            "performance": PerformanceEvaluator(),
            "staged": StagedEvaluator()
        }
    
    async def evaluate_staged(self, code: str, test_code: str = None) -> float:
        """Run staged evaluation."""
        return await self.staged_evaluator.evaluate_staged(code, test_code)
    
    async def evaluate_parallel(self, code: str, evaluator_types: List[str] = None) -> Dict[str, float]:
        """Run parallel evaluation with multiple evaluators."""
        if evaluator_types is None:
            evaluator_types = ["unit", "performance"]
        
        # Run evaluators in parallel
        tasks = []
        for eval_type in evaluator_types:
            if eval_type in self.evaluators:
                task = self._evaluate_with_type(code, eval_type)
                tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Combine results
        combined_results = {}
        for i, eval_type in enumerate(evaluator_types):
            if i < len(results):
                combined_results[eval_type] = results[i]
        
        return combined_results
    
    async def _evaluate_with_type(self, code: str, eval_type: str) -> float:
        """Evaluate with specific evaluator type."""
        evaluator = self.evaluators.get(eval_type)
        if not evaluator:
            return 0.0
        
        if eval_type == "unit":
            result = await evaluator.evaluate(code)
            return result.score
        elif eval_type == "performance":
            result = await evaluator.evaluate(code)
            return result.score
        else:
            return 0.0
