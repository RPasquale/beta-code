#!/usr/bin/env python3
"""
Simple test for UE-SEA system.

Tests basic functionality without requiring full setup.
"""

import asyncio
import tempfile
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ue_sea.locagent import LOCAGENT_Graph
from ue_sea.alphaevolve import AlphaEvolve_Controller
from ue_sea.evaluation import EvaluatorPool, GuardrailSystem
from ue_sea.orchestrator import UE_SEA_Orchestrator, CycleConfig


async def test_locagent():
    """Test LOCAGENT graph functionality."""
    print("Testing LOCAGENT...")
    
    # Create temporary repository
    temp_dir = Path(tempfile.mkdtemp(prefix="ue_sea_test_"))
    
    # Create sample Python file
    sample_code = '''
def hello_world():
    """Print hello world."""
    print("Hello, World!")

class Calculator:
    def add(self, a, b):
        return a + b
    
    def multiply(self, a, b):
        return a * b

if __name__ == "__main__":
    hello_world()
    calc = Calculator()
    print(calc.add(2, 3))
'''
    
    sample_file = temp_dir / "test.py"
    sample_file.write_text(sample_code)
    
    # Test LOCAGENT
    graph = LOCAGENT_Graph(use_gpu=False)  # Disable GPU for testing
    
    try:
        await graph.build_from_repository(str(temp_dir))
        
        # Test search
        search_results = await graph.search_entity.search(["hello_world"])
        print(f"  Search results: {len(search_results)} entities found")
        
        # Test traverse
        if search_results:
            entity_id = search_results[0].entity_id
            traverse_results = await graph.traverse_graph.traverse([entity_id], hops=1)
            print(f"  Traverse results: {len(traverse_results.entities)} entities")
        
        # Test retrieve
        if search_results:
            entity_id = search_results[0].entity_id
            retrieve_results = await graph.retrieve_entity.retrieve([entity_id])
            print(f"  Retrieve results: {len(retrieve_results)} entities")
        
        print("  LOCAGENT test passed!")
        return True
        
    except Exception as e:
        print(f"  LOCAGENT test failed: {e}")
        return False
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


async def test_alphaevolve():
    """Test AlphaEvolve controller."""
    print("Testing AlphaEvolve...")
    
    try:
        controller = AlphaEvolve_Controller()
        
        # Test single improvement
        program = '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
'''
        
        task = "Optimize the Fibonacci function for better performance"
        
        improved_program, score = await controller.single_improvement(program, task)
        
        print(f"  Original program length: {len(program)} chars")
        print(f"  Improved program length: {len(improved_program)} chars")
        print(f"  Improvement score: {score:.3f}")
        
        print("  AlphaEvolve test passed!")
        return True
        
    except Exception as e:
        print(f"  AlphaEvolve test failed: {e}")
        return False


async def test_evaluation():
    """Test evaluation system."""
    print("Testing Evaluation System...")
    
    try:
        evaluator = EvaluatorPool()
        guardrails = GuardrailSystem()
        
        # Test code
        test_code = '''
def calculate_sum(numbers):
    """Calculate sum of numbers."""
    total = 0
    for num in numbers:
        total += num
    return total
'''
        
        # Test evaluation
        score = await evaluator.evaluate_staged(test_code)
        print(f"  Evaluation score: {score:.3f}")
        
        # Test guardrails
        guardrail_result = await guardrails.evaluate_guardrails(test_code, {"score": score})
        print(f"  Guardrail passed: {guardrail_result.passed}")
        print(f"  Safety score: {guardrail_result.safety_score:.3f}")
        print(f"  Hacking detections: {len(guardrail_result.hacking_detections)}")
        
        print("  Evaluation test passed!")
        return True
        
    except Exception as e:
        print(f"  Evaluation test failed: {e}")
        return False


async def test_orchestrator():
    """Test orchestrator initialization."""
    print("Testing Orchestrator...")
    
    try:
        config = CycleConfig(
            max_cycles=1,  # Minimal for testing
            cycle_timeout=60,
            evolution_budget=10,
            training_budget=5,
            gpu_enabled=False
        )
        
        orchestrator = UE_SEA_Orchestrator(config)
        
        # Test status
        status = orchestrator.get_status()
        print(f"  Orchestrator initialized")
        print(f"  Current cycle: {status['current_cycle']}")
        print(f"  Program registry size: {status['program_registry_size']}")
        
        print("  Orchestrator test passed!")
        return True
        
    except Exception as e:
        print(f"  Orchestrator test failed: {e}")
        return False


async def run_all_tests():
    """Run all tests."""
    print("=== UE-SEA System Tests ===\n")
    
    tests = [
        ("LOCAGENT Graph", test_locagent),
        ("AlphaEvolve Controller", test_alphaevolve),
        ("Evaluation System", test_evaluation),
        ("Orchestrator", test_orchestrator),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running {test_name} test...")
        try:
            result = await test_func()
            if result:
                passed += 1
            print()
        except Exception as e:
            print(f"  {test_name} test error: {e}\n")
    
    print(f"=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("All tests passed! UE-SEA system is working correctly.")
        return True
    else:
        print("Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
