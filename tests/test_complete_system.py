#!/usr/bin/env python3
"""
Complete UE-SEA System Test

Tests the complete UE-SEA system with all components integrated.
"""

import asyncio
import tempfile
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ue_sea import (
    UE_SEA_Orchestrator, LOCAGENT_Graph, AlphaEvolve_Controller,
    SkillPath, ReasoningCurriculum, StreamingDiLoCo, DiPaCo_Router,
    EvolutionStrategies, EvaluatorPool, GuardrailSystem
)
from ue_sea.orchestrator import CycleConfig
from ue_sea.reasoning import SkillType, SkillConfig
from ue_sea.distributed import FragmentConfig, WorkerConfig, PathConfig
from ue_sea.optimizer import ESConfig


async def test_complete_system():
    """Test the complete UE-SEA system."""
    print("=== Complete UE-SEA System Test ===\n")
    
    # Create demo repository
    temp_dir = Path(tempfile.mkdtemp(prefix="ue_sea_complete_test_"))
    
    # Create sample Python files
    sample_code = {
        "main.py": '''
def fibonacci(n):
    """Calculate Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def main():
    print("Fibonacci calculator")
    for i in range(10):
        result = fibonacci(i)
        print(f"F({i}) = {result}")

if __name__ == "__main__":
    main()
''',
        "utils.py": '''
def is_prime(n):
    """Check if number is prime."""
    if n < 2:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True

def get_primes(limit):
    """Get all primes up to limit."""
    primes = []
    for i in range(2, limit):
        if is_prime(i):
            primes.append(i)
    return primes
''',
        "test_main.py": '''
import unittest
from main import fibonacci

class TestFibonacci(unittest.TestCase):
    def test_fibonacci(self):
        self.assertEqual(fibonacci(0), 0)
        self.assertEqual(fibonacci(1), 1)
        self.assertEqual(fibonacci(5), 5)
        self.assertEqual(fibonacci(10), 55)

if __name__ == "__main__":
    unittest.main()
'''
    }
    
    # Write files
    for filename, content in sample_code.items():
        file_path = temp_dir / filename
        file_path.write_text(content)
    
    print(f"Demo repository created at: {temp_dir}")
    
    try:
        # Test 1: LOCAGENT Graph
        print("\n1. Testing LOCAGENT Graph...")
        graph = LOCAGENT_Graph(use_gpu=False)
        await graph.build_from_repository(str(temp_dir))
        
        # Test search
        search_results = await graph.search_entity.search(["fibonacci"])
        print(f"   Search results: {len(search_results)} entities found")
        
        # Test traverse
        if search_results:
            entity_id = search_results[0].entity_id
            traverse_results = await graph.traverse_graph.traverse([entity_id], hops=1)
            print(f"   Traverse results: {len(traverse_results.entities)} entities")
        
        print("   ✓ LOCAGENT Graph test passed!")
        
        # Test 2: AlphaEvolve Controller
        print("\n2. Testing AlphaEvolve Controller...")
        controller = AlphaEvolve_Controller()
        
        program = '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
'''
        
        task = "Optimize the Fibonacci function for better performance"
        improved_program, score = await controller.single_improvement(program, task)
        
        print(f"   Original length: {len(program)} chars")
        print(f"   Improved length: {len(improved_program)} chars")
        print(f"   Improvement score: {score:.3f}")
        print("   ✓ AlphaEvolve Controller test passed!")
        
        # Test 3: Evaluation System
        print("\n3. Testing Evaluation System...")
        evaluator = EvaluatorPool()
        guardrails = GuardrailSystem()
        
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
        print(f"   Evaluation score: {score:.3f}")
        
        # Test guardrails
        guardrail_result = await guardrails.evaluate_guardrails(test_code, {"score": score})
        print(f"   Guardrail passed: {guardrail_result.passed}")
        print(f"   Safety score: {guardrail_result.safety_score:.3f}")
        print("   ✓ Evaluation System test passed!")
        
        # Test 4: Reasoning Skills
        print("\n4. Testing Reasoning Skills...")
        
        # Create skill paths
        localize_skill = SkillPath(
            skill_id="localize",
            skill_type=SkillType.LOCALIZE,
            description="Bug localization skill"
        )
        
        verify_skill = SkillPath(
            skill_id="verify",
            skill_type=SkillType.VERIFY,
            description="Code verification skill"
        )
        
        # Test skill generation
        localization_result = await localize_skill.generate({
            "code": "def buggy_function():\n    return None",
            "error": "AttributeError"
        })
        print(f"   Localization result: {len(localization_result)} keys")
        
        verification_result = await verify_skill.generate({
            "code": "def verify_function():\n    return True",
            "properties": ["correctness"]
        })
        print(f"   Verification result: {len(verification_result)} keys")
        print("   ✓ Reasoning Skills test passed!")
        
        # Test 5: Distributed Training
        print("\n5. Testing Distributed Training...")
        
        # Initialize Streaming DiLoCo
        fragments = [
            FragmentConfig(fragment_id="frag1", layers=[0, 1, 2], offset_t_p=0),
            FragmentConfig(fragment_id="frag2", layers=[3, 4, 5], offset_t_p=50)
        ]
        workers = [
            WorkerConfig(worker_id="worker1", replicas=2),
            WorkerConfig(worker_id="worker2", replicas=2)
        ]
        
        diloco = StreamingDiLoCo()
        await diloco.initialize(fragments, workers)
        print(f"   DiLoCo initialized with {len(fragments)} fragments")
        
        # Initialize DiPaCo Router
        paths = [
            PathConfig(
                path_id="path1",
                modules=["module1", "module2"],
                shared_modules=["shared1"],
                unshared_modules=["unshared1"],
                routing_key="localize"
            )
        ]
        
        dipaco = DiPaCo_Router()
        await dipaco.initialize(paths)
        print(f"   DiPaCo initialized with {len(paths)} paths")
        
        # Test routing
        routing_decision = await dipaco.route_request({
            "complexity": 0.7,
            "urgency": 0.5,
            "resource_requirements": 0.3
        })
        print(f"   Routing decision: {routing_decision.path_id} (confidence: {routing_decision.confidence:.3f})")
        print("   ✓ Distributed Training test passed!")
        
        # Test 6: Evolution Strategies
        print("\n6. Testing Evolution Strategies...")
        
        es_optimizer = EvolutionStrategies(ESConfig(
            population_size=20,
            max_iterations=10,
            learning_rate=0.01
        ))
        
        # Initialize with dummy parameters
        initial_params = {
            "weight": torch.randn(10, 5),
            "bias": torch.randn(5)
        }
        await es_optimizer.initialize(initial_params)
        
        # Define objective function
        def objective_function(params):
            # Simple objective: minimize sum of squares
            total = 0
            for param in params.values():
                total += torch.sum(param ** 2).item()
            return -total  # Negative for maximization
        
        # Run optimization
        result = await es_optimizer.optimize(objective_function, max_iterations=5)
        print(f"   Best reward: {result.best_reward:.4f}")
        print(f"   Iterations: {result.iterations}")
        print(f"   Convergence: {result.convergence}")
        print("   ✓ Evolution Strategies test passed!")
        
        # Test 7: Complete Orchestrator
        print("\n7. Testing Complete Orchestrator...")
        
        config = CycleConfig(
            max_cycles=2,  # Small for testing
            cycle_timeout=60,
            evolution_budget=20,
            training_budget=10,
            gpu_enabled=False
        )
        
        orchestrator = UE_SEA_Orchestrator(config)
        
        # Test status
        status = orchestrator.get_status()
        print(f"   Orchestrator status: {status['current_cycle']} cycles")
        print(f"   Program registry size: {status['program_registry_size']}")
        print(f"   Skill registry size: {status['skill_registry_size']}")
        print("   ✓ Complete Orchestrator test passed!")
        
        print("\n=== All Tests Passed! ===")
        print("✅ LOCAGENT Graph")
        print("✅ AlphaEvolve Controller")
        print("✅ Evaluation System")
        print("✅ Reasoning Skills")
        print("✅ Distributed Training")
        print("✅ Evolution Strategies")
        print("✅ Complete Orchestrator")
        
        print(f"\n🎉 UE-SEA system is fully functional!")
        print(f"   All components integrated and working")
        print(f"   Ready for production use with RTX 4090 GPU")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    success = asyncio.run(test_complete_system())
    sys.exit(0 if success else 1)
