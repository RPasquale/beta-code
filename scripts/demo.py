#!/usr/bin/env python3
"""
UE-SEA Demo: Demonstration of the Unified Evolutionary Software Engineering Agent.

This script shows how to use UE-SEA to improve code in a repository.
"""

import asyncio
import tempfile
import os
from pathlib import Path
from src.ue_sea.orchestrator import UE_SEA_Orchestrator, CycleConfig


async def create_demo_repository():
    """Create a demo repository with sample code."""
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp(prefix="ue_sea_demo_"))
    
    # Create sample Python files
    sample_code = {
        "main.py": '''
def calculate_fibonacci(n):
    """Calculate Fibonacci number."""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def main():
    print("Fibonacci calculator")
    for i in range(10):
        result = calculate_fibonacci(i)
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
from main import calculate_fibonacci

class TestFibonacci(unittest.TestCase):
    def test_fibonacci(self):
        self.assertEqual(calculate_fibonacci(0), 0)
        self.assertEqual(calculate_fibonacci(1), 1)
        self.assertEqual(calculate_fibonacci(5), 5)
        self.assertEqual(calculate_fibonacci(10), 55)

if __name__ == "__main__":
    unittest.main()
'''
    }
    
    # Write files
    for filename, content in sample_code.items():
        file_path = temp_dir / filename
        file_path.write_text(content)
    
    return temp_dir


async def run_demo():
    """Run the UE-SEA demo."""
    print("=== UE-SEA Demo ===")
    print("Creating demo repository...")
    
    # Create demo repository
    repo_path = await create_demo_repository()
    print(f"Demo repository created at: {repo_path}")
    
    # Configure UE-SEA
    config = CycleConfig(
        max_cycles=5,  # Small number for demo
        cycle_timeout=300,  # 5 minutes
        evolution_budget=50,  # Small budget for demo
        training_budget=20,
        deployment_threshold=0.7,
        parallel_workers=2,
        gpu_enabled=True
    )
    
    # Initialize orchestrator
    orchestrator = UE_SEA_Orchestrator(config)
    
    # Task description
    task_description = """
    Improve the code quality and performance of this Python repository:
    
    1. Optimize the Fibonacci calculation (currently inefficient recursive implementation)
    2. Improve the prime number checking algorithm
    3. Add proper error handling and input validation
    4. Enhance code documentation and type hints
    5. Improve test coverage and quality
    
    Focus on:
    - Performance optimization
    - Code readability and maintainability
    - Proper error handling
    - Comprehensive testing
    """
    
    # Context for the task
    context = {
        "language": "python",
        "focus": "performance",
        "requirements": ["optimization", "testing", "documentation"],
        "constraints": ["maintain_backward_compatibility", "preserve_api"]
    }
    
    try:
        print("\nStarting UE-SEA evolution...")
        print(f"Task: {task_description.strip()}")
        print(f"Repository: {repo_path}")
        print(f"Configuration: {config.max_cycles} cycles, {config.evolution_budget} evaluations per cycle")
        
        # Run evolution
        await orchestrator.start(str(repo_path), task_description, context)
        
        # Print results
        print("\n=== Evolution Complete ===")
        status = orchestrator.get_status()
        print(f"Total cycles completed: {status['total_cycles']}")
        print(f"Programs in registry: {status['program_registry_size']}")
        print(f"Skills in registry: {status['skill_registry_size']}")
        
        # Show best programs
        if orchestrator.program_registry:
            print("\n=== Best Programs ===")
            sorted_programs = sorted(
                orchestrator.program_registry.items(),
                key=lambda x: x[1]["score"],
                reverse=True
            )
            
            for i, (program_id, program_data) in enumerate(sorted_programs[:3]):
                print(f"\nProgram {i+1} (ID: {program_id}):")
                print(f"  Score: {program_data['score']:.3f}")
                print(f"  Timestamp: {program_data['timestamp']}")
        
        # Show skill paths
        if orchestrator.skill_registry:
            print("\n=== Skill Paths ===")
            for skill_id, skill in orchestrator.skill_registry.items():
                print(f"  {skill_id}: {skill.description}")
        
        print(f"\nResults saved to: {repo_path}")
        print("Demo complete!")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print(f"\nCleaning up demo repository: {repo_path}")
        import shutil
        shutil.rmtree(repo_path, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(run_demo())
