"""
Test runner for LOCAGENT tests.

Runs all LOCAGENT tests and provides a summary.
"""

import sys
import subprocess
from pathlib import Path


def run_tests():
    """Run all LOCAGENT tests."""
    print("🚀 Running LOCAGENT Tests")
    print("=" * 50)
    
    # Add src to path
    src_path = Path(__file__).parent.parent.parent / "src"
    sys.path.insert(0, str(src_path))
    
    # Run tests with pytest
    test_files = [
        "test_entities.py",
        "test_indices.py", 
        "test_agent.py",
        "test_training.py",
        "test_integration.py"
    ]
    
    all_passed = True
    
    for test_file in test_files:
        print(f"\n📋 Running {test_file}...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                f"tests/locagent/{test_file}", 
                "-v", "--tb=short"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {test_file} passed")
            else:
                print(f"❌ {test_file} failed")
                print(result.stdout)
                print(result.stderr)
                all_passed = False
                
        except Exception as e:
            print(f"❌ Error running {test_file}: {e}")
            all_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All LOCAGENT tests passed!")
        print("\nLOCAGENT is fully operational with:")
        print("✓ Entity and relation management")
        print("✓ Hierarchical indexing (ID, Name, BM25)")
        print("✓ Agent reasoning loop with tool usage")
        print("✓ Training pipeline for fine-tuning")
        print("✓ Data loading for SWE-Bench and LocBench")
        print("✓ Integration testing")
        
        print("\nNext steps:")
        print("1. Build graph: python -c \"import asyncio; from src.ue_sea.locagent.graph import LOCAGENT_Graph; graph = LOCAGENT_Graph(); asyncio.run(graph.build_from_repository('src/ue_sea/locagent'))\"")
        print("2. Test localization: Use the agent to localize code issues")
        print("3. Train the agent: Use the training pipeline with your data")
        
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
