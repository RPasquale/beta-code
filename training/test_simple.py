#!/usr/bin/env python3
"""
Simple test script to verify the training setup works.
"""

import torch
import json
from pathlib import Path

def test_gpu():
    """Test GPU availability and memory."""
    print("=== GPU Test ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Test memory allocation
        try:
            # Allocate a small tensor to test memory
            test_tensor = torch.randn(1000, 1000).cuda()
            print("GPU memory allocation test passed")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"GPU memory allocation test failed: {e}")
            return False
    
    return True

def test_data():
    """Test data loading."""
    print("\n=== Data Test ===")
    
    # Check if data files exist
    data_dir = Path("data/processed")
    if not data_dir.exists():
        print("Data directory not found")
        return False
    
    files = list(data_dir.glob("*.jsonl"))
    print(f"Found {len(files)} data files:")
    for file in files:
        print(f"  - {file.name}")
        
        # Test loading a few samples
        try:
            with open(file, 'r') as f:
                samples = [json.loads(line) for line in f.readlines()[:3]]
            print(f"    Loaded {len(samples)} samples")
        except Exception as e:
            print(f"    Error loading {file.name}: {e}")
            return False
    
    return True

def test_imports():
    """Test importing required libraries."""
    print("\n=== Import Test ===")
    
    try:
        import transformers
        print("transformers imported")
    except ImportError as e:
        print(f"transformers import failed: {e}")
        return False
    
    try:
        import accelerate
        print("accelerate imported")
    except ImportError as e:
        print(f"accelerate import failed: {e}")
        return False
    
    try:
        import peft
        print("peft imported")
    except ImportError as e:
        print(f"peft import failed: {e}")
        return False
    
    try:
        import bitsandbytes
        print("bitsandbytes imported")
    except ImportError as e:
        print(f"bitsandbytes import failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("Testing UE-SEA Training Setup")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_gpu,
        test_data
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            print(f"\nTest failed: {test.__name__}")
            break
    
    print(f"\n{'='*50}")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("All tests passed! Ready for training.")
        return True
    else:
        print("Some tests failed. Please fix issues before training.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
