#!/usr/bin/env python3
"""
Verify that PyTorch is installed with CUDA support.
Run this script to check if your environment has CUDA-enabled PyTorch.
"""

import sys

def check_cuda_pytorch():
    """Check if PyTorch has CUDA support."""
    print("=" * 60)
    print("PyTorch CUDA Verification")
    print("=" * 60)
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
    except ImportError:
        print("❌ ERROR: PyTorch is not installed!")
        print("\nTo install PyTorch with CUDA support:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False
    
    # Check if it's CPU-only version
    if "+cpu" in torch.__version__:
        print(f"⚠️  WARNING: CPU-only version detected ({torch.__version__})")
        print("\nThis is the CPU version of PyTorch. CUDA support is NOT available.")
        print("\nTo fix this, uninstall and reinstall with CUDA support:")
        print("  pip uninstall torch torchvision torchaudio")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("\nOr for CUDA 12.1:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"✅ CUDA available: Yes")
        print(f"✅ CUDA version: {torch.version.cuda}")
        print(f"✅ cuDNN version: {torch.backends.cudnn.version()}")
        print(f"✅ Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"          Memory: {props.total_memory / 1024**3:.1f} GB")
        
        # Test GPU computation
        try:
            x = torch.randn(100, 100).cuda()
            y = torch.matmul(x, x)
            print(f"\n✅ GPU computation test: PASSED")
            del x, y
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"\n⚠️  GPU computation test: FAILED - {e}")
            return False
        
        print("\n" + "=" * 60)
        print("✅ SUCCESS: PyTorch with CUDA support is correctly installed!")
        print("=" * 60)
        return True
    else:
        print(f"❌ CUDA available: No")
        if "+cpu" not in torch.__version__:
            print("\n⚠️  PyTorch was compiled with CUDA support, but CUDA runtime is not available.")
            print("This could mean:")
            print("  1. CUDA drivers are not installed")
            print("  2. CUDA runtime version doesn't match PyTorch build")
            print("  3. GPU is not detected")
            print("\nCheck your CUDA installation:")
            print("  nvidia-smi  # Should show your GPU")
        return False

if __name__ == "__main__":
    success = check_cuda_pytorch()
    sys.exit(0 if success else 1)

