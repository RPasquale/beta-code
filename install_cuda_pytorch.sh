#!/bin/bash
# Bash script to install PyTorch with CUDA support on Linux/Mac
# Run this in your activated virtual environment

echo "Installing PyTorch with CUDA support..."

# Uninstall existing CPU version if present
echo -e "\nStep 1: Uninstalling existing PyTorch packages..."
pip uninstall -y torch torchvision torchaudio

# Install CUDA version
# CUDA 11.8 version (most compatible)
echo -e "\nStep 2: Installing PyTorch with CUDA 11.8 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Alternative: CUDA 12.1 version (if you have newer CUDA drivers)
# Uncomment the line below and comment out the cu118 line if you prefer CUDA 12.1
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo -e "\nStep 3: Verifying installation..."
python check_cuda_pytorch.py

echo -e "\n✅ Installation complete!"
echo "Run 'python check_cuda_pytorch.py' anytime to verify CUDA support."

