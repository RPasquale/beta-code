

#!/bin/bash
# Setup script for installing GNN.py dependencies with CUDA 12.6 support
# Make sure you're in the koder directory when running this

set -e  # Exit on error

echo "Step 1: Installing PyTorch with CUDA 12.6 support..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

if [ $? -ne 0 ]; then
    echo "Failed to install PyTorch. Please check your connection and try again."
    exit 1
fi

echo ""
echo "Step 2: Installing remaining dependencies..."
uv sync

if [ $? -ne 0 ]; then
    echo "Failed to sync dependencies."
    exit 1
fi

echo ""
echo "Step 3: Verifying installation..."
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "Done! You can now run:"
echo "  uv run python GNN.py --path <your-data-file>"

