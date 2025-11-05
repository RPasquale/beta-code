# PowerShell script to install PyTorch with CUDA support on Windows
# Run this in your activated virtual environment

Write-Host "Installing PyTorch with CUDA support..." -ForegroundColor Green

# Uninstall existing CPU version if present
Write-Host "`nStep 1: Uninstalling existing PyTorch packages..." -ForegroundColor Yellow
pip uninstall -y torch torchvision torchaudio

# Install CUDA version
# CUDA 11.8 version (most compatible)
Write-Host "`nStep 2: Installing PyTorch with CUDA 11.8 support..." -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Alternative: CUDA 12.1 version (if you have newer CUDA drivers)
# Uncomment the line below and comment out the cu118 line if you prefer CUDA 12.1
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Write-Host "`nStep 3: Verifying installation..." -ForegroundColor Yellow
python check_cuda_pytorch.py

Write-Host "`n✅ Installation complete!" -ForegroundColor Green
Write-Host "Run 'python check_cuda_pytorch.py' anytime to verify CUDA support." -ForegroundColor Cyan

