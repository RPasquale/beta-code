# Setup script for installing GNN.py dependencies with CUDA 12.6 support (PowerShell)
# Make sure you're in the koder directory when running this

Write-Host "Step 1: Installing PyTorch with CUDA 12.6 support..." -ForegroundColor Green
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to install PyTorch. Please check your connection and try again." -ForegroundColor Red
    exit 1
}

Write-Host "`nStep 2: Installing remaining dependencies..." -ForegroundColor Green
uv sync

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to sync dependencies." -ForegroundColor Red
    exit 1
}

Write-Host "`nStep 3: Verifying installation..." -ForegroundColor Green
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

Write-Host "`nDone! You can now run:" -ForegroundColor Green
Write-Host "  uv run python GNN.py --path <your-data-file>" -ForegroundColor Yellow

