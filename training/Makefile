# UE-SEA Training Makefile for RTX 4090
# Safe, crash-proof training pipeline

.PHONY: setup download-data sft-diff sft-tools sft-tests rl eval clean help

# Default model (start with 3B, can upgrade to 7B later)
MODEL := Qwen/Qwen2.5-Coder-3B

# Output directories
OUTPUT_DIR := outputs
DATA_DIR := data

help:
	@echo "UE-SEA Training Commands for RTX 4090:"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Setup environment and install dependencies"
	@echo "  make download-data  - Download training datasets"
	@echo ""
	@echo "SFT Training (run one at a time):"
	@echo "  make sft-diff       - Train diff generation model"
	@echo "  make sft-tools      - Train tool-use model"
	@echo "  make sft-tests      - Train test generation model"
	@echo ""
	@echo "RL Training:"
	@echo "  make rl             - Run RL training (after SFT)"
	@echo ""
	@echo "Evaluation:"
	@echo "  make eval           - Run evaluation on trained models"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean          - Clean up temporary files"
	@echo "  make status         - Check training status"
	@echo "  make monitor        - Monitor GPU usage and temperature"

setup:
	@echo "Setting up environment for RTX 4090..."
	python -m venv ue4090_env
	@echo "Environment created. Activate with:"
	@echo "  source ue4090_env/bin/activate  # Linux/Mac"
	@echo "  ue4090_env\\Scripts\\activate   # Windows"
	@echo ""
	@echo "Then run: make install-deps"

install-deps:
	@echo "Installing dependencies..."
	pip install --upgrade pip
	pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
	pip install transformers==4.43.3 accelerate==0.33.0 datasets==2.20.0 peft==0.11.1 bitsandbytes==0.43.1 trl==0.9.6
	pip install einops sentencepiece tiktoken tree_sitter==0.21.3 whoosh fastapi uvicorn pydantic pytest coverage hypothesis
	@echo "Dependencies installed successfully!"

download-data:
	@echo "Downloading datasets..."
	mkdir -p $(DATA_DIR)/raw
	mkdir -p $(DATA_DIR)/processed
	python scripts/download_datasets.py
	@echo "Datasets downloaded!"

sft-diff:
	@echo "Training diff generation model..."
	mkdir -p $(OUTPUT_DIR)/sft_diff
	accelerate launch sft/train_diff_safe.py \
		--config_file configs/sft_safe_4090.yaml \
		--model_name $(MODEL) \
		--output_dir $(OUTPUT_DIR)/sft_diff \
		--train_file $(DATA_DIR)/processed/diffs.jsonl \
		--max_train_samples 10000
	@echo "Diff model training completed!"

sft-tools:
	@echo "Training tool-use model..."
	mkdir -p $(OUTPUT_DIR)/sft_tools
	accelerate launch sft/train_tools_safe.py \
		--config_file configs/sft_safe_4090.yaml \
		--model_name $(MODEL) \
		--output_dir $(OUTPUT_DIR)/sft_tools \
		--train_file $(DATA_DIR)/processed/tool_use.jsonl \
		--max_train_samples 10000
	@echo "Tool-use model training completed!"

sft-tests:
	@echo "Training test generation model..."
	mkdir -p $(OUTPUT_DIR)/sft_tests
	accelerate launch sft/train_tests_safe.py \
		--config_file configs/sft_safe_4090.yaml \
		--model_name $(MODEL) \
		--output_dir $(OUTPUT_DIR)/sft_tests \
		--train_file $(DATA_DIR)/processed/tests.jsonl \
		--max_train_samples 10000
	@echo "Test generation model training completed!"

rl:
	@echo "Running RL training..."
	mkdir -p $(OUTPUT_DIR)/rl
	accelerate launch rl/run_grpo_safe.py \
		--config_file configs/rl_safe_4090.yaml \
		--model_name $(OUTPUT_DIR)/sft_diff \
		--output_dir $(OUTPUT_DIR)/rl \
		--tasks_file $(DATA_DIR)/processed/rl_tasks.jsonl
	@echo "RL training completed!"

eval:
	@echo "Running evaluation..."
	python scripts/evaluate_models.py \
		--diff_model $(OUTPUT_DIR)/sft_diff \
		--tools_model $(OUTPUT_DIR)/sft_tools \
		--tests_model $(OUTPUT_DIR)/sft_tests \
		--rl_model $(OUTPUT_DIR)/rl \
		--output_dir $(OUTPUT_DIR)/eval_results
	@echo "Evaluation completed!"

status:
	@echo "Checking training status..."
	@echo "Output directories:"
	@ls -la $(OUTPUT_DIR)/ 2>/dev/null || echo "No output directory found"
	@echo ""
	@echo "Data directories:"
	@ls -la $(DATA_DIR)/ 2>/dev/null || echo "No data directory found"
	@echo ""
	@echo "GPU status:"
	@nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "nvidia-smi not available"

monitor:
	@echo "Monitoring GPU usage and temperature..."
	@echo "Press Ctrl+C to stop"
	@while true; do \
		clear; \
		echo "=== GPU Status ==="; \
		nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "nvidia-smi not available"; \
		echo ""; \
		echo "=== Memory Usage ==="; \
		free -h 2>/dev/null || echo "free command not available"; \
		echo ""; \
		echo "=== Training Processes ==="; \
		ps aux | grep -E "(python|accelerate)" | grep -v grep || echo "No training processes found"; \
		echo ""; \
		echo "Last updated: $$(date)"; \
		sleep 5; \
	done

clean:
	@echo "Cleaning up temporary files..."
	rm -rf $(OUTPUT_DIR)/*/recovery/
	rm -rf $(OUTPUT_DIR)/*/logs/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf *.pyc
	@echo "Cleanup completed!"

clean-all:
	@echo "Cleaning up all outputs..."
	rm -rf $(OUTPUT_DIR)/
	rm -rf $(DATA_DIR)/processed/
	@echo "All outputs cleaned!"

# Safety checks
check-gpu:
	@echo "Checking GPU availability..."
	@python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

check-memory:
	@echo "Checking available memory..."
	@python -c "import psutil; print(f'CPU Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB'); print(f'Available: {psutil.virtual_memory().available / 1024**3:.1f} GB')"

# Training sequence (run in order)
train-all: setup download-data sft-diff sft-tools sft-tests rl eval
	@echo "Complete training pipeline finished!"

# Quick test (small dataset)
test-training:
	@echo "Running quick test training..."
	make sft-diff MODEL=$(MODEL) max_train_samples=100
	@echo "Test training completed!"
