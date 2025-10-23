#!/usr/bin/env python3
"""
Safe SFT training for diff generation on RTX 4090.

Features:
- Memory monitoring and automatic cleanup
- Crash recovery
- Automatic batch size adjustment
- Temperature monitoring
- Gradient clipping for stability
"""

import os
import sys
import json
import yaml
import argparse
import logging
import psutil
import gc
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Core training imports
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, 
    Trainer, DataCollatorForLanguageModeling, EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import accelerate
from accelerate import Accelerator
from accelerate.utils import set_seed

# Custom imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.memory_monitor import MemoryMonitor
from utils.crash_recovery import CrashRecovery
from utils.temperature_monitor import TemperatureMonitor


@dataclass
class SafeTrainingConfig:
    """Safe training configuration for RTX 4090."""
    model_name: str = "Qwen/Qwen2.5-Coder-3B"
    max_memory_usage: float = 0.85  # Use max 85% of VRAM
    auto_batch_size: bool = True
    crash_recovery: bool = True
    temperature_threshold: float = 85.0  # Celsius
    gradient_clip_norm: float = 1.0
    save_every_n_steps: int = 200
    max_seq_len: int = 1536
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1.5e-4
    num_train_epochs: int = 2
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    output_dir: str = "outputs/sft_diff_safe"
    train_file: str = "data/processed/diffs.jsonl"
    validation_file: str = "data/processed/diffs_val.jsonl"
    max_train_samples: int = 10000
    max_eval_samples: int = 1000
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Additional config fields that might be in YAML
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    bf16: bool = True
    dataloader_pin_memory: bool = False
    dataloader_num_workers: int = 2
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 500
    save_total_limit: int = 3
    remove_unused_columns: bool = True
    group_by_length: bool = True
    packing: bool = True
    report_to: str = None
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    save_safetensors: bool = True
    dataloader_drop_last: bool = True
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])


class SafeTrainer:
    """Safe trainer with memory monitoring and crash recovery."""
    
    def __init__(self, config: SafeTrainingConfig):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.crash_recovery = CrashRecovery(config.output_dir)
        self.temperature_monitor = TemperatureMonitor()
        
        # Training state
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.accelerator = None
        
        # Safety flags
        self.training_stopped = False
        self.oom_count = 0
        self.max_oom_retries = 3
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for training."""
        log_dir = Path(self.config.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _check_system_requirements(self) -> bool:
        """Check if system meets requirements for safe training."""
        self.logger.info("Checking system requirements...")
        
        # Check GPU
        if not torch.cuda.is_available():
            self.logger.error("CUDA not available!")
            return False
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        self.logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 20:
            self.logger.error(f"GPU memory too low: {gpu_memory:.1f} GB (need at least 20 GB)")
            return False
        
        # Check CPU memory
        cpu_memory = psutil.virtual_memory().total / 1024**3
        self.logger.info(f"CPU Memory: {cpu_memory:.1f} GB")
        
        if cpu_memory < 16:
            self.logger.warning(f"CPU memory low: {cpu_memory:.1f} GB (recommend 16+ GB)")
        
        # Check disk space
        disk_usage = psutil.disk_usage('/').free / 1024**3
        self.logger.info(f"Disk space: {disk_usage:.1f} GB")
        
        if disk_usage < 50:
            self.logger.warning(f"Disk space low: {disk_usage:.1f} GB (recommend 50+ GB)")
        
        return True
    
    def _setup_model_and_tokenizer(self):
        """Setup model and tokenizer with safety measures."""
        self.logger.info("Setting up model and tokenizer...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                padding_side="right"
            )
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with 4-bit quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
            
            # Setup LoRA
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
            self.logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def _load_dataset(self) -> Dataset:
        """Load and prepare dataset."""
        self.logger.info("Loading dataset...")
        
        # Load training data
        train_data = []
        if Path(self.config.train_file).exists():
            with open(self.config.train_file, 'r') as f:
                for line in f:
                    if line.strip():
                        train_data.append(json.loads(line))
        
        # Limit samples for initial training
        if len(train_data) > self.config.max_train_samples:
            train_data = train_data[:self.config.max_train_samples]
        
        self.logger.info(f"Loaded {len(train_data)} training samples")
        
        # Create dataset
        dataset = Dataset.from_list(train_data)
        
        return dataset
    
    def _prepare_training_arguments(self) -> TrainingArguments:
        """Prepare training arguments with safety measures."""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            max_grad_norm=self.config.gradient_clip_norm,
            warmup_ratio=0.03,
            weight_decay=0.0,
            logging_steps=10,
            save_steps=self.config.save_every_n_steps,
            eval_steps=500,
            save_total_limit=3,
            remove_unused_columns=True,
            group_by_length=True,
            dataloader_pin_memory=False,
            dataloader_num_workers=2,
            bf16=True,
            gradient_checkpointing=True,
            report_to=None,  # Disable wandb for stability
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_safetensors=True,
            dataloader_drop_last=True,
        )
    
    def _create_data_collator(self):
        """Create data collator for training."""
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
    
    def _preprocess_function(self, examples):
        """Preprocess examples for training."""
        # Format: instruction + current_program -> target_output
        inputs = []
        targets = []
        
        for example in examples:
            instruction = example.get("instruction", "")
            current_program = example.get("current_program", "")
            target_output = example.get("target_output", [])
            
            # Create input prompt
            prompt = f"Instruction: {instruction}\n\nCurrent Program:\n{current_program}\n\nGenerate diff:\n"
            inputs.append(prompt)
            
            # Create target (join diff blocks)
            if isinstance(target_output, list):
                target = "\n".join(target_output)
            else:
                target = str(target_output)
            targets.append(target)
        
        # Tokenize
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.config.max_seq_len,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize targets
        labels = self.tokenizer(
            targets,
            max_length=self.config.max_seq_len,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Set labels
        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs
    
    def _monitor_training(self):
        """Monitor training for safety."""
        # Check memory usage
        memory_usage = self.memory_monitor.get_gpu_memory_usage()
        if memory_usage > self.config.max_memory_usage:
            self.logger.warning(f"High memory usage: {memory_usage:.2%}")
            
            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()
        
        # Check temperature
        if self.temperature_monitor.is_overheating():
            self.logger.warning("GPU temperature high, pausing training...")
            import time
            time.sleep(30)  # Cool down
        
        # Check for OOM
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.logger.error("OOM detected!")
                    self.oom_count += 1
                    if self.oom_count >= self.max_oom_retries:
                        self.logger.error("Too many OOM errors, stopping training")
                        self.training_stopped = True
                    else:
                        self.logger.info("Reducing batch size and retrying...")
                        # Reduce batch size
                        self.config.per_device_train_batch_size = max(1, self.config.per_device_train_batch_size // 2)
                        self.config.gradient_accumulation_steps *= 2
    
    def train(self):
        """Main training loop with safety measures."""
        self.logger.info("Starting safe training...")
        
        # Check system requirements
        if not self._check_system_requirements():
            return False
        
        try:
            # Setup model and tokenizer
            self._setup_model_and_tokenizer()
            
            # Load dataset
            dataset = self._load_dataset()
            
            # Preprocess dataset
            processed_dataset = dataset.map(
                self._preprocess_function,
                batched=True,
                remove_columns=dataset.column_names
            )
            
            # Setup training arguments
            training_args = self._prepare_training_arguments()
            
            # Create trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=processed_dataset,
                data_collator=self._create_data_collator(),
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )
            
            # Start training with monitoring
            self.logger.info("Starting training...")
            
            # Add monitoring hook
            def on_log(logs):
                self._monitor_training()
                if self.training_stopped:
                    self.logger.info("Training stopped due to safety concerns")
                    return False
            
            # Override trainer's log method
            original_log = self.trainer.log
            self.trainer.log = lambda logs: on_log(logs) and original_log(logs)
            
            # Train
            self.trainer.train()
            
            # Save final model
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)
            
            self.logger.info("Training completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            self.crash_recovery.save_crash_info(str(e))
            return False
        
        finally:
            # Cleanup
            if self.model:
                del self.model
            if self.tokenizer:
                del self.tokenizer
            if self.trainer:
                del self.trainer
            
            gc.collect()
            torch.cuda.empty_cache()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Safe SFT training for RTX 4090")
    parser.add_argument("--config_file", type=str, required=True, help="Path to config file")
    parser.add_argument("--model_name", type=str, help="Override model name")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--train_file", type=str, help="Override training file")
    parser.add_argument("--max_train_samples", type=int, help="Override max training samples")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Override with command line args
    if args.model_name:
        config_dict["model_name"] = args.model_name
    if args.output_dir:
        config_dict["output_dir"] = args.output_dir
    if args.train_file:
        config_dict["train_file"] = args.train_file
    if args.max_train_samples:
        config_dict["max_train_samples"] = args.max_train_samples
    
    # Create config object
    config = SafeTrainingConfig(**config_dict)
    
    # Create trainer and start training
    trainer = SafeTrainer(config)
    success = trainer.train()
    
    if success:
        print("Training completed successfully!")
        sys.exit(0)
    else:
        print("Training failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
