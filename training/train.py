#!/usr/bin/env python3
"""
Unified UE-SEA training entry point.

This script consolidates every previous training variant into a single, fully
configured pipeline that:
  * loads data for all UE-SEA training objectives (core diffing, evolutionary,
    AlphaCode-style tool use, reasoning, and RL-inspired tasks);
  * runs supervised fine-tuning with Hugging Face Transformers + PEFT LoRA;
  * tracks extended metrics per objective and system statistics; and
  * logs everything to both structured JSONL files and Weights & Biases.
It also supports configurable validation splits so you can monitor held-out
performance alongside training loss.

Usage (defaults are safe for an RTX 4090):

    python train.py --output-dir outputs/unified_run

Override any argument from the CLI; run `python train.py --help` for details.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import math
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import psutil
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model

try:  # Optional dependency; if missing we'll warn and disable later.
    import wandb  # type: ignore[import]
except ImportError:  # pragma: no cover - handled at runtime
    wandb = None  # type: ignore[assignment]


LOGGER = logging.getLogger("ue_sea.train")

# Relative paths are resolved from the repository / training directory.
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"


OBJECTIVE_ORDER = ["core", "evolutionary", "alpha_code", "reasoning", "rl"]
OBJECTIVE_INDEX = {name: idx for idx, name in enumerate(OBJECTIVE_ORDER)}

# Each objective lists JSONL files that contain samples for that objective.
OBJECTIVE_SOURCES: Dict[str, List[Path]] = {
    "core": [DATA_DIR / "processed" / "diffs.jsonl"],
    "evolutionary": [DATA_DIR / "evolutionary" / "evolutionary_data.jsonl"],
    "alpha_code": [DATA_DIR / "processed" / "tool_use.jsonl"],
    "reasoning": [DATA_DIR / "reasoning" / "reasoning_data.jsonl"],
    "rl": [DATA_DIR / "rl" / "rl_data.jsonl"],
}


def parse_args() -> "TrainingConfig":
    """Parse CLI arguments into a TrainingConfig instance."""

    parser = argparse.ArgumentParser(description="Unified UE-SEA training script.")
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-Coder-3B",
        help="Base model to fine-tune.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/unified_training",
        help="Directory to store checkpoints, logs, and metrics.",
    )
    parser.add_argument(
        "--max-samples-per-objective",
        type=int,
        default=None,
        help="Optional cap for samples loaded per objective.",
    )
    parser.add_argument(
        "--eval-max-samples-per-objective",
        type=int,
        default=None,
        help="Optional cap for validation samples per objective (after splitting).",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Fraction of data per objective reserved for validation (0 disables eval).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=768,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        help="Per-device batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1.5e-4,
        help="Peak learning rate.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=1.0,
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Ratio of total steps used for LR warmup.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Interval (in optimizer steps) to log metrics.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Interval (in optimizer steps) to save checkpoints.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help="Maximum number of checkpoints to retain.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--wandb-project",
        default="ue-sea-training",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional custom Weights & Biases run name.",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Run without initializing Weights & Biases.",
    )
    parser.add_argument(
        "--compute-grad-norm",
        action="store_true",
        help="Compute gradient norm each step (adds overhead).",
    )

    args = parser.parse_args()

    if not 0.0 <= args.validation_ratio < 1.0:
        parser.error("--validation-ratio must be in the range [0.0, 1.0).")
    if args.eval_max_samples_per_objective is not None and args.eval_max_samples_per_objective <= 0:
        parser.error("--eval-max-samples-per-objective must be a positive integer.")

    return TrainingConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_samples_per_objective=args.max_samples_per_objective,
        eval_max_samples_per_objective=args.eval_max_samples_per_objective,
        validation_ratio=args.validation_ratio,
        max_length=args.max_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        use_wandb=not args.disable_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        compute_grad_norm=args.compute_grad_norm,
    )


@dataclass
class TrainingConfig:
    """Runtime configuration for UE-SEA training."""

    model_name: str
    output_dir: str
    max_samples_per_objective: Optional[int]
    eval_max_samples_per_objective: Optional[int]
    validation_ratio: float
    max_length: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_train_epochs: float
    warmup_ratio: float
    weight_decay: float
    logging_steps: int
    save_steps: int
    save_total_limit: int
    seed: int
    use_wandb: bool
    wandb_project: str
    wandb_run_name: Optional[str]
    compute_grad_norm: bool

    def to_serializable_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable version of the config."""
        data = asdict(self)
        return data


class TrainingMetricsTracker:
    """Track system metrics, objective coverage, and training statistics."""

    def __init__(
        self,
        output_dir: Path,
        objective_names: List[str],
        wandb_run: Optional["wandb.sdk.wandb_run.Run"] = None,
    ) -> None:
        self.output_dir = output_dir
        self.metrics_path = output_dir / "training_metrics.jsonl"
        self.objective_names = list(objective_names)
        self.objective_stats = {
            name: {"seen": 0, "last_batch": 0} for name in self.objective_names
        }
        self.system_metrics: Dict[str, float] = {
            "gpu_memory_gb": 0.0,
            "gpu_temperature": 0.0,
            "cpu_usage": 0.0,
            "training_loss": 0.0,
            "learning_rate": 0.0,
            "gradient_norm": 0.0,
        }
        self.start_time = time.time()
        self.wandb_run = wandb_run

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _update_system_metrics(self) -> None:
        """Refresh GPU + CPU telemetry if available."""
        # GPU metrics via nvidia-smi
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                gpu_memory, gpu_temp = result.stdout.strip().split(", ")
                self.system_metrics["gpu_memory_gb"] = float(gpu_memory) / 1024.0
                self.system_metrics["gpu_temperature"] = float(gpu_temp)
        except Exception:  # pragma: no cover - telemetry best effort
            pass

        # CPU metrics
        try:
            self.system_metrics["cpu_usage"] = psutil.cpu_percent()
        except Exception:  # pragma: no cover
            self.system_metrics["cpu_usage"] = 0.0

    def record_step(
        self,
        *,
        step: int,
        loss: float,
        learning_rate: float,
        grad_norm: float,
        objective_batch: Counter,
    ) -> None:
        """Update internal state and emit logs for the current step."""
        self.system_metrics["training_loss"] = loss
        self.system_metrics["learning_rate"] = learning_rate
        self.system_metrics["gradient_norm"] = grad_norm

        for objective_name, count in objective_batch.items():
            stats = self.objective_stats.setdefault(
                objective_name, {"seen": 0, "last_batch": 0}
            )
            stats["seen"] += count
            stats["last_batch"] = count

        self._update_system_metrics()
        self._emit(step)

    def _emit(self, step: int) -> None:
        """Persist metrics to disk and optionally to W&B."""
        timestamp = datetime.utcnow().isoformat()
        elapsed = time.time() - self.start_time

        payload = {
            "timestamp": timestamp,
            "step": step,
            "elapsed_seconds": elapsed,
            "system": self.system_metrics,
            "objectives": self.objective_stats,
        }

        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

        if self.wandb_run:
            wandb_payload = {
                "train/loss": self.system_metrics["training_loss"],
                "train/learning_rate": self.system_metrics["learning_rate"],
                "train/gradient_norm": self.system_metrics["gradient_norm"],
                "system/gpu_memory_gb": self.system_metrics["gpu_memory_gb"],
                "system/gpu_temperature_c": self.system_metrics["gpu_temperature"],
                "system/cpu_usage_pct": self.system_metrics["cpu_usage"],
                "time/elapsed_seconds": elapsed,
            }
            for objective_name, stats in self.objective_stats.items():
                wandb_payload[f"objectives/{objective_name}_seen"] = stats["seen"]
                wandb_payload[f"objectives/{objective_name}_last_batch"] = stats[
                    "last_batch"
                ]
            wandb.log(wandb_payload, step=step)

        LOGGER.info(
            "step=%d loss=%.4f lr=%.2e | objectives: %s",
            step,
            self.system_metrics["training_loss"],
            self.system_metrics["learning_rate"],
            ", ".join(
                f"{name}:{stats['seen']}"
                for name, stats in self.objective_stats.items()
            ),
        )

    def summary(self) -> Dict[str, Any]:
        """Return a summary snapshot of metrics collected so far."""
        return {
            "total_time_seconds": time.time() - self.start_time,
            "system": self.system_metrics,
            "objectives": self.objective_stats,
            "metrics_path": str(self.metrics_path),
        }


class MetricsTrainer(Trainer):
    """Trainer subclass that wires objective tracking + rich logging."""

    def __init__(
        self,
        *,
        metrics_tracker: TrainingMetricsTracker,
        objective_names: List[str],
        compute_grad_norm: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.metrics_tracker = metrics_tracker
        self.objective_names = objective_names
        self.compute_grad_norm = compute_grad_norm
        self._tracking_step = 0

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        objective_ids = inputs.pop("objective_id", None)

        result = super().training_step(model, inputs)
        loss_value = result.detach().float().cpu().item()

        learning_rate = self._get_learning_rate()

        grad_norm = 0.0
        if self.compute_grad_norm:
            grad_norm = self._compute_grad_norm(model)

        batch_counter: Counter = Counter()
        if objective_ids is not None:
            if isinstance(objective_ids, torch.Tensor):
                objective_ids = objective_ids.detach().cpu().tolist()
            for idx in objective_ids:
                try:
                    name = self.objective_names[int(idx)]
                except (IndexError, ValueError):
                    continue
                batch_counter[name] += 1

        self._tracking_step += 1
        self.metrics_tracker.record_step(
            step=self._tracking_step,
            loss=loss_value,
            learning_rate=learning_rate,
            grad_norm=grad_norm,
            objective_batch=batch_counter,
        )
        return result

    @staticmethod
    def _compute_grad_norm(model: torch.nn.Module) -> float:
        total_norm_sq = 0.0
        for param in model.parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            param_norm = grad.data.norm(2)
            total_norm_sq += param_norm.item() ** 2
        return total_norm_sq ** 0.5


def set_seed(seed: int) -> None:
    """Seed random number generators for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - device specific
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path, *, max_items: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    """Yield parsed JSON objects from a JSONL file."""
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning("Skipping invalid JSON in %s", path)
                continue
            yield record
            count += 1
            if max_items is not None and count >= max_items:
                break


def format_core_sample(sample: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    instruction = sample.get("instruction", "")
    current_program = sample.get("current_program", "")
    target = sample.get("target_output", "")

    if isinstance(target, list):
        target_text = "\n".join(str(item) for item in target)
    else:
        target_text = str(target)

    prompt = (
        "[Objective: CORE DIFF]\n"
        f"Instruction: {instruction}\n\n"
        "Current Program:\n"
        f"{current_program}\n\n"
        "Produce the unified diff that applies the requested change:"
    )
    return prompt, target_text


def format_evolutionary_sample(sample: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    prompt_text = sample.get("prompt")
    current_code = sample.get("code")
    target_code = sample.get("target")

    if not prompt_text or not target_code:
        return None

    prompt = (
        "[Objective: EVOLUTIONARY]\n"
        f"{prompt_text}\n\n"
        "Current Implementation:\n"
        f"{current_code}\n\n"
        "Return the improved implementation:"
    )
    return prompt, str(target_code)


def _format_demo_step(step: Dict[str, Any]) -> str:
    call = step.get("call", "ToolCall")
    args = step.get("args", {})
    obs = step.get("obs", {})
    args_repr = json.dumps(args, sort_keys=True)
    obs_repr = json.dumps(obs, sort_keys=True)
    return f"{call}(args={args_repr}) => {obs_repr}"


def format_alpha_code_sample(sample: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    instruction = sample.get("instruction")
    demo = sample.get("demonstration", [])
    target = sample.get("target")

    if not instruction or target is None:
        return None

    demo_text = "\n".join(
        f"Step {idx + 1}: {_format_demo_step(step)}" for idx, step in enumerate(demo)
    )

    if isinstance(target, dict):
        final_entities = target.get("final_entity_ids") or target.get("entities", [])
        target_text = "\n".join(str(entity) for entity in final_entities) or json.dumps(
            target, indent=2, sort_keys=True
        )
    else:
        target_text = str(target)

    prompt = (
        "[Objective: ALPHACODE TOOL USE]\n"
        f"Instruction: {instruction}\n\n"
        "Tool Demonstration:\n"
        f"{demo_text}\n\n"
        "Summarize the entities identified by the tool usage:"
    )
    return prompt, target_text


def format_reasoning_sample(sample: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    task = sample.get("task", "reasoning")
    prompt_text = sample.get("prompt")
    code = sample.get("code")
    target = sample.get("target")

    if not prompt_text or target is None:
        return None

    prompt = (
        f"[Objective: REASONING - {task.upper()}]\n"
        f"{prompt_text}\n\n"
        "Context Code:\n"
        f"{code}\n\n"
        "Provide the reasoning-driven improvement or analysis:"
    )
    return prompt, str(target)


def format_rl_sample(sample: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    prompt_text = sample.get("prompt")
    tests = sample.get("tests", [])
    expected_reward = sample.get("expected_reward")
    domain = sample.get("domain", "general")

    if not prompt_text:
        return None

    tests_text = "\n".join(str(test) for test in tests)
    target = (
        f"# Objective domain: {domain}\n"
        f"# Expected reward: {expected_reward}\n"
        "# Ensure the implementation satisfies the following checks:\n"
        f"{tests_text}"
    )

    prompt = (
        "[Objective: RL TASK]\n"
        f"{prompt_text}\n\n"
        "Respond with guidance that maximizes the reward signal:"
    )
    return prompt, target


FORMATTERS = {
    "core": format_core_sample,
    "evolutionary": format_evolutionary_sample,
    "alpha_code": format_alpha_code_sample,
    "reasoning": format_reasoning_sample,
    "rl": format_rl_sample,
}


def build_objective_examples(
    *,
    max_samples_per_objective: Optional[int],
) -> Dict[str, List[Dict[str, Any]]]:
    """Load and normalize samples grouped by objective."""
    grouped_examples: Dict[str, List[Dict[str, Any]]] = {key: [] for key in OBJECTIVE_ORDER}
    for objective in OBJECTIVE_ORDER:
        formatter = FORMATTERS[objective]
        available_paths = [path for path in OBJECTIVE_SOURCES[objective] if path.exists()]

        if not available_paths:
            LOGGER.warning(
                "No data found for objective '%s'. Expected one of: %s",
                objective,
                ", ".join(str(p) for p in OBJECTIVE_SOURCES[objective]),
            )
            continue

        sample_cap = max_samples_per_objective
        loaded = 0

        for path in available_paths:
            for record in load_jsonl(path):
                formatted = formatter(record)
                if formatted is None:
                    continue
                prompt, target = formatted
                grouped_examples[objective].append(
                    {
                        "prompt": prompt,
                        "response": target,
                        "objective": objective,
                        "objective_id": OBJECTIVE_INDEX[objective],
                    }
                )
                loaded += 1
                if sample_cap is not None and loaded >= sample_cap:
                    break
            if sample_cap is not None and loaded >= sample_cap:
                break

    total_loaded = sum(len(items) for items in grouped_examples.values())
    if total_loaded == 0:
        raise RuntimeError(
            "No training data found. Ensure the dataset download step completed "
            "and the JSONL files exist under training/data/."
        )

    for items in grouped_examples.values():
        random.shuffle(items)

    return grouped_examples


def split_examples(
    *,
    grouped_examples: Dict[str, List[Dict[str, Any]]],
    validation_ratio: float,
    eval_max_samples_per_objective: Optional[int],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, int],
    Dict[str, int],
]:
    """Split grouped examples into train and validation sets."""
    train_examples: List[Dict[str, Any]] = []
    eval_examples: List[Dict[str, Any]] = []
    train_distribution: Dict[str, int] = {objective: 0 for objective in OBJECTIVE_ORDER}
    eval_distribution: Dict[str, int] = {objective: 0 for objective in OBJECTIVE_ORDER}

    for objective, items in grouped_examples.items():
        if not items:
            continue

        eval_count = 0
        if validation_ratio > 0 and len(items) > 1:
            eval_count = max(1, int(len(items) * validation_ratio))
            eval_count = min(eval_count, len(items) - 1)

        eval_subset = items[:eval_count] if eval_count else []
        train_subset = items[eval_count:] if eval_count else items

        if eval_max_samples_per_objective is not None and eval_subset:
            kept_eval = eval_subset[:eval_max_samples_per_objective]
            overflow = eval_subset[eval_max_samples_per_objective:]
            train_subset = overflow + train_subset
            eval_subset = kept_eval

        train_examples.extend(train_subset)
        eval_examples.extend(eval_subset)

        train_distribution[objective] = len(train_subset)
        eval_distribution[objective] = len(eval_subset)

    random.shuffle(train_examples)
    random.shuffle(eval_examples)

    return train_examples, eval_examples, train_distribution, eval_distribution


def prepare_datasets(
    *,
    tokenizer: AutoTokenizer,
    max_length: int,
    max_samples_per_objective: Optional[int],
    validation_ratio: float,
    eval_max_samples_per_objective: Optional[int],
) -> Tuple[
    Dataset,
    Optional[Dataset],
    Dict[str, int],
    Dict[str, int],
]:
    """Create Hugging Face datasets for training and validation."""
    grouped = build_objective_examples(
        max_samples_per_objective=max_samples_per_objective
    )
    train_examples, eval_examples, train_distribution, eval_distribution = split_examples(
        grouped_examples=grouped,
        validation_ratio=validation_ratio,
        eval_max_samples_per_objective=eval_max_samples_per_objective,
    )


    def preprocess_batch(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        texts = [
            f"{prompt}\n\n### Response:\n{response}"
            for prompt, response in zip(batch["prompt"], batch["response"])
        ]
        tokenized = tokenizer(
            texts,
            max_length=max_length,
            padding=False,
            truncation=True,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        labels = [ids.copy() for ids in input_ids]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "objective_id": batch["objective_id"],
        }

    train_dataset = Dataset.from_list(train_examples)
    train_dataset = train_dataset.map(
        preprocess_batch,
        batched=True,
        remove_columns=["prompt", "response", "objective"],
        desc="Tokenizing train dataset",
    )

    eval_dataset = None
    if eval_examples:
        eval_dataset = Dataset.from_list(eval_examples)
        eval_dataset = eval_dataset.map(
            preprocess_batch,
            batched=True,
            remove_columns=["prompt", "response", "objective"],
            desc="Tokenizing validation dataset",
        )

    return train_dataset, eval_dataset, train_distribution, eval_distribution


class ObjectiveAwareCollator:
    """Wrap a data collator so it preserves the objective ids."""

    def __init__(self, base_collator: DataCollatorForLanguageModeling):
        self.base_collator = base_collator

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        objective_ids = [feature["objective_id"] for feature in features]

        stripped_features = [
            {k: v for k, v in feature.items() if k != "objective_id"}
            for feature in features
        ]

        batch = self.base_collator(stripped_features)
        batch["objective_id"] = torch.tensor(objective_ids, dtype=torch.long)
        return batch


def init_wandb(
    config: TrainingConfig,
    train_distribution: Dict[str, int],
    eval_distribution: Dict[str, int],
) -> Optional["wandb.sdk.wandb_run.Run"]:
    """Initialise a Weights & Biases run if requested."""
    if not config.use_wandb:
        return None

    if wandb is None or not hasattr(wandb, "init"):
        LOGGER.warning(
            "Weights & Biases is unavailable or incompatible (module=%r). "
            "Disable W&B logging with --disable-wandb or install/upgrade wandb.",
            wandb,
        )
        return None

    run_name = config.wandb_run_name or datetime.utcnow().strftime("ue-sea-%Y%m%d-%H%M%S")
    mode = os.getenv("WANDB_MODE", "offline")
    wandb_run = wandb.init(
        project=config.wandb_project,
        name=run_name,
        mode=mode,
        config={
            **config.to_serializable_dict(),
            "train_objective_distribution": train_distribution,
            "eval_objective_distribution": eval_distribution,
        },
    )
    return wandb_run


def setup_logging(output_dir: Path) -> None:
    """Configure root logging for console + file output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "training.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def configure_model(model_name: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load tokenizer and base model, applying LoRA."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.train()
    return tokenizer, model


def main() -> None:
    config = parse_args()

    output_dir = (SCRIPT_DIR / config.output_dir).resolve()
    setup_logging(output_dir)
    LOGGER.info("Starting UE-SEA unified training with config: %s", config)

    set_seed(config.seed)

    tokenizer, model = configure_model(config.model_name)
    LOGGER.info("Loaded model %s with tokenizer %s", config.model_name, type(tokenizer).__name__)

    train_dataset, eval_dataset, train_distribution, eval_distribution = prepare_datasets(
        tokenizer=tokenizer,
        max_length=config.max_length,
        max_samples_per_objective=config.max_samples_per_objective,
        validation_ratio=config.validation_ratio,
        eval_max_samples_per_objective=config.eval_max_samples_per_objective,
    )

    LOGGER.info("Training dataset prepared with %d samples", len(train_dataset))
    for objective, count in train_distribution.items():
        LOGGER.info("Objective %-12s : %d samples", objective, count)

    if eval_dataset is not None:
        LOGGER.info("Validation dataset prepared with %d samples", len(eval_dataset))
        for objective, count in eval_distribution.items():
            LOGGER.info("Eval Objective %-12s : %d samples", objective, count)
    else:
        LOGGER.info("Validation disabled (no held-out samples).")

    wandb_run = init_wandb(config, train_distribution, eval_distribution)
    metrics_tracker = TrainingMetricsTracker(
        output_dir=output_dir,
        objective_names=OBJECTIVE_ORDER,
        wandb_run=wandb_run,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        max_grad_norm=1.0,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        remove_unused_columns=False,
        group_by_length=True,
        bf16=True,
        gradient_checkpointing=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        report_to=[],
        save_safetensors=True,
        dataloader_drop_last=True,
        logging_dir=str(output_dir / "logs"),
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )
    collator = ObjectiveAwareCollator(data_collator)

    trainer = MetricsTrainer(
        metrics_tracker=metrics_tracker,
        objective_names=OBJECTIVE_ORDER,
        compute_grad_norm=config.compute_grad_norm,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    eval_results: Optional[Dict[str, Any]] = None

    try:
        trainer.train()
    except KeyboardInterrupt:
        LOGGER.warning("Training interrupted by user.")
    finally:
        if eval_dataset is not None and len(eval_dataset) > 0:
            eval_results = trainer.evaluate(metric_key_prefix="eval")
            if "eval_loss" in eval_results:
                try:
                    eval_results["eval_perplexity"] = math.exp(eval_results["eval_loss"])
                except (OverflowError, ValueError):  # pragma: no cover - numerical edge
                    LOGGER.warning("Could not compute perplexity from eval_loss")
            eval_path = output_dir / "eval_results.json"
            with eval_path.open("w", encoding="utf-8") as handle:
                json.dump(eval_results, handle, indent=2)
            LOGGER.info("Validation results: %s", eval_results)
            if wandb_run:
                step = trainer.state.global_step or 0
                wandb_run.log(eval_results, step=step)
                wandb_run.summary.update(eval_results)

        LOGGER.info("Saving model artifacts to %s", output_dir)
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)

        summary = metrics_tracker.summary()
        if eval_results is not None:
            summary["evaluation"] = eval_results
        LOGGER.info("Training summary: %s", summary)
        if wandb_run:
            wandb_run.log({"training/total_time_seconds": summary["total_time_seconds"]})
            wandb_run.finish()

        LOGGER.info("Unified training completed. Metrics logged to %s", summary["metrics_path"])


if __name__ == "__main__":
    main()
