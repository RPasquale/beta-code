"""
LLM Ensemble: Manages multiple LLMs for code generation.

Implements ensemble of fast and powerful models for diff generation.
"""

import asyncio
import os
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import time

# Mock imports - replace with actual LLM libraries
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    hf_pipeline = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ModelType(Enum):
    """Types of models in the ensemble."""
    FAST = "fast"          # Fast, lightweight models
    POWERFUL = "powerful"  # Powerful, slower models
    SPECIALIZED = "specialized"  # Task-specific models


@dataclass
class ModelConfig:
    """Configuration for a model in the ensemble."""
    name: str
    model_type: ModelType
    api_key: Optional[str] = None
    model_path: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.3
    top_p: float = 0.9
    timeout: int = 30
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class GenerationResult:
    """Result from LLM generation."""
    text: str
    model_name: str
    tokens_used: int
    generation_time: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LLM_Ensemble:
    """
    Ensemble of LLMs for code generation.
    
    Manages multiple models with different capabilities and selects
    appropriate models for different generation tasks.
    """
    
    def __init__(self, configs: List[ModelConfig] = None):
        self.models = {}
        self.model_configs = {}
        
        if configs is None:
            configs = self._get_default_configs()
        
        # Initialize models
        for config in configs:
            self._initialize_model(config)
    
    def _get_default_configs(self) -> List[ModelConfig]:
        """Get default model configurations."""
        configs = []
        
        # Fast models for quick iterations
        if OPENAI_AVAILABLE:
            configs.append(ModelConfig(
                name="gpt-3.5-turbo",
                model_type=ModelType.FAST,
                max_tokens=1024,
                temperature=0.3
            ))
        
        # Powerful models for complex tasks
        if OPENAI_AVAILABLE:
            configs.append(ModelConfig(
                name="gpt-4",
                model_type=ModelType.POWERFUL,
                max_tokens=2048,
                temperature=0.3
            ))
        
        # Local models for specialized tasks
        if TRANSFORMERS_AVAILABLE:
            local_model_id = os.getenv(
                "UE_SEA_LOCAL_MODEL",
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
            )
            max_tokens = int(os.getenv("UE_SEA_LOCAL_MODEL_MAX_TOKENS", "2048"))
            model_name = local_model_id.split("/")[-1] or local_model_id
            configs.append(ModelConfig(
                name=model_name,
                model_type=ModelType.SPECIALIZED,
                model_path=local_model_id,
                max_tokens=max_tokens,
                temperature=0.3,
                metadata={"model_id": local_model_id}
            ))
        
        return configs
    
    def _initialize_model(self, config: ModelConfig) -> None:
        """Initialize a model with the given configuration."""
        try:
            if config.model_type == ModelType.FAST or config.model_type == ModelType.POWERFUL:
                # API-based models
                self.models[config.name] = {
                    "type": "api",
                    "config": config,
                    "initialized": True
                }
            elif config.model_type == ModelType.SPECIALIZED:
                # Local models
                if TRANSFORMERS_AVAILABLE and config.model_path:
                    print(f"Loading local model '{config.model_path}' for ensemble generation...")
                    tokenizer = AutoTokenizer.from_pretrained(
                        config.model_path,
                        trust_remote_code=True
                    )

                    model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
                    dtype_preference = os.getenv("UE_SEA_LOCAL_MODEL_DTYPE", "").lower()
                    if TORCH_AVAILABLE:
                        if dtype_preference == "float16" and hasattr(torch, "float16"):
                            model_kwargs["torch_dtype"] = torch.float16
                        elif dtype_preference == "bfloat16" and hasattr(torch, "bfloat16"):
                            model_kwargs["torch_dtype"] = torch.bfloat16
                        device_preference = os.getenv("UE_SEA_LOCAL_MODEL_DEVICE", "").lower()
                        if device_preference == "cuda" and torch.cuda.is_available():
                            model_kwargs["device_map"] = "auto"

                    model = AutoModelForCausalLM.from_pretrained(
                        config.model_path,
                        **model_kwargs
                    )

                    generation_pipeline = None
                    if hf_pipeline:
                        try:
                            generation_pipeline = hf_pipeline(
                                "text-generation",
                                model=model,
                                tokenizer=tokenizer
                            )
                        except Exception as pipeline_error:
                            print(f"Warning: Failed to create pipeline for {config.name}: {pipeline_error}")

                    self.models[config.name] = {
                        "type": "local",
                        "config": config,
                        "tokenizer": tokenizer,
                        "model": model,
                        "pipeline": generation_pipeline,
                        "initialized": True
                    }
                else:
                    print(f"Warning: Could not initialize model {config.name}")
                    self.models[config.name] = {
                        "type": "mock",
                        "config": config,
                        "initialized": False
                    }
            
            self.model_configs[config.name] = config
            
        except Exception as e:
            print(f"Error initializing model {config.name}: {e}")
            self.models[config.name] = {
                "type": "mock",
                "config": config,
                "initialized": False
            }
    
    async def generate_variation(self, prompt: str) -> Optional[str]:
        """Generate a variation using the ensemble."""
        # Select appropriate model for variation
        model_name = self._select_model_for_task("variation")
        
        # Generate using selected model
        result = await self._generate_with_model(model_name, prompt)
        
        if result:
            return result.text
        return None
    
    async def generate_mutation(self, prompt: str) -> Optional[str]:
        """Generate a mutation using the ensemble."""
        model_name = self._select_model_for_task("mutation")
        result = await self._generate_with_model(model_name, prompt)
        
        if result:
            return result.text
        return None
    
    async def generate_crossover(self, prompt: str) -> Optional[str]:
        """Generate a crossover using the ensemble."""
        model_name = self._select_model_for_task("crossover")
        result = await self._generate_with_model(model_name, prompt)
        
        if result:
            return result.text
        return None
    
    async def generate_improvement(self, prompt: str) -> Optional[str]:
        """Generate an improvement using the ensemble."""
        model_name = self._select_model_for_task("improvement")
        result = await self._generate_with_model(model_name, prompt)
        
        if result:
            return result.text
        return None
    
    def _select_model_for_task(self, task: str) -> str:
        """Select appropriate model for a task."""
        available_models = [
            name for name, model in self.models.items()
            if model["initialized"]
        ]
        
        if not available_models:
            return list(self.models.keys())[0]  # Fallback
        
        # Task-specific model selection
        if task in ["variation", "improvement"]:
            # Prefer powerful models for complex tasks
            powerful_models = [
                name for name in available_models
                if self.model_configs[name].model_type == ModelType.POWERFUL
            ]
            if powerful_models:
                return random.choice(powerful_models)
        
        elif task in ["mutation", "crossover"]:
            # Prefer fast models for simple tasks
            fast_models = [
                name for name in available_models
                if self.model_configs[name].model_type == ModelType.FAST
            ]
            if fast_models:
                return random.choice(fast_models)
        
        # Fallback to random selection
        return random.choice(available_models)
    
    async def _generate_with_model(self, model_name: str, prompt: str) -> Optional[GenerationResult]:
        """Generate text using a specific model."""
        if model_name not in self.models:
            return None
        
        model_info = self.models[model_name]
        config = self.model_configs[model_name]
        
        if not model_info["initialized"]:
            return None
        
        start_time = time.time()
        
        try:
            if model_info["type"] == "api":
                result = await self._generate_with_api(model_name, prompt, config)
            elif model_info["type"] == "local":
                result = await self._generate_with_local(model_name, prompt, config, model_info)
            else:
                result = await self._generate_mock(model_name, prompt, config)
            
            generation_time = time.time() - start_time
            
            return GenerationResult(
                text=result,
                model_name=model_name,
                tokens_used=len(prompt.split()) + len(result.split()),  # Rough estimate
                generation_time=generation_time
            )
            
        except Exception as e:
            print(f"Error generating with model {model_name}: {e}")
            return None
    
    async def _generate_with_api(self, model_name: str, prompt: str, config: ModelConfig) -> str:
        """Generate text using API-based models."""
        if not OPENAI_AVAILABLE:
            return await self._generate_mock(model_name, prompt, config)
        
        try:
            # Mock API call - replace with actual implementation
            await asyncio.sleep(0.1)  # Simulate API call
            
            # Simple mock response
            return f"# Generated by {model_name}\n# TODO: Implement actual generation\nprint('Hello from {model_name}')"
            
        except Exception as e:
            print(f"API generation error: {e}")
            return await self._generate_mock(model_name, prompt, config)
    
    async def _generate_with_local(self, model_name: str, prompt: str, 
                                 config: ModelConfig, model_info: Dict[str, Any]) -> str:
        """Generate text using local models."""
        try:
            tokenizer = model_info["tokenizer"]
            model = model_info["model"]
            generation_pipeline = model_info.get("pipeline")
            
            system_prompt = os.getenv(
                "UE_SEA_LOCAL_SYSTEM_PROMPT",
                "You are a senior software engineer generating precise code updates."
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            
            max_new_tokens = min(
                config.max_tokens,
                int(os.getenv("UE_SEA_LOCAL_MODEL_MAX_NEW_TOKENS", "256"))
            )
            
            if hasattr(tokenizer, "apply_chat_template"):
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                )
            else:
                prompt_text = f"{system_prompt}\n\nUser:\n{prompt}\nAssistant:"
            
            if generation_pipeline is not None:
                outputs = generation_pipeline(
                    prompt_text,
                    max_new_tokens=max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    do_sample=True,
                    return_full_text=False
                )
                if outputs:
                    generated = outputs[0].get("generated_text") or outputs[0].get("text", "")
                    return generated.strip()
            
            # Tokenize input
            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch is required for local generation but is not available.")
            
            if hasattr(tokenizer, "apply_chat_template"):
                chat_inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt"
                )
                if isinstance(chat_inputs, dict):
                    inputs = chat_inputs
                else:
                    inputs = {"input_ids": chat_inputs}
            else:
                inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=config.max_tokens
            )
            
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if pad_token_id is None and hasattr(tokenizer, "eos_token_id"):
                pad_token_id = tokenizer.eos_token_id
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    do_sample=True,
                    pad_token_id=pad_token_id
                )
            
            # Decode output
            if hasattr(tokenizer, "apply_chat_template"):
                generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
                result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            else:
                result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return result.strip()
            
        except Exception as e:
            print(f"Local generation error: {e}")
            return await self._generate_mock(model_name, prompt, config)
    
    async def _generate_mock(self, model_name: str, prompt: str, config: ModelConfig) -> str:
        """Generate mock text for testing."""
        # Simple mock generation based on prompt content
        if "improve" in prompt.lower():
            return f"# Improved version by {model_name}\n# Enhanced functionality\nprint('Improved code')"
        elif "mutate" in prompt.lower():
            return f"# Mutated version by {model_name}\n# Modified implementation\nprint('Mutated code')"
        else:
            return f"# Generated by {model_name}\n# New implementation\nprint('Generated code')"
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get statistics about model usage."""
        return {
            "total_models": len(self.models),
            "initialized_models": sum(1 for model in self.models.values() if model["initialized"]),
            "model_types": {
                "api": sum(1 for model in self.models.values() if model["type"] == "api"),
                "local": sum(1 for model in self.models.values() if model["type"] == "local"),
                "mock": sum(1 for model in self.models.values() if model["type"] == "mock"),
            }
        }
    
    def add_model(self, config: ModelConfig) -> None:
        """Add a new model to the ensemble."""
        self._initialize_model(config)
    
    def remove_model(self, model_name: str) -> None:
        """Remove a model from the ensemble."""
        if model_name in self.models:
            del self.models[model_name]
        if model_name in self.model_configs:
            del self.model_configs[model_name]
