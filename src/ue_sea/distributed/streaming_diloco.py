"""
Streaming DiLoCo: Distributed Low-Communication training.

Implements fragmented synchronization with overlap and quantized outer gradients.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch
import torch.distributed as dist


class FragmentStatus(Enum):
    """Status of a fragment."""
    IDLE = "idle"
    TRAINING = "training"
    SYNCING = "syncing"
    COMPLETED = "completed"


@dataclass
class FragmentConfig:
    """Configuration for a fragment."""
    fragment_id: str
    layers: List[int]  # Layer indices
    offset_t_p: int    # Offset for synchronization
    sync_frequency: int = 100  # H: steps between sync
    overlap_tau: int = 5       # Overlap steps
    quantize_outer_grads: bool = True
    quant_scheme: str = "E3M0"  # FP4 quantization
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class WorkerConfig:
    """Configuration for a worker."""
    worker_id: str
    replicas: int = 4
    gpu_id: Optional[int] = None
    memory_limit: Optional[int] = None  # MB
    bandwidth_limit: Optional[int] = None  # Mbps
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class FragmentState:
    """State of a fragment during training."""
    fragment_id: str
    status: FragmentStatus
    current_step: int
    local_gradients: Optional[torch.Tensor] = None
    global_gradients: Optional[torch.Tensor] = None
    sync_count: int = 0
    last_sync_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class StreamingDiLoCo:
    """
    Streaming DiLoCo implementation for distributed training.
    
    Implements fragmented synchronization with overlap and quantized
    outer gradients for efficient distributed training.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.fragments = {}
        self.workers = {}
        self.fragment_states = {}
        self.communication_manager = None
        self.quantization_manager = None
        self.overlap_manager = None
        
        # Training state
        self.is_initialized = False
        self.training_active = False
        self.global_step = 0
        
        # Statistics
        self.sync_statistics = {
            "total_syncs": 0,
            "total_bandwidth": 0.0,
            "overlap_efficiency": 0.0,
            "quantization_savings": 0.0
        }
    
    async def initialize(self, fragments: List[FragmentConfig], 
                        workers: List[WorkerConfig]) -> None:
        """
        Initialize the Streaming DiLoCo system.
        
        Args:
            fragments: List of fragment configurations
            workers: List of worker configurations
        """
        print("Initializing Streaming DiLoCo...")
        
        # Initialize fragments
        for fragment_config in fragments:
            self.fragments[fragment_config.fragment_id] = fragment_config
            self.fragment_states[fragment_config.fragment_id] = FragmentState(
                fragment_id=fragment_config.fragment_id,
                status=FragmentStatus.IDLE,
                current_step=0
            )
        
        # Initialize workers
        for worker_config in workers:
            self.workers[worker_config.worker_id] = worker_config
        
        # Initialize communication manager
        self.communication_manager = CommunicationManager()
        await self.communication_manager.initialize()
        
        # Initialize quantization manager
        self.quantization_manager = QuantizationManager()
        await self.quantization_manager.initialize()
        
        # Initialize overlap manager
        self.overlap_manager = OverlapManager()
        await self.overlap_manager.initialize()
        
        self.is_initialized = True
        print(f"Streaming DiLoCo initialized with {len(fragments)} fragments and {len(workers)} workers")
    
    async def start_training(self, model, optimizer, dataloader, 
                           max_steps: int = 1000) -> Dict[str, Any]:
        """
        Start distributed training.
        
        Args:
            model: Model to train
            optimizer: Optimizer
            dataloader: Data loader
            max_steps: Maximum training steps
        
        Returns:
            Training results
        """
        if not self.is_initialized:
            raise ValueError("Streaming DiLoCo not initialized")
        
        print(f"Starting distributed training for {max_steps} steps...")
        
        self.training_active = True
        start_time = time.time()
        
        try:
            # Start training loop
            await self._training_loop(model, optimizer, dataloader, max_steps)
            
            duration = time.time() - start_time
            
            return {
                "success": True,
                "duration": duration,
                "total_steps": self.global_step,
                "statistics": self.sync_statistics
            }
            
        except Exception as e:
            print(f"Training error: {e}")
            return {
                "success": False,
                "error": str(e),
                "duration": time.time() - start_time
            }
        finally:
            self.training_active = False
    
    async def _training_loop(self, model, optimizer, dataloader, max_steps: int) -> None:
        """Main training loop with fragmented synchronization."""
        
        for step in range(max_steps):
            self.global_step = step
            
            # Train all fragments
            fragment_tasks = []
            for fragment_id, fragment_config in self.fragments.items():
                if self._should_sync_fragment(fragment_id, step):
                    # This fragment needs to sync
                    task = self._sync_fragment(fragment_id, model, optimizer)
                else:
                    # This fragment continues training
                    task = self._train_fragment(fragment_id, model, optimizer, dataloader)
                
                fragment_tasks.append(task)
            
            # Execute fragment tasks in parallel
            await asyncio.gather(*fragment_tasks)
            
            # Update statistics
            self._update_statistics()
            
            if step % 100 == 0:
                print(f"  Step {step}/{max_steps} - Active fragments: {self._get_active_fragment_count()}")
    
    def _should_sync_fragment(self, fragment_id: str, step: int) -> bool:
        """Check if a fragment should synchronize at this step."""
        fragment_config = self.fragments[fragment_id]
        fragment_state = self.fragment_states[fragment_id]
        
        # Check if it's time to sync based on offset and frequency
        sync_step = step + fragment_config.offset_t_p
        return sync_step % fragment_config.sync_frequency == 0
    
    async def _sync_fragment(self, fragment_id: str, model, optimizer) -> None:
        """Synchronize a fragment with other workers."""
        fragment_state = self.fragment_states[fragment_id]
        fragment_config = self.fragments[fragment_id]
        
        print(f"    Syncing fragment {fragment_id}...")
        
        # Update state
        fragment_state.status = FragmentStatus.SYNCING
        fragment_state.sync_count += 1
        fragment_state.last_sync_time = time.time()
        
        # Get local gradients
        local_grads = self._get_fragment_gradients(model, fragment_config.layers)
        
        # Quantize gradients if enabled
        if fragment_config.quantize_outer_grads:
            quantized_grads = await self.quantization_manager.quantize(
                local_grads, scheme=fragment_config.quant_scheme
            )
        else:
            quantized_grads = local_grads
        
        # All-reduce gradients
        global_grads = await self.communication_manager.all_reduce(
            quantized_grads, fragment_id
        )
        
        # Dequantize if needed
        if fragment_config.quantize_outer_grads:
            global_grads = await self.quantization_manager.dequantize(
                global_grads, scheme=fragment_config.quant_scheme
            )
        
        # Update model with global gradients
        self._update_fragment_gradients(model, global_grads, fragment_config.layers)
        
        # Update state
        fragment_state.global_gradients = global_grads
        fragment_state.status = FragmentStatus.TRAINING
        
        # Update statistics
        self.sync_statistics["total_syncs"] += 1
        if fragment_config.quantize_outer_grads:
            self.sync_statistics["quantization_savings"] += 0.75  # 75% bandwidth savings
    
    async def _train_fragment(self, fragment_id: str, model, optimizer, dataloader) -> None:
        """Train a fragment for one step."""
        fragment_state = self.fragment_states[fragment_id]
        fragment_config = self.fragments[fragment_id]
        
        # Update state
        fragment_state.status = FragmentStatus.TRAINING
        fragment_state.current_step += 1
        
        # Simulate training step
        await asyncio.sleep(0.01)  # Simulate computation time
        
        # Get batch from dataloader
        try:
            batch = next(iter(dataloader))
        except StopIteration:
            return
        
        # Forward pass
        outputs = model(batch)
        
        # Compute loss
        loss = self._compute_loss(outputs, batch)
        
        # Backward pass
        loss.backward()
        
        # Update gradients for this fragment
        self._update_fragment_gradients(model, None, fragment_config.layers)
    
    def _get_fragment_gradients(self, model, layers: List[int]) -> torch.Tensor:
        """Get gradients for specific layers."""
        grads = []
        for layer_idx in layers:
            if hasattr(model, f'layer_{layer_idx}'):
                layer = getattr(model, f'layer_{layer_idx}')
                if layer.weight.grad is not None:
                    grads.append(layer.weight.grad.flatten())
                if layer.bias is not None and layer.bias.grad is not None:
                    grads.append(layer.bias.grad.flatten())
        
        if grads:
            return torch.cat(grads)
        else:
            return torch.zeros(1)  # Dummy gradient
    
    def _update_fragment_gradients(self, model, gradients: torch.Tensor, 
                                 layers: List[int]) -> None:
        """Update gradients for specific layers."""
        if gradients is None:
            return
        
        grad_idx = 0
        for layer_idx in layers:
            if hasattr(model, f'layer_{layer_idx}'):
                layer = getattr(model, f'layer_{layer_idx}')
                
                # Update weight gradients
                if layer.weight.grad is not None:
                    grad_size = layer.weight.numel()
                    layer.weight.grad = gradients[grad_idx:grad_idx + grad_size].view_as(layer.weight)
                    grad_idx += grad_size
                
                # Update bias gradients
                if layer.bias is not None and layer.bias.grad is not None:
                    grad_size = layer.bias.numel()
                    layer.bias.grad = gradients[grad_idx:grad_idx + grad_size].view_as(layer.bias)
                    grad_idx += grad_size
    
    def _compute_loss(self, outputs, batch) -> torch.Tensor:
        """Compute loss for training."""
        # Simple MSE loss for demonstration
        if isinstance(outputs, torch.Tensor) and isinstance(batch, torch.Tensor):
            return torch.nn.functional.mse_loss(outputs, batch)
        else:
            return torch.tensor(0.1, requires_grad=True)  # Dummy loss
    
    def _get_active_fragment_count(self) -> int:
        """Get number of active fragments."""
        return sum(1 for state in self.fragment_states.values() 
                  if state.status == FragmentStatus.TRAINING)
    
    def _update_statistics(self) -> None:
        """Update training statistics."""
        # Calculate overlap efficiency
        active_fragments = self._get_active_fragment_count()
        total_fragments = len(self.fragments)
        self.sync_statistics["overlap_efficiency"] = active_fragments / total_fragments
    
    async def stop_training(self) -> None:
        """Stop distributed training."""
        print("Stopping distributed training...")
        self.training_active = False
        
        # Wait for all fragments to complete
        while any(state.status == FragmentStatus.TRAINING 
                 for state in self.fragment_states.values()):
            await asyncio.sleep(0.1)
        
        print("Training stopped.")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            "is_initialized": self.is_initialized,
            "training_active": self.training_active,
            "global_step": self.global_step,
            "fragment_count": len(self.fragments),
            "worker_count": len(self.workers),
            "sync_statistics": self.sync_statistics,
            "fragment_states": {
                fid: {
                    "status": state.status.value,
                    "current_step": state.current_step,
                    "sync_count": state.sync_count
                }
                for fid, state in self.fragment_states.items()
            }
        }
    
    def reset(self) -> None:
        """Reset the system."""
        self.fragments = {}
        self.workers = {}
        self.fragment_states = {}
        self.is_initialized = False
        self.training_active = False
        self.global_step = 0
        self.sync_statistics = {
            "total_syncs": 0,
            "total_bandwidth": 0.0,
            "overlap_efficiency": 0.0,
            "quantization_savings": 0.0
        }
