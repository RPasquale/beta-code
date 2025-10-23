"""
Memory monitoring utilities for safe training on RTX 4090.

Monitors GPU and CPU memory usage to prevent crashes.
"""

import psutil
import torch
import gc
from typing import Dict, Any, Optional
import logging


class MemoryMonitor:
    """Monitor memory usage during training."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.memory_history = []
        self.peak_memory = 0.0
        
    def get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage as fraction (0.0 to 1.0)."""
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            # Get memory info
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            
            # Calculate usage fraction
            usage = allocated / total
            
            # Track peak memory
            if usage > self.peak_memory:
                self.peak_memory = usage
            
            # Store in history
            self.memory_history.append(usage)
            if len(self.memory_history) > 100:  # Keep last 100 measurements
                self.memory_history.pop(0)
            
            return usage
            
        except Exception as e:
            self.logger.warning(f"Error getting GPU memory usage: {e}")
            return 0.0
    
    def get_cpu_memory_usage(self) -> float:
        """Get current CPU memory usage as fraction (0.0 to 1.0)."""
        try:
            memory = psutil.virtual_memory()
            return memory.percent / 100.0
        except Exception as e:
            self.logger.warning(f"Error getting CPU memory usage: {e}")
            return 0.0
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        stats = {
            "gpu_usage": self.get_gpu_memory_usage(),
            "cpu_usage": self.get_cpu_memory_usage(),
            "peak_gpu_usage": self.peak_memory,
            "gpu_allocated": 0.0,
            "gpu_reserved": 0.0,
            "gpu_total": 0.0
        }
        
        if torch.cuda.is_available():
            try:
                stats["gpu_allocated"] = torch.cuda.memory_allocated() / 1024**3  # GB
                stats["gpu_reserved"] = torch.cuda.memory_reserved() / 1024**3   # GB
                stats["gpu_total"] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            except Exception as e:
                self.logger.warning(f"Error getting GPU memory stats: {e}")
        
        return stats
    
    def is_memory_critical(self, threshold: float = 0.9) -> bool:
        """Check if memory usage is critical."""
        gpu_usage = self.get_gpu_memory_usage()
        cpu_usage = self.get_cpu_memory_usage()
        
        return gpu_usage > threshold or cpu_usage > threshold
    
    def cleanup_memory(self):
        """Force memory cleanup."""
        try:
            # Python garbage collection
            gc.collect()
            
            # PyTorch cache cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            self.logger.info("Memory cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"Error during memory cleanup: {e}")
    
    def get_memory_recommendations(self) -> Dict[str, str]:
        """Get memory optimization recommendations."""
        recommendations = {}
        
        gpu_usage = self.get_gpu_memory_usage()
        cpu_usage = self.get_cpu_memory_usage()
        
        if gpu_usage > 0.8:
            recommendations["gpu"] = "High GPU memory usage. Consider reducing batch size or sequence length."
        
        if cpu_usage > 0.8:
            recommendations["cpu"] = "High CPU memory usage. Consider reducing data loader workers."
        
        if gpu_usage > 0.9:
            recommendations["critical"] = "Critical GPU memory usage. Training may crash soon."
        
        return recommendations
    
    def log_memory_status(self):
        """Log current memory status."""
        stats = self.get_memory_stats()
        
        self.logger.info(f"Memory Status:")
        self.logger.info(f"  GPU: {stats['gpu_usage']:.1%} ({stats['gpu_allocated']:.1f}GB / {stats['gpu_total']:.1f}GB)")
        self.logger.info(f"  CPU: {stats['cpu_usage']:.1%}")
        self.logger.info(f"  Peak GPU: {stats['peak_gpu_usage']:.1%}")
        
        # Log recommendations
        recommendations = self.get_memory_recommendations()
        if recommendations:
            for key, rec in recommendations.items():
                self.logger.warning(f"  {key.upper()}: {rec}")
