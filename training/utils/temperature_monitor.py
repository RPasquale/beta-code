"""
Temperature monitoring for RTX 4090 to prevent overheating.

Monitors GPU temperature and implements cooling strategies.
"""

import subprocess
import time
import logging
from typing import Dict, Any, Optional


class TemperatureMonitor:
    """Monitor GPU temperature to prevent overheating."""
    
    def __init__(self, max_temp: float = 85.0):
        self.max_temp = max_temp
        self.logger = logging.getLogger(__name__)
        self.temp_history = []
        self.overheat_count = 0
        self.max_overheat_retries = 5
        
    def get_gpu_temperature(self) -> Optional[float]:
        """Get current GPU temperature in Celsius."""
        try:
            # Try nvidia-smi first
            result = subprocess.run([
                "nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                temp = float(result.stdout.strip())
                self.temp_history.append(temp)
                
                # Keep only last 50 measurements
                if len(self.temp_history) > 50:
                    self.temp_history.pop(0)
                
                return temp
            
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError) as e:
            self.logger.warning(f"Error getting GPU temperature via nvidia-smi: {e}")
        
        # Fallback: try PyTorch (less reliable)
        try:
            import torch
            if torch.cuda.is_available():
                # This is a fallback - PyTorch doesn't directly expose temperature
                # but we can use it as a proxy for GPU activity
                return None
        except ImportError:
            pass
        
        return None
    
    def is_overheating(self) -> bool:
        """Check if GPU is overheating."""
        temp = self.get_gpu_temperature()
        if temp is None:
            return False
        
        return temp > self.max_temp
    
    def get_temperature_stats(self) -> Dict[str, Any]:
        """Get temperature statistics."""
        current_temp = self.get_gpu_temperature()
        
        stats = {
            "current_temp": current_temp,
            "max_temp": self.max_temp,
            "is_overheating": current_temp > self.max_temp if current_temp else False,
            "overheat_count": self.overheat_count,
            "avg_temp": sum(self.temp_history) / len(self.temp_history) if self.temp_history else None,
            "max_observed": max(self.temp_history) if self.temp_history else None
        }
        
        return stats
    
    def handle_overheating(self) -> bool:
        """Handle overheating situation. Returns True if training should continue."""
        self.overheat_count += 1
        
        if self.overheat_count > self.max_overheat_retries:
            self.logger.error("Too many overheating events. Stopping training for safety.")
            return False
        
        self.logger.warning(f"GPU overheating detected (attempt {self.overheat_count}/{self.max_overheat_retries})")
        
        # Implement cooling strategies
        self._cool_down()
        
        return True
    
    def _cool_down(self):
        """Implement cooling strategies."""
        self.logger.info("Implementing cooling strategies...")
        
        # Strategy 1: Pause training
        self.logger.info("Pausing training for 30 seconds to cool down...")
        time.sleep(30)
        
        # Strategy 2: Force garbage collection
        import gc
        gc.collect()
        
        # Strategy 3: Clear GPU cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass
        
        # Strategy 4: Reduce GPU utilization
        self.logger.info("Reducing GPU utilization...")
        time.sleep(10)
        
        self.logger.info("Cooling strategies completed")
    
    def log_temperature_status(self):
        """Log current temperature status."""
        stats = self.get_temperature_stats()
        
        if stats["current_temp"] is not None:
            self.logger.info(f"GPU Temperature: {stats['current_temp']:.1f}°C (max: {stats['max_temp']:.1f}°C)")
            
            if stats["is_overheating"]:
                self.logger.warning(f"⚠️  GPU OVERHEATING: {stats['current_temp']:.1f}°C > {stats['max_temp']:.1f}°C")
            elif stats["current_temp"] > stats["max_temp"] - 10:
                self.logger.warning(f"⚠️  GPU temperature high: {stats['current_temp']:.1f}°C")
            else:
                self.logger.info(f"✅ GPU temperature normal: {stats['current_temp']:.1f}°C")
        else:
            self.logger.warning("Could not read GPU temperature")
    
    def reset_overheat_count(self):
        """Reset overheat counter (call when temperature is normal)."""
        if self.get_gpu_temperature() and self.get_gpu_temperature() < self.max_temp - 5:
            self.overheat_count = 0
            self.logger.info("GPU temperature normalized, reset overheat counter")
