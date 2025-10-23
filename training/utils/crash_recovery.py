"""
Crash recovery utilities for training stability.

Saves training state and enables recovery from crashes.
"""

import json
import pickle
import torch
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime


class CrashRecovery:
    """Handle crash recovery for training."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.recovery_dir = self.output_dir / "recovery"
        self.recovery_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Recovery files
        self.state_file = self.recovery_dir / "training_state.json"
        self.checkpoint_file = self.recovery_dir / "last_checkpoint.pkl"
        self.crash_file = self.recovery_dir / "crash_info.json"
    
    def save_training_state(self, state: Dict[str, Any]):
        """Save current training state."""
        try:
            # Convert tensors to serializable format
            serializable_state = self._make_serializable(state)
            
            # Save state
            with open(self.state_file, 'w') as f:
                json.dump(serializable_state, f, indent=2)
            
            self.logger.info("Training state saved for recovery")
            
        except Exception as e:
            self.logger.warning(f"Error saving training state: {e}")
    
    def load_training_state(self) -> Optional[Dict[str, Any]]:
        """Load training state from recovery file."""
        if not self.state_file.exists():
            return None
        
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            self.logger.info("Training state loaded from recovery")
            return state
            
        except Exception as e:
            self.logger.warning(f"Error loading training state: {e}")
            return None
    
    def save_checkpoint(self, checkpoint: Dict[str, Any]):
        """Save model checkpoint."""
        try:
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            self.logger.info("Checkpoint saved for recovery")
            
        except Exception as e:
            self.logger.warning(f"Error saving checkpoint: {e}")
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load model checkpoint."""
        if not self.checkpoint_file.exists():
            return None
        
        try:
            with open(self.checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            
            self.logger.info("Checkpoint loaded from recovery")
            return checkpoint
            
        except Exception as e:
            self.logger.warning(f"Error loading checkpoint: {e}")
            return None
    
    def save_crash_info(self, error_message: str, additional_info: Dict[str, Any] = None):
        """Save crash information."""
        crash_info = {
            "timestamp": datetime.now().isoformat(),
            "error_message": error_message,
            "additional_info": additional_info or {},
            "recovery_files": {
                "state_file": str(self.state_file),
                "checkpoint_file": str(self.checkpoint_file)
            }
        }
        
        try:
            with open(self.crash_file, 'w') as f:
                json.dump(crash_info, f, indent=2)
            
            self.logger.error(f"Crash information saved: {error_message}")
            
        except Exception as e:
            self.logger.warning(f"Error saving crash info: {e}")
    
    def load_crash_info(self) -> Optional[Dict[str, Any]]:
        """Load crash information."""
        if not self.crash_file.exists():
            return None
        
        try:
            with open(self.crash_file, 'r') as f:
                crash_info = json.load(f)
            
            return crash_info
            
        except Exception as e:
            self.logger.warning(f"Error loading crash info: {e}")
            return None
    
    def can_recover(self) -> bool:
        """Check if recovery is possible."""
        return self.state_file.exists() and self.checkpoint_file.exists()
    
    def get_recovery_info(self) -> Dict[str, Any]:
        """Get recovery information."""
        return {
            "can_recover": self.can_recover(),
            "state_file_exists": self.state_file.exists(),
            "checkpoint_file_exists": self.checkpoint_file.exists(),
            "crash_file_exists": self.crash_file.exists(),
            "recovery_dir": str(self.recovery_dir)
        }
    
    def cleanup_recovery_files(self):
        """Clean up recovery files after successful training."""
        try:
            if self.state_file.exists():
                self.state_file.unlink()
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
            if self.crash_file.exists():
                self.crash_file.unlink()
            
            self.logger.info("Recovery files cleaned up")
            
        except Exception as e:
            self.logger.warning(f"Error cleaning up recovery files: {e}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return {
                "_tensor": True,
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "data": obj.detach().cpu().tolist() if obj.numel() < 1000 else "large_tensor"
            }
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj
