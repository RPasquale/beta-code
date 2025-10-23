#!/usr/bin/env python3
"""
Safe training launcher for RTX 4090.

Monitors system resources and ensures stable training.
"""

import os
import sys
import time
import subprocess
import psutil
import logging
from pathlib import Path
from typing import List, Dict, Any
import argparse


class SafeTrainingLauncher:
    """Safe launcher for training on RTX 4090."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # System requirements
        self.min_gpu_memory = 20  # GB
        self.min_cpu_memory = 16  # GB
        self.max_gpu_temp = 85   # Celsius
        self.max_cpu_temp = 80   # Celsius
        
        # Training state
        self.training_process = None
        self.monitoring = False
    
    def setup_logging(self):
        """Setup logging for the launcher."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training_launcher.log'),
                logging.StreamHandler()
            ]
        )
    
    def check_system_requirements(self) -> bool:
        """Check if system meets requirements for safe training."""
        self.logger.info("Checking system requirements...")
        
        # Check GPU
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                gpu_memory = int(result.stdout.strip()) / 1024  # Convert MB to GB
                self.logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
                
                if gpu_memory < self.min_gpu_memory:
                    self.logger.error(f"GPU memory too low: {gpu_memory:.1f} GB (need {self.min_gpu_memory} GB)")
                    return False
            else:
                self.logger.error("Could not check GPU memory")
                return False
        except Exception as e:
            self.logger.error(f"Error checking GPU: {e}")
            return False
        
        # Check CPU memory
        cpu_memory = psutil.virtual_memory().total / 1024**3
        self.logger.info(f"CPU Memory: {cpu_memory:.1f} GB")
        
        if cpu_memory < self.min_cpu_memory:
            self.logger.warning(f"CPU memory low: {cpu_memory:.1f} GB (recommend {self.min_cpu_memory} GB)")
        
        # Check disk space
        disk_usage = psutil.disk_usage('/').free / 1024**3
        self.logger.info(f"Disk space: {disk_usage:.1f} GB")
        
        if disk_usage < 50:
            self.logger.warning(f"Disk space low: {disk_usage:.1f} GB (recommend 50+ GB)")
        
        return True
    
    def check_temperature(self) -> bool:
        """Check if system temperature is safe for training."""
        try:
            # Check GPU temperature
            result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                gpu_temp = int(result.stdout.strip())
                self.logger.info(f"GPU Temperature: {gpu_temp}°C")
                
                if gpu_temp > self.max_gpu_temp:
                    self.logger.error(f"GPU temperature too high: {gpu_temp}°C (max {self.max_gpu_temp}°C)")
                    return False
            else:
                self.logger.warning("Could not check GPU temperature")
        
        except Exception as e:
            self.logger.warning(f"Error checking temperature: {e}")
        
        return True
    
    def monitor_training(self):
        """Monitor training process for safety."""
        self.monitoring = True
        self.logger.info("Starting training monitoring...")
        
        while self.monitoring and self.training_process and self.training_process.poll() is None:
            try:
                # Check GPU memory
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,temperature.gpu', 
                                       '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    gpu_used, gpu_total, gpu_temp = map(int, result.stdout.strip().split(', '))
                    gpu_usage = gpu_used / gpu_total
                    
                    self.logger.info(f"GPU: {gpu_usage:.1%} ({gpu_used}MB/{gpu_total}MB), Temp: {gpu_temp}°C")
                    
                    # Check for high memory usage
                    if gpu_usage > 0.9:
                        self.logger.warning("High GPU memory usage detected!")
                    
                    # Check for overheating
                    if gpu_temp > self.max_gpu_temp:
                        self.logger.error(f"GPU overheating: {gpu_temp}°C")
                        self.stop_training()
                        break
                
                # Check CPU memory
                cpu_memory = psutil.virtual_memory()
                self.logger.info(f"CPU Memory: {cpu_memory.percent:.1f}%")
                
                if cpu_memory.percent > 90:
                    self.logger.warning("High CPU memory usage detected!")
                
                # Check if training process is still running
                if self.training_process.poll() is not None:
                    self.logger.info("Training process completed")
                    break
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error during monitoring: {e}")
                time.sleep(60)  # Wait longer on error
    
    def stop_training(self):
        """Stop training process safely."""
        if self.training_process and self.training_process.poll() is None:
            self.logger.info("Stopping training process...")
            self.training_process.terminate()
            
            # Wait for graceful shutdown
            try:
                self.training_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.logger.warning("Training process did not stop gracefully, forcing...")
                self.training_process.kill()
        
        self.monitoring = False
    
    def launch_training(self, command: List[str], working_dir: str = None) -> bool:
        """Launch training with safety monitoring."""
        if not self.check_system_requirements():
            return False
        
        if not self.check_temperature():
            self.logger.error("System temperature too high for training")
            return False
        
        self.logger.info(f"Launching training: {' '.join(command)}")
        
        try:
            # Start training process
            self.training_process = subprocess.Popen(
                command,
                cwd=working_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Start monitoring in background
            import threading
            monitor_thread = threading.Thread(target=self.monitor_training)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            # Stream output
            for line in iter(self.training_process.stdout.readline, ''):
                print(line, end='')
                
                # Check for error patterns
                if "out of memory" in line.lower() or "cuda error" in line.lower():
                    self.logger.error("Training error detected!")
                    self.stop_training()
                    return False
            
            # Wait for completion
            return_code = self.training_process.wait()
            
            if return_code == 0:
                self.logger.info("Training completed successfully!")
                return True
            else:
                self.logger.error(f"Training failed with return code: {return_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error launching training: {e}")
            return False
        
        finally:
            self.monitoring = False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Safe training launcher for RTX 4090")
    parser.add_argument("--command", nargs="+", required=True, help="Training command to run")
    parser.add_argument("--working_dir", type=str, help="Working directory for training")
    parser.add_argument("--check_only", action="store_true", help="Only check system requirements")
    
    args = parser.parse_args()
    
    launcher = SafeTrainingLauncher()
    
    if args.check_only:
        success = launcher.check_system_requirements() and launcher.check_temperature()
        if success:
            print("✅ System requirements met for safe training")
            sys.exit(0)
        else:
            print("❌ System requirements not met")
            sys.exit(1)
    
    success = launcher.launch_training(args.command, args.working_dir)
    
    if success:
        print("✅ Training completed successfully!")
        sys.exit(0)
    else:
        print("❌ Training failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
