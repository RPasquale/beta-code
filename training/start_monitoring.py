#!/usr/bin/env python3
"""
Start monitoring tools for UE-SEA training.

Provides multiple monitoring options: console, web, and simple.
"""

import os
import sys
import argparse
import subprocess
import threading
import time
from pathlib import Path

def start_console_monitor(output_dir: str = None, interval: int = 2):
    """Start console-based monitoring."""
    print("Starting console monitor...")
    cmd = [sys.executable, "monitor/simple_monitor.py"]
    if output_dir:
        cmd.extend(["--output_dir", output_dir])
    cmd.extend(["--interval", str(interval)])
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nStopping console monitor...")

def start_web_monitor(output_dir: str = None, host: str = "localhost", port: int = 8000):
    """Start web-based monitoring."""
    print("Starting web monitor...")
    cmd = [sys.executable, "monitor/web_dashboard.py"]
    if output_dir:
        cmd.extend(["--output_dir", output_dir])
    cmd.extend(["--host", host, "--port", str(port)])
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nStopping web monitor...")

def start_training_with_monitoring():
    """Start training with monitoring."""
    print("Starting training with monitoring...")
    
    # Start training in background
    training_cmd = [sys.executable, "train_working.py"]
    training_process = subprocess.Popen(training_cmd)
    
    # Start monitoring
    monitor_cmd = [sys.executable, "monitor/simple_monitor.py", "--output_dir", "outputs/working_training"]
    monitor_process = subprocess.Popen(monitor_cmd)
    
    try:
        # Wait for training to complete
        training_process.wait()
        print("Training completed!")
    except KeyboardInterrupt:
        print("\nStopping training and monitoring...")
        training_process.terminate()
        monitor_process.terminate()
    finally:
        monitor_process.terminate()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="UE-SEA Training Monitor")
    parser.add_argument("--mode", choices=["console", "web", "training"], default="console",
                       help="Monitoring mode")
    parser.add_argument("--output_dir", type=str, help="Training output directory to monitor")
    parser.add_argument("--interval", type=int, default=2, help="Update interval in seconds")
    parser.add_argument("--host", type=str, default="localhost", help="Host for web monitor")
    parser.add_argument("--port", type=int, default=8000, help="Port for web monitor")
    
    args = parser.parse_args()
    
    if args.mode == "console":
        start_console_monitor(args.output_dir, args.interval)
    elif args.mode == "web":
        start_web_monitor(args.output_dir, args.host, args.port)
    elif args.mode == "training":
        start_training_with_monitoring()

if __name__ == "__main__":
    main()
