#!/usr/bin/env python3
"""
Dr.GRPO Pipeline Monitor

Real-time monitoring of the Dr.GRPO training pipeline with:
- Progress tracking
- Metrics visualization
- Performance monitoring
- Alert system
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np


class DrGRPOMonitor:
    """Monitor for Dr.GRPO training pipeline."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.metrics_history = {
            "rewards": [],
            "advantages": [],
            "reasoning_quality": [],
            "evolution_fitness": [],
            "timestamps": []
        }
        self.start_time = time.time()
        
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update metrics from orchestrator."""
        current_time = time.time() - self.start_time
        
        if "metrics" in metrics:
            m = metrics["metrics"]
            self.metrics_history["rewards"].extend(m.get("rewards", []))
            self.metrics_history["advantages"].extend(m.get("advantages", []))
            self.metrics_history["reasoning_quality"].extend(m.get("reasoning_quality", []))
            self.metrics_history["evolution_fitness"].extend(m.get("evolution_fitness", []))
            self.metrics_history["timestamps"].extend([current_time] * len(m.get("rewards", [])))
    
    def print_status(self, metrics: Dict[str, Any]) -> None:
        """Print current training status."""
        print(f"\n{'='*60}")
        print(f"📊 Dr.GRPO Training Status - {time.strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        
        print(f"🔄 Training Step: {metrics.get('training_step', 0)}")
        print(f"🧠 Reasoning Traces: {metrics.get('reasoning_traces', 0)}")
        print(f"👥 Group Responses: {metrics.get('group_responses', 0)}")
        print(f"📈 Curriculum Stage: {metrics.get('curriculum_stage', 0)}")
        
        if "metrics" in metrics:
            m = metrics["metrics"]
            if m.get("rewards"):
                avg_reward = np.mean(m["rewards"])
                print(f"🎯 Average Reward: {avg_reward:.3f}")
            
            if m.get("advantages"):
                avg_advantage = np.mean(m["advantages"])
                print(f"📊 Average Advantage: {avg_advantage:.3f}")
            
            if m.get("reasoning_quality"):
                avg_quality = np.mean(m["reasoning_quality"])
                print(f"🧠 Average Reasoning Quality: {avg_quality:.3f}")
            
            if m.get("evolution_fitness"):
                avg_fitness = np.mean(m["evolution_fitness"])
                print(f"🧬 Average Evolution Fitness: {avg_fitness:.3f}")
        
        print(f"{'='*60}")
    
    def save_plots(self) -> None:
        """Save training progress plots."""
        if not any(self.metrics_history["rewards"]):
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Dr.GRPO Training Progress", fontsize=16)
        
        # Rewards plot
        if self.metrics_history["rewards"]:
            axes[0, 0].plot(self.metrics_history["timestamps"], self.metrics_history["rewards"], 'b-', alpha=0.7)
            axes[0, 0].set_title("Rewards Over Time")
            axes[0, 0].set_xlabel("Time (seconds)")
            axes[0, 0].set_ylabel("Reward")
            axes[0, 0].grid(True)
        
        # Advantages plot
        if self.metrics_history["advantages"]:
            axes[0, 1].plot(self.metrics_history["timestamps"], self.metrics_history["advantages"], 'g-', alpha=0.7)
            axes[0, 1].set_title("Advantages Over Time")
            axes[0, 1].set_xlabel("Time (seconds)")
            axes[0, 1].set_ylabel("Advantage")
            axes[0, 1].grid(True)
        
        # Reasoning quality plot
        if self.metrics_history["reasoning_quality"]:
            axes[1, 0].plot(self.metrics_history["timestamps"], self.metrics_history["reasoning_quality"], 'r-', alpha=0.7)
            axes[1, 0].set_title("Reasoning Quality Over Time")
            axes[1, 0].set_xlabel("Time (seconds)")
            axes[1, 0].set_ylabel("Quality Score")
            axes[1, 0].grid(True)
        
        # Evolution fitness plot
        if self.metrics_history["evolution_fitness"]:
            axes[1, 1].plot(self.metrics_history["timestamps"], self.metrics_history["evolution_fitness"], 'm-', alpha=0.7)
            axes[1, 1].set_title("Evolution Fitness Over Time")
            axes[1, 1].set_xlabel("Time (seconds)")
            axes[1, 1].set_ylabel("Fitness Score")
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = self.output_dir / "training_progress.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Training plots saved to {plot_path}")
    
    def check_alerts(self, metrics: Dict[str, Any]) -> None:
        """Check for training alerts."""
        alerts = []
        
        if "metrics" in metrics:
            m = metrics["metrics"]
            
            # Low reward alert
            if m.get("rewards") and np.mean(m["rewards"]) < 0.3:
                alerts.append("⚠️ Low average reward detected")
            
            # High advantage variance alert
            if m.get("advantages") and len(m["advantages"]) > 10:
                advantage_std = np.std(m["advantages"])
                if advantage_std > 0.5:
                    alerts.append("⚠️ High advantage variance detected")
            
            # Low reasoning quality alert
            if m.get("reasoning_quality") and np.mean(m["reasoning_quality"]) < 0.4:
                alerts.append("⚠️ Low reasoning quality detected")
        
        if alerts:
            print("\n🚨 Training Alerts:")
            for alert in alerts:
                print(f"   {alert}")


def monitor_pipeline(output_dir: str, refresh_interval: int = 30) -> None:
    """Monitor the Dr.GRPO pipeline."""
    monitor = DrGRPOMonitor(output_dir)
    
    print(f"🔍 Starting Dr.GRPO pipeline monitoring...")
    print(f"📁 Monitoring directory: {output_dir}")
    print(f"⏱️ Refresh interval: {refresh_interval} seconds")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            # Check for metrics file
            metrics_file = Path(output_dir) / "training_metrics.json"
            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    
                    monitor.update_metrics(metrics)
                    monitor.print_status(metrics)
                    monitor.check_alerts(metrics)
                    
                except Exception as e:
                    print(f"❌ Error reading metrics: {e}")
            
            # Check for results file
            results_file = Path(output_dir) / "drgrpo_results.json"
            if results_file.exists():
                print("✅ Pipeline completed! Final results available.")
                monitor.save_plots()
                break
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
        monitor.save_plots()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Monitor Dr.GRPO Pipeline")
    parser.add_argument("--output", type=str, default="drgrpo_output", help="Output directory to monitor")
    parser.add_argument("--refresh", type=int, default=30, help="Refresh interval in seconds")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output):
        print(f"❌ Output directory {args.output} does not exist")
        return
    
    monitor_pipeline(args.output, args.refresh)


if __name__ == "__main__":
    main()
