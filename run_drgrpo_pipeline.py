#!/usr/bin/env python3
"""
Dr.GRPO Full Pipeline Runner

This script runs the complete end-to-end Dr.GRPO RL training pipeline:
1. SFT with reasoning traces
2. Reward model training
3. Dr.GRPO RL training
4. AlphaCode evolution
5. End-to-end learning loop

Usage:
    python run_drgrpo_pipeline.py [--config CONFIG_FILE] [--data DATA_PATH] [--output OUTPUT_DIR]
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ue_sea.rl.drgrpo_orchestrator import DrGRPOOrchestrator, DrGRPOConfig


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('drgrpo_pipeline.log', encoding='utf-8')
        ]
    )


def load_config(config_path: Optional[str] = None) -> DrGRPOConfig:
    """Load configuration from file or use defaults."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return DrGRPOConfig(**config_dict)
    else:
        # Use optimized defaults for full pipeline
        return DrGRPOConfig(
            # Model configuration
            model_name="Qwen/Qwen2.5-Coder-3B",
            max_length=128000,
            temperature=1.0,
            top_p=0.9,
            
            # Dr.GRPO specific parameters
            group_size=8,
            ppo_epsilon=0.2,
            entropy_coef=0.01,
            kl_coef=0.0,
            max_grad_norm=1.0,
            
            # Training configuration
            learning_rate=1e-6,
            batch_size=4,
            num_epochs=3,
            warmup_steps=100,
            
            # Curriculum configuration
            difficulty_stages=["moderate", "extreme"],
            curriculum_threshold=0.8,
            
            # Reward configuration
            use_reasoning_traces=True,
            reward_model_lr=1e-5,
            reward_model_epochs=2,
            
            # AlphaCode evolution
            evolution_budget=100,
            evolution_temperature=1.2,
            
            # LOCAGENT integration
            use_locagent=True,
            locagent_confidence_threshold=0.7,
            
            # Evaluation
            eval_frequency=100,
            save_frequency=500
        )


def find_training_data(data_path: Optional[str] = None) -> str:
    """Find the best training data file."""
    if data_path and os.path.exists(data_path):
        return data_path
    
    # Look for training data in order of preference
    data_files = [
        "training/data/processed/full_dataset.jsonl",
        "training/data/processed/rl_tasks.jsonl", 
        "training/data/reasoning/reasoning_data.jsonl",
        "training/data/processed/code_contests.jsonl"
    ]
    
    for data_file in data_files:
        if os.path.exists(data_file):
            return data_file
    
    raise FileNotFoundError("No training data found. Please specify --data or ensure training data exists.")


async def run_pipeline(config: DrGRPOConfig, data_path: str, output_dir: str) -> Dict[str, Any]:
    """Run the complete Dr.GRPO pipeline."""
    logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize orchestrator
    logger.info("🚀 Initializing Dr.GRPO RL Orchestrator...")
    orchestrator = DrGRPOOrchestrator(config)
    
    # Initialize with training data
    await orchestrator.initialize(data_path)
    
    # Run end-to-end training
    logger.info("🔄 Starting end-to-end Dr.GRPO training...")
    start_time = time.time()
    
    try:
        results = await orchestrator.run_end_to_end_training()
        
        # Save results
        results_file = os.path.join(output_dir, "drgrpo_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save metrics
        metrics_file = os.path.join(output_dir, "training_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(orchestrator.get_metrics(), f, indent=2, default=str)
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"✅ Pipeline completed successfully in {duration:.2f} seconds")
        logger.info(f"📊 Results saved to {output_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise


def print_results_summary(results: Dict[str, Any]) -> None:
    """Print a summary of the training results."""
    print("\n" + "="*60)
    print("🎯 Dr.GRPO Pipeline Results Summary")
    print("="*60)
    
    # SFT Results
    if "sft_results" in results:
        sft = results["sft_results"]
        print(f"📚 SFT Stage:")
        print(f"   • Reasoning traces generated: {sft.get('traces_generated', 0)}")
        print(f"   • Average reasoning quality: {sft.get('reasoning_quality', 0):.3f}")
        if "sft_metrics" in sft:
            print(f"   • Training loss: {sft['sft_metrics'].get('training_loss', 0):.4f}")
    
    # Reward Model Results
    if "reward_model_results" in results:
        rm = results["reward_model_results"]
        print(f"🎯 Reward Model Stage:")
        print(f"   • Training examples: {rm.get('training_examples', 0)}")
        print(f"   • Reward model loss: {rm.get('reward_model_loss', 0):.4f}")
        print(f"   • Traces used: {rm.get('traces_used', 0)}")
    
    # Dr.GRPO Results
    if "drgrpo_results" in results:
        drgrpo = results["drgrpo_results"]
        print(f"🚀 Dr.GRPO RL Stage:")
        print(f"   • Average reward: {drgrpo.get('avg_reward', 0):.3f}")
        print(f"   • Average advantage: {drgrpo.get('avg_advantage', 0):.3f}")
        print(f"   • Group responses: {drgrpo.get('group_responses', 0)}")
        print(f"   • Total rewards: {drgrpo.get('total_rewards', 0)}")
        print(f"   • ES triggered count: {drgrpo.get('es_triggered_count', 0)}")
        print(f"   • Average ES improvement: {drgrpo.get('avg_es_improvement', 0):.3f}")
        print(f"   • RL stagnation count: {drgrpo.get('rl_stagnation_count', 0)}")
    
    # AlphaCode Evolution Results
    if "alphacode_results" in results:
        alphacode = results["alphacode_results"]
        print(f"🧬 AlphaCode Evolution Stage:")
        print(f"   • Evolution results: {alphacode.get('evolution_results', 0)}")
        print(f"   • Average fitness: {alphacode.get('avg_fitness', 0):.3f}")
        print(f"   • Average improvement: {alphacode.get('avg_improvement', 0):.3f}")
        print(f"   • Successful evolutions: {alphacode.get('successful_evolutions', 0)}")
        print(f"   • Mutations applied: {alphacode.get('mutations_applied', 0)}")
        print(f"   • AlphaCode objectives trained: {alphacode.get('alphacode_objectives_trained', False)}")
    
    # Final Metrics
    if "final_metrics" in results:
        final = results["final_metrics"]
        print(f"🔄 End-to-End Learning:")
        print(f"   • Learning iterations: {final.get('iterations', 0)}")
        print(f"   • Final avg reward: {final.get('avg_reward', 0):.3f}")
        print(f"   • Final avg advantage: {final.get('avg_advantage', 0):.3f}")
        print(f"   • Final reasoning quality: {final.get('reasoning_quality', 0):.3f}")
        print(f"   • Final evolution fitness: {final.get('evolution_fitness', 0):.3f}")
    
    print("="*60)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Dr.GRPO Full Pipeline")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--data", type=str, help="Path to training data file")
    parser.add_argument("--output", type=str, default="drgrpo_output", help="Output directory")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--dry-run", action="store_true", help="Show configuration without running")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"[CONFIG] Loaded configuration: {config}")
        
        # Find training data
        data_path = find_training_data(args.data)
        logger.info(f"[DATA] Using training data: {data_path}")
        
        if args.dry_run:
            print("[DRY RUN] Configuration:")
            print(f"   Model: {config.model_name}")
            print(f"   Group size: {config.group_size}")
            print(f"   Learning rate: {config.learning_rate}")
            print(f"   Data path: {data_path}")
            print(f"   Output dir: {args.output}")
            return
        
        # Run pipeline
        results = await run_pipeline(config, data_path, args.output)
        
        # Print summary
        print_results_summary(results)
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
