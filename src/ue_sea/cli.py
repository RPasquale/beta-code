"""
UE-SEA CLI: Command-line interface for the system.

Provides easy access to all UE-SEA functionality.
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import Dict, Any

from .orchestrator import UE_SEA_Orchestrator, CycleConfig


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Unified Evolutionary Software Engineering Agent (UE-SEA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Traditional mode - Start UE-SEA on a repository
  python -m ue_sea.cli --repo /path/to/repo --task "Improve code quality"
  
  # End-to-end training mode - Train agent using training data
  python -m ue_sea.cli --train-end-to-end --training-data training/data/processed/full_dataset.jsonl
  
  # SPCT training - Self-Principled Critique Tuning
  python -m ue_sea.cli --train-spct --training-data training/data/processed/full_dataset.jsonl --rft-samples 8 --inference-samples 8
  
  # SPCT training with custom configuration
  python -m ue_sea.cli --train-spct --training-data training/data/processed/full_dataset.jsonl --spct-config spct_config.json
  
  # Full RL training with Dr.GRPO - Complete end-to-end pipeline
  python -m ue_sea.cli --train-rl --training-data training/data/processed/full_dataset.jsonl --group-size 8 --ppo-epsilon 0.2
  
  # RL training with custom configuration
  python -m ue_sea.cli --train-rl --training-data training/data/processed/full_dataset.jsonl --rl-config rl_config.json
  
  # End-to-end training with custom configuration
  python -m ue_sea.cli --train-end-to-end --training-data training/data/processed/full_dataset.jsonl --batch-size 20 --spct --spct-frequency 2
  
  # Run with custom configuration
  python -m ue_sea.cli --repo /path/to/repo --task "Optimize performance" --max-cycles 50 --gpu
  
  # Initialize system only
  python -m ue_sea.cli --init --repo /path/to/repo
        """
    )
    
    # Main arguments
    parser.add_argument("--repo", type=str, required=False,
                       help="Path to the repository to analyze (optional for end-to-end training)")
    parser.add_argument("--task", type=str, default=None,
                       help="Description of the improvement task (optional for end-to-end training)")
    parser.add_argument("--init", action="store_true",
                       help="Initialize system only (don't run evolution)")
    
    # End-to-end training arguments
    parser.add_argument("--train-end-to-end", action="store_true",
                       help="Run end-to-end training using training data")
    parser.add_argument("--training-data", type=str,
                       help="Path to training data file (JSONL format) for end-to-end training")
    parser.add_argument("--batch-size", type=int, default=10,
                       help="Batch size for processing training tasks")
    
    # Configuration arguments
    parser.add_argument("--max-cycles", type=int, default=100,
                       help="Maximum number of evolution cycles")
    parser.add_argument("--cycle-timeout", type=int, default=3600,
                       help="Timeout per cycle in seconds")
    parser.add_argument("--evolution-budget", type=int, default=1000,
                       help="Maximum evaluations per cycle")
    parser.add_argument("--training-budget", type=int, default=100,
                       help="Maximum training steps per cycle")
    parser.add_argument("--deployment-threshold", type=float, default=0.8,
                       help="Minimum score for deployment")
    parser.add_argument("--parallel-workers", type=int, default=4,
                       help="Number of parallel workers")
    
    # GPU and performance
    parser.add_argument("--gpu", action="store_true", default=True,
                       help="Enable GPU acceleration")
    parser.add_argument("--no-gpu", action="store_true",
                       help="Disable GPU acceleration")
    
    # SPCT options
    parser.add_argument("--spct", action="store_true", default=True,
                       help="Enable SPCT (Self-Programming Code Training)")
    parser.add_argument("--no-spct", action="store_true",
                       help="Disable SPCT training")
    parser.add_argument("--spct-frequency", type=int, default=3,
                       help="Run SPCT every N cycles")
    parser.add_argument("--spct-budget", type=int, default=200,
                       help="Maximum SPCT training steps per cycle")
    
    # RL training arguments
    parser.add_argument("--train-rl", action="store_true",
                       help="Run full end-to-end RL training with Dr.GRPO")
    parser.add_argument("--rl-config", type=str,
                       help="Path to RL configuration JSON file")
    parser.add_argument("--group-size", type=int, default=8,
                       help="Group size for Dr.GRPO (default: 8)")
    parser.add_argument("--ppo-epsilon", type=float, default=0.2,
                       help="PPO epsilon for Dr.GRPO (default: 0.2)")
    parser.add_argument("--rl-learning-rate", type=float, default=1e-6,
                       help="Learning rate for RL training (default: 1e-6)")
    
    # SPCT training arguments
    parser.add_argument("--train-spct", action="store_true",
                       help="Run SPCT (Self-Principled Critique Tuning) training")
    parser.add_argument("--spct-config", type=str,
                       help="Path to SPCT configuration JSON file")
    parser.add_argument("--rft-samples", type=int, default=8,
                       help="Number of RFT samples per data point (default: 8)")
    parser.add_argument("--inference-samples", type=int, default=8,
                       help="Number of inference samples for scaling (default: 8)")
    parser.add_argument("--use-meta-rm", action="store_true", default=True,
                       help="Use Meta-RM for sample filtering")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="./ue_sea_output",
                       help="Output directory for results")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.train_spct:
        # SPCT training mode
        if not args.training_data:
            print("Error: --training-data is required for SPCT training")
            sys.exit(1)
        
        training_data_path = Path(args.training_data)
        if not training_data_path.exists():
            print(f"Error: Training data file does not exist: {training_data_path}")
            sys.exit(1)
        
        repo_path = args.repo if args.repo else "."  # Use current directory if no repo specified
    elif args.train_rl:
        # RL training mode
        if not args.training_data:
            print("Error: --training-data is required for RL training")
            sys.exit(1)
        
        training_data_path = Path(args.training_data)
        if not training_data_path.exists():
            print(f"Error: Training data file does not exist: {training_data_path}")
            sys.exit(1)
        
        repo_path = args.repo if args.repo else "."  # Use current directory if no repo specified
    elif args.train_end_to_end:
        # End-to-end training mode
        if not args.training_data:
            print("Error: --training-data is required for end-to-end training")
            sys.exit(1)
        
        training_data_path = Path(args.training_data)
        if not training_data_path.exists():
            print(f"Error: Training data file does not exist: {training_data_path}")
            sys.exit(1)
        
        repo_path = args.repo if args.repo else "."  # Use current directory if no repo specified
    else:
        # Traditional mode - require repo
        if not args.repo:
            print("Error: --repo is required for traditional mode")
            sys.exit(1)
        
        repo_path = Path(args.repo)
        if not repo_path.exists():
            print(f"Error: Repository path does not exist: {repo_path}")
            sys.exit(1)
        
        if not repo_path.is_dir():
            print(f"Error: Repository path is not a directory: {repo_path}")
            sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure system
    config = CycleConfig(
        max_cycles=args.max_cycles,
        cycle_timeout=args.cycle_timeout,
        evolution_budget=args.evolution_budget,
        training_budget=args.training_budget,
        spct_budget=args.spct_budget,
        deployment_threshold=args.deployment_threshold,
        parallel_workers=args.parallel_workers,
        gpu_enabled=args.gpu and not args.no_gpu,
        spct_enabled=args.spct and not args.no_spct,
        spct_frequency=args.spct_frequency
    )
    
    # Initialize orchestrator
    orchestrator = UE_SEA_Orchestrator(config)
    
    try:
        if args.init:
            # Initialize system only
            print("Initializing UE-SEA system...")
            await orchestrator._initialize_system(str(repo_path), args.task)
            print("System initialization complete!")
            
        elif args.train_spct:
            # Run SPCT training
            print("🚀 Starting SPCT (Self-Principled Critique Tuning)...")
            
            # Prepare SPCT configuration
            spct_config = {
                "rft_samples": args.rft_samples,
                "inference_samples": args.inference_samples,
                "use_meta_rm_filter": args.use_meta_rm,
                "model_name": "Qwen/Qwen2.5-Coder-3B"
            }
            
            # Load custom SPCT config if provided
            if args.spct_config:
                import json
                with open(args.spct_config, 'r') as f:
                    custom_config = json.load(f)
                    spct_config.update(custom_config)
            
            results = await orchestrator.train_spct(str(training_data_path), spct_config)
            
            print("✅ SPCT training complete!")
            print(f"\nSPCT Training Results:")
            print(f"  RFT Results: {results['spct_results'].get('rft_results', {})}")
            print(f"  Dr.GRPO Results: {results['spct_results'].get('drgrpo_results', {})}")
            print(f"  Meta-RM Results: {results['spct_results'].get('meta_rm_results', {})}")
            print(f"  Inference Results: {results['spct_results'].get('inference_results', {})}")
            print(f"  Agent capabilities: {results['agent_capabilities']} skills")
            
        elif args.train_rl:
            # Run RL training
            print("🚀 Starting Full End-to-End RL Training with Dr.GRPO...")
            
            # Prepare RL configuration
            rl_config = {
                "group_size": args.group_size,
                "ppo_epsilon": args.ppo_epsilon,
                "learning_rate": args.rl_learning_rate,
                "use_locagent": True,
                "use_reasoning_traces": True
            }
            
            # Load custom RL config if provided
            if args.rl_config:
                import json
                with open(args.rl_config, 'r') as f:
                    custom_config = json.load(f)
                    rl_config.update(custom_config)
            
            results = await orchestrator.train_end_to_end_rl(str(training_data_path), rl_config)
            
            print("✅ Full RL training complete!")
            print(f"\nRL Training Results:")
            print(f"  SFT Results: {results['rl_results'].get('sft_results', {})}")
            print(f"  Reward Model: {results['rl_results'].get('reward_model_results', {})}")
            print(f"  Dr.GRPO Results: {results['rl_results'].get('drgrpo_results', {})}")
            print(f"  Evolution Results: {results['rl_results'].get('evolution_results', {})}")
            print(f"  Agent capabilities: {results['agent_capabilities']} skills")
            
        elif args.train_end_to_end:
            # Run end-to-end training
            print("🚀 Starting End-to-End Agent Training...")
            context = {
                "batch_size": args.batch_size,
                "gpu_enabled": config.gpu_enabled,
                "spct_enabled": config.spct_enabled
            }
            
            results = await orchestrator.train_end_to_end(str(training_data_path), context)
            
            print("✅ End-to-end training complete!")
            print(f"\nTraining Results:")
            print(f"  Total tasks processed: {results['total_tasks']}")
            print(f"  Training cycles: {results['training_cycles']}")
            print(f"  Successful cycles: {results['successful_cycles']}")
            print(f"  Agent capabilities: {len(results['agent_capabilities'])} skills")
            
        else:
            # Run traditional evolution
            print("Starting UE-SEA evolution...")
            await orchestrator.start(str(repo_path), args.task)
            print("Evolution complete!")
        
        # Print final status
        status = orchestrator.get_status()
        print(f"\nFinal Status:")
        print(f"  Total cycles: {status['total_cycles']}")
        print(f"  Programs in registry: {status['program_registry_size']}")
        print(f"  Skills in registry: {status['skill_registry_size']}")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user. Stopping...")
        await orchestrator.stop()
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
