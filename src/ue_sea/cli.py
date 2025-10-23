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
  # Start UE-SEA on a repository
  python -m ue_sea.cli --repo /path/to/repo --task "Improve code quality"
  
  # Run with custom configuration
  python -m ue_sea.cli --repo /path/to/repo --task "Optimize performance" --max-cycles 50 --gpu
  
  # Initialize system only
  python -m ue_sea.cli --init --repo /path/to/repo
        """
    )
    
    # Main arguments
    parser.add_argument("--repo", type=str, required=True,
                       help="Path to the repository to analyze")
    parser.add_argument("--task", type=str, default="Improve code quality and performance",
                       help="Description of the improvement task")
    parser.add_argument("--init", action="store_true",
                       help="Initialize system only (don't run evolution)")
    
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
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="./ue_sea_output",
                       help="Output directory for results")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Validate arguments
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
        deployment_threshold=args.deployment_threshold,
        parallel_workers=args.parallel_workers,
        gpu_enabled=args.gpu and not args.no_gpu
    )
    
    # Initialize orchestrator
    orchestrator = UE_SEA_Orchestrator(config)
    
    try:
        if args.init:
            # Initialize system only
            print("Initializing UE-SEA system...")
            await orchestrator._initialize_system(str(repo_path), args.task)
            print("System initialization complete!")
            
        else:
            # Run full evolution
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
