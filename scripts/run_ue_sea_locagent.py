#!/usr/bin/env python3
"""
UE-SEA with LOCAGENT Integration Demo Script

This script demonstrates the complete integration of LOCAGENT with UE-SEA,
providing intelligent code localization, semantic understanding, and
real-time feedback for evolutionary code improvement.
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ue_sea.orchestrator import UE_SEA_Orchestrator, CycleConfig


async def main():
    """Main function to run UE-SEA with LOCAGENT integration."""
    parser = argparse.ArgumentParser(description="Run UE-SEA with LOCAGENT Integration")
    parser.add_argument("--repo-path", required=True, help="Path to the repository to analyze")
    parser.add_argument("--task", required=True, help="Description of the improvement task")
    parser.add_argument("--use-locagent", action="store_true", default=True, 
                       help="Enable LOCAGENT integration")
    parser.add_argument("--gpu-enabled", action="store_true", default=True,
                       help="Enable GPU acceleration")
    parser.add_argument("--localization-threshold", type=float, default=0.8,
                       help="LOCAGENT localization confidence threshold")
    parser.add_argument("--semantic-search", action="store_true", default=True,
                       help="Enable semantic search")
    parser.add_argument("--real-time-feedback", action="store_true", default=True,
                       help="Enable real-time feedback")
    parser.add_argument("--evolution-budget", type=int, default=1000,
                       help="Evolution budget (max evaluations)")
    parser.add_argument("--training-budget", type=int, default=100,
                       help="Training budget (max training steps)")
    parser.add_argument("--max-cycles", type=int, default=10,
                       help="Maximum number of evolution cycles")
    
    args = parser.parse_args()
    
    # Create configuration
    config = CycleConfig(
        max_cycles=args.max_cycles,
        evolution_budget=args.evolution_budget,
        training_budget=args.training_budget,
        gpu_enabled=args.gpu_enabled,
        parallel_workers=4,
        deployment_threshold=0.8
    )
    
    # Create orchestrator
    orchestrator = UE_SEA_Orchestrator(config=config)
    
    # Prepare context with LOCAGENT settings
    context = {
        "use_locagent": args.use_locagent,
        "localization_threshold": args.localization_threshold,
        "semantic_search": args.semantic_search,
        "real_time_feedback": args.real_time_feedback,
        "gpu_acceleration": args.gpu_enabled
    }
    
    print("🚀 UE-SEA with LOCAGENT Integration")
    print("=" * 50)
    print(f"Repository: {args.repo_path}")
    print(f"Task: {args.task}")
    print(f"LOCAGENT Integration: {args.use_locagent}")
    print(f"GPU Acceleration: {args.gpu_enabled}")
    print(f"Semantic Search: {args.semantic_search}")
    print(f"Real-time Feedback: {args.real_time_feedback}")
    print("=" * 50)
    
    try:
        # Run UE-SEA with LOCAGENT integration
        results = await orchestrator.run(
            repo_path=args.repo_path,
            task_description=args.task,
            context=context
        )
        
        # Display results
        print("\n🎉 UE-SEA with LOCAGENT Integration Complete!")
        print("=" * 50)
        print(f"Total Cycles: {results['total_cycles']}")
        print(f"Successful Cycles: {results['successful_cycles']}")
        print(f"LOCAGENT Enhanced: {results['locagent_enhanced']}")
        print(f"Semantic Understanding: {results['semantic_understanding']}")
        print(f"GPU Acceleration: {results['gpu_acceleration']}")
        print(f"Real-time Feedback: {results['real_time_feedback']}")
        
        if results['final_metrics']:
            print("\n📊 Final Metrics:")
            for key, value in results['final_metrics'].items():
                print(f"  {key}: {value}")
        
        print("\n✅ Integration successful! LOCAGENT + UE-SEA = Supercharged AI Code Evolution!")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
