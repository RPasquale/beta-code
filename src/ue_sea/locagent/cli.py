"""
LOCAGENT CLI: Command-line interface for testing and training LOCAGENT.

Provides commands for building graphs, training agents, and running localization.
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from .graph import LOCAGENT_Graph
from .agent import LOCAGENT_Agent
from .training import train_locagent_agent, load_swe_bench_data, load_locbench_data
from .data_loader import create_sample_dataset, load_training_data
from .github_integration import github_train, check_github_dependencies


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


async def build_graph(repo_path: str, output_path: Optional[str] = None, use_gpu: bool = True):
    """Build graph from repository."""
    print(f"Building graph from repository: {repo_path}")
    
    # Create graph
    graph = LOCAGENT_Graph(use_gpu=use_gpu)
    
    # Build from repository
    await graph.build_from_repository(repo_path)
    
    # Print statistics
    stats = graph.get_entity_statistics()
    print(f"Graph built successfully!")
    print(f"Entities: {stats['total_entities']}")
    print(f"Relations: {stats['total_relations']}")
    print(f"Entity types: {stats['entity_types']}")
    print(f"Relation types: {stats['relation_types']}")
    
    # Save if output path provided
    if output_path:
        graph.save_to_file(output_path)
        print(f"Graph saved to: {output_path}")


async def test_localization(repo_path: str, issue_description: str, use_gpu: bool = True):
    """Test localization on a repository."""
    print(f"Testing localization for: {issue_description}")
    
    # Build graph
    graph = LOCAGENT_Graph(use_gpu=use_gpu)
    await graph.build_from_repository(repo_path)
    
    # Create agent
    agent = LOCAGENT_Agent(graph)
    
    # Run localization
    result = await agent.localize(issue_description)
    
    # Print results
    print(f"\nLocalization Results:")
    print(f"Relevant entities found: {len(result.relevant_entities)}")
    print(f"Final ranking:")
    for i, (entity_id, score) in enumerate(result.final_ranking[:10]):
        print(f"  {i+1}. {entity_id} (score: {score:.3f})")
    
    # Print reasoning trace
    print(f"\nReasoning trace:")
    for i, obs in enumerate(result.reasoning_trace):
        print(f"  Step {i+1}: {obs.action.action_type.value} - {obs.action.reasoning}")
        if not obs.success:
            print(f"    Error: {obs.error_message}")


async def train_agent(repo_path: str, training_data_path: str, output_dir: str, 
                     model_name: str = "microsoft/DialoGPT-medium", 
                     num_epochs: int = 3, batch_size: int = 4, use_gpu: bool = True):
    """Train the LOCAGENT agent."""
    print(f"Training LOCAGENT agent...")
    print(f"Repository: {repo_path}")
    print(f"Training data: {training_data_path}")
    print(f"Output directory: {output_dir}")
    
    # Build graph
    print("Building graph...")
    graph = LOCAGENT_Graph(use_gpu=use_gpu)
    await graph.build_from_repository(repo_path)
    
    # Train agent
    await train_locagent_agent(
        graph=graph,
        training_data_path=training_data_path,
        output_dir=output_dir,
        model_name=model_name,
        num_epochs=num_epochs,
        batch_size=batch_size
    )
    
    print("Training completed!")


def create_sample_data(output_path: str, num_examples: int = 100):
    """Create sample training data."""
    print(f"Creating sample dataset with {num_examples} examples...")
    create_sample_dataset(output_path, num_examples)
    print(f"Sample dataset created at: {output_path}")


async def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="LOCAGENT CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Build graph command
    build_parser = subparsers.add_parser("build", help="Build graph from repository")
    build_parser.add_argument("repo_path", help="Path to repository")
    build_parser.add_argument("--output", "-o", help="Output path for graph")
    build_parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    
    # Test localization command
    test_parser = subparsers.add_parser("test", help="Test localization")
    test_parser.add_argument("repo_path", help="Path to repository")
    test_parser.add_argument("issue", help="Issue description")
    test_parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    
    # Train agent command
    train_parser = subparsers.add_parser("train", help="Train agent")
    train_parser.add_argument("repo_path", help="Path to repository")
    train_parser.add_argument("training_data", help="Path to training data")
    train_parser.add_argument("output_dir", help="Output directory for trained model")
    train_parser.add_argument("--model", default="microsoft/DialoGPT-medium", help="Base model name")
    train_parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    train_parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    
    # Create sample data command
    sample_parser = subparsers.add_parser("create-sample", help="Create sample training data")
    sample_parser.add_argument("output_path", help="Output path for sample data")
    sample_parser.add_argument("--num-examples", type=int, default=100, help="Number of examples to create")
    
    # GitHub training command
    github_parser = subparsers.add_parser("github-train", help="Train LOCAGENT on a GitHub repository")
    github_parser.add_argument("repo_url", help="GitHub repository URL")
    github_parser.add_argument("output_dir", help="Output directory for trained model")
    github_parser.add_argument("--github-token", help="GitHub API token (optional)")
    github_parser.add_argument("--issues-limit", type=int, default=100, help="Maximum number of issues to fetch")
    github_parser.add_argument("--issues-labels", nargs="+", help="Filter issues by labels")
    github_parser.add_argument("--issues-state", default="open", choices=["open", "closed", "all"], help="Issue state filter")
    github_parser.add_argument("--include-prs", action="store_true", help="Include pull requests")
    github_parser.add_argument("--min-comments", type=int, default=1, help="Minimum number of comments per issue")
    github_parser.add_argument("--model", default="microsoft/DialoGPT-medium", help="Base model name")
    github_parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    github_parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    github_parser.add_argument("--temp-dir", help="Temporary directory for cloning")
    github_parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    
    # Global options
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Execute command
    if args.command == "build":
        await build_graph(
            repo_path=args.repo_path,
            output_path=args.output,
            use_gpu=not args.no_gpu
        )
    
    elif args.command == "test":
        await test_localization(
            repo_path=args.repo_path,
            issue_description=args.issue,
            use_gpu=not args.no_gpu
        )
    
    elif args.command == "train":
        await train_agent(
            repo_path=args.repo_path,
            training_data_path=args.training_data,
            output_dir=args.output_dir,
            model_name=args.model,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            use_gpu=not args.no_gpu
        )
    
    elif args.command == "create-sample":
        create_sample_data(
            output_path=args.output_path,
            num_examples=args.num_examples
        )
    
    elif args.command == "github-train":
        # Check GitHub dependencies
        deps_available, missing_deps = check_github_dependencies()
        if not deps_available:
            print(f"Warning: Missing GitHub dependencies: {missing_deps}")
            print("Install with: pip install requests PyGithub")
            print("Continuing with limited functionality...")
        
        await github_train(
            repo_url=args.repo_url,
            output_dir=args.output_dir,
            github_token=args.github_token,
            issues_limit=args.issues_limit,
            issues_labels=args.issues_labels,
            issues_state=args.issues_state,
            include_prs=args.include_prs,
            min_issue_comments=args.min_comments,
            model_name=args.model,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            use_gpu=not args.no_gpu,
            temp_dir=args.temp_dir
        )


if __name__ == "__main__":
    asyncio.run(main())
