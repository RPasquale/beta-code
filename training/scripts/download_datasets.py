#!/usr/bin/env python3
"""
Download datasets for UE-SEA training.

Downloads and processes datasets for SFT and RL training.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datasets import load_dataset
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_swe_bench_lite(data_dir: Path) -> None:
    """Download SWE-bench Lite dataset."""
    logger.info("Downloading SWE-bench Lite dataset...")
    
    try:
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="train")
        logger.info(f"Downloaded {len(dataset)} samples from SWE-bench Lite")
        
        # Save to local directory
        dataset.save_to_disk(data_dir / "raw" / "swe_bench_lite")
        
        # Process into diff format
        processed_data = []
        for item in dataset:
            processed_item = {
                "task_id": item.get("instance_id", ""),
                "instruction": f"Fix the issue: {item.get('problem_statement', '')}",
                "current_program": item.get('patch', ''),
                "target_output": item.get('patch', ''),
                "metadata": {
                    "repository": item.get('repo', ''),
                    "base_commit": item.get('base_commit', ''),
                    "problem_statement": item.get('problem_statement', '')
                }
            }
            processed_data.append(processed_item)
        
        # Save processed data
        output_file = data_dir / "processed" / "diffs.jsonl"
        with open(output_file, 'w') as f:
            for item in processed_data:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Processed {len(processed_data)} diff samples to {output_file}")
        
    except Exception as e:
        logger.error(f"Error downloading SWE-bench Lite: {e}")


def download_competition_math(data_dir: Path) -> None:
    """Download Competition Math dataset."""
    logger.info("Downloading Competition Math dataset...")
    
    try:
        dataset = load_dataset("hendrycks/competition_math", split="train")
        logger.info(f"Downloaded {len(dataset)} samples from Competition Math")
        
        # Save to local directory
        dataset.save_to_disk(data_dir / "raw" / "competition_math")
        
        # Process into math format
        processed_data = []
        for item in dataset:
            processed_item = {
                "task_id": f"math_{item.get('id', '')}",
                "instruction": f"Solve this math problem: {item.get('problem', '')}",
                "context": {
                    "problem": item.get('problem', ''),
                    "level": item.get('level', ''),
                    "type": item.get('type', '')
                },
                "target_output": item.get('solution', ''),
                "metadata": {
                    "level": item.get('level', ''),
                    "type": item.get('type', '')
                }
            }
            processed_data.append(processed_item)
        
        # Save processed data
        output_file = data_dir / "processed" / "math_problems.jsonl"
        with open(output_file, 'w') as f:
            for item in processed_data:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Processed {len(processed_data)} math samples to {output_file}")
        
    except Exception as e:
        logger.error(f"Error downloading Competition Math: {e}")


def create_tool_use_data(data_dir: Path) -> None:
    """Create synthetic tool-use data."""
    logger.info("Creating synthetic tool-use data...")
    
    # Sample tool-use scenarios
    tool_scenarios = [
        {
            "task_id": "tool_001",
            "instruction": "Find where HTTP redirects are implemented",
            "demonstration": [
                {
                    "call": "SearchEntity",
                    "args": {"keywords": ["redirect", "http"]},
                    "obs": {"hits": [{"entity_id": "django/shortcuts.py:redirect"}]}
                },
                {
                    "call": "TraverseGraph",
                    "args": {
                        "start_ids": ["django/shortcuts.py:redirect"],
                        "direction": "both",
                        "hops": 2,
                        "entity_types": ["function", "class"],
                        "relation_types": ["invoke", "import"]
                    },
                    "obs": {"subgraph": {"nodes": [], "edges": []}}
                },
                {
                    "call": "RetrieveEntity",
                    "args": {"entity_ids": ["django/shortcuts.py:redirect"]},
                    "obs": {"code": "def redirect(to, *args, **kwargs): ..."}
                }
            ],
            "target": {"final_entity_ids": ["django/shortcuts.py:redirect"]}
        },
        {
            "task_id": "tool_002",
            "instruction": "Find all functions that handle authentication",
            "demonstration": [
                {
                    "call": "SearchEntity",
                    "args": {"keywords": ["auth", "login", "authenticate"]},
                    "obs": {"hits": [{"entity_id": "auth/views.py:login"}]}
                },
                {
                    "call": "RetrieveEntity",
                    "args": {"entity_ids": ["auth/views.py:login"]},
                    "obs": {"code": "def login(request): ..."}
                }
            ],
            "target": {"final_entity_ids": ["auth/views.py:login"]}
        }
    ]
    
    # Save tool-use data
    output_file = data_dir / "processed" / "tool_use.jsonl"
    with open(output_file, 'w') as f:
        for item in tool_scenarios:
            f.write(json.dumps(item) + '\n')
    
    logger.info(f"Created {len(tool_scenarios)} tool-use samples to {output_file}")


def create_test_data(data_dir: Path) -> None:
    """Create synthetic test generation data."""
    logger.info("Creating synthetic test generation data...")
    
    # Sample test generation scenarios
    test_scenarios = [
        {
            "task_id": "test_001",
            "instruction": "Write tests for the redirect function",
            "context": {
                "code_excerpt": "def redirect(to, *args, **kwargs):\n    return HttpResponseRedirect(to)"
            },
            "target_output": "def test_redirect():\n    response = redirect('/home/')\n    assert response.status_code == 302\n    assert response.url == '/home/'\n\ndef test_redirect_with_args():\n    response = redirect('/profile/', permanent=True)\n    assert response.status_code == 301"
        },
        {
            "task_id": "test_002",
            "instruction": "Write tests for the fibonacci function",
            "context": {
                "code_excerpt": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
            },
            "target_output": "def test_fibonacci():\n    assert fibonacci(0) == 0\n    assert fibonacci(1) == 1\n    assert fibonacci(5) == 5\n    assert fibonacci(10) == 55\n\ndef test_fibonacci_edge_cases():\n    assert fibonacci(0) == 0\n    assert fibonacci(1) == 1"
        }
    ]
    
    # Save test data
    output_file = data_dir / "processed" / "tests.jsonl"
    with open(output_file, 'w') as f:
        for item in test_scenarios:
            f.write(json.dumps(item) + '\n')
    
    logger.info(f"Created {len(test_scenarios)} test generation samples to {output_file}")


def create_rl_tasks(data_dir: Path) -> None:
    """Create RL training tasks."""
    logger.info("Creating RL training tasks...")
    
    # Sample RL tasks
    rl_tasks = [
        {
            "id": "rl_001",
            "domain": "code",
            "prompt": "Add support for HTTP 307/308 redirects to the redirect function. CURRENT PROGRAM: def redirect(to, *args, **kwargs): return HttpResponseRedirect(to)",
            "tests": ["test_redirect_307", "test_redirect_308"],
            "metadata": {"difficulty": "medium", "category": "http"}
        },
        {
            "id": "rl_002",
            "domain": "code",
            "prompt": "Optimize the fibonacci function for better performance. CURRENT PROGRAM: def fibonacci(n): if n <= 1: return n; return fibonacci(n-1) + fibonacci(n-2)",
            "tests": ["test_fibonacci_performance", "test_fibonacci_correctness"],
            "metadata": {"difficulty": "hard", "category": "optimization"}
        },
        {
            "id": "rl_101",
            "domain": "math",
            "prompt": "Solve: What is the sum of all positive integers less than 100 that are divisible by 3 or 5? (Final answer as \\boxed{...})",
            "metadata": {"difficulty": "medium", "category": "arithmetic"}
        }
    ]
    
    # Save RL tasks
    output_file = data_dir / "processed" / "rl_tasks.jsonl"
    with open(output_file, 'w') as f:
        for item in rl_tasks:
            f.write(json.dumps(item) + '\n')
    
    logger.info(f"Created {len(rl_tasks)} RL tasks to {output_file}")


def main():
    """Main function to download and process all datasets."""
    parser = argparse.ArgumentParser(description="Download datasets for UE-SEA training")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--datasets", nargs="+", default=["all"], 
                       choices=["all", "swe_bench", "math", "tools", "tests", "rl"],
                       help="Datasets to download")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "raw").mkdir(exist_ok=True)
    (data_dir / "processed").mkdir(exist_ok=True)
    
    logger.info(f"Downloading datasets to {data_dir}")
    
    if "all" in args.datasets or "swe_bench" in args.datasets:
        download_swe_bench_lite(data_dir)
    
    if "all" in args.datasets or "math" in args.datasets:
        download_competition_math(data_dir)
    
    if "all" in args.datasets or "tools" in args.datasets:
        create_tool_use_data(data_dir)
    
    if "all" in args.datasets or "tests" in args.datasets:
        create_test_data(data_dir)
    
    if "all" in args.datasets or "rl" in args.datasets:
        create_rl_tasks(data_dir)
    
    logger.info("Dataset download and processing completed!")
    
    # Print summary
    print("\n=== Dataset Summary ===")
    for file_path in (data_dir / "processed").glob("*.jsonl"):
        with open(file_path, 'r') as f:
            count = sum(1 for _ in f)
        print(f"{file_path.name}: {count} samples")


if __name__ == "__main__":
    main()
