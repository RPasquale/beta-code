"""
Comprehensive LOCAGENT Training Script

Uses MAXIMUM training data from codebases by extracting from ALL possible sources:
- GitHub issues, PRs, commits, discussions
- Code comments, docstrings, TODOs
- Test files and their descriptions  
- Documentation and README files
- Git history and commit messages
- Synthetic examples
- And much more!
"""

import sys
import os
import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from ue_sea.locagent.comprehensive_data_extractor import (
        ComprehensiveDataExtractor, 
        extract_comprehensive_training_data,
        TrainingExample
    )
    from ue_sea.locagent.training import LOCAGENT_Trainer, LocalizationExample, LocalizationDataset
    from ue_sea.locagent.graph import LOCAGENT_Graph
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
    import torch
    import random
    import numpy as np
    
    print("✅ All dependencies available")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure LOCAGENT components are available")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


async def main():
    print("🚀 COMPREHENSIVE LOCAGENT Training Pipeline")
    print("==================================================")
    print("This will extract training data from ALL possible sources!")
    print()
    
    # Configuration
    repo_url = input("Enter GitHub repository URL (or press Enter for Django): ").strip()
    if not repo_url:
        repo_url = "https://github.com/django/django"
    
    github_token = input("Enter GitHub token (optional, for more data): ").strip() or None
    
    max_examples = int(input("Enter max examples to extract (default 1000): ") or "1000")
    
    num_epochs = int(input("Enter training epochs (default 3): ") or "3")
    
    output_dir = f"outputs/comprehensive_locagent_training_{Path(repo_url.split('/')[-1]).stem}"
    
    print(f"\n📊 Configuration:")
    print(f"   Repository: {repo_url}")
    print(f"   Max examples: {max_examples}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Output: {output_dir}")
    print()
    
    # Step 1: Clone repository
    print("📥 Step 1: Cloning repository...")
    temp_repo_path = Path(os.environ.get("TEMP", "/tmp")) / f"tmp_comprehensive_repo_{os.getpid()}"
    
    try:
        import subprocess
        subprocess.run(["git", "clone", repo_url, str(temp_repo_path)], check=True)
        print(f"✅ Repository cloned to {temp_repo_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to clone repository: {e}")
        return
    
    # Step 2: Extract comprehensive training data
    print("\n🔍 Step 2: Extracting comprehensive training data...")
    print("   This will extract from ALL possible sources:")
    print("   - GitHub issues, PRs, commits, discussions")
    print("   - Code comments, docstrings, TODOs")
    print("   - Test files and their descriptions")
    print("   - Documentation and README files")
    print("   - Git history and commit messages")
    print("   - Synthetic examples")
    print("   - And much more!")
    
    try:
        extractor = ComprehensiveDataExtractor(str(temp_repo_path), github_token)
        training_examples = await extractor.extract_all_training_data(
            max_examples=max_examples,
            include_synthetic=True
        )
        
        print(f"✅ Extracted {len(training_examples)} comprehensive training examples")
        
        # Show breakdown by source type
        source_counts = {}
        for example in training_examples:
            source_type = example.source_type
            source_counts[source_type] = source_counts.get(source_type, 0) + 1
        
        print("\n📊 Training data breakdown:")
        for source_type, count in sorted(source_counts.items()):
            print(f"   {source_type}: {count} examples")
        
    except Exception as e:
        print(f"❌ Error extracting training data: {e}")
        return
    
    # Step 3: Convert to LocalizationExample format
    print("\n🔄 Step 3: Converting to training format...")
    
    localization_examples = []
    for example in training_examples:
        loc_example = LocalizationExample(
            issue_description=example.issue_description,
            repository_path=example.repository_path,
            target_entities=example.target_entities,
            context=example.context,
            difficulty=example.difficulty,
            metadata={
                "source_type": example.source_type,
                "source_id": example.source_id,
                "confidence": example.confidence,
                **example.metadata
            }
        )
        localization_examples.append(loc_example)
    
    print(f"✅ Converted {len(localization_examples)} examples to training format")
    
    # Step 4: Build code graph
    print("\n🏗️ Step 4: Building code graph...")
    
    try:
        graph = LOCAGENT_Graph(use_gpu=True)
        await graph.build_from_repository(str(temp_repo_path))
        print(f"✅ Built graph with {len(graph.get_all_entities())} entities")
    except Exception as e:
        print(f"❌ Error building graph: {e}")
        return
    
    # Step 5: Train the model
    print("\n🚀 Step 5: Training LOCAGENT model...")
    
    try:
        # Initialize trainer
        trainer = LOCAGENT_Trainer(model_name="microsoft/DialoGPT-medium")
        trainer.load_model()
        
        # Prepare dataset
        train_dataset, val_dataset = trainer.prepare_dataset(localization_examples)
        
        print(f"📊 Training examples: {len(train_dataset)}")
        print(f"📊 Validation examples: {len(val_dataset)}")
        print(f"🔄 Epochs: {num_epochs}")
        print(f"📦 Batch size: {trainer.batch_size}")
        print(f"📁 Output directory: {output_dir}")
        
        # Train the model
        trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_dir=output_dir,
            num_epochs=num_epochs,
            batch_size=trainer.batch_size
        )
        
        print("✅ Model training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return
    
    # Step 6: Save comprehensive training data
    print("\n💾 Step 6: Saving comprehensive training data...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training examples
    with open(Path(output_dir) / "comprehensive_training_data.jsonl", "w") as f:
        for example in training_examples:
            f.write(json.dumps({
                "issue_description": example.issue_description,
                "repository_path": example.repository_path,
                "target_entities": example.target_entities,
                "source_type": example.source_type,
                "source_id": example.source_id,
                "context": example.context,
                "difficulty": example.difficulty,
                "confidence": example.confidence,
                "metadata": example.metadata
            }) + "\n")
    
    # Save training summary
    training_summary = {
        "total_examples": len(training_examples),
        "source_breakdown": source_counts,
        "repository": repo_url,
        "extraction_time": "N/A",  # Could add timing
        "training_epochs": num_epochs,
        "model_name": "microsoft/DialoGPT-medium",
        "output_directory": output_dir
    }
    
    with open(Path(output_dir) / "training_summary.json", "w") as f:
        json.dump(training_summary, f, indent=2)
    
    print(f"✅ Comprehensive training data saved to {output_dir}")
    
    # Step 7: Cleanup
    print("\n🧹 Step 7: Cleaning up...")
    
    try:
        import shutil
        shutil.rmtree(temp_repo_path)
        print(f"✅ Cleaned up temporary repository: {temp_repo_path}")
    except Exception as e:
        print(f"⚠️  Warning: Could not clean up {temp_repo_path}: {e}")
    
    print("\n🎉 COMPREHENSIVE LOCAGENT TRAINING COMPLETED!")
    print("==================================================")
    print(f"📁 Model saved to: {output_dir}")
    print(f"📊 Total examples: {len(training_examples)}")
    print(f"🔍 Sources used: {len(source_counts)} different types")
    print(f"🎯 Training epochs: {num_epochs}")
    print()
    print("The model has been trained on the MAXIMUM amount of data possible!")
    print("This includes issues, PRs, commits, comments, tests, docs, and more!")


if __name__ == "__main__":
    asyncio.run(main())
