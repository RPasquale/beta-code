#!/usr/bin/env python3
"""
SLM Training Script - Easy training interface

Usage examples:
    python train_cli.py
    python train_cli.py --config custom_config.yaml
    python train_cli.py --steps 10000 --batch_size 64
"""

import argparse
import sys
import os

# Add the slm package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slm.train_slm import main as train_main
from slm.slm import Config, set_seed


def main():
    parser = argparse.ArgumentParser(
        description="SLM Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Train with default config
  %(prog)s --steps 10000 --batch_size 64     # Override specific parameters
  %(prog)s --seed 123 --device cuda          # Set seed and device
        """
    )
    
    # Training parameters
    parser.add_argument("--steps", type=int, default=None, 
                       help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=None, 
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=None, 
                       help="Learning rate")
    parser.add_argument("--device", type=str, default=None, 
                       choices=["cpu", "cuda", "mps"], 
                       help="Device to use")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    
    # Model parameters
    parser.add_argument("--d_model", type=int, default=None, 
                       help="Model dimension")
    parser.add_argument("--nhead", type=int, default=None, 
                       help="Number of attention heads")
    parser.add_argument("--n_layers_rep", type=int, default=None, 
                       help="Number of representation layers")
    parser.add_argument("--n_layers_dyn", type=int, default=None, 
                       help="Number of dynamics layers")
    
    # Data parameters
    parser.add_argument("--train_samples", type=int, default=None, 
                       help="Number of training samples")
    parser.add_argument("--vocab_topk", type=int, default=None, 
                       help="Top-k vocabulary size")
    parser.add_argument("--max_seq_len", type=int, default=None, 
                       help="Maximum sequence length")
    
    # Output options
    parser.add_argument("--save_path", type=str, default="slm_model.pth", 
                       help="Path to save the trained model")
    parser.add_argument("--quiet", action="store_true", 
                       help="Suppress verbose output")
    
    args = parser.parse_args()
    
    try:
        # Create config with overrides
        config = Config()
        
        # Apply overrides
        if args.steps is not None:
            config.train_steps = args.steps
        if args.batch_size is not None:
            config.batch_size = args.batch_size
        if args.learning_rate is not None:
            config.learning_rate = args.learning_rate
        if args.device is not None:
            config.device = args.device
        if args.d_model is not None:
            config.d_model = args.d_model
        if args.nhead is not None:
            config.nhead = args.nhead
        if args.n_layers_rep is not None:
            config.n_layers_rep = args.n_layers_rep
        if args.n_layers_dyn is not None:
            config.n_layers_dyn = args.n_layers_dyn
        if args.train_samples is not None:
            config.train_samples = args.train_samples
        if args.vocab_topk is not None:
            config.vocab_topk = args.vocab_topk
        if args.max_seq_len is not None:
            config.max_seq_len = args.max_seq_len
            
        # Set seed
        set_seed(args.seed)
        
        if not args.quiet:
            print("SLM Training Configuration:")
            print(f"  Steps: {config.train_steps}")
            print(f"  Batch size: {config.batch_size}")
            print(f"  Learning rate: {config.learning_rate}")
            print(f"  Device: {config.device}")
            print(f"  Model dim: {config.d_model}")
            print(f"  Attention heads: {config.nhead}")
            print(f"  Rep layers: {config.n_layers_rep}")
            print(f"  Dyn layers: {config.n_layers_dyn}")
            print(f"  Train samples: {config.train_samples}")
            print(f"  Vocab top-k: {config.vocab_topk}")
            print(f"  Max seq len: {config.max_seq_len}")
            print("-" * 50)
        
        # Run training
        train_main()
        
        if not args.quiet:
            print(f"\nTraining completed! Model saved to: {args.save_path}")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
