#!/usr/bin/env python3
"""
SLM CLI - Easy inference interface for the Small Language Model

Usage examples:
    python slm_cli.py --model_path model.pth --prompt "The quick brown fox"
    python slm_cli.py --model_path model.pth --prompt "Once upon a time" --method beam --beam_width 8
    python slm_cli.py --model_path model.pth --prompt "The future of AI" --method constrained --temperature 0.8
"""

import argparse
import sys
import os

# Add the slm package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slm.inference_slm import SLMInference


def main():
    parser = argparse.ArgumentParser(
        description="SLM CLI - Easy inference for Small Language Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model_path model.pth --prompt "The quick brown fox"
  %(prog)s --model_path model.pth --prompt "Once upon a time" --method beam --beam_width 8
  %(prog)s --model_path model.pth --prompt "The future of AI" --method constrained --temperature 0.8
        """
    )
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to model checkpoint (.pth file)")
    parser.add_argument("--prompt", type=str, required=True, 
                       help="Input text prompt")
    
    # Generation method
    parser.add_argument("--method", type=str, 
                       choices=["greedy", "beam", "constrained"], 
                       default="greedy", 
                       help="Generation method (default: greedy)")
    
    # Generation parameters
    parser.add_argument("--max_tokens", type=int, default=50, 
                       help="Maximum new tokens to generate (default: 50)")
    parser.add_argument("--beam_width", type=int, default=4, 
                       help="Beam width for beam search (default: 4)")
    parser.add_argument("--temperature", type=float, default=1.0, 
                       help="Sampling temperature (default: 1.0)")
    parser.add_argument("--top_k", type=int, default=50, 
                       help="Top-k for constrained generation (default: 50)")
    parser.add_argument("--end_token", type=str, default=None, 
                       help="End token to stop generation (e.g., '\\n', '.')")
    parser.add_argument("--length_penalty", type=float, default=0.7, 
                       help="Length penalty for beam search (default: 0.7)")
    
    # Output options
    parser.add_argument("--quiet", action="store_true", 
                       help="Suppress status messages")
    parser.add_argument("--full_text", action="store_true", 
                       help="Show full text (prompt + generated)")
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    try:
        # Initialize inference
        if not args.quiet:
            print(f"Loading model from {args.model_path}...")
            
        inference = SLMInference(args.model_path)
        
        if not args.quiet:
            print("Model loaded successfully!")
            print(f"Prompt: {args.prompt}")
            print(f"Method: {args.method}")
            print("-" * 60)
        
        # Generate text
        if args.method == "greedy":
            result = inference.greedy_generate(
                args.prompt, 
                max_new_tokens=args.max_tokens,
                end_token=args.end_token
            )
        elif args.method == "beam":
            result = inference.beam_search_generate(
                args.prompt,
                beam_width=args.beam_width,
                max_new_tokens=args.max_tokens,
                end_token=args.end_token,
                length_penalty=args.length_penalty
            )
        elif args.method == "constrained":
            result = inference.constrained_generate(
                args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                end_token=args.end_token
            )
        
        # Output results
        if args.full_text:
            print(f"{args.prompt}{result}")
        else:
            print(result)
            
        if not args.quiet:
            print("-" * 60)
            print(f"Generated {len(result.split())} words")
            
    except Exception as e:
        print(f"Error: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
