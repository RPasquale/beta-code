# inference_slm.py - Inference with KV cache and beam search
import argparse
import torch
import torch.nn.functional as F
from typing import List, Optional
from transformers import AutoTokenizer

from .slm import MuZeroTransformer, Config


# ============================== Inference Components ============================== #
class Hypothesis:
    __slots__ = ("tokens", "score", "caches", "pos")
    def __init__(self, tokens, score, caches, pos):
        self.tokens = tokens      # python list[int]
        self.score = score        # float (log-prob sum)
        self.caches = caches      # list of (K,V) for each rep block
        self.pos = pos            # next absolute position int


class SLMInference:
    def __init__(self, model_path: str, config: Optional[Config] = None):
        """
        Initialize SLM inference with a trained model.
        
        Args:
            model_path: Path to the saved model checkpoint
            config: Optional config override
        """
        self.config = config or Config()
        self.device = self.config.device
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model
        self.model = MuZeroTransformer(
            vocab_size=self.tokenizer.vocab_size, 
            cfg=self.config
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.eval()
        
        # Get top-k vocabulary for constrained generation
        self.vocab_allowed = self._get_topk_vocab()
        self.allowed_tensor = torch.tensor(self.vocab_allowed, device=self.device, dtype=torch.long)
        
    def _get_topk_vocab(self) -> List[int]:
        """Get top-k most frequent tokens for constrained generation"""
        # This is a simplified version - in practice you'd want to load this from training
        # For now, we'll use a reasonable subset of the vocabulary
        vocab_size = self.tokenizer.vocab_size
        topk = min(self.config.vocab_topk, vocab_size)
        return list(range(topk))
    
    @torch.no_grad()
    def greedy_generate(self, prompt: str, max_new_tokens: int = 50, end_token: Optional[str] = None) -> str:
        """
        Generate text using greedy decoding with KV cache.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            end_token: Optional token to stop generation (e.g., "\n")
            
        Returns:
            Generated text continuation
        """
        # Tokenize prompt
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) > self.config.max_seq_len:
            prompt_ids = prompt_ids[-self.config.max_seq_len:]
            
        inp = torch.tensor([prompt_ids], device=self.device, dtype=torch.long)
        
        # Prefill rep stack + get initial policy
        s0, caches, pos = self.model.encode_prefill_with_cache(inp)
        
        # Use CLS from s0 for logits
        logits = self.model.head_policy(s0[:, :1, :]).squeeze(1)  # [B,V]
        next_id = torch.argmax(logits, dim=-1)  # [B]
        
        generated_ids = []
        for _ in range(max_new_tokens):
            tid = int(next_id.item())
            generated_ids.append(tid)
            
            # Check for end token
            if end_token:
                token_text = self.tokenizer.decode([tid])
                if end_token in token_text:
                    break
                    
            # One-step decode via caches on rep stack
            _, caches, pos = self.model.encode_decode_with_cache(next_id, caches, pos)
            
            # Score from current CLS (simple choice: use the *existing* s0 CLS)
            logits = self.model.head_policy(s0[:, :1, :]).squeeze(1)
            next_id = torch.argmax(logits, dim=-1)
            
        # Decode and return
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text
    
    @torch.no_grad()
    def beam_search_generate(
        self, 
        prompt: str, 
        beam_width: int = 4,
        max_new_tokens: int = 50,
        end_token: Optional[str] = None,
        length_penalty: float = 0.7
    ) -> str:
        """
        Generate text using beam search with KV cache.
        
        Args:
            prompt: Input text prompt
            beam_width: Number of beams to maintain
            max_new_tokens: Maximum number of new tokens to generate
            end_token: Optional token to stop generation
            length_penalty: Length penalty for beam scoring
            
        Returns:
            Generated text continuation
        """
        # Tokenize prompt
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) > self.config.max_seq_len:
            prompt_ids = prompt_ids[-self.config.max_seq_len:]
            
        inp = torch.tensor([prompt_ids], device=self.device, dtype=torch.long)
        s0, caches0, pos0 = self.model.encode_prefill_with_cache(inp)
        
        logits0 = self.model.head_policy(s0[:, :1, :]).squeeze(1)  # [1,V]
        log_probs0 = F.log_softmax(logits0, dim=-1)
        
        topk_scores, topk_ids = torch.topk(log_probs0, k=beam_width, dim=-1)
        active = []
        for i in range(beam_width):
            t = int(topk_ids[0, i].item())
            s = float(topk_scores[0, i].item())
            active.append(Hypothesis(tokens=[t], score=s, caches=caches0, pos=pos0))
        finished = []
        
        for _ in range(max_new_tokens - 1):
            if not active:
                break
            candidates = []
            
            # Simple unbatched loop for clarity
            for b in active:
                last = torch.tensor([b.tokens[-1]], device=self.device, dtype=torch.long)
                _, new_caches, new_pos = self.model.encode_decode_with_cache(last, b.caches, b.pos)
                
                # Score next tokens from CLS proxy
                logits = self.model.head_policy(s0[:, :1, :]).squeeze(1)  # [1,V]
                logp = F.log_softmax(logits, dim=-1)
                topk_s, topk_i = torch.topk(logp, k=beam_width, dim=-1)
                
                for j in range(beam_width):
                    nid = int(topk_i[0, j].item())
                    sc = float(topk_s[0, j].item())
                    new_beam = Hypothesis(
                        tokens=b.tokens + [nid],
                        score=b.score + sc,
                        caches=new_caches,
                        pos=new_pos
                    )
                    
                    # Check for end token
                    if end_token:
                        token_text = self.tokenizer.decode([nid])
                        if end_token in token_text:
                            finished.append(new_beam)
                        else:
                            candidates.append(new_beam)
                    else:
                        candidates.append(new_beam)
            
            candidates.sort(key=lambda z: z.score, reverse=True)
            active = candidates[:beam_width]
        
        finished.extend(active)
        
        # Apply length penalty
        for b in finished:
            b.score = b.score / (len(b.tokens) ** max(1e-6, length_penalty))
            
        best = max(finished, key=lambda z: z.score)
        generated_text = self.tokenizer.decode(best.tokens, skip_special_tokens=True)
        return generated_text
    
    @torch.no_grad()
    def constrained_generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        end_token: Optional[str] = None
    ) -> str:
        """
        Generate text with constrained vocabulary (top-k tokens only).
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Number of top tokens to consider
            end_token: Optional token to stop generation
            
        Returns:
            Generated text continuation
        """
        # Tokenize prompt
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) > self.config.max_seq_len:
            prompt_ids = prompt_ids[-self.config.max_seq_len:]
            
        inp = torch.tensor([prompt_ids], device=self.device, dtype=torch.long)
        
        # Prefill rep stack
        s0, caches, pos = self.model.encode_prefill_with_cache(inp)
        
        generated_ids = []
        for _ in range(max_new_tokens):
            # Get logits from CLS
            logits = self.model.head_policy(s0[:, :1, :]).squeeze(1)  # [V]
            
            # Constrain to allowed vocabulary
            allowed_logits = logits.index_select(0, self.allowed_tensor)
            
            # Apply temperature
            if temperature != 1.0:
                allowed_logits = allowed_logits / temperature
                
            # Top-k filtering
            if top_k < len(allowed_logits):
                topk_logits, topk_indices = torch.topk(allowed_logits, top_k)
                # Create filtered logits
                filtered_logits = torch.full_like(allowed_logits, float('-inf'))
                filtered_logits[topk_indices] = topk_logits
                allowed_logits = filtered_logits
            
            # Sample
            probs = F.softmax(allowed_logits, dim=-1)
            token_idx = torch.multinomial(probs, 1).item()
            token_id = self.allowed_tensor[token_idx].item()
            
            generated_ids.append(token_id)
            
            # Check for end token
            if end_token:
                token_text = self.tokenizer.decode([token_id])
                if end_token in token_text:
                    break
                    
            # One-step decode
            _, caches, pos = self.model.encode_decode_with_cache(
                torch.tensor([token_id], device=self.device), caches, pos
            )
            
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text


# ============================== CLI Interface ============================== #
def main():
    parser = argparse.ArgumentParser(description="SLM Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--method", type=str, choices=["greedy", "beam", "constrained"], 
                       default="greedy", help="Generation method")
    parser.add_argument("--max_tokens", type=int, default=50, help="Maximum new tokens")
    parser.add_argument("--beam_width", type=int, default=4, help="Beam width for beam search")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k for constrained generation")
    parser.add_argument("--end_token", type=str, default=None, help="End token (e.g., '\\n')")
    parser.add_argument("--length_penalty", type=float, default=0.7, help="Length penalty for beam search")
    
    args = parser.parse_args()
    
    # Initialize inference
    print(f"Loading model from {args.model_path}...")
    inference = SLMInference(args.model_path)
    print("Model loaded successfully!")
    
    # Generate text
    print(f"\nPrompt: {args.prompt}")
    print(f"Method: {args.method}")
    print("-" * 50)
    
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
    
    print(f"Generated: {result}")
    print("-" * 50)
    print(f"Full text: {args.prompt}{result}")


if __name__ == "__main__":
    main()
