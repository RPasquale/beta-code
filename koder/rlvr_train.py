# rlvr_train.py
# RL with Verifiable Rewards for K2-mini.
# Objective matches §3.2.3: sample K responses per prompt, compute rewards r(x,y_i),
# center by mean r̄(x), and include a log-ratio term τ * log π(yi|x)/π_old(yi|x) inside the squared bracket.
# Includes token-budget truncation (budget control) and a stub PTX loss hook (disabled by default).
# Kimi K2: §3.2.1 (verifiable rewards) and §3.2.3 (algorithm). 

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
# rlvr_train.py
from k2_mini import K2Mini, K2Config, MuonClip
from k2_config import k2_4090_fast
from deepcoder_data import DeepCoderAdapter, deepcoder_reward


def make_prompt(raw_prompt: str, entry: str = "solve") -> str:
    return (
        "You are a Python coding assistant.\n"
        f"Write valid Python 3 code that defines a function `{entry}` and returns the answer.\n"
        "Do not print; do not use input(); just return the result.\n"
        "Use only the standard library. Keep it concise.\n"
        "Task:\n"
        f"{raw_prompt}\n\n"
        "# Your code below:\n"
    )


def run_sft(model: K2Mini, tokenizer, data: DeepCoderAdapter, device: str, steps: int = 1000,
            lr: float = 5e-5, max_prompt_len: int = 512, max_label_len: int = 256):
    model.train()
    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    for step in range(steps):
        idx = random.randrange(len(data))
        ref = data.get_reference(idx)
        vi = ref.get("verification_info")
        entry = "solve"
        if isinstance(vi, dict) and isinstance(vi.get("entry"), str):
            entry = vi["entry"]
        prompt = make_prompt(data.get_prompt(idx), entry=entry)
        label = ref.get("solution") or ""
        text = prompt + label
        ids = tokenizer.encode(text, max_len=max_prompt_len + max_label_len)
        if not ids:
            continue
        input_ids = torch.tensor(ids, device=device).unsqueeze(0)
        labels = input_ids.clone()
        prompt_ids = tokenizer.encode(prompt, max_len=max_prompt_len)
        cutoff = len(prompt_ids)
        if cutoff > labels.size(1):
            cutoff = labels.size(1)
        if cutoff >= labels.size(1):
            continue
        labels[:, :cutoff] = -100
        attn = torch.ones_like(input_ids)
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=input_ids.is_cuda):
            _, aux, ce = model(input_ids, attention_mask=attn, labels=labels)
        if ce is None:
            continue
        if not torch.isfinite(ce):
            continue
        opt.zero_grad()
        ce.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 100 == 0:
            print(f"[SFT] step {step+1}/{steps} | loss={ce.item():.4f} | aux={aux.item():.4f}")
    del opt


def quick_eval(model: K2Mini, tokenizer, data: DeepCoderAdapter, device: str,
               max_prompt_len: int = 512, max_gen_len: int = 128, n: int = 32) -> float:
    model.eval()
    good = 0
    for _ in range(n):
        idx = random.randrange(len(data))
        ref = data.get_reference(idx)
        vi = ref.get("verification_info")
        entry = "solve"
        if isinstance(vi, dict) and isinstance(vi.get("entry"), str):
            entry = vi["entry"]
        prompt = make_prompt(data.get_prompt(idx), entry=entry)
        input_ids, attn, prompt_lens = prepare_batch(tokenizer, [prompt], max_prompt_len, device)
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=input_ids.is_cuda):
            gen = model.generate(input_ids, max_new_tokens=max_gen_len, temperature=0.7, top_p=0.95)
        gen_only = gen[0, prompt_lens[0]:].tolist()
        code = tokenizer.decode(gen_only)
        reward = deepcoder_reward(code, ref)
        good += int(reward > 0.0)
    model.train()
    return good / max(1, n)

# -------- Tokenizer ----------
# For simplicity, use a byte-level fallback tokenizer (you can switch to HF tokenizers easily).
class ByteTokenizer:
    def __init__(self, vocab_size=32768):
        # naive byte tokenizer
        self.vocab_size = vocab_size
    def encode(self, text: str, max_len: int = None) -> List[int]:
        ids = [c for c in text.encode("utf-8")]
        if max_len is not None:
            ids = ids[:max_len]
        return ids
    def decode(self, ids: List[int]) -> str:
        return bytes([i % 256 for i in ids]).decode("utf-8", errors="ignore")

# -------- RL Config ----------
@dataclass
class RLConfig:
    lr: float = 1e-4
    weight_decay: float = 0.1
    batch_size: int = 2
    max_prompt_len: int = 512
    max_gen_len: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    K: int = 4                          # samples per prompt
    tau_reg: float = 0.005              # τ in paper objective (smaller is gentler)
    token_budget: int = 512             # "budget control" (§3.2.3)
    steps: int = 200
    use_muon_like: bool = False         # turn on to try Muon-like precond (heavy)
    qk_clip_tau: float = 100.0          # QK-Clip threshold

def prepare_batch(tokenizer, prompts: List[str], max_prompt_len: int, device):
    ids = [tokenizer.encode(p, max_len=max_prompt_len) for p in prompts]
    lengths = [len(x) for x in ids]
    maxlen = max(lengths)
    B = len(ids)
    input_ids = torch.full((B, maxlen), 0, dtype=torch.long, device=device)
    attn = torch.zeros((B, maxlen), dtype=torch.long, device=device)
    for i, seq in enumerate(ids):
        L = len(seq)
        input_ids[i, :L] = torch.tensor(seq, device=device)
        attn[i, :L] = 1
    return input_ids, attn, lengths

def sample_and_logprobs(model: K2Mini, input_ids, K, max_new, temperature, top_p):
    """
    Returns lists (length B) of generated token sequences, their logprobs, and the concatenated input->output for training.
    """
    B = input_ids.size(0)
    all_gen_tokens = []
    all_logprobs = []
    all_concat_ids = []
    for _ in range(K):
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=input_ids.is_cuda):
                gen_ids = model.generate(input_ids, max_new_tokens=max_new, temperature=temperature, top_p=top_p)
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=input_ids.is_cuda):
            logp = model.logprobs(gen_ids)
        all_gen_tokens.append(gen_ids)
        all_logprobs.append(logp)
        all_concat_ids.append(gen_ids)
    return all_gen_tokens, all_logprobs, all_concat_ids

def main():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = k2_4090_fast       # <-- start with the fast preset; switch to k2_4090_top8 after it runs
    rl = RLConfig()
    rl.batch_size = 1        # <= keep small on 24 GB
    rl.K = 2
    rl.max_gen_len = 128
    rl.token_budget = 128

    # ----- DATA -----
    DATASET_DIR = r"C:\Users\Admin\.cache\huggingface\datasets\PrimeIntellect___deepcoder-gold-standard-solutions\default\0.0.0"
    data = DeepCoderAdapter(DATASET_DIR)

    # ----- MODEL -----
    model = K2Mini(cfg).to(device)  # your model class; same file. :contentReference[oaicite:4]{index=4}
    # BF16 weights for memory; keep norms in FP32 for stability.
    model.to(dtype=torch.bfloat16)
    for m in model.modules():
        if m.__class__.__name__ in ("RMSNorm",):
            m.to(dtype=torch.float32)
    tokenizer = ByteTokenizer(vocab_size=cfg.vocab_size)

    print("Starting short SFT warm-start...")
    run_sft(
        model,
        tokenizer,
        data,
        device,
        steps=500,
        lr=5e-5,
        max_prompt_len=rl.max_prompt_len,
        max_label_len=rl.max_gen_len,
    )
    print("SFT done. Switching to RL.")

    model.train()

    base_optim = AdamW(model.parameters(), lr=rl.lr, weight_decay=rl.weight_decay)
    optim = MuonClip(model, base_optim, tau=cfg.qk_clip_tau, use_muon_like=rl.use_muon_like)

    model.train()

    for step in range(rl.steps):
        # sample a mini-batch of problems
        idxs = random.sample(range(len(data)), rl.batch_size)
        prompts = []
        refs = []
        for i in idxs:
            ref = data.get_reference(i)
            vi = ref.get("verification_info")
            entry = "solve"
            if isinstance(vi, dict) and isinstance(vi.get("entry"), str):
                entry = vi["entry"]
            raw = data.get_prompt(i)
            prompts.append(make_prompt(raw, entry=entry))
            refs.append(ref)

        input_ids, attn, prompt_lens = prepare_batch(tokenizer, prompts, rl.max_prompt_len, device)

        # SAMPLE from π_old (current model in practice; for multi-iter you’d snapshot π_old)
        gen_seqs, seq_logps, concat_ids = sample_and_logprobs(
            model, input_ids, rl.K, rl.max_gen_len, rl.temperature, rl.top_p
        )

        # build rewards per (B,K)
        rewards = torch.zeros((rl.batch_size, rl.K), device=device)
        for ki in range(rl.K):
            # decode only the *generated* suffix for verification
            for b in range(rl.batch_size):
                prompt_len = prompt_lens[b]
                gen_only_ids = gen_seqs[ki][b, prompt_len:].tolist()
                code = tokenizer.decode(gen_only_ids)
                r = deepcoder_reward(code, refs[b])
                rewards[b, ki] = r

        # mean-center per prompt
        rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
        r_bar = rewards.mean(dim=1, keepdim=True)   # (B,1)
        advantages = rewards - r_bar                # (B,K)
        mean_reward = rewards.mean().item()

        # Compute log π(yi|x) for each sample and build objective:
        # L = E[ ( r - r̄ - τ * log π(yi|x)/π_old(yi|x) )^2 ] ; with π_old=π (single policy variant) => -τ * log π(yi|x)
        obj = torch.zeros((), device=device, dtype=torch.float32)
        count = 0
        for ki in range(rl.K):
            lp = seq_logps[ki].float()  # (B, L-1)
            lp = torch.nan_to_num(lp, nan=0.0, posinf=0.0, neginf=-50.0)
            if lp.numel() == 0:
                continue
            # Apply token budget truncation (§3.2.3): cap tokens used in the squared term by budget
            # Compute mean logp per sample up to the budget suffix
            tokens_slice = lp[:, -min(lp.size(1), rl.token_budget):]
            used = tokens_slice.mean(dim=-1)  # (B,)
            used = torch.nan_to_num(used, nan=0.0, posinf=0.0, neginf=-50.0)
            term = (advantages[:, ki] - rl.tau_reg * used).pow(2).mean()
            term = torch.nan_to_num(term, nan=0.0, posinf=1e4, neginf=0.0)
            obj = obj + term
            count += 1
        obj = obj / max(1, count)

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=input_ids.is_cuda):
            _, aux_loss, _ = model(input_ids, attention_mask=attn)
        aux_loss = aux_loss.float()
        if not torch.isfinite(obj):
            print("Non-finite obj detected; skipping step.")
            continue
        if not torch.isfinite(aux_loss):
            print("Non-finite aux_loss detected; skipping step.")
            continue
        loss = obj + 1e-3 * aux_loss
        if not torch.isfinite(loss):
            print(f"Non-finite loss detected; obj={obj.item()} aux={aux_loss.item()} rewards_mean={mean_reward:.4f}")
            continue

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if (step + 1) % 20 == 0:
            sample_prompt = prompts[0][:400]
            sample_gen = tokenizer.decode(gen_seqs[0][0, prompt_lens[0]:].tolist())[:600]
            print("\n--- SAMPLE ---")
            print("PROMPT:\n", sample_prompt, "...")
            print("GEN:\n", sample_gen, "...\n")

        if (step + 1) % 10 == 0:
            print(f"step {step+1}/{rl.steps} | loss={loss.item():.4f} | obj={obj.item():.4f} | meanR={mean_reward:.4f}")

        if (step + 1) % 100 == 0:
            torch.save(
                {"model": model.state_dict(), "step": step + 1},
                f"ckpt_rl_step_{step+1}.pt",
            )
            eval_pass = quick_eval(
                model,
                tokenizer,
                data,
                device,
                max_prompt_len=rl.max_prompt_len,
                max_gen_len=rl.max_gen_len,
                n=32,
            )
            print(f"[EVAL] pass@1 ≈ {eval_pass:.2f}")

if __name__ == "__main__":
    main()
