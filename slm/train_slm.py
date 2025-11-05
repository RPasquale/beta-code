# train_slm.py - Training functions and main training loop
import math
import random
from typing import List, Tuple
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from .slm import (
    Config, MuZeroTransformer, MuonClip, patch_model_for_qk_logging,
    set_seed, topk_by_frequency, grad_summary, causal_mask
)


# ============================== Training Components ============================== #
class NextTokenEnv:
    def __init__(self, token_ids: List[int], vocab_allowed: List[int], max_seq_len: int):
        self.ids = token_ids
        self.allowed = vocab_allowed
        self.max_seq_len = max_seq_len
        center = len(self.ids) // 2
        if center == 0 and len(self.ids) > 1:
            center = 1
        self.pos = min(max_seq_len // 2, center)
        self.pos = min(self.pos, max(len(self.ids) - 1, 0))

    def get_obs(self) -> List[int]:
        s = max(0, self.pos - self.max_seq_len)
        return self.ids[s:self.pos]

    def step(self, action_id: int):
        if self.pos >= len(self.ids) - 1:
            return 0, True
        gt = self.ids[self.pos]
        r = 1 if action_id == gt else 0
        self.pos += 1
        return r, self.pos >= len(self.ids) - 1


class Replay:
    def __init__(self, cap):
        self.buf = []
        self.cap = cap
    def add(self, x):
        if len(self.buf) >= self.cap:
            self.buf.pop(0)
        self.buf.append(x)
    def sample(self, bs):
        return random.sample(self.buf, bs)
    def __len__(self): return len(self.buf)


class CosineWarmup:
    def __init__(self, optimizer, warmup, total_steps, base_lr):
        self.opt = optimizer
        self.warmup = warmup
        self.total = total_steps
        self.base = base_lr
        self.s = 0
        
    def step(self):
        self.s += 1
        if self.s <= self.warmup:
            lr = self.base * self.s / max(1, self.warmup)
        else:
            t = (self.s - self.warmup) / max(1, self.total - self.warmup)
            lr = 0.5 * self.base * (1 + math.cos(math.pi * t))
        for g in self.opt.param_groups:
            g["lr"] = lr


def current_bc_coef(step, cfg: Config):
    """Linear warm-down of BC coefficient"""
    warmdown = int(0.1 * cfg.train_steps)  # 10% of training
    if step < warmdown:
        return cfg.bc_coef * (1 - step / warmdown)
    return 0.0


# ============================== Training Functions ============================== #
@torch.no_grad()
def run_mcts(net, s_root, allowed, cfg):
    p_logits, _ = net.f(s_root)               # (1,V)
    p_logits = p_logits.squeeze(0)
    p_allowed = p_logits.index_select(0, allowed)
    probs = F.softmax(p_allowed, dim=-1)      # soft π target
    a_idx = torch.multinomial(probs, 1).item()
    action = allowed[a_idx].item()
    return action, probs.detach().cpu().tolist()


def nstep_bootstrap(rews: List[int], values: List[float], gamma: float, n: int, start: int) -> float:
    G = 0.0
    T = len(rews)
    for i in range(n):
        t = start + i
        if t < T: G += (gamma ** i) * rews[t]
        else: break
    tB = start + n
    if tB < len(values): G += (gamma ** n) * values[tB]
    return G


def muzero_loss(batch, net, cfg: Config, allowed, bc_coef=None):
    B = len(batch)
    device = cfg.device
    # pack obs
    maxlen = max(len(x["obs"]) for x in batch)
    obs = torch.full((B, min(cfg.max_seq_len, maxlen)), 0, dtype=torch.long, device=device)
    for i, x in enumerate(batch):
        toks = x["obs"][-cfg.max_seq_len:]
        obs[i, -len(toks):] = torch.tensor(toks, device=device)

    s = net.h(obs)
    v_losses, r_losses, p_losses, bc_losses = [], [], [], []

    # precompute values for bootstrap
    with torch.no_grad():
        p0, v0 = net.f(s)
    values_cache = [v0.detach()]

    # unroll
    for k in range(cfg.unroll_K):
        # pack step targets
        a_k, r_true, gt_ids = [], [], []
        for i in range(B):
            if k < len(batch[i]["actions"]):
                a_k.append(batch[i]["actions"][k])
                r_true.append(batch[i]["rewards"][k])
                gt_ids.append(batch[i]["gt_ids"][k])
            else:
                a_k.append(0); r_true.append(0); gt_ids.append(-1)

        a_k = torch.tensor(a_k, device=device, dtype=torch.long)
        r_true = torch.tensor(r_true, device=device, dtype=torch.float)
        gt_ids_t = torch.tensor(gt_ids, device=device, dtype=torch.long)

        # one-step
        r_logit, s = net.g(s, a_k)
        p_pred, v_pred = net.f(s)                   # (B,V), (B,)
        values_cache.append(v_pred.detach())

        # reward
        r_losses.append(F.binary_cross_entropy_with_logits(r_logit, r_true, reduction="mean"))

        # value (n-step bootstrap from k)
        z_k = []
        for i in range(B):
            if k < len(batch[i]["rewards"]):
                z = nstep_bootstrap(batch[i]["rewards"],
                                    [v[i].item() for v in values_cache],
                                    cfg.gamma, cfg.n_step_bootstrap, k)
                z_k.append(z)
            else:
                z_k.append(0.0)
        z_k = torch.tensor(z_k, device=device, dtype=torch.float)
        v_losses.append(F.mse_loss(v_pred, z_k))

        # policy (search-improved; restricted to allowed)
        p_allowed = p_pred.index_select(1, allowed)              # (B,A)
        pi_targets = torch.tensor(
            [batch[i]["mcts_pi"][k] if k < len(batch[i]["mcts_pi"]) else [0]*len(allowed)
             for i in range(B)], device=device, dtype=torch.float
        )
        logp = F.log_softmax(p_allowed, dim=-1)
        p_losses.append(-(pi_targets * logp).sum(dim=-1).mean())

        # behavior cloning warm-start (full vocab) if enabled
        current_bc = bc_coef if bc_coef is not None else cfg.bc_coef
        if current_bc > 0:
            mask = (gt_ids_t >= 0)
            if mask.any():
                bc_losses.append(F.cross_entropy(p_pred[mask], gt_ids_t[mask]))

    loss_v = sum(v_losses)/max(1,len(v_losses))
    loss_r = sum(r_losses)/max(1,len(r_losses))
    loss_p = sum(p_losses)/max(1,len(p_losses))
    loss_bc = (sum(bc_losses)/max(1,len(bc_losses))) if bc_losses else torch.tensor(0.0, device=device)
    current_bc = bc_coef if bc_coef is not None else cfg.bc_coef
    return loss_v + loss_r + loss_p + current_bc * loss_bc, (loss_v.item(), loss_r.item(), loss_p.item(), loss_bc.item())


def drgrpo_step(
    net, opt,
    tokenized: List[List[int]],
    allowed: torch.Tensor,
    allowed_set: set,
    tok,
    cfg: Config,
    step_idx: int,
    USE_AMP: bool = False,
    AMP_DTYPE = None,
    scaler = None,
    MAX_NORM: float = 1.0
):
    device = cfg.device
    net.train()
    opt.zero_grad(set_to_none=True)

    allowed_cpu = allowed.detach().cpu().tolist()
    EPS_ADV = 0.05  # small shaping to break all-zero advantage ties

    # 1) sample contexts
    contexts = []
    for _ in range(cfg.drgrpo_batch):
        seq = random.choice(tokenized)
        pos = len(seq) // 2
        pos = min(max(cfg.max_seq_len // 2, 1), max(len(seq)-1, 1), pos)
        while pos < len(seq) - 1 and seq[pos] not in allowed_set:
            pos += 1
        contexts.append((seq, pos))

    # 2) collect rollouts
    # Each group = list of (obs_tokens, a_global, logp_old, r, gt)
    batch_groups = []
    actionable_steps = 0
    token_terms = 0

    for (seq, start_pos) in contexts:
        ctx_groups = []
        for _ in range(cfg.drgrpo_G):
            pos = start_pos
            traj = []
            for t in range(cfg.drgrpo_T):
                if pos >= len(seq) - 1: break
                gt = seq[pos]

                # Debug logging for GT coverage
                if (t == 0) and (cfg.debug_level >= 2):
                    print("GT outside allowed:", gt not in allowed_set)

                obs_tokens = seq[max(0, pos-cfg.max_seq_len):pos]
                obs_t = torch.tensor([obs_tokens], device=device, dtype=torch.long)
                s = net.h(obs_t)
                with torch.no_grad():
                    p_logits, _ = net.f(s)               # (1,V)
                    # Per-step candidate set: allowed ∪ {gt} (if requested)
                    if cfg.include_gt_in_aux and (gt not in allowed_set):
                        allowed_step = torch.tensor(allowed_cpu + [gt], device=device, dtype=torch.long)
                    else:
                        allowed_step = allowed

                    p_allowed = p_logits.index_select(1, allowed_step)  # (1,A_step)
                    if cfg.aux_temp != 1.0:
                        p_allowed = p_allowed / cfg.aux_temp
                    probs = F.softmax(p_allowed, dim=-1)
                    a_local = torch.multinomial(probs, 1).item()
                    logp_old = torch.log(probs[0, a_local] + 1e-12).item()
                    a_global = int(allowed_step[a_local].item())
                r = 1 if a_global == gt else 0
                traj.append((obs_tokens, a_global, logp_old, r, gt))
                actionable_steps += 1
                pos += 1
            ctx_groups.append(traj)
        batch_groups.append(ctx_groups)

    # 3) compute PPO loss with either per-step or per-group A
    losses = []
    ent_terms = []

    if cfg.per_step_adv:
        # compute mean reward per (ctx,t)
        for ctx_groups in batch_groups:
            L = max((len(traj) for traj in ctx_groups), default=0)
            if L == 0:
                continue
            # mean r_t across groups
            means = []
            for t in range(L):
                vals = [traj[t][3] for traj in ctx_groups if t < len(traj)]
                means.append(sum(vals)/len(vals) if vals else 0.0)

            for traj in ctx_groups:
                for t, step in enumerate(traj):
                    A_t = step[3] - means[t]  # r_t - mean_t
                    obs_tokens, a_global, logp_old, _, gt = step
                    
                    # Add exploration advantage to break ties
                    if A_t == 0.0:
                        A_t = EPS_ADV if (a_global == gt) else -EPS_ADV
                    
                    # current policy
                    obs_t = torch.tensor([obs_tokens], device=device, dtype=torch.long)
                    s = net.h(obs_t)
                    p_logits, _ = net.f(s)
                    if cfg.include_gt_in_aux and (gt not in allowed_set):
                        allowed_step = torch.tensor(allowed_cpu + [gt], device=device, dtype=torch.long)
                    else:
                        allowed_step = allowed
                    p_allowed = p_logits.index_select(1, allowed_step)
                    if cfg.aux_temp != 1.0:
                        p_allowed = p_allowed / cfg.aux_temp
                    probs = F.softmax(p_allowed, dim=-1)
                    
                    # try to find local idx of a_global (may fail in rare edge cases)
                    try:
                        a_local = (allowed_step == a_global).nonzero(as_tuple=False).flatten()[0].item()
                    except Exception:
                        a_local = None
                    
                    if a_local is not None:
                        logp_new = torch.log(probs[0, a_local] + 1e-12)
                        ratio = torch.exp(logp_new - torch.tensor([logp_old], device=device))
                        unclipped = ratio * A_t
                        clipped   = torch.clamp(ratio, 1 - cfg.drgrpo_clip, 1 + cfg.drgrpo_clip) * A_t
                        losses.append(-torch.min(unclipped, clipped))
                        token_terms += 1
                    
                    # Even when A_t == 0, still push entropy to avoid 'skip'
                    if cfg.aux_entropy_coef > 0:
                        ent_terms.append(-(probs * (probs.clamp_min(1e-12).log())).sum())

    else:
        # per-group advantage: A_i = R_i - mean(R)
        centered_adv = []
        for ctx_groups in batch_groups:
            returns = [sum(step[3] for step in traj) for traj in ctx_groups]
            if len(returns) == 0:
                centered_adv.append([])
            else:
                mR = sum(returns)/len(returns)
                centered_adv.append([R - mR for R in returns])

        for ctx_idx, ctx_groups in enumerate(batch_groups):
            A_list = centered_adv[ctx_idx]
            for traj, A_i in zip(ctx_groups, A_list):
                # Add exploration advantage to break ties
                if A_i == 0.0:
                    # Use average reward across trajectory to determine direction
                    avg_reward = sum(step[3] for step in traj) / len(traj) if traj else 0
                    A_i = EPS_ADV if avg_reward > 0 else -EPS_ADV
                
                for (obs_tokens, a_global, logp_old, _, gt) in traj:
                    # current policy
                    obs_t = torch.tensor([obs_tokens], device=device, dtype=torch.long)
                    s = net.h(obs_t)
                    p_logits, _ = net.f(s)
                    if cfg.include_gt_in_aux and (gt not in allowed_set):
                        allowed_step = torch.tensor(allowed_cpu + [gt], device=device, dtype=torch.long)
                    else:
                        allowed_step = allowed
                    p_allowed = p_logits.index_select(1, allowed_step)
                    if cfg.aux_temp != 1.0:
                        p_allowed = p_allowed / cfg.aux_temp
                    probs = F.softmax(p_allowed, dim=-1)
                    
                    try:
                        a_local = (allowed_step == a_global).nonzero(as_tuple=False).flatten()[0].item()
                    except Exception:
                        a_local = None
                    
                    if a_local is not None:
                        logp_new = torch.log(probs[0, a_local] + 1e-12)
                        ratio = torch.exp(logp_new - torch.tensor([logp_old], device=device))
                        unclipped = ratio * A_i
                        clipped   = torch.clamp(ratio, 1 - cfg.drgrpo_clip, 1 + cfg.drgrpo_clip) * A_i
                        losses.append(-torch.min(unclipped, clipped))
                        token_terms += 1
                    
                    # Always add entropy term
                    if cfg.aux_entropy_coef > 0:
                        ent_terms.append(-(probs * (probs.clamp_min(1e-12).log())).sum())

    if len(losses) == 0:
        if cfg.debug_level >= 1:
            print(f"[Dr.GRPO] skip: token_terms=0 | actionable_steps={actionable_steps}")
        return 0.0

    aux_loss = torch.stack(losses).mean()
    if ent_terms:
        aux_loss = aux_loss - cfg.aux_entropy_coef * torch.stack(ent_terms).mean()

    loss_div = aux_loss  # typically do one step per DR-GRPO call
    if USE_AMP and AMP_DTYPE==torch.float16:
        scaler.scale(loss_div).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), MAX_NORM)
        scaler.step(opt)
        opt._apply_qkclip_on_model(net)
        scaler.update()
    else:
        loss_div.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), MAX_NORM)
        opt.step_with_qkclip(net)

    gnorm, gcount = grad_summary(net)

    if cfg.debug_level >= 1:
        print(f"[Dr.GRPO] aux loss={aux_loss.item():.4f} | token_terms={token_terms} "
              f"actionable_steps={actionable_steps} grad_norm={gnorm:.3f} nonzero_grad_elems={int(gcount)}")

    return float(aux_loss.item())


# ============================== Data Loading ============================== #
def load_data(cfg: Config):
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Use more data for scaled experiment
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:10%]")  # 10x more data
    texts = [t["text"] for t in ds if t["text"]]
    tokenized = [
        tok(t, truncation=True, max_length=cfg.max_seq_len + 10, add_special_tokens=False)["input_ids"]
        for t in texts if t.strip()
    ]
    min_len = max(cfg.eval_steps + 2, cfg.rollout_steps + 2)
    tokenized = [x for x in tokenized if len(x) > min_len]
    if cfg.train_samples and len(tokenized) > cfg.train_samples:
        tokenized = tokenized[:cfg.train_samples]
    # build allowed
    vocab_allowed = topk_by_frequency(tokenized, tok, cfg.vocab_topk)
    allowed = torch.tensor(vocab_allowed, device=cfg.device, dtype=torch.long)
    allowed_set = set(vocab_allowed)
    return tok, tokenized, allowed, allowed_set


# ============================== Main Training Loop ============================== #
def main():
    cfg = Config()
    tok, tokenized, allowed, allowed_set = load_data(cfg)

    net = MuZeroTransformer(vocab_size=tok.vocab_size, cfg=cfg).to(cfg.device)

    # Patch once so we can measure attention logits during forwards (for QK-Clip)
    patch_model_for_qk_logging(net)
    
    # Optional: thin logging overhead
    from .slm import MultiHeadSelfAttention
    for m in net.modules():
        if isinstance(m, MultiHeadSelfAttention):
            m._muonclip_log_every = 2

    # Use MuonClip as the optimizer wrapper
    opt = MuonClip(
        net.parameters(),
        base_opt_ctor=torch.optim.AdamW,
        base_opt_kwargs=dict(lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=(0.9, 0.999), eps=1e-8),
        qk_tau=90.0,  # slightly lower threshold for stability
        qk_alpha=0.6  # bias toward Q scaling
    )
    
    # Create cosine scheduler
    sched = CosineWarmup(opt.base_opt, warmup=cfg.warmup_steps, total_steps=cfg.train_steps, base_lr=cfg.learning_rate)
    
    # Optional: compile for speed (Linux/macOS only)
    import platform
    if hasattr(torch, "compile") and platform.system() != "Windows":
        net = torch.compile(net, mode="reduce-overhead")
        print("Model compiled with torch.compile")
    elif platform.system() == "Windows":
        print("torch.compile not supported on Windows, skipping compilation")

    replay = Replay(cfg.replay_size)

    # AMP and gradient accumulation setup
    USE_AMP = (cfg.use_mixed_precision and (torch.cuda.is_available() or torch.backends.mps.is_available()))
    AMP_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp_autocast = (
        torch.autocast(device_type="cuda", dtype=AMP_DTYPE) if torch.cuda.is_available()
        else torch.autocast(device_type="cpu", dtype=AMP_DTYPE)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and AMP_DTYPE==torch.float16)
    ACCUM = max(1, getattr(cfg, "gradient_accumulation_steps", 1))
    MAX_NORM = getattr(cfg, "max_grad_norm", 1.0)

    print(f"Running on device: {cfg.device}")
    print(f"AMP enabled: {USE_AMP}, dtype: {AMP_DTYPE}, grad_accum: {ACCUM}")

    print("Collecting and training ...")
    for step in tqdm(range(cfg.train_steps)):
        # --- Collect one short episode (actionable steps only)
        seq = random.choice(tokenized)
        env = NextTokenEnv(seq, allowed.detach().cpu().tolist(), cfg.max_seq_len)

        # move to actionable start
        while env.pos < len(seq) - 1 and seq[env.pos] not in allowed_set:
            env.pos += 1

        init_obs = env.get_obs()
        acts, rews, pis, gt_ids = [], [], [], []

        for _ in range(cfg.rollout_steps):
            if env.pos >= len(seq) - 1:
                break
            gt = seq[env.pos]
            if gt not in allowed_set:
                env.pos += 1
                continue
            obs_t = torch.tensor([env.get_obs()[-cfg.max_seq_len:]], device=cfg.device)
            s0 = net.h(obs_t)
            a, pi = run_mcts(net, s0, allowed, cfg)
            r, done = env.step(a)
            acts.append(a); rews.append(r); pis.append(pi); gt_ids.append(gt)
            if done:
                break

        if acts:
            replay.add({
                "obs": init_obs,
                "actions": acts[:cfg.unroll_K],
                "rewards": rews[:cfg.unroll_K],
                "mcts_pi": pis[:cfg.unroll_K],
                "gt_ids": gt_ids[:cfg.unroll_K],
            })

        # --- MuZero update
        if len(replay) >= cfg.batch_size:
            batch = replay.sample(cfg.batch_size)
            
            if (step % ACCUM) == 0:
                opt.zero_grad(set_to_none=True)

            with (amp_autocast if USE_AMP else nullcontext()):
                current_bc = current_bc_coef(step, cfg)
                loss, parts = muzero_loss(batch, net, cfg, allowed, bc_coef=current_bc)
                loss_to_backprop = loss / ACCUM

            if USE_AMP and AMP_DTYPE==torch.float16:
                scaler.scale(loss_to_backprop).backward()
                if ((step + 1) % ACCUM) == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), MAX_NORM)
                    scaler.step(opt)
                    opt._apply_qkclip_on_model(net)
                    scaler.update()
                    sched.step()  # update learning rate
            else:
                loss_to_backprop.backward()
                if ((step + 1) % ACCUM) == 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), MAX_NORM)
                    opt.step_with_qkclip(net)
                    sched.step()  # update learning rate
            
            if (step + 1) % cfg.log_every == 0:
                lv, lrw, lp, lbc = parts
                print(f"[MuZero] step {step + 1}: loss={loss.item():.4f} "
                      f"(v={lv:.4f}, r={lrw:.4f}, p={lp:.4f}, bc={lbc:.4f}) replay={len(replay)}")

        # --- Dr.GRPO aux update
        if (step + 1) % cfg.drgrpo_every == 0:
            _ = drgrpo_step(net, opt, tokenized, allowed, allowed_set, tok, cfg, step_idx=step+1,
                           USE_AMP=USE_AMP, AMP_DTYPE=AMP_DTYPE, scaler=scaler, MAX_NORM=MAX_NORM)

    # --- Quick eval
    print(f"Quick eval on {cfg.eval_samples} samples...")
    accs = []
    with torch.no_grad():
        for _ in range(cfg.eval_samples):
            seq = random.choice(tokenized)
            env = NextTokenEnv(seq, allowed.detach().cpu().tolist(), cfg.max_seq_len)
            while env.pos < len(seq) - 1 and seq[env.pos] not in allowed_set:
                env.pos += 1

            correct, total = 0, 0
            for _ in range(cfg.eval_steps):
                if env.pos >= len(seq) - 1: break
                gt = seq[env.pos]
                if gt not in allowed_set:
                    env.pos += 1; continue
                obs_t = torch.tensor([env.get_obs()[-cfg.max_seq_len:]], device=cfg.device)
                s0 = net.h(obs_t)
                a, _ = run_mcts(net, s0, allowed, cfg)
                correct += int(a == gt)
                total += 1
                _, done = env.step(a)
                if done: break
            if total:
                accs.append(correct / total)
    print(f"Mean step accuracy: {sum(accs)/len(accs):.3f}" if accs else "No evaluation episodes produced steps.")


if __name__ == "__main__":
    set_seed(42)
    main()
