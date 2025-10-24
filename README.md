# Unified Evolutionary Software Engineering Agent (UE-SEA)

A self-improving software-engineering agent that understands large codebases, evolves code using evolutionary algorithms, and trains modular reasoning skills at scale.

## Architecture

UE-SEA consists of five main components:

1. **Code World Model & Tools (LOCAGENT)**: Graph-based code understanding with sparse indices
2. **Evolutionary Code Improver (AlphaEvolve)**: LLM ensemble with structured diffs and evaluators
3. **Reasoning Skills Curriculum**: SFT→RL training with temperature-entropy targeting
4. **Distributed Training Substrate**: Streaming DiLoCo + DiPaCo for scalable training
5. **Global Optimizer**: Evolution Strategies for robust parameter optimization

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize the system
python -m ue_sea.orchestrator --init

# Start the orchestrator
python -m ue_sea.orchestrator --start
```

## SPCT-Style Generative Reward Model (GRM) for Code

> **Note:** This extends the existing pipeline without altering the directory layout. Attach the reward/acceptance metadata during preprocessing and reuse our current training entrypoints (`training/train.py --stage {rft,rl}`) plus W&B logging.

### 0. Outcome

Build a pointwise GRM (LLM-as-a-judge) that, for each programming prompt and N candidate solutions, outputs:

1. Task-specific self-criteria (principles)
2. A concise critique
3. Integer scores in `[1,10]` per candidate

Training uses two stages: **Rejective Fine-Tuning (RFT/SFT)** followed by **rule-based online RL (GRPO/PPO)**. Inference-time scaling runs k parallel judge samples, optionally meta-filters them, and votes via score summation.

### 1. Data Normalisation

#### 1.1 Recommended sources

| Category | Datasets |
| --- | --- |
| Verifiable (unit tests) | BigCodeBench, MBPP / MBPP-Plus, APPS, CodeContests, DS-1000 |
| Critique / feedback | CodeFeedback, curated PR review comments (optional) |
| (Optional) General code chat | Code Q&A corpora for richer criteria vocabulary |

#### 1.2 Unified JSONL schema (per problem)

```json
{
  "task_id": "mbpp_0421",
  "language": "python",
  "prompt": "Write function ...",
  "candidates": [
    {"code": "def f(...): ...", "exec_pass": true,  "stderr": "", "tests_passed": 12, "tests_total": 12},
    {"code": "def f(...): ...", "exec_pass": false, "stderr": "Traceback...", "tests_passed": 0,  "tests_total": 12}
  ],
  "gold_best_index": 0,
  "meta": {"source": "MBPP", "difficulty": "easy"}
}
```

* `gold_best_index` = argmax by pass rate; break ties via fewer runtime errors then shorter code.
* Single-candidate problems become `(label = tests_passed == tests_total ? 1 : 0)`.

#### 1.3 Stage-specific splits

* **RFT**: all multi-candidate items plus some single-response, verifiable samples
* **RL**: same pool (or held-out subset) ensuring each record has `gold_best_index` or a binary label

### 2. Prompt template & JSON IO

**System message**

```
You are a code-quality judge. Score candidates strictly. Penalize hallucinations, unsafe code, and over-verbosity. Prefer correctness, robustness, clarity, and efficiency. Use integer scores 1..10. Keep critique ≤ 1400 chars total.
```

**User message**

```
TASK (language=${language}):
${prompt}

CANDIDATES (N=${len(candidates)}):
[BEGIN CANDIDATE 1]
<code block 1>
[END CANDIDATE 1]
...
[BEGIN CANDIDATE N]
<code block N>
[END CANDIDATE N]

Return STRICT JSON with keys:
- self_criteria: array of {"name": string, "weight": float, "rationale": string}; 3–5 items, weights sum≈1.0
- critique: short paragraph(s) comparing each candidate against all criteria
- scores: array<int> length N, each in [1..10], (same order as candidates)

Scoring policy hints (soft):
- Correctness (tests likely to pass), Robustness (edge-cases), Safety (no insecure/undefined behavior), Clarity (readability), Efficiency (time/space), Maintainability.
```

**Expected JSON**

```json
{
  "self_criteria": [
    {"name": "Correctness", "weight": 0.45, "rationale": "pass tests, handle edge-cases"},
    {"name": "Clarity", "weight": 0.20, "rationale": "readable, idiomatic"},
    {"name": "Efficiency", "weight": 0.20, "rationale": "avoid O(n^2) if O(n) possible"},
    {"name": "Safety", "weight": 0.15, "rationale": "no eval/exec, input checks"}
  ],
  "critique": "Cand#1 passes obvious cases ...",
  "scores": [9, 4, 7]
}
```

### 3. Stage A — Rejective Fine-Tuning (RFT)

1. Sample each item `N_RFT = 3` times at temperature 0.5; shuffle candidate order each time.
2. Parse JSON and reject invalid outputs.
3. Accept trace only if:
   * For `n ≥ 2`: `argmax(scores) == gold_best_index` (after unshuffle)
   * For `n == 1`: `(label==1 → score ≥ 6)` and `(label==0 → score ≤ 4)`
4. Optionally drop “too-easy” items where all sampled traces are correct.
5. Store accepted traces as SFT pairs `(prompt → JSON)` and fine-tune with teacher forcing.

**Typical hyperparameters**

| Setting | Value |
| --- | --- |
| LR | `5e-6` |
| Batch | 1024 tokens |
| Steps | ~1k |
| Optimiser | AdamW, weight decay `1e-2` |
| Base model | 7–27B code-specialised LLM |

Monitor JSON validity and exact-match winner accuracy on a dev split.

### 4. Stage B — Rule-based Online RL (GRPO/PPO)

**Reward**

```
if n ≥ 2:
    reward = +1 if winner_pred == gold_best_index else -1
else:
    reward = +1 if (label==1 and score>=6) or (label==0 and score<=4) else -1
```

Optional shaping: -0.25 for critique > 1400 chars; -0.25 if scores out of range or all equal. Standardise advantages per batch.

**Objective**

```
L_clip = E[min(rho * A, clip(rho, 1±ε) * A)]
L_kl   = β * KL(πθ || πref)   # πref = frozen RFT model
J      = L_clip - L_kl
```

| Parameter | 7–13B | 27B |
| --- | --- | --- |
| RL LR | 1e-6 → 6e-7 | 4e-7 |
| KL β | 0.005–0.02 | 0.08 |
| Group size | 4 | 4 |
| Generation | `temp=0.5`, `top_p=0.95`, max 700–900 tokens |

Use a strong KL to keep format stable. Stop when JSON validity ≥ 99%, dev accuracy improves, KL remains bounded, and critique length is stable.

### 5. Inference-time Scaling

1. Sample k judge runs with different seeds and shuffled ordering.
2. Discard invalid JSON, unshuffle, and compute `final[i] = Σ_j scores_j[i]`.
3. `argmax final[i]` selects the best candidate.
4. Optional meta-filter: small classifier over `(prompt, candidates, judge JSON)` to drop low-quality judge outputs before voting.

### 6. Implementation Notes

* Enforce strict JSON parsing and canonical shuffle/unshuffle mapping.
* Attach `accepted` / `reward` metadata during preprocessing—no runtime reward model required.
* Run `training/train.py --stage rft ...` for SFT and `--stage rl ... --kl-beta <value>` for RL (metrics logged to W&B as `rft/acceptance`, `rl/avg_reward`, KL, etc.).
* Keep critique concise; log critique length distribution to detect verbosity drift.

### 7. Evaluation and Monitoring

* Judge accuracy@1 on held-out preference items (`k=1`, `k=8`)
* ROC-AUC for single-response items
* k-scaling curve for k ∈ {1,2,4,8,16}
* Bias checks: correlation with code length, shuffle invariance, critique length, safety patterns
* Optional downstream: pass@k lift when using the judge to rank code-generation samples

### 8. Ablations

* Remove principle generation → expect drop in both greedy and k-scaled accuracy
* Compare hinted vs non-hinted RFT sampling
* Add reference snippets for verifiable tasks and measure accuracy gain

### 9. Practical Tips

* Use hidden tests to prevent overfitting
* Increase KL β if format or bias drifts
* Penalise overly long critiques to keep judge outputs compact
* Maintain the current repository structure—extend existing scripts instead of introducing new directories

## 🚀 LOCAGENT Implementation Status - COMPLETE!

### ✅ **IMPLEMENTATION COMPLETE** 
All core LOCAGENT components have been successfully implemented and are fully operational!

### **Core Implementation Files**
- `src/ue_sea/locagent/` - All core LOCAGENT components
- `tests/locagent/` - Comprehensive test suite

### **Clean Repository Structure**
```
src/ue_sea/locagent/
├── __init__.py              # Main package exports
├── agent.py                 # LOCAGENT_Agent implementation
├── cli.py                   # Command-line interface
├── data_loader.py           # Data loading utilities
├── entities.py              # Entity and Relation models
├── github_integration.py    # GitHub integration
├── graph.py                 # Graph construction
├── indices.py               # Indexing system
├── parser.py                # Code parsing utilities
├── tools.py                 # Agent tools
└── training.py              # Training pipeline

tests/locagent/
├── __init__.py
├── run_tests.py             # Test runner
├── test_agent.py            # Agent tests
├── test_entities.py        # Entity tests
├── test_github_integration.py # GitHub tests
├── test_indices.py          # Index tests
├── test_integration.py      # Integration tests
└── test_training.py        # Training tests
```

## Components

### LOCAGENT (Code World Model) ✅
- **Graph Builder**: Parses repositories into entity/relation graphs
- **Sparse Indices**: ID, name, and BM25 indices for fast retrieval
- **Tools**: SearchEntity, TraverseGraph, RetrieveEntity APIs
- **GitHub Integration**: Clone repositories and train on real issues
- **CLI Interface**: Complete command-line interface for all operations

### AlphaEvolve (Evolutionary Improver)
- **Prompt Sampler**: Builds rich prompts from parent solutions
- **LLM Ensemble**: Generates SEARCH/REPLACE diffs
- **Evaluators**: Cascaded machine-grade evaluation
- **Evolutionary DB**: MAP-elites selection and lineage tracking

### Reasoning Skills
- **Skill Paths**: Specialized modules for localization, verification, etc.
- **Curriculum**: SFT scaling → staged RL with GRPO
- **Entropy Control**: Temperature targeting ~0.3 for exploration/exploitation balance

### Distributed Training
- **Streaming DiLoCo**: Fragment-based synchronization with overlap
- **DiPaCo**: Modular path routing and discriminative re-sharding
- **Quantization**: FP4 outer gradients for bandwidth efficiency

### Evolution Strategies
- **Parameter Perturbation**: In-place layer-wise noise injection
- **Z-score Normalization**: Stable reward scaling across tasks
- **Greedy Decoding**: Deterministic evaluation for consistent gradients

## API Reference

### Graph Tools
```python
# Search for entities by keywords
entities = await search_entity(["csrf", "middleware"])

# Traverse the code graph
subgraph = await traverse_graph(
    start_ids=["django/middleware/csrf.py:CsrfViewMiddleware"],
    direction="both",
    hops=2
)

# Retrieve full entity details
details = await retrieve_entity(["django/shortcuts.py:redirect"])
```

### Evolution Controller
```python
# Generate evolutionary improvement
diff = await evolve(parent_id="prog_123", task_id="fix_456")

# Evaluate candidate
metrics = await evaluate(program_id="candidate_789", stage=1)

# Select elite
await select(program_id="candidate_789", score=0.873)
```

## Development

```bash
# Run tests
pytest

# Type checking
mypy src/

# Code formatting
black src/ tests/

# Linting
flake8 src/ tests/
```

## License

MIT License - see LICENSE file for details.
