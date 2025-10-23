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

## Components

### LOCAGENT (Code World Model)
- **Graph Builder**: Parses repositories into entity/relation graphs
- **Sparse Indices**: ID, name, and BM25 indices for fast retrieval
- **Tools**: SearchEntity, TraverseGraph, RetrieveEntity APIs

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
