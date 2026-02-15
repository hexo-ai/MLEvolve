# AutoMLGen Paper vs MLEvolve Code: Key Discrepancies

## Context

MLEvolve is described as "an advanced version based on [AutoMLGen](https://arxiv.org/abs/2510.08511)." This document compares what the AutoMLGen paper (arXiv:2510.08511) describes versus what the MLEvolve codebase actually implements, highlighting discrepancies users should be aware of.

---

## 1. Performance & LLM Backend

| Aspect | AutoMLGen Paper | MLEvolve Code |
|--------|----------------|---------------|
| Medal rate (MLE-bench) | 36.4% avg medal rate | 61.33% avg medal rate |
| LLM | DeepSeek-R1-0528 | Gemini-3-Pro-Preview (config default) |
| Temperature | 0.5 | 1.0 |
| Max draft nodes from root | 7 | 5 (`num_drafts` in config) |

The paper reports results with DeepSeek-R1; the code ships configured for Gemini-3-Pro-Preview. The temperature is doubled (0.5 → 1.0), likely to increase solution diversity.

---

## 2. Operator Set Differences

### Paper describes 3 Improve variants; Code has a single unified Improve agent

The paper defines fine-grained improve operators:
- **Improve-Normal** — small adjustments like switching optimizers
- **Improve-FE** — feature engineering focus
- **Improve-CS** — competition strategies from knowledge base

The MLEvolve code (`agents/improve_agent.py`) has a **single `improve_agent.run()`** function. There is no explicit routing to FE or CS sub-variants. Instead, the prompt broadly asks for "advanced architectures, ensemble techniques, feature engineering, hyperparameter optimization." The specialization described in the paper is collapsed into one general-purpose improve operator.

### Paper describes an Ensemble operator; Code has no Ensemble

The paper (Section 3.3.3) describes a dedicated **Ensemble operator** that combines Top-K solutions near termination into a robust final solution. The MLEvolve codebase has **no ensemble operator** in `engine/` or `agents/`. There is a `utils/submission_fusion_utils.py` file, but it is a standalone utility, not integrated as a search operator.

### Code adds an Evolution agent not in the paper

MLEvolve introduces `agents/evolution_agent.py` — an **intra-branch evolution** operator triggered on stagnation (before the time threshold for fusion). The paper's "intra-branch evolution" (Section 3.3.2, Eq. 6) is described as a mode of expansion within MCGS, not a separate operator. In the code, it is a distinct agent with its own prompts and logic.

---

## 3. Knowledge Base: Paper has 3 Dimensions; Code has Only Model-Level

The paper describes an ML domain knowledge base across **three dimensions**:
1. **Model-level** — model recommendations by domain
2. **Data-level** — modality-specific preprocessing and feature engineering principles
3. **Strategy-level** — competition tricks (TTA, ensembling, pseudo-labeling)

The MLEvolve codebase's cold-start system (`engine/coldstart/`) contains:
- `models_guidance_classified.json` — model recommendations (model-level only)
- `competition_tag_classified.json` — task classification tags

**Data-level and strategy-level knowledge bases are absent** from the code. The paper's `kb_init_ref_prob = 0.8` heuristic probability for KB reference at initialization is also not present in the config (cold-start is a simple boolean `use_coldstart: True`).

---

## 4. Experience-Driven Global Memory (New in MLEvolve, Not in Paper)

The paper describes **node/branch/graph-level memory maintenance** as part of MCGS (storing top-k nodes at each level). This is simple bookkeeping.

MLEvolve adds a **full experience-driven global memory system** (`agents/memory/`) that is NOT described in the paper:
- `GlobalMemoryLayer` with persistent storage of plan, code, metrics, and success/failure labels
- **BM25 + FAISS hybrid retrieval** for finding similar/dissimilar past experiences
- `HybridRetriever` with embedding models (BAAI/bge-base-en-v1.5)
- **Memory-enhanced two-stage planning** (`agents/planner/planner_with_memory.py`):
  - Stage 1: Generate initial plan guided by dissimilar records (for novelty)
  - Stage 2: Refine to structured JSON using similar success/fail records
- Different agents query memory differently to encourage diverse approaches

This is a significant architectural addition. The config has it **disabled by default** (`use_global_memory: False`).

---

## 5. Code Generation Strategies

### Paper: Single-pass code generation

The paper describes a simple pipeline: Plan Generation → Code Generation & Review → Code Execution → Results Update. Code generation is single-pass.

### MLEvolve: Three code generation modes

The code supports three strategies (not described in the paper):
1. **Single-pass full generation** (base mode)
2. **Stepwise multi-agent pipeline** (`agents/coder/stepwise_coder.py`) — specialized agents handle data_processing, model_design, and training_evaluation stages separately
3. **SEARCH/REPLACE diff patching** (`agents/coder/diff_coder/`) — incremental patches via diff generation + application, enabled by `use_diff_mode: True` in config

The diff mode is the default for improvements, allowing targeted edits rather than full rewrites.

---

## 6. Search/Selection Enhancements

### Paper: Standard UCT with fixed exploration constant

The paper uses UCT with `C = 1.414` throughout.

### MLEvolve: Progressive UCT with soft explore-exploit switching

The code adds several mechanisms not in the paper:
- **Piecewise exploration constant decay** (`node_selection._piecewise_decay`): C decays from 1.414 to 0.5 over the search
- **Time-aware soft switching** (`select_with_soft_switch`): probabilistically switches between UCT exploration and Top-K exploitation based on elapsed time
  - Before 50% time: pure exploration
  - 50–70%: linear transition
  - After 70%: mostly exploitation (80%)
- **Top-K weighted selection** with branch diversity constraints
- **Per-node expansion limits** differentiated for normal vs top-k triggered nodes

### Paper: Graph structure with reference edges (E_ref)

The paper formalizes MCGS with explicit reference edges `E_ref` connecting nodes across branches. The code does NOT maintain an explicit graph structure with reference edges. Cross-branch information flow happens through:
- The fusion agent reading top nodes from other branches
- The evolution agent reading intra-branch history
- Global top-K candidate tracking

The reference edges are conceptual in the code, not a data structure.

---

## 7. Hyperparameter Differences

| Parameter | Paper (Table 4) | MLEvolve Config |
|-----------|-----------------|-----------------|
| max_draft_num | 7 | 5 |
| branch_top_k | 5 | 20 (`top_candidates_size`) |
| global_top_k | 10 | 20 |
| max_history_num | 7 | Not explicitly configurable |
| max_ref_num | 7 | Not explicitly configurable |
| max_agg_num | 7 | Not explicitly configurable |
| ensemble_num | 6 | No ensemble operator |
| kb_init_ref_prob | 0.8 | Not present (binary on/off) |
| max_parallel_workers | 3 | 3 (matches) |
| exploration_constant | 1.414 | 1.414 initial, decays to 0.5 |

---

## 8. Additional Code Features Not in Paper

1. **Data leakage detection** (`agents/data_leakage_agent.py`) — validates solutions don't leak test data
2. **Aggregation agent** (`agents/aggregation_agent.py`) — multi-branch aggregation as a distinct agent (paper describes this as part of MCGS expansion)
3. **Result parse agent** (`agents/result_parse_agent.py`) with metric direction auto-detection
4. **Deferred execution** — code generation and execution can be decoupled (pipelined)
5. **Plateau/stagnation detection** with patience counters and magnitude-based prompts
6. **Force backpropagation** heuristics at configurable time thresholds

---

## Summary of Key Discrepancies

1. **The fine-grained operator set from the paper (Improve-FE, Improve-CS, Ensemble) is NOT implemented** — improvements are handled by a single general-purpose agent, and there is no ensemble operator.
2. **The 3-dimensional knowledge base is reduced to model-level only** — data-level and strategy-level KB are missing.
3. **The global memory system is entirely new** — not described in the paper, and disabled by default.
4. **Multi-mode code generation (stepwise, diff) is new** — the paper only describes single-pass generation.
5. **UCT selection is significantly enhanced** with progressive decay and time-aware switching, going well beyond what the paper describes.
6. **Reference edges (E_ref) are conceptual, not implemented as data structures** — cross-branch info flow happens through agent prompts, not graph edges.
7. **Different LLM and temperature** — the paper used DeepSeek-R1 at temp=0.5; the code defaults to Gemini-3-Pro-Preview at temp=1.0.
