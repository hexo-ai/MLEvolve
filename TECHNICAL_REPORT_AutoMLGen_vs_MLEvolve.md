# Technical Report: From AutoMLGen to MLEvolve — Architectural Discrepancies and Novel Contributions

**Prepared as reference material for a revised paper submission.**

---

## Abstract

MLEvolve is described as "an advanced version based on AutoMLGen" (arXiv:2510.08511). While both systems share the same Monte Carlo Graph Search (MCGS) foundation for automated Machine Learning Engineering (MLE), the MLEvolve codebase introduces several substantial architectural changes, drops certain paper-described components, and adds entirely new subsystems. This report systematically catalogues every discrepancy between the AutoMLGen paper and the MLEvolve implementation, and provides detailed technical descriptions of the three most significant novel contributions: (1) the experience-driven global memory system, (2) the multi-mode code generation framework, and (3) the progressive UCT selection with soft explore–exploit switching.

---

## Table of Contents

1. [High-Level Summary of Discrepancies](#1-high-level-summary-of-discrepancies)
2. [Operator Set: What Changed](#2-operator-set-what-changed)
3. [Knowledge Base: Reduced Scope](#3-knowledge-base-reduced-scope)
4. [Hyperparameter and Configuration Differences](#4-hyperparameter-and-configuration-differences)
5. [Graph Structure: Conceptual vs. Implemented](#5-graph-structure-conceptual-vs-implemented)
6. [Deep Dive — Experience-Driven Global Memory System](#6-deep-dive--experience-driven-global-memory-system)
7. [Deep Dive — Multi-Mode Code Generation](#7-deep-dive--multi-mode-code-generation)
8. [Deep Dive — Progressive UCT with Soft Explore–Exploit Switching](#8-deep-dive--progressive-uct-with-soft-exploreexploit-switching)
9. [Additional MLEvolve Features Not in AutoMLGen](#9-additional-mlevolve-features-not-in-automlgen)
10. [Consolidated Discrepancy Table](#10-consolidated-discrepancy-table)
11. [Recommendations for the Revised Paper](#11-recommendations-for-the-revised-paper)

---

## 1. High-Level Summary of Discrepancies

| Dimension | AutoMLGen Paper | MLEvolve Code |
|-----------|----------------|---------------|
| **MLE-bench medal rate** | 36.4% (DeepSeek-R1, 12 h) | 61.33% (Gemini-3-Pro-Preview, 12 h) |
| **Default LLM** | DeepSeek-R1-0528 | Gemini-3-Pro-Preview |
| **Decoding temperature** | 0.5 | 1.0 |
| **Improve operator** | 3 specialised variants (Normal, FE, CS) | 1 unified agent |
| **Ensemble operator** | Dedicated operator (§3.3.3) | Not implemented |
| **Evolution agent** | Described as MCGS expansion mode (Eq. 6) | Standalone agent (`evolution_agent.py`) |
| **Knowledge base** | 3 dimensions (model, data, strategy) | Model-level only |
| **Global memory** | Not described | Full BM25+FAISS retrieval system |
| **Code generation** | Single-pass | 3 modes (single-pass, stepwise, diff) |
| **UCT exploration constant** | Fixed C = 1.414 | Piecewise decay 1.414 → 0.5 |
| **Selection policy** | Standard UCT | UCT + time-aware soft switching to Top-K exploitation |
| **Reference edges E_ref** | Explicit graph data structure | Conceptual; info flows through prompts |

---

## 2. Operator Set: What Changed

### 2.1 Improve Variants — Collapsed into One

The paper defines three improve sub-operators in Section 3.3.3:

- **Improve-Normal**: small adjustments such as switching optimisers or tuning hyperparameters.
- **Improve-FE (Feature Enhancement)**: focused on data augmentation, feature engineering (categorical encodings, feature aggregation).
- **Improve-CS (Competition Strategies)**: injects competition-style practices from the knowledge base (pseudo-labelling, ensembling, TTA).

In the MLEvolve codebase, `agents/improve_agent.py` exposes a single `run()` entry point. There is no routing logic to select among FE, CS, or Normal sub-strategies. Instead, the prompt instructs the LLM broadly:

> "Acceptable: Advanced architectures, ensemble techniques, feature engineering, hyperparameter optimization."

The specialised knowledge-base references that would power Improve-CS are absent because data-level and strategy-level KBs were not implemented (see Section 3). Thus, the three-way dispatch described in the paper has been collapsed into one general-purpose agent.

**Implication for the paper**: If the ablation study reports separate contributions of Improve-FE and Improve-CS, these claims cannot be reproduced from the released code.

### 2.2 Ensemble Operator — Not Implemented

The paper states (§3.3.3):

> "During search, a global Top-K set of candidate nodes is maintained. Near termination, the best solutions are heuristically combined to produce a more robust final solution."

The MLEvolve codebase contains `utils/submission_fusion_utils.py` (19.6 KB), a standalone utility for combining submission CSV files. However, this utility is **not** integrated into the search loop as an operator—it is never called by `AgentSearch` or any agent. There is no `ensemble_agent.py` or equivalent.

**Implication**: The paper's reported ensemble_num = 6 hyperparameter has no counterpart in the code.

### 2.3 Evolution Agent — Promoted to Standalone

The paper describes "intra-branch evolution" (Section 3.3.2, Eq. 6) as one of four MCGS expansion modes. In the code, it is promoted to a first-class agent (`agents/evolution_agent.py`) with its own prompt engineering and trigger logic. It is dispatched from `agent_search.py:_run_single_step()` when a branch is stagnant but the elapsed time has not yet reached the fusion threshold:

```python
# agent_search.py:173-177
if is_branch_stagnant(self, parent_node.branch_id, threshold=stagnation_threshold):
    if can_use_fusion:
        if random.random() < self.acfg.fusion_vs_evolution_prob:  # default 0.3
            result_node = fusion_agent.run(self, parent_node)
        else:
            result_node = evolution_agent.run(self, parent_node)
    else:
        result_node = evolution_agent.run(self, parent_node)
```

The probability split between fusion and evolution is governed by `fusion_vs_evolution_prob` (default 0.3 for fusion, 0.7 for evolution), a parameter not mentioned in the paper.

---

## 3. Knowledge Base: Reduced Scope

### Paper's Three-Dimensional Design

The paper describes a curated ML domain knowledge base spanning three dimensions (§3.2.1):

1. **Model-level**: categorises models by application domain with concise usage guidelines.
2. **Data-level**: summarises modality-specific constraints and preprocessing methods; highlights feature-engineering principles.
3. **Strategy-level**: distils practical competition tricks (TTA, ensembling, pseudo-labelling) from ML competition discussions.

Section 3.2.2 further specifies a probabilistic injection mechanism: model-level knowledge is injected at initialisation with probability `kb_init_ref_prob = 0.8`, while data- and strategy-level knowledge guide the iterative search.

### MLEvolve's Implementation

The cold-start system (`engine/coldstart/`) contains only:

| File | Content | Paper Dimension |
|------|---------|-----------------|
| `models_guidance_classified.json` | Model recommendations per task category | Model-level |
| `competition_tag_classified.json` | Task classification tags | (classification helper) |

**Data-level and strategy-level knowledge bases are entirely absent.** The injection mechanism is a binary toggle (`use_coldstart: True/False` in `config.yaml`) rather than the probabilistic `kb_init_ref_prob = 0.8` described in the paper.

**Implication**: The ablation row "+ knowledge base" in Table 3 of the paper reflects a richer system than what the code provides.

---

## 4. Hyperparameter and Configuration Differences

| Parameter | Paper (Table 4) | MLEvolve `config.yaml` | Notes |
|-----------|-----------------|------------------------|-------|
| `max_steps` | 500 | 500 | Matches |
| `exploration_constant` | 1.414 (fixed) | 1.414 initial → 0.5 (decayed) | Major change; see §8 |
| `temperature` | 0.5 | 1.0 | Doubled |
| `max_parallel_workers` | 3 | 3 (`parallel_search_num`) | Matches |
| `max_draft_num` | 7 | 5 (`num_drafts`) | Reduced |
| `max_debug_num` | 20 | 20 (`max_debug_depth`) | Matches |
| `branch_top_k` | 5 | 20 (`top_candidates_size`) | 4× larger |
| `global_top_k` | 10 | 20 (`top_candidates_size`) | 2× larger (shared param) |
| `max_history_num` | 7 | Not configurable | — |
| `max_ref_num` | 7 | Not configurable | — |
| `max_agg_num` | 7 | Not configurable | — |
| `ensemble_num` | 6 | N/A | No ensemble operator |
| `kb_init_ref_prob` | 0.8 | N/A (binary on/off) | Simplified |

---

## 5. Graph Structure: Conceptual vs. Implemented

### Paper's Formalisation

The paper defines (Eq. 3):

```
G = (V, E),   E = E_T ∪ E_ref
```

- **Primary edges E_T**: parent–child generative relationships; used for UCT selection and backpropagation.
- **Reference edges E_ref**: cross-branch or non-adjacent information flow; do not participate in backpropagation.

The four expansion modes (primary, intra-branch evolution, cross-branch reference, multi-branch aggregation) each produce specific `E_ref` edges as formalised in Equations 5–8.

### Code's Implementation

The `SearchNode` class (`engine/search_node.py`) stores only tree-structural relationships:

```python
parent: Optional["SearchNode"]       # single parent (E_T edge)
children: set["SearchNode"]          # children set (E_T edges)
```

There is **no data structure for reference edges**. Cross-branch information transfer happens entirely through prompt construction:

- The **fusion agent** reads top nodes from other branches by querying `agent.branch_successful_nodes` and injecting their plans/code into the LLM prompt.
- The **evolution agent** reads intra-branch history via `parent_node.fetch_child_memory()`.
- The **aggregation agent** reads global top candidates via `agent.top_candidates`.

Backpropagation (`engine/evaluation.py:backpropagate()`) traverses `.parent` pointers—equivalent to the paper's E_T-only policy—but there are no E_ref edges to exclude in the first place.

**Implication**: The graph formalism in the paper is a mathematical abstraction of prompt-level information flow, not a concrete data structure in the code.

---

## 6. Deep Dive — Experience-Driven Global Memory System

This subsystem is the most significant addition in MLEvolve with no counterpart in the AutoMLGen paper. The paper mentions "memory maintenance" only in the context of keeping branch-level and graph-level top-k node lists—simple bookkeeping. MLEvolve introduces a full retrieval-augmented memory layer that persistently stores, indexes, and retrieves node-level experiences to guide future search decisions.

### 6.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    GlobalMemoryLayer                        │
│                  (agents/memory/global_memory.py)           │
│                                                             │
│  ┌──────────────┐   ┌──────────────────────────────────┐   │
│  │  MemRecord[]  │   │        HybridRetriever           │   │
│  │  (record.py)  │   │       (retriever.py)             │   │
│  │               │   │                                  │   │
│  │ - record_id   │   │  ┌──────────┐  ┌─────────────┐  │   │
│  │ - title       │   │  │ BM25Okapi│  │ FAISS Index │  │   │
│  │ - description │   │  │ (keyword)│  │  (vector)   │  │   │
│  │ - method      │   │  └─────┬────┘  └──────┬──────┘  │   │
│  │ - label       │   │        │               │         │   │
│  │ - timestamp   │   │        └───────┬───────┘         │   │
│  └──────────────┘   │          RRF Fusion               │   │
│                      │   (Reciprocal Rank Fusion)        │   │
│  ┌──────────────┐   └──────────────────────────────────┘   │
│  │ Metadata Map │                                          │
│  │ node_id →    │   ┌──────────────────────────────────┐   │
│  │  exec_time   │   │       EmbeddingModel             │   │
│  │  parent_met  │   │    (embedding_models.py)          │   │
│  │  current_met │   │                                   │   │
│  │  parent_err  │   │  Supports: local (SentenceTrans), │   │
│  └──────────────┘   │  OpenAI, Azure, Custom            │   │
│                      └──────────────────────────────────┘   │
│                                                             │
│  Persistence: records.json (flat file, auto-load/save)     │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Data Model: MemRecord

Each search node that passes quality gates is serialised into a `MemRecord` (`agents/memory/record.py`):

```python
@dataclass
class MemRecord:
    record_id: str          # "node_{uuid}"
    title: str              # "{stage} - {node_id[:8]}"
    description: str        # The node's plan text
    method: str             # Code summary (classes/functions extracted, or plan[:500])
    label: int              # +1 = success, 0 = neutral, -1 = failure
    timestamp: str          # ISO-8601
```

A separate `node_metadata_map` stores execution-level metadata (exec_time, parent_metric, current_metric, parent_error for debug nodes) without polluting the retrieval-indexed fields.

### 6.3 Labelling Logic

Labels are determined automatically by `_determine_label()` based on the node's stage and metric delta:

| Stage | Label = +1 | Label = -1 | Label = 0 |
|-------|-----------|-----------|----------|
| `draft`, `fusion_draft` | Always | — | — |
| `debug` | Parent was buggy and node is not | Otherwise | — |
| `improve`, `evolution`, `fusion` | Metric improved over parent | Metric worsened | No change |

Quality gate: buggy nodes and nodes with `metric.value = None` are never saved.

### 6.4 Hybrid Retrieval: BM25 + FAISS with Reciprocal Rank Fusion

The `HybridRetriever` (`agents/memory/retriever.py`) implements a two-channel search fused via Reciprocal Rank Fusion (RRF):

**Channel 1 — BM25 (keyword match):**
- Tokenisation: simple whitespace split, lowercased.
- Implementation: `rank_bm25.BM25Okapi`.
- Captures exact term overlap (e.g., model names, metric names, library references).

**Channel 2 — FAISS (semantic vector search):**
- Default embedding model: `BAAI/bge-base-en-v1.5` (768-dimensional, via `sentence-transformers`).
- Index type: `faiss.IndexFlatL2` (exact L2 nearest-neighbour; no approximation).
- L2 distances are negated to produce similarity scores.

**Fusion formula (RRF):**

```
RRF_score(d) = α × 1/(k + rank_BM25(d)) + (1 − α) × 1/(k + rank_FAISS(d))
```

where `α` (default 0.5) controls the BM25-vs-vector weight and `k = 60` is the RRF constant.

**Index maintenance:**
- On first record: full index build (`build_index()`).
- On subsequent records: incremental update (`add_to_index()`) — new embeddings are computed and appended to the FAISS index; BM25 is rebuilt (lightweight).
- Persistence: `records.json` is re-written after each save.

### 6.5 Retrieval Modes

The `retrieve_similar_records()` method supports multiple modes controlled by parameters:

| Mode | `dissimilar` | `label_filter` | Use Case |
|------|-------------|---------------|----------|
| Similar successes | `False` | `1` | "What worked before for a similar plan?" |
| Similar failures | `False` | `-1` | "What failed before for a similar plan?" |
| Dissimilar records | `True` | `None` | "What hasn't been tried yet?" (novelty) |
| Stage-specific | either | + `stage_filter` | e.g., only `improve` records |

When `dissimilar=True`, results are sorted by ascending score (least similar first), promoting exploration of under-explored plan regions.

### 6.6 Memory-Enhanced Two-Stage Planning

The most sophisticated consumer of global memory is `agents/planner/planner_with_memory.py`, which replaces the standard single-stage planner when `use_global_memory: True`.

**Stage 1 — `generate_initial_plan()`:**
- Produces a free-text improvement plan.
- Optionally guided by **dissimilar** historical records retrieved from memory, injected into the `Memory` prompt section.
- Intent: encourage the agent to explore novel directions that have not yet been attempted.

**Stage 2 — `refine_plan_to_json()`:**
- Takes the Stage 1 text plan as input.
- Retrieves **similar success records** (`label_filter=1`, top_k=2) and **similar failure records** (`label_filter=-1`, top_k=2) from memory.
- Builds a "Historical Experience of Similar Experiments" guidance block:
  - "Successful Similar Experiments" — approaches that worked for similar plans.
  - "Failed Similar Experiments (Avoid)" — approaches that failed for similar plans.
- Instructs the LLM to refine the text plan into structured JSON (module selections + per-module modification plans), informed by both successes to emulate and failures to avoid.
- Includes retry logic (up to 3 attempts) with JSON schema validation.

**Information flow diagram:**

```
                    ┌──────────────────┐
                    │  Global Memory   │
                    │  (all records)   │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │ dissimilar   │ similar+label │
              ▼              │              ▼
    ┌─────────────────┐      │    ┌──────────────────┐
    │  Stage 1:       │      │    │  Stage 2:        │
    │  Free-text plan │──────┘    │  JSON plan       │
    │  (novelty-      │           │  (refined with   │
    │   guided)       │───────────│   success/fail   │
    └─────────────────┘           │   references)    │
                                  └──────────────────┘
                                           │
                                           ▼
                                  Code Generation
                                  (diff or rewrite)
```

### 6.7 Configuration

```yaml
agent:
  use_global_memory: False            # disabled by default
  memory_similarity_threshold: 0.7    # minimum score for similar retrieval
  memory_embedding_device: cuda       # GPU-accelerated embedding
  memory_embedding_model_path: "BAAI/bge-base-en-v1.5"
```

**Design note**: The system is disabled by default, suggesting it was either experimental at the time of release or incurs overhead (embedding computation, FAISS indexing) that may not be justified for all scenarios.

---

## 7. Deep Dive — Multi-Mode Code Generation

The AutoMLGen paper describes a single code generation pipeline: Plan Generation → Code Generation & Review → Code Execution → Results Update. MLEvolve replaces this with a three-mode code generation framework, where each mode is suited to different search stages and contexts.

### 7.1 Mode Overview

| Mode | File | When Used | Key Property |
|------|------|-----------|--------------|
| **Single-pass** (base) | `agents/coder/base_coder.py` | Drafts, fallback | One LLM call → plan + complete code |
| **Stepwise** | `agents/coder/stepwise_coder.py` | Drafts (when enabled) | 3 specialised agents + 1 meta-agent |
| **Diff (SEARCH/REPLACE)** | `agents/coder/diff_coder/` | Improvements, evolution, fusion | Incremental patches on existing code |

### 7.2 Mode 1: Single-Pass Generation (`base_coder.py`)

The simplest strategy. A single LLM call receives the full prompt (task description, data preview, memory, instructions) and produces a natural-language plan followed by a complete Python code block.

```
LLM Input:  [Introduction] [Task] [Data preview] [Memory] [Instructions]
LLM Output: "Brief plan...\n```python\n<complete code>\n```"
```

Post-processing extracts the plan text (everything before the code block) and the code (content within triple backticks). Up to 3 retries if extraction fails.

This mode serves as the default for initial drafts and the fallback when other modes fail.

### 7.3 Mode 2: Stepwise Multi-Agent Pipeline (`stepwise_coder.py`)

This mode decomposes code generation into three specialised stages, each handled by a dedicated `StepAgent`, followed by a `MetaAgent` that merges the outputs into a single runnable script.

**Pipeline stages:**

```
┌──────────────────────────────────────┐
│  Step 1: data_processing_and_        │
│          feature_engineering         │
│  "Load data, clean, create features,│
│   split into train/val/test"         │
└──────────────┬───────────────────────┘
               │ (plan + code)
               ▼
┌──────────────────────────────────────┐
│  Step 2: model_design                │
│  "Design architecture, loss, optim" │
│  (receives Step 1 output as context)│
└──────────────┬───────────────────────┘
               │ (plan + code)
               ▼
┌──────────────────────────────────────┐
│  Step 3: training_evaluation         │
│  "Training loop, validation, metric │
│   tracking, submission generation"   │
│  (receives Steps 1-2 as context)    │
└──────────────┬───────────────────────┘
               │ (plan + code)
               ▼
┌──────────────────────────────────────┐
│  MetaAgent: merge                    │
│  "Combine all step outputs into one │
│   cohesive, runnable Python script"  │
└──────────────────────────────────────┘
```

**Key design decisions:**

1. **Sequential dependency**: Each StepAgent receives all previous steps' plans and code as context (`previous_steps`), ensuring coherent variable naming and interface compatibility.

2. **Cold-start awareness**: During draft stage, if cold-start is enabled:
   - The `model_design` agent receives explicit instructions to prioritise recommended pretrained models.
   - The `data_processing` agent receives guidance to prepare data in formats compatible with pretrained models (image sizes, normalisation, tokenisation).

3. **Improvement mode**: When used for improvements (not just drafts), each StepAgent can receive the previous module-specific code and an improvement strategy, enabling targeted per-module refinement.

4. **MetaAgent merge**: The merge step ensures:
   - Duplicate imports are removed.
   - Variable names are consistent across steps.
   - Execution flow is logical (data → model → training).
   - The final script is self-contained and runnable.

**Relationship to the paper**: The paper's "Improve-FE" and "Improve-CS" specialisation is partially captured by this stepwise design—each step agent focuses on its domain. However, the stepwise coder does not explicitly route through "feature engineering" vs. "competition strategy" knowledge bases as the paper describes.

### 7.4 Mode 3: SEARCH/REPLACE Diff Patching (`diff_coder/`)

This is the **default mode for improvements** (`use_diff_mode: True` in config). Instead of regenerating the entire solution, the LLM produces targeted patches in a structured SEARCH/REPLACE format.

#### 7.4.1 Diff Format

```
<<<<<<< SEARCH
# Exact original code to find (must match verbatim)
=======
# Replacement code
>>>>>>> REPLACE
```

Multiple independent patches can be chained in a single response.

#### 7.4.2 Patch Application Engine (`patcher.py`)

The `SearchReplacePatcher` class implements a multi-strategy matching engine:

1. **Exact match**: Direct string search for the SEARCH pattern in the source code.
2. **Indentation-aware match**: If exact match fails, strips leading whitespace from both SEARCH pattern and source code lines, matches by content, then re-applies the detected indentation to the REPLACE block.
3. **Fuzzy match (diagnostics)**: If both above fail, uses `difflib.SequenceMatcher` with a 0.6 similarity threshold to find the closest match and generate a diagnostic diff showing where the SEARCH pattern diverged from the actual code. This is used for error messages, not for automatic application.

Special case: An empty SEARCH block triggers an **append** operation—the REPLACE content is appended to the end of the file.

#### 7.4.3 Generation and Retry Pipeline (`diff_generate.py`)

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Build diff  │     │  LLM call    │     │  Apply patches  │
│  prompt with │────▶│  (generate   │────▶│  with retry     │
│  plan + code │     │   diffs)     │     │  (max 3)        │
│  + context   │     └──────────────┘     └────────┬────────┘
└─────────────┘                                     │
                                          ┌─────────▼─────────┐
                                          │  Success?          │
                                          │  Yes → return      │
                                          │  No  → regenerate  │
                                          │  with error note   │
                                          └───────────────────┘
```

The `diff_generate_and_apply()` function:

1. Formats the improvement plan into a structured prompt with diff instructions.
2. Calls the LLM to generate SEARCH/REPLACE patches.
3. Calls `apply_diff_with_retry()`:
   - Attempts to apply all patches.
   - If any SEARCH pattern fails to match, captures the `PatchError` diagnostic and triggers a **regeneration call** with the error message prepended to the prompt, asking the LLM to fix its SEARCH patterns.
   - Up to `max_diff_retries` (default 3) regeneration attempts.
4. Returns the final patched code and the count of successfully applied patches.
5. Falls back to returning the original (unmodified) code if all attempts fail.

**Advantages over full rewrite:**
- **Preserves working code**: Only the targeted sections are modified, reducing the risk of introducing new bugs.
- **Smaller LLM output**: The model generates only the changed regions, not the entire (potentially multi-hundred-line) solution.
- **Explicit change tracking**: Each SEARCH/REPLACE block makes the modification intent clear.

#### 7.4.4 Shared Across Agents

The diff pipeline is used by four different agents—improve, evolution, fusion, and aggregation—each providing only stage-specific text parameters:

| Parameter | Purpose |
|-----------|---------|
| `introduction` | System-level description of the agent's role |
| `extra_context` | Trajectory history, cross-branch references, etc. |
| `extra_user_sections` | Additional prompt sections unique to the stage |
| `learning_guidance` | Extra instruction appended to diff guidelines |

This shared architecture ensures consistent diff formatting and application logic across all improvement stages.

---

## 8. Deep Dive — Progressive UCT with Soft Explore–Exploit Switching

The AutoMLGen paper describes standard UCT (Upper Confidence Trees) with a fixed exploration constant C = 1.414 (Eq. 4). MLEvolve fundamentally redesigns the selection policy with three layered mechanisms that adapt search behaviour over time.

### 8.1 Standard UCT (Paper Baseline)

For reference, the paper's selection criterion:

```
UCT(i) = Q_i / (N_i + ε) + C × √(ln(N_parent + 1) / (N_i + ε))
```

where C = 1.414 throughout the search.

### 8.2 Mechanism 1: Piecewise Exploration Constant Decay

**File**: `engine/node_selection.py:_piecewise_decay()`

Instead of a fixed C, MLEvolve decays the exploration constant over three phases:

```
         C(t)
    1.414 ┤━━━━━━━━━━━━┓
          │             ┃  Phase 1: Constant
          │             ┃  (t < T₁)
          │             ┃
          │             ┗━━━━━━━━━━━━━━━┓  Phase 2: Linear decay
          │                              ┃  (T₁ ≤ t ≤ T₂)
          │                              ┃  C(t) = C₀ − α(t − T₁)
    0.5   ┤                              ┗━━━━━━━━━━━━━━━━  Phase 3: Constant
          │                                                  (t > T₂)
          └─────────────┬───────────────┬───────────────────── step t
                       T₁              T₂
```

**Phase boundaries** are computed dynamically:

```python
T₁ = min(num_drafts × num_improves², steps × phase_ratios[0])   # e.g., min(45, 150)
T₂ = steps × phase_ratios[1]                                      # e.g., 350
```

**Default configuration:**

```yaml
decay:
  exploration_constant: 1.414   # C₀
  lower_bound: 0.5              # C_min
  alpha: 0.01                   # decay rate per step
  phase_ratios: [0.3, 0.7]     # T₁ and T₂ as fractions of max_steps
```

**Rationale**: In the early phase, high C encourages broad exploration of diverse branches. As the search matures and more information is gathered, reducing C shifts UCT toward exploiting the most promising branches, concentrating compute on refining high-performing solutions.

### 8.3 Mechanism 2: Time-Aware Soft Explore–Exploit Switching

**File**: `engine/node_selection.py:select_with_soft_switch()`

This is the highest-level selection mechanism. Instead of always using UCT, MLEvolve probabilistically switches between two selection strategies based on wall-clock time progress:

**Strategy A — Exploration (UCT)**: Standard tree traversal via `select()`, choosing nodes using the decayed UCT criterion.

**Strategy B — Exploitation (Top-K)**: Selects from the globally best-performing nodes, biased toward exploiting proven solutions.

The switching probability is controlled by `get_exploration_weight()`:

```
    exploration_weight(t)
    1.0 ┤━━━━━━━━━━━━━━━━━━━━━┓
        │                      ┃  Pure exploration
        │                      ┃  (t < switch_start)
        │                      ┃
        │                      ┗━━━━━━━━━━━━━━┓  Linear transition
        │                                      ┃  (switch_start ≤ t < switch_end)
    0.2 ┤                                      ┗━━━━━━━━━━━━━━━━  Mostly exploitation
        │                                                          (t ≥ switch_end)
        └──────────────────────┬──────────────┬──────────────────── time progress
                             0.5            0.7
                         (switch_start)  (switch_end)
```

**Default configuration:**

```yaml
search:
  explore_switch_start: 0.5    # Start transitioning at 50% of time budget
  explore_switch_end: 0.7      # Complete transition at 70%
  min_exploration_weight: 0.2  # Always keep 20% exploration probability
```

At each step:
1. Compute `exploration_weight` from the elapsed fraction of the time budget.
2. With probability = `exploration_weight`, use UCT (Strategy A).
3. With probability = 1 − `exploration_weight`, use Top-K exploitation (Strategy B).

### 8.4 Mechanism 3: Top-K Weighted Selection with Branch Diversity

When exploitation mode is triggered, `select_with_soft_switch()` calls `get_top_k_nodes_global()` to assemble the candidate set.

**Top-K assembly** (`get_top_k_nodes_global()`):

1. Collect all non-buggy nodes with valid metrics across all branches.
2. Sort by metric value (descending for maximise, ascending for minimise).
3. Apply a **per-branch cap** (`max_from_same_branch`) to ensure diversity: no single branch can dominate the Top-K set.
4. Return the top-k nodes with their branch IDs, metrics, and ranks.

**Weighted random selection** (`select_from_top_k_weighted()`):

The node is chosen by weighted random sampling where the weight is inversely proportional to rank:

```
P(select node at rank r) = (1/r) / Σ(1/r_i)
```

This ensures that the top-ranked node is most likely to be selected, but lower-ranked nodes still have a chance, maintaining some diversity within exploitation.

**Phase-dependent parameters:**

| Phase | Time Progress | k | max_per_branch |
|-------|--------------|---|----------------|
| Early-mid | < 70% | 5 (`topk_early_k`) | 3 (`topk_early_max_per_branch`) |
| Late | ≥ 70% | 3 (`topk_late_k`) | 2 (`topk_late_max_per_branch`) |

In the late phase, the candidate set narrows (smaller k, tighter per-branch cap), focusing compute on the very best solutions.

### 8.5 Additional Selection Mechanisms

**Root lock for drafts**: When a draft node is selected from the root, it is `lock`ed to prevent concurrent threads from re-selecting it before execution completes. The lock is released during backpropagation.

**Expansion limit checks**: `select()` checks `reached_child_limit()` before deciding whether to expand or recurse deeper. If the root is fully expanded for regular drafts and multi-branch fusion conditions are met (time window, minimum successful branches, global stagnation), the root is returned for aggregation instead.

**Top-K triggered relaxation**: Nodes selected via Top-K have `_topk_triggered = True` set, which causes `_run_single_step()` to use a relaxed stagnation threshold (`topk_stagnation_threshold = 6` vs. the default `branch_stagnation_threshold = 3`), allowing more improvement attempts before triggering fusion/evolution.

### 8.6 Force Backpropagation Heuristics

**File**: `engine/evaluation.py:check_improvement()`

To prevent deep non-improving chains from consuming the time budget, MLEvolve introduces probabilistic forced backpropagation that is not described in the paper:

| Time Progress | Trigger | Probability |
|--------------|---------|-------------|
| ≥ 80% (`force_backprop_late_threshold`) | Every node | 50% (`force_backprop_late_prob`) |
| ≥ 40% (`force_backprop_mid_threshold`) | Every 3rd node per branch (`force_backprop_mid_modulo`) | 100% |

**Smart backprop exception**: If the current node is the best among the last `recent_best_window` (default 4) nodes, forced backpropagation is skipped to avoid cutting off an active improvement chain.

### 8.7 Complete Selection Flow Diagram

```
                  ┌──────────────────────────┐
                  │  select_with_soft_switch  │
                  └────────────┬─────────────┘
                               │
                   ┌───────────▼───────────┐
                   │  Compute exploration  │
                   │  weight from time     │
                   └───────────┬───────────┘
                               │
                    ┌──────────┴──────────┐
            P=weight│                     │P=1-weight
                    ▼                     ▼
           ┌──────────────┐    ┌────────────────────┐
           │  UCT select  │    │  Top-K exploitation │
           │  (decayed C) │    │                    │
           └──────┬───────┘    │  1. Get top-k      │
                  │            │     globally        │
                  │            │  2. Apply branch    │
                  │            │     diversity cap   │
                  │            │  3. Filter by       │
                  │            │     expansion limit │
                  │            │  4. Weighted random │
                  │            │     selection       │
                  │            └─────────┬──────────┘
                  │                      │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  _run_single_step()  │
                  │  (draft / improve /  │
                  │   debug / evolve /   │
                  │   fuse / aggregate)  │
                  └──────────────────────┘
```

---

## 9. Additional MLEvolve Features Not in AutoMLGen

Beyond the three deep-dive topics above, MLEvolve includes several additional features with no counterpart in the paper:

### 9.1 Data Leakage Detection (`agents/data_leakage_agent.py`)

A dedicated agent that validates generated solutions do not leak test data into training. Enabled by `check_data_leakage: True` in config.

### 9.2 Aggregation Agent (`agents/aggregation_agent.py`)

The paper describes "multi-branch aggregation" as an MCGS expansion mode (Eq. 8). In MLEvolve, this is a standalone agent with its own trigger conditions:

```python
# engine/conditions.py:should_trigger_branch_fusion()
# Conditions: time in [6h, 10h], ≥2 branches with ≥2 successful nodes, global stagnation
```

Limited to `max_fusion_drafts = 2` aggregation branches per run.

### 9.3 Deferred (Pipelined) Execution

The main entry point (`run.py`) uses a two-phase pipeline with `ThreadPoolExecutor`:
- **Phase 1**: Generate and review code for multiple initial drafts (no execution).
- **Phase 2**: Execute generated code in parallel.

This decoupling is enabled by `execute_immediately=False` in `step()` and `execute_deferred_node()` for later execution, allowing LLM generation and code execution to overlap.

### 9.4 Plateau Detection with Patience Counters

`agents/triggers.py` provides `get_patience_counter()` which tracks:
- `success_patience`: consecutive non-improving steps from successful nodes.
- `total_patience`: total non-improving steps across all attempts.

When thresholds are exceeded (`success_patience ≥ 2` or `total_patience ≥ 5`), the improve agent switches to "Magnitude-Based prompts" that explicitly request larger, more ambitious changes.

### 9.5 Code Review Agent (`agents/code_review_agent.py`)

Every generated node passes through code review before execution:

```python
# agent_search.py:186-189
reviewed_code = code_review_agent.run(self, result_node)
if reviewed_code.strip() != result_node.code.strip():
    result_node.code = reviewed_code
```

This catches data leakage patterns, import errors, and metric–task mismatches before execution.

---

## 10. Consolidated Discrepancy Table

| # | Paper Claim / Feature | Code Reality | Category |
|---|----------------------|--------------|----------|
| 1 | Improve-Normal / Improve-FE / Improve-CS operators | Single unified improve agent | Simplified |
| 2 | Ensemble operator (Top-K combination near termination) | Not implemented | Missing |
| 3 | Intra-branch evolution as MCGS expansion mode | Standalone `evolution_agent.py` | Restructured |
| 4 | 3-dimensional knowledge base (model + data + strategy) | Model-level only | Reduced |
| 5 | `kb_init_ref_prob = 0.8` probabilistic KB injection | Binary `use_coldstart: True/False` | Simplified |
| 6 | Explicit reference edges E_ref as graph data structure | Prompt-level info flow only | Conceptual |
| 7 | Fixed exploration constant C = 1.414 | Piecewise decay 1.414 → 0.5 | Enhanced |
| 8 | Standard UCT selection only | UCT + time-aware soft Top-K switching | Enhanced |
| 9 | Single-pass code generation | 3 modes (single-pass, stepwise, diff) | Enhanced |
| 10 | Simple top-k bookkeeping for memory | Full BM25+FAISS retrieval-augmented memory | New |
| 11 | DeepSeek-R1, temperature 0.5 | Gemini-3-Pro-Preview, temperature 1.0 | Changed |
| 12 | max_draft_num = 7 | num_drafts = 5 | Changed |
| 13 | branch_top_k = 5, global_top_k = 10 | top_candidates_size = 20 (shared) | Changed |
| 14 | No data leakage detection | `data_leakage_agent.py` | New |
| 15 | No deferred/pipelined execution | Phase 1 (generate) / Phase 2 (execute) | New |
| 16 | No plateau detection with magnitude prompts | Patience counters + magnitude-based prompts | New |
| 17 | No force backpropagation heuristics | Time-aware probabilistic force backprop | New |

---

## 11. Recommendations for the Revised Paper

Based on this analysis, the following areas warrant attention in a revised paper:

1. **Operator set description**: Align the paper's fine-grained operator taxonomy (Improve-FE, Improve-CS, Ensemble) with the actual implementation, or clearly mark them as ablated/future-work components.

2. **Knowledge base scope**: Clarify that the released code provides model-level cold-start only; the data-level and strategy-level KBs are planned/internal.

3. **New contributions to highlight**: The global memory system, multi-mode code generation, and progressive UCT with soft switching are substantial contributions that deserve formal description and ablation. Each could be a distinct subsection.

4. **Graph formalism vs. implementation**: Consider noting that E_ref edges are implemented as prompt-level information injection rather than as a persistent graph data structure. This is arguably a more practical design but differs from the mathematical formalisation.

5. **Hyperparameter reconciliation**: Update Table 4 to match the deployed configuration, or clearly label paper values as "paper experiment settings" vs. "released defaults."

6. **Ablation coverage**: The released code supports ablating the memory system (`use_global_memory`), diff mode (`use_diff_mode`), cold-start (`use_coldstart`), and the explore-exploit switching (`explore_switch_*`). These offer natural ablation axes for the new contributions.

---

*Report generated from source-level analysis of the MLEvolve codebase (commit 22b2d5a) and AutoMLGen paper (arXiv:2510.08511v1).*
