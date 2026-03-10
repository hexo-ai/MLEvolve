# MLEvolve Architecture Guide

This document provides a comprehensive technical overview of the MLEvolve codebase — an agentic ML engineering system that automatically solves Kaggle-style competitions through Monte Carlo Graph Search (MCGS) with multi-agent collaboration.

## Table of Contents

- [High-Level Overview](#high-level-overview)
- [Directory Structure](#directory-structure)
- [Core Concepts](#core-concepts)
- [Module Reference](#module-reference)
  - [Entry Points](#entry-points)
  - [Engine](#engine)
  - [Agents](#agents)
  - [Memory System](#memory-system)
  - [LLM Interface](#llm-interface)
  - [Configuration](#configuration)
  - [Utilities](#utilities)
- [Data Flow](#data-flow)
- [Search Algorithm Details](#search-algorithm-details)
- [Configuration Reference](#configuration-reference)

---

## High-Level Overview

MLEvolve operates as a tree search system where each node represents a candidate ML solution (plan + code + metric). The system:

1. **Drafts** initial solutions from a virtual root node
2. **Executes** each solution in an isolated subprocess
3. **Evaluates** solutions by parsing metrics from execution output
4. **Selects** the next node to expand using UCT (Upper Confidence Bound for Trees) with time-aware soft switching between exploration and exploitation
5. **Improves**, **debugs**, **evolves**, or **fuses** solutions depending on the node's state and search progress
6. **Persists** the best solutions and top-K candidates to disk

```
                        ┌──────────────┐
                        │ Virtual Root │
                        └──────┬───────┘
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
          ┌─────────┐    ┌─────────┐    ┌─────────┐
          │ Draft 1 │    │ Draft 2 │    │ Draft 3 │    (Branch 1, 2, 3)
          └────┬────┘    └────┬────┘    └────┬────┘
               ▼              ▼              ▼
          ┌─────────┐   ┌──────────┐   ┌──────────┐
          │ Improve │   │  Debug   │   │ Improve  │
          └────┬────┘   └────┬─────┘   └────┬─────┘
               ▼              ▼              ▼
          ┌─────────┐   ┌──────────┐   ┌──────────┐
          │Evolution│   │ Improve  │   │  Fusion  │
          └─────────┘   └──────────┘   └──────────┘
```

---

## Directory Structure

```
MLEvolve/
├── run.py                      # Main entry point (CLI)
├── __init__.py                 # Programmatic API (Experiment class)
├── run_single_task.sh          # Shell wrapper for single-task runs
├── launch_server.sh            # Launches validation format server
│
├── config/
│   ├── __init__.py             # Config loading, workspace setup, run saving
│   └── config.yaml             # Default configuration
│
├── engine/                     # Core search engine
│   ├── agent_search.py         # AgentSearch: main search coordinator
│   ├── search_node.py          # SearchNode and Journal data structures
│   ├── node_selection.py       # UCT selection, Top-K exploitation, soft switch
│   ├── evaluation.py           # Backpropagation, reward computation, improvement checks
│   ├── execution.py            # Post-execution validation (submission checks)
│   ├── solution_manager.py     # Best solution tracking, Top-K persistence
│   ├── conditions.py           # Stagnation detection, fusion trigger conditions
│   ├── executor.py             # Subprocess-based Python code executor
│   ├── validation/             # Submission format validation (client/server)
│   │   ├── format_client.py
│   │   ├── format_server.py
│   │   └── quality_check.py
│   └── coldstart/              # Cold-start model recommendations
│       ├── __init__.py
│       ├── classify_tasks.py
│       ├── knowledge.py
│       ├── competition_tag_classified.json
│       └── models_guidance_classified.json
│
├── agents/                     # LLM-powered agents
│   ├── draft_agent.py          # Initial solution generation
│   ├── improve_agent.py        # Solution improvement (diff or full rewrite)
│   ├── debug_agent.py          # Bug fixing (diff or full rewrite)
│   ├── evolution_agent.py      # Trajectory-aware improvement
│   ├── fusion_agent.py         # Cross-branch solution merging
│   ├── aggregation_agent.py    # Multi-branch synthesis (new root-level drafts)
│   ├── code_review_agent.py    # Post-generation code review
│   ├── result_parse_agent.py   # Execution output parsing and metric extraction
│   ├── data_leakage_agent.py   # Data leakage detection
│   ├── triggers.py             # Node registration and branch management
│   │
│   ├── coder/                  # Code generation strategies
│   │   ├── __init__.py         # plan_and_code_query, stepwise_plan_and_code_query
│   │   ├── base_coder.py       # Single-pass code generation
│   │   ├── stepwise_coder.py   # Multi-agent stepwise pipeline
│   │   └── diff_coder/         # SEARCH/REPLACE diff patching
│   │       ├── __init__.py
│   │       ├── diff_generate.py
│   │       ├── apply.py
│   │       ├── patcher.py
│   │       └── prompts.py
│   │
│   ├── planner/                # Planning strategies
│   │   ├── __init__.py
│   │   ├── base_planner.py     # Direct JSON planning
│   │   └── planner_with_memory.py  # Two-stage RAG planning
│   │
│   ├── memory/                 # Experience-driven memory
│   │   ├── __init__.py
│   │   ├── global_memory.py    # GlobalMemoryLayer (save, retrieve, guidance)
│   │   ├── record.py           # MemRecord data structure
│   │   ├── retriever.py        # HybridRetriever (BM25 + FAISS)
│   │   └── embedding_models.py # Embedding model wrapper
│   │
│   └── prompts/                # Shared prompt templates
│       ├── __init__.py
│       ├── environment.py      # Environment/system prompt
│       ├── impl_guideline.py   # Implementation guidelines
│       ├── shared.py           # Shared prompt fragments
│       └── validation_template_prompts.py
│
├── llm/                        # LLM backend
│   ├── __init__.py             # query() dispatcher
│   └── gemini.py               # Gemini API client
│
├── utils/                      # Utilities
│   ├── __init__.py             # copytree, preproc_data helpers
│   ├── data_preview.py         # Dataset preview generation
│   ├── logging_config.py       # Logging setup
│   ├── metric.py               # MetricValue, WorstMetricValue
│   ├── response.py             # Response parsing (extract code, plan)
│   ├── seed.py                 # Global random seed
│   ├── serialize.py            # JSON serialization for Journal
│   ├── submission_fusion_utils.py  # Submission file fusion
│   └── visualization.py        # Search tree visualization
│
├── assets/                     # Static assets (logo)
├── docs/                       # Project page (HTML)
├── requirements_base.txt       # Core dependencies
├── requirements_ml.txt         # ML framework dependencies
└── requirements_domain.txt     # Domain-specific dependencies
```

---

## Core Concepts

### SearchNode

A `SearchNode` (`engine/search_node.py`) represents a single candidate solution in the search tree. Key fields:

| Field | Description |
|-------|-------------|
| `code` | The complete Python solution script |
| `plan` | Natural language description of the approach |
| `stage` | One of: `root`, `draft`, `improve`, `debug`, `evolution`, `fusion`, `fusion_draft` |
| `metric` | `MetricValue` with the validation score and maximize/minimize direction |
| `is_buggy` | Whether execution failed or produced invalid output |
| `parent` / `children` | Tree structure links |
| `branch_id` | Which search branch this node belongs to |
| `visits` / `total_reward` | MCTS statistics for UCT computation |
| `local_best_node` | Best node found along this improvement chain |
| `is_terminal` | Whether this node can still be expanded |

### Journal

A `Journal` (`engine/search_node.py`) is an ordered list of all `SearchNode`s produced during a run. It provides:
- `get_best_node()` — returns the node with the highest metric
- `good_nodes` — non-buggy nodes
- `draft_nodes` — initial solution nodes
- Filtering and path extraction for serialization

### AgentSearch

`AgentSearch` (`engine/agent_search.py`) is the main coordinator. It:
- Maintains the search tree (journal, virtual root, branch tracking)
- Delegates node selection to `node_selection`
- Dispatches to the appropriate agent (draft, improve, debug, evolution, fusion, aggregation)
- Invokes code review and result parsing after each step
- Tracks the global best solution and top-K candidates

---

## Module Reference

### Entry Points

#### `run.py` — CLI Entry Point

The main execution loop:

1. Loads configuration from `config/config.yaml` + CLI overrides
2. Optionally loads cold-start model guidance
3. Prepares the workspace (copies data, creates directories)
4. **Phase 1**: Sequentially generates initial draft solutions (code only, no execution) for diversity
5. **Phase 2**: Parallel execution pipeline using `ThreadPoolExecutor`:
   - Submits pending draft executions with 10s stagger
   - Fills remaining slots with `step_task()` calls
   - On completion, saves journal and submits next task
   - Handles `KeyboardInterrupt` gracefully (terminates subprocesses)

#### `__init__.py` — Programmatic API

Provides the `Experiment` class for using MLEvolve as a library:

```python
from MLEvolve import Experiment

exp = Experiment(data_dir="/path/to/data", goal="Predict house prices")
solution = exp.run(steps=50)
print(solution.code, solution.valid_metric)
```

#### `run_single_task.sh`

Shell wrapper that sets up environment variables and invokes `run.py` for a single MLE-bench task:

```bash
bash run_single_task.sh <EXP_ID> <DATASET_DIR> [SERVER_ID]
```

#### `launch_server.sh`

Starts the validation format server (`engine/validation/format_server.py`) for submission format checking.

---

### Engine

#### `agent_search.py` — AgentSearch

The central coordinator class. Key method: `step(node, exec_callback)`:

1. If no data preview exists, generates one and records search start time
2. If `node` is the root, calls `node_selection.select_with_soft_switch()` to pick a node
3. Calls `_run_single_step()` which dispatches to the appropriate agent:
   - **Root node not at child limit** → `draft_agent`
   - **Root node at child limit + fusion conditions met** → `aggregation_agent`
   - **Buggy parent** → `debug_agent`
   - **Non-buggy parent, stagnant branch, after 6h** → `fusion_agent` or `evolution_agent` (probabilistic)
   - **Non-buggy parent, stagnant branch, before 6h** → `evolution_agent`
   - **Non-buggy parent, not stagnant** → `improve_agent`
4. After code generation, runs `code_review_agent`
5. Executes code via `exec_callback` (subprocess interpreter)
6. Parses results via `result_parse_agent`
7. Validates the executed node (submission existence, metric sanity)
8. Checks improvement and backpropagates rewards
9. Updates best solution and top-K candidates

Also provides `execute_deferred_node()` for Phase 1 drafts that were generated but not yet executed.

#### `search_node.py` — SearchNode & Journal

The `SearchNode` dataclass includes:

- **UCT computation**: `uct_value(exploration_constant)` — standard UCB1 formula with configurable exploration constant
- **Child limit checks**: `reached_child_limit(scfg)` — enforces `num_drafts`, `num_bugs`, `num_improves` limits
- **Memory helpers**: `fetch_child_memory()`, `fetch_parent_memory()` — build context strings for LLM prompts
- **Trajectory**: `get_root_to_current_trajectory()` — returns the full improvement history from root to current node
- **Bayesian sampling**: `alpha`/`beta` parameters with `update_beta()` and `p_mean()`
- **Thread safety**: `child_count_lock` for concurrent child count updates

#### `node_selection.py` — Node Selection

Three selection strategies:

1. **`select(agent, node)`** — Standard UCT tree policy:
   - Recurses down the tree selecting the child with highest UCT value
   - Uses piecewise decay for the exploration constant (high early, lower later)
   - Draft nodes get locked on selection to prevent double-expansion

2. **`select_with_soft_switch(agent)`** — Time-aware exploration/exploitation:
   - Before `explore_switch_start` (default 50% of time): pure exploration (UCT)
   - Between `explore_switch_start` and `explore_switch_end` (50-70%): linear blend
   - After `explore_switch_end`: mostly exploitation with `min_exploration_weight` (20%)
   - Exploitation uses `get_top_k_nodes_global()` + weighted random selection

3. **`get_top_k_nodes_global(agent, k, max_from_same_branch)`** — Branch-diverse top-K:
   - Sorts all non-buggy nodes by metric globally
   - Caps nodes per branch to ensure diversity
   - Returns `[{node, branch_id, metric, rank}]`

#### `evaluation.py` — Evaluation & Backpropagation

- **`backpropagate(node, value)`**: Propagates reward up the tree, updating `visits`, `total_reward`, unlocking draft nodes, resetting `improve_failure_depth`
- **`get_node_reward(agent, node)`**: Computes reward: +1.5 for global best improvement, +1.5 for debug success, +1 for non-buggy execution, -1 for buggy/null metric
- **`check_improvement(agent, cur_node, parent_node)`**: Determines whether to continue the improvement chain or backpropagate:
  - Tracks `improve_failure_depth` — after `max_improve_failure` consecutive non-improvements, marks node as terminal
  - Implements **force backpropagation** at mid/late search stages to redistribute search effort
  - Smart backprop skip: if current node is recent best, continue its improvement chain

#### `conditions.py` — Search Conditions

- **`should_trigger_branch_fusion(agent)`**: Checks time window (6-10h), minimum successful branches (2), global stagnation, and fusion draft count
- **`is_branch_stagnant(agent, branch_id, threshold)`**: True if last `threshold` successful nodes didn't improve on branch best
- **`is_globally_stagnant(agent)`**: True if no improvement in last `stagnation_window` nodes

#### `executor.py` — Code Execution

The `Interpreter` class:
- Executes Python code in **subprocesses** (not fork/multiprocessing) to avoid CUDA issues
- Supports `max_parallel_run` concurrent slots with CPU pinning (`os.sched_setaffinity`)
- Isolates submission paths (`submission_{node_id}.csv`) and model paths to avoid write conflicts
- Captures stdout/stderr, parses exception types from stderr
- Enforces timeout with SIGINT → SIGKILL escalation

#### `execution.py` — Post-Execution Validation

`validate_executed_node()`:
- Checks that `submission_{node_id}.csv` exists
- Flags metric=0.0 with maximize=True as buggy (complete failure)
- Registers successful nodes to their branch

#### `solution_manager.py` — Solution Persistence

- **`update_top_candidates(agent, node)`**: Maintains a top-N list with branch diversity (max 5 per branch)
- **`save_top_candidates(agent)`**: Writes top-N solutions to `workspace/top_solution/top{rank}/`
- **`save_best_solution(agent, node, submission_path)`**: Saves best solution code, submission, and metadata
- **`update_best_solution(agent, node)`**: Orchestrates top-K update + best solution update

#### `coldstart/` — Cold-Start Knowledge

Provides pretrained model recommendations based on task category:
- `classify_tasks.py`: Classifies competition by domain (vision, NLP, tabular, etc.)
- `knowledge.py`: Builds guidance description from `models_guidance_classified.json`
- JSON files map competition tags to recommended pretrained models

---

### Agents

All agents follow the pattern: build a prompt → call the LLM → parse the response → create a new `SearchNode`.

#### `draft_agent.py` — Initial Solution

Generates the first solutions from scratch. Key features:
- Persona: "Kaggle Grandmaster"
- Requires scientific reasoning (WHAT/WHY/HOW)
- Enforces novelty vs. existing drafts in memory
- Supports cold-start model recommendations
- Uses stepwise code generation by default

#### `improve_agent.py` — Solution Improvement

Improves a successful parent node. Two modes:
- **Standard**: Scientific approach with structured CHANGES format
- **Plateau**: When `success_patience >= 2` or `total_patience >= 5`, triggers Magnitude-Based Reasoning requiring Tier 2/3 changes (no hyperparameter tuning)

Supports both full rewrite and **diff mode** (SEARCH/REPLACE patching) via a two-stage pipeline:
1. Planner generates a JSON plan identifying modules to change
2. Diff coder applies targeted SEARCH/REPLACE patches

#### `debug_agent.py` — Bug Fixing

Fixes buggy nodes. Features:
- Diff mode with up to 3 retries, fallback to full rewrite
- Global memory lookup for similar past bugs and their fixes
- Requires complete, self-contained executable scripts

#### `evolution_agent.py` — Trajectory-Aware Improvement

Like `improve_agent` but provides the full branch evolution trajectory (root → current) as context, enabling the LLM to learn from the history of what worked and what failed. Falls back to `improve_agent` if trajectory has fewer than 2 steps.

#### `fusion_agent.py` — Cross-Branch Fusion

Merges insights from other branches:
- `_get_fusion_candidates()`: Collects top-2 nodes from each other branch
- `fuse_two_nodes()`: Single reference fusion
- `_fuse_with_multiple_references()`: Multi-reference fusion (up to 5)
- Falls back to `improve_agent` if no candidates found

#### `aggregation_agent.py` — Multi-Branch Aggregation

Creates entirely new root-level solutions by synthesizing insights from all branches:
- Triggered when regular draft limit is reached AND fusion conditions are met
- Two modes: `node` (best solutions only) or `trajectory` (full evolution paths)
- Creates `fusion_draft` stage nodes
- Limited by `max_fusion_drafts` (default 2)

#### `code_review_agent.py` — Code Review

Reviews generated code for correctness before execution. Can modify the code if issues are found.

#### `result_parse_agent.py` — Result Parsing

Parses execution output to extract:
- Validation metric value
- Whether the metric should be maximized or minimized
- Bug detection and analysis
- Code summary for memory

Also determines metric direction at initialization via `determine_metric_direction()`.

#### `data_leakage_agent.py` — Data Leakage Detection

Checks generated code for potential data leakage (using test data during training).

#### `triggers.py` — Node Registration

`register_node(agent, node, prompt, new_branch, parent_node)`:
- Assigns `branch_id` (new or inherited from parent)
- Registers node in `branch_all_nodes` and `branch_successful_nodes`
- Saves the node to global memory (if enabled)
- Stores prompt input for debugging

Also provides `get_patience_counter()` for plateau detection.

---

### Code Generation Strategies

Located in `agents/coder/`:

1. **Single-pass** (`base_coder.py`): One LLM call generates plan + code
2. **Stepwise** (`stepwise_coder.py`): Multi-agent pipeline where planning and coding are separate steps
3. **Diff patching** (`diff_coder/`):
   - `diff_generate.py`: Generates SEARCH/REPLACE patches
   - `patcher.py`: `SearchReplacePatcher` applies patches with fuzzy matching
   - `apply.py`: Orchestrates planning → diff generation → application
   - Up to 3 retries with context about previous failures

### Planning Strategies

Located in `agents/planner/`:

1. **Direct** (`base_planner.py`): Single LLM call returns JSON with `module` and `plan` keys
2. **Memory-augmented** (`planner_with_memory.py`): Two-stage RAG:
   - Stage 1: Generate initial plan
   - Stage 2: Retrieve similar/dissimilar experiences from global memory, refine plan

---

### Memory System

Located in `agents/memory/`:

#### `global_memory.py` — GlobalMemoryLayer

Task-scoped experience store:
- **Save**: Records plan, code summary, stage, label (success/failure/neutral), metric change
- **Retrieve**: Hybrid BM25 + FAISS search with optional label/stage filters
- **Guidance**: Generates prompt text summarizing relevant historical experiences
- Persists to `global_memory/records.json` in the workspace

#### `retriever.py` — HybridRetriever

Combines sparse (BM25) and dense (FAISS) retrieval:
- BM25 for keyword matching
- FAISS for semantic similarity using embedding vectors
- Configurable alpha blending between the two

#### `embedding_models.py` — Embedding Models

Wraps local sentence-transformer models (default: `BAAI/bge-base-en-v1.5`).

#### `record.py` — MemRecord

Data structure for memory records with fields: `record_id`, `title`, `description`, `method`, `label`, `timestamp`.

---

### LLM Interface

Located in `llm/`:

- `__init__.py`: `query()` function dispatches to the Gemini backend
- `gemini.py`: OpenAI-compatible client for Gemini API
  - Supports function calling via `FunctionSpec`
  - `generate()` convenience function for simple completions
  - `compile_prompt_to_md()` recursively converts nested dict/list prompts to Markdown

---

### Configuration

Located in `config/`:

#### `config.yaml` — Default Settings

See [Configuration Reference](#configuration-reference) below.

#### `__init__.py` — Config Management

Key functions:
- `load_cfg()` / `_load_cfg()`: Load YAML + CLI overrides
- `prep_cfg()`: Validate and resolve paths, create experiment directory
- `load_task_desc()`: Load task description from file or config
- `prep_agent_workspace()`: Create `input/`, `working/`, `submission/` directories; copy/symlink data
- `save_run()`: Persist journal, filtered journal, config, and best solution to log directory

---

### Utilities

| File | Purpose |
|------|---------|
| `data_preview.py` | Generates dataset preview (file listing, CSV head, shapes) for LLM context |
| `logging_config.py` | Sets up file + console logging with Rich formatting |
| `metric.py` | `MetricValue` (value + maximize flag + comparison), `WorstMetricValue` sentinel |
| `response.py` | Extracts code blocks, plans, and diff responses from LLM output; `trim_long_string()` |
| `seed.py` | Sets global random seeds (Python, NumPy, PyTorch) |
| `serialize.py` | Custom JSON encoder for `SearchNode`/`Journal` serialization |
| `submission_fusion_utils.py` | Merges/averages multiple submission CSV files |
| `visualization.py` | `journal_to_string_tree()` — renders the search tree as an ASCII tree |

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                          run.py (main loop)                         │
│                                                                     │
│  1. load_cfg() → Config                                            │
│  2. load_task_desc(cfg) → task description                         │
│  3. build_guidance_description(cfg) → cold-start guidance          │
│  4. prep_agent_workspace(cfg) → workspace/input, working, submission│
│                                                                     │
│  Phase 1: Sequential draft generation (code only)                  │
│  ┌──────────────────────────────────────────────┐                  │
│  │  agent.step(exec_callback, node=None,        │                  │
│  │            execute_immediately=False)         │                  │
│  │    → draft_agent.run(agent) → SearchNode     │                  │
│  │    → code_review_agent.run(agent, node)      │                  │
│  │    → node.pending_execution = True           │                  │
│  └──────────────────────────────────────────────┘                  │
│                                                                     │
│  Phase 2: Parallel execution pipeline                              │
│  ┌──────────────────────────────────────────────┐                  │
│  │  ThreadPoolExecutor(max_workers)              │                  │
│  │    ├─ execute_draft_node(node)               │                  │
│  │    │    → exec_callback(code) → ExecutionResult                 │
│  │    │    → result_parse_agent.run() → metrics │                  │
│  │    │    → validate_executed_node()           │                  │
│  │    │    → check_improvement() → backpropagate│                  │
│  │    │    → update_best_solution()             │                  │
│  │    │                                         │                  │
│  │    └─ step_task(node)                        │                  │
│  │         → select_with_soft_switch() → node   │                  │
│  │         → agent dispatch → new SearchNode    │                  │
│  │         → exec → parse → validate → evaluate │                  │
│  └──────────────────────────────────────────────┘                  │
│                                                                     │
│  save_run(cfg, journal) after each completion                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Search Algorithm Details

### UCT with Piecewise Exploration Decay

The exploration constant C follows a piecewise schedule:

```
C(t) = │ C_initial                          if t < T1
       │ max(C_initial - α(t - T1), C_min)  if T1 ≤ t ≤ T2
       │ C_min                               if t > T2
```

Where:
- `C_initial = 1.414` (√2, standard UCT)
- `T1 = min(num_drafts × num_improves², steps × phase_ratios[0])`
- `T2 = steps × phase_ratios[1]`
- `α = 0.01`, `C_min = 0.5`

### Soft Exploration-Exploitation Switch

Time-based blending:
- **Exploration weight** = 1.0 until 50% of time budget
- Linear decay to 0.2 by 70% of time budget
- Below 0.2 for the remaining 30%

When in exploitation mode:
- **Early-mid phase** (<70%): Top-5 nodes, max 3 per branch
- **Late phase** (≥70%): Top-3 nodes, max 2 per branch
- Weighted random selection: weight = 1/rank

### Force Backpropagation

Prevents branches from running indefinitely:
- **Mid stage** (≥40% time): Every 3rd node in a branch triggers backpropagation
- **Late stage** (≥80% time): 50% chance per node
- **Smart skip**: If node is the best in the last 4 nodes, skip forced backprop

### Stagnation-Triggered Actions

When a branch shows no improvement over its best for `threshold` consecutive successful nodes:
- **Before 6h**: `evolution_agent` (trajectory-aware improvement)
- **After 6h**: 30% chance `fusion_agent`, 70% chance `evolution_agent`
- Global stagnation triggers `aggregation_agent` at the root level

---

## Configuration Reference

### Data & Paths

| Key | Default | Description |
|-----|---------|-------------|
| `data_dir` | null | Path to task data directory (required) |
| `dataset_dir` | `""` | MLE-bench dataset root |
| `desc_file` | null | Task description file path |
| `goal` | null | Task goal string (alternative to desc_file) |
| `eval` | null | Evaluation description |
| `log_dir` | `./runs` | Log output directory |
| `workspace_dir` | `./runs` | Workspace directory |
| `torch_hub_dir` | `""` | PyTorch Hub cache directory |
| `pretrain_model_dir` | `""` | Pretrained model weights directory |

### Execution

| Key | Default | Description |
|-----|---------|-------------|
| `exec.timeout` | 32400 (9h) | Per-execution timeout in seconds |
| `exec.agent_file_name` | `runfile.py` | Base name for generated Python scripts |

### Agent

| Key | Default | Description |
|-----|---------|-------------|
| `agent.steps` | 500 | Maximum number of search steps |
| `agent.time_limit` | 43200 (12h) | Total time budget in seconds |
| `agent.initial_drafts` | 3 | Number of initial draft solutions |
| `agent.seed` | 42 | Random seed |
| `agent.code.model` | `gemini-3-pro-preview` | LLM model for code generation |
| `agent.code.temp` | 1 | Temperature for code generation |
| `agent.code.base_url` | `""` | API endpoint (required) |
| `agent.code.api_key` | `""` | API key (required) |
| `agent.feedback.model` | `gemini-3-pro-preview` | LLM model for feedback/parsing |
| `agent.use_diff_mode` | true | Use SEARCH/REPLACE diff patching |
| `agent.use_global_memory` | false | Enable experience-driven memory |
| `agent.memory_embedding_model_path` | `BAAI/bge-base-en-v1.5` | Embedding model for memory retrieval |

### Search Tree

| Key | Default | Description |
|-----|---------|-------------|
| `search.parallel_search_num` | 3 | Max concurrent executions |
| `search.num_drafts` | 5 | Max initial drafts from root |
| `search.num_bugs` | 1 | Max debug attempts per buggy node |
| `search.num_improves` | 3 | Max improve attempts per node |
| `search.max_debug_depth` | 20 | Max recursive debug depth |
| `search.back_debug_depth` | 3 | Debug depth before backpropagation |
| `search.max_improve_failure` | 3 | Consecutive non-improvements before terminal |
| `search.branch_stagnation_threshold` | 3 | Nodes without improvement to trigger stagnation |
| `search.stagnation_window` | 4 | Window for global stagnation check |

### Exploration-Exploitation

| Key | Default | Description |
|-----|---------|-------------|
| `search.explore_switch_start` | 0.5 | Time fraction to start exploitation blend |
| `search.explore_switch_end` | 0.7 | Time fraction for full exploitation |
| `search.min_exploration_weight` | 0.2 | Minimum exploration probability |
| `search.topk_early_k` | 5 | Top-K size in early-mid phase |
| `search.topk_late_k` | 3 | Top-K size in late phase |

### Decay

| Key | Default | Description |
|-----|---------|-------------|
| `decay.exploration_constant` | 1.414 | Initial UCT exploration constant |
| `decay.lower_bound` | 0.5 | Minimum exploration constant |
| `decay.alpha` | 0.01 | Decay rate |
| `decay.phase_ratios` | [0.3, 0.7] | Phase transition points as fraction of total steps |

### Cold-Start

| Key | Default | Description |
|-----|---------|-------------|
| `coldstart.use_coldstart` | true | Enable cold-start model recommendations |
| `coldstart.task_json_path` | `engine/coldstart/competition_tag_classified.json` | Task classification data |
| `coldstart.model_json_path` | `engine/coldstart/models_guidance_classified.json` | Model recommendation data |
