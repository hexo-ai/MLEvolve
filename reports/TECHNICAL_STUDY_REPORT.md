# MLEvolve: Technical Study Report

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Configuration and Launch](#3-configuration-and-launch)
4. [The Search Tree: Data Structures](#4-the-search-tree-data-structures)
5. [Search Strategy: Selection and Evaluation](#5-search-strategy-selection-and-evaluation)
6. [The Coordinator: AgentSearch](#6-the-coordinator-agentsearch)
7. [Agent Subsystem](#7-agent-subsystem)
8. [Code Generation Pipeline](#8-code-generation-pipeline)
9. [LLM Interface and Prompt Engineering](#9-llm-interface-and-prompt-engineering)
10. [Execution and Validation](#10-execution-and-validation)
11. [Memory and Cold-Start](#11-memory-and-cold-start)
12. [Post-Processing and Utilities](#12-post-processing-and-utilities)
13. [Concurrency Model](#13-concurrency-model)
14. [Key Design Patterns](#14-key-design-patterns)
15. [End-to-End Walkthrough](#15-end-to-end-walkthrough)
16. [Glossary](#16-glossary)

---

## 1. Executive Summary

MLEvolve is an autonomous ML engineering agent that solves Kaggle-style competitions by framing solution development as a **tree search problem**. It is currently ranked #1 on the MLE-bench leaderboard within a 12-hour budget.

**Core Idea**: Each node in the search tree is a complete Python solution (code + plan + metric). The system uses UCT-based selection (from Monte Carlo Tree Search) to decide which solution to expand, dispatches specialized LLM-powered agents to generate or refine code, executes the code in sandboxed subprocesses, and backpropagates rewards to guide future exploration.

**What "Evolve" means**: Despite the name, MLEvolve does not use evolutionary algorithms or genetic programming. "Evolve" refers to the iterative improvement of solutions guided by their branch history — the trajectory of changes from root to the current node.

### Key Numbers

| Parameter | Default Value |
|---|---|
| Total budget | 12 hours (43,200s) |
| Max steps | 500 |
| Initial drafts | 3 |
| Parallel execution slots | 3 |
| Execution timeout per run | 9 hours (32,400s) |
| Top-K candidates tracked | 20 |
| UCT exploration constant | 1.414 (decays to 0.5) |
| LLM | Gemini 3 Pro Preview |

---

## 2. System Architecture

### Directory Structure

```
MLEvolve/
├── run.py                      # Main entry point
├── run_single_task.sh          # Shell wrapper (launches server + run.py)
├── launch_server.sh            # Starts the grading server
├── __init__.py                 # Experiment class (alternative API)
├── config/
│   ├── __init__.py             # Config loading, workspace setup, save_run
│   └── config.yaml             # All tuneable parameters
├── engine/
│   ├── agent_search.py         # Central coordinator (AgentSearch)
│   ├── search_node.py          # SearchNode and Journal
│   ├── executor.py             # Subprocess execution with CPU pinning
│   ├── node_selection.py       # UCT selection + Top-K exploitation
│   ├── evaluation.py           # Reward computation + backpropagation
│   ├── execution.py            # Post-run validation
│   ├── solution_manager.py     # Top-K candidate management
│   ├── conditions.py           # Stagnation detection + fusion triggers
│   ├── coldstart/              # Task/model knowledge base
│   └── validation/             # Grading server client + quality checks
├── agents/
│   ├── draft_agent.py          # Initial solution generation
│   ├── improve_agent.py        # Refine working solutions
│   ├── debug_agent.py          # Fix buggy solutions
│   ├── evolution_agent.py      # Trajectory-guided improvement
│   ├── fusion_agent.py         # Cross-branch idea merging
│   ├── aggregation_agent.py    # Multi-branch fresh drafts
│   ├── code_review_agent.py    # Post-generation code review
│   ├── result_parse_agent.py   # Output parsing + metric extraction
│   ├── data_leakage_agent.py   # Data leakage detection
│   ├── triggers.py             # Patience counters + node registration
│   ├── planner/                # Plan generation (base + memory-enhanced)
│   ├── coder/                  # Code generation (single-shot, stepwise, diff)
│   ├── memory/                 # Global memory (BM25 + FAISS)
│   └── prompts/                # Shared prompt fragments
├── llm/
│   ├── __init__.py             # query() interface
│   └── gemini.py               # Gemini API client (streaming + structured)
└── utils/
    ├── data_preview.py         # Dataset preview for prompts
    ├── submission_fusion_utils.py  # Post-run ensemble
    ├── visualization.py        # Tree rendering
    ├── metric.py               # MetricValue (direction-aware comparison)
    ├── response.py             # Code/text extraction from LLM responses
    ├── seed.py                 # Global random seed
    ├── serialize.py            # JSON serialization
    └── logging_config.py       # Logging setup
```

### High-Level Data Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│  run.py                                                                  │
│                                                                          │
│  1. load_cfg() ──────────────────> Config (YAML + CLI overrides)         │
│  2. load_task_desc() ────────────> Task description (markdown)           │
│  3. build_guidance_description() > Cold-start model recommendations      │
│  4. prep_agent_workspace() ──────> ./runs/<timestamp>/workspace/         │
│  5. AgentSearch(task_desc, cfg) ─> Tree search coordinator               │
│  6. Interpreter(workspace, cfg) ─> Subprocess executor                   │
│                                                                          │
│  Phase 1: Sequential draft generation (code only, no execution)          │
│  Phase 2: Parallel pipelined execution (ThreadPoolExecutor)              │
│           ┌─────────────────────────────────────────────┐                │
│           │  agent.step(exec_callback)                  │                │
│           │    ├─ select node (UCT or Top-K)            │                │
│           │    ├─ dispatch agent (draft/improve/debug/…) │                │
│           │    ├─ code_review_agent                     │                │
│           │    ├─ exec_callback (subprocess)            │                │
│           │    ├─ result_parse_agent                    │                │
│           │    ├─ validate_executed_node                │                │
│           │    ├─ check_improvement (backprop)          │                │
│           │    └─ update_best_solution                  │                │
│           └─────────────────────────────────────────────┘                │
│  7. save_run() ──────────────────> journal.json, best_solution.py        │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Configuration and Launch

### Entry Points

There are two ways to run MLEvolve:

**1. Shell script** (`run_single_task.sh`): The standard method for MLE-bench tasks.
- Starts the grading server on port `5005 + SERVER_ID`
- Waits for health check (up to 30s)
- Runs `run.py` with a 12-hour timeout via the `timeout` command
- After the run, calls `submission_fusion_utils.py` to ensemble top solutions

**2. Python API** (`__init__.py`): `Experiment(data_dir, goal, eval).run(steps)` for programmatic use.

### Configuration System

Configuration is handled via OmegaConf with three layers:
1. **YAML defaults** (`config/config.yaml`) — all tuneable parameters
2. **CLI overrides** — `python run.py data_dir=... exp_id=...`
3. **Typed validation** — dataclasses (`Config`, `AgentConfig`, `SearchConfig`, `DecayConfig`, etc.) enforce types

Key configuration sections:

| Section | Purpose | Key Parameters |
|---|---|---|
| `data_dir`, `dataset_dir` | Competition data paths | Paths to MLE-bench data |
| `exec` | Code execution | `timeout=32400` (9h), `agent_file_name=runfile.py` |
| `agent` | Search budget | `steps=500`, `time_limit=43200` (12h), `initial_drafts=3` |
| `agent.code` / `agent.feedback` | LLM endpoints | `model=gemini-3-pro-preview`, `temp=1` |
| `agent.search` | Tree parameters | `parallel_search_num=3`, `num_drafts=5`, stagnation thresholds |
| `agent.decay` | UCT decay | `exploration_constant=1.414`, `lower_bound=0.5`, `alpha=0.01` |
| `coldstart` | Knowledge base | `use_coldstart=True`, JSON paths for task/model mapping |

### Workspace Preparation

`prep_agent_workspace(cfg)` creates:
```
workspace/
├── input/       # Symlinked competition data (or copied if copy_data=True)
├── working/     # Temporary files during execution
└── submission/   # Generated submission CSVs
```

The run directory is:
```
./runs/<YYYYMMDD_HHMMSS>_<exp_name>/
├── logs/
│   ├── journal.json            # Full search tree
│   ├── filtered_journal.json   # Best-path only
│   ├── config.yaml             # Saved config
│   └── best_solution.py        # Best code
└── workspace/
    ├── input/
    ├── working/
    ├── submission/
    ├── best_solution/
    ├── best_submission/
    └── top_solution/            # Top-K ranked solutions
```

---

## 4. The Search Tree: Data Structures

### SearchNode (`engine/search_node.py`)

Every candidate solution is a `SearchNode` — the fundamental unit of the search tree.

```
SearchNode
├── Identity
│   ├── id: str (UUID)
│   ├── step: int (journal index)
│   ├── ctime: float (creation timestamp)
│   └── stage: "root" | "draft" | "improve" | "debug" | "evolution" | "fusion" | "fusion_draft"
│
├── Tree Links
│   ├── parent: Optional[SearchNode]
│   └── children: set[SearchNode]    # Auto-registered on __post_init__
│
├── Solution Content
│   ├── code: str                    # Complete Python script
│   ├── plan: str                    # Natural language plan
│   └── prompt_input: str            # Serialized prompt (for debugging)
│
├── Execution Results
│   ├── _term_out: list[str]         # stdout + stderr
│   ├── exec_time: float             # Wall-clock seconds
│   ├── exc_type: str | None         # Exception class name
│   ├── exc_info: dict | None        # Error details
│   └── exc_stack: list[tuple]       # Stack trace
│
├── Evaluation
│   ├── metric: MetricValue          # Validation score (direction-aware)
│   ├── is_buggy: bool               # Whether code failed
│   ├── is_valid: bool               # Whether submission passed format check
│   └── analysis: str                # LLM analysis of the output
│
├── MCTS State
│   ├── visits: int                  # Visit count (N(s,a))
│   ├── total_reward: float          # Cumulative reward (Q(s,a))
│   ├── is_terminal: bool            # No further expansion
│   ├── continue_improve: bool       # Whether to keep improving this branch
│   ├── improve_failure_depth: int   # Consecutive failures without improvement
│   ├── local_best_node: SearchNode  # Best node on this branch
│   ├── lock: bool                   # Prevents concurrent selection
│   └── expected_child_count: int    # In-flight + completed children
│
├── Bayesian Sampling
│   ├── alpha: int = 1               # Beta distribution parameter
│   └── beta: int = 1                # Beta distribution parameter
│
└── Branch Management
    ├── branch_id: int               # Which branch this node belongs to
    ├── from_topk: bool              # Whether selected via Top-K exploitation
    ├── code_summary: str            # Brief code description
    └── work_dir: str                # Per-node working directory
```

**Key methods**:

- `uct_value(C)`: Computes `Q/N + C * sqrt(ln(N_parent)/N)`. Unvisited nodes return infinity.
- `reached_child_limit(scfg)`: Checks draft/improve/debug expansion limits.
- `fetch_child_memory()`: Builds a formatted string of all children's plans, metrics, and outcomes — used as "Memory" in prompts.
- `fetch_parent_memory()`: Returns the parent's plan, metric, and analysis.
- `get_root_to_current_trajectory()`: Walks from root to this node, building a step-by-step evolution history.

### MetricValue (`utils/metric.py`)

Direction-aware metric comparison:

```python
MetricValue(value=0.85, maximize=True) > MetricValue(value=0.80, maximize=True)   # True
MetricValue(value=0.10, maximize=False) > MetricValue(value=0.20, maximize=False)  # True (lower is better)
WorstMetricValue()  # Always compares worse than any valid metric
```

The `maximize` field is determined at initialization by `result_parse_agent.determine_metric_direction()`, which reads the task description and evaluation criteria.

### Journal

An ordered list of all `SearchNode`s ever created:

```python
journal = Journal()
journal.append(virtual_root)   # step=0
journal.append(draft_node_1)   # step=1
journal.append(debug_node)     # step=2
...
journal.get_best_node()        # Returns node with highest metric
```

`filter_journal()` extracts just the path from root to the best node, for lightweight serialization.

---

## 5. Search Strategy: Selection and Evaluation

### Node Selection (`engine/node_selection.py`)

The system uses a **soft explore-exploit switch** that transitions from exploration to exploitation over time.

#### Phase Transitions

```
Time:  0%          50%              70%             100%
       |-----------|----------------|----------------|
       | Pure UCT  |  Linear blend  | 80% exploit    |
       | (explore) | (UCT + Top-K)  | 20% explore    |
```

**`select_with_soft_switch(agent)`** — the main entry point:

1. Computes `exploration_weight` based on wall-clock time progress:
   - Before 50%: weight = 1.0 (pure exploration)
   - 50%–70%: linear decay from 1.0 to 0.2
   - After 70%: weight = 0.2 (mostly exploitation)
2. With probability `exploration_weight`, uses **UCT selection** (exploration)
3. Otherwise, uses **Top-K selection** (exploitation)

#### UCT Selection (`select()`)

Recursively descends from the virtual root:
- At each node, picks the child with the highest UCT value
- If the node hasn't reached its child limit, returns it for expansion
- The exploration constant `C` decays via piecewise schedule:
  - Full strength (1.414) until ~30% of steps
  - Linear decay to lower bound (0.5) by ~70% of steps

#### Top-K Selection

Selects from the globally best K nodes (across all branches) with diversity constraints:

| Phase | K | Max per branch |
|---|---|---|
| Early-mid (<70% time) | 5 | 3 |
| Late (≥70% time) | 3 | 2 |

Uses weighted random selection: `weight = 1/rank`, so the best node is most likely to be chosen but not guaranteed.

### Evaluation and Backpropagation (`engine/evaluation.py`)

#### Reward Function

```python
def get_node_reward(agent, node):
    if node.is_buggy or metric is None:
        return -1
    if metric beats global best:
        return +1.5
    if parent was buggy (debug success):
        return +1.5
    else:
        return +1
```

#### Backpropagation

`backpropagate(node, reward)` walks from the node up to the root:
- Increments `visits` and `total_reward` at each ancestor
- Propagates `is_debug_success` upward (so parent buggy nodes know their child fixed the bug)
- Propagates `continue_improve` (whether the branch should keep being expanded)
- Unlocks draft nodes (clears `lock = True` set during selection)

#### Improvement Checking

`check_improvement(agent, cur_node, parent_node)` makes the critical continue/stop decision:

1. **Improvement found**: Updates `local_best_node`, sets `continue_improve = True`
2. **No improvement, under failure limit**: Increments `improve_failure_depth`, continues
3. **No improvement, at failure limit** (default 3): Marks node as `is_terminal`, triggers backpropagation
4. **Force backprop** (mid/late game): Probabilistically forces backprop to break stagnant chains, but skips if the node is the recent best (to avoid killing hot branches)

### Stagnation Detection (`engine/conditions.py`)

Three levels of stagnation:

| Condition | What it checks | Consequence |
|---|---|---|
| `is_branch_stagnant()` | Last N nodes in a branch didn't beat branch best | Triggers evolution or fusion agent |
| `is_globally_stagnant()` | No improvement in last `window_size` nodes globally | Required for aggregation trigger |
| `should_trigger_branch_fusion()` | Global stagnation + 6-10h window + ≥2 successful branches | Triggers aggregation (new draft from all branches) |

---

## 6. The Coordinator: AgentSearch

`AgentSearch` (`engine/agent_search.py`) is the central orchestrator that ties everything together.

### Initialization

Creates:
- A `virtual_root` node (step 0) as the tree's anchor
- Branch tracking: `branch_all_nodes`, `branch_successful_nodes` (dicts keyed by `branch_id`)
- Top-K candidate list (`top_candidates`)
- Metric direction determination (calls LLM to figure out if higher or lower is better)
- Optional global memory layer (BM25 + FAISS)

### The Step Function

```python
def step(self, node, exec_callback, execute_immediately=True):
    # 1. Initialize data preview on first call
    if first_call:
        self.update_data_preview()
        self.search_start_time = time.time()

    # 2. Select node if none provided
    if node is None or node is root:
        node = select_with_soft_switch(self)

    # 3. Run the step
    _root, result_node = self._run_single_step(node, exec_callback, execute_immediately)

    # 4. Update best solution
    if result_node has valid metric:
        solution_manager.update_best_solution(self, result_node)

    # 5. Return result (or virtual_root to trigger fresh selection)
    return result_node if not _root else self.virtual_root
```

### Action Routing (`_run_single_step`)

This is the decision engine:

```
Parent Node State              → Agent Dispatched
─────────────────────────────────────────────────
Root + under draft limit       → draft_agent
Root + at draft limit + fusion → aggregation_agent
Buggy or invalid               → debug_agent
Not buggy + stagnant + >6h     → fusion_agent (30%) or evolution_agent (70%)
Not buggy + stagnant + <6h     → evolution_agent
Not buggy + not stagnant       → improve_agent
```

After the agent generates code:
1. `code_review_agent.run()` — reviews and possibly modifies the code
2. `exec_callback(code, node_id, True)` — executes in subprocess
3. `result_parse_agent.run()` — parses output, extracts metric, detects bugs
4. `execution.validate_executed_node()` — checks submission exists, metric != 0
5. `evaluation.check_improvement()` — decides continue/terminal/backprop
6. Appends to journal (thread-safe via `journal_lock`)

---

## 7. Agent Subsystem

### 7.1 Draft Agent (`agents/draft_agent.py`)

**Purpose**: Creates initial solutions from scratch (from the virtual root).

**Prompt structure**:
- "Kaggle Grandmaster" persona
- Task description
- Memory (previous drafts and their outcomes)
- Novelty/diversity requirement: "MUST be fundamentally different from existing attempts"
- Cold-start model recommendations (if available)
- Implementation guidelines (time budget, output format, anti-leakage rules)

**Code generation**: Uses `stepwise_plan_and_code_query()` by default (three specialized agents + merge), falling back to single-shot `plan_and_code_query()`.

**Output**: A new `SearchNode` with `stage="draft"` and a new `branch_id`.

### 7.2 Improve Agent (`agents/improve_agent.py`)

**Purpose**: Refines a working (non-buggy) solution.

**Key feature — Plateau detection**:
- Computes `success_patience` (successful nodes since branch best) and `total_patience` (all nodes since branch best)
- If `success_patience >= 2` or `total_patience >= 5`: switches to "Magnitude-Based Reasoning" prompt
  - Demands Tier 2 (component swap) or Tier 3 (paradigm shift) changes
  - Explicitly forbids Tier 1 (hyperparameter tuning)
- Otherwise: uses structured "Scientific Approach" prompt (WHAT/WHY/HOW)

**Code generation**: Supports two modes:
1. **Diff mode** (default): Planner → JSON plan → SEARCH/REPLACE patches via `diff_generate_and_apply()`
2. **Full rewrite mode**: Single-shot `plan_and_code_query()`

### 7.3 Debug Agent (`agents/debug_agent.py`)

**Purpose**: Fixes buggy solutions.

**Key features**:
- Receives the buggy code, error output, and LLM analysis
- Supports diff-based debugging with up to 3 retries (accumulating partial patches across retries)
- Falls back to full code rewrite if all diff attempts fail
- Can retrieve similar past errors from global memory for historical debug guidance
- Requires complete, self-contained executable scripts

### 7.4 Evolution Agent (`agents/evolution_agent.py`)

**Purpose**: Intra-branch improvement when a branch is stagnant.

**What distinguishes it from improve**: The prompt includes the full **branch trajectory** — the step-by-step history from root to the current node showing what was tried, what worked, and what failed.

- Falls back to normal improve if trajectory has fewer than 2 steps
- Includes the same plateau detection (Magnitude-Based Reasoning) as improve
- Prompt instructs: "Build on successes, avoid repeating failures"

### 7.5 Fusion Agent (`agents/fusion_agent.py`)

**Purpose**: Cross-branch idea merging when a branch is stagnant (and >6h elapsed).

**Candidate selection**: Gathers top-2 nodes from each *other* branch. Falls back to same-branch candidates.

**Two fusion modes**:
- **Two-node fusion** (`fuse_two_nodes()`): Source node + one reference
- **Multi-reference fusion** (`_fuse_with_multiple_references()`): Source + up to 5 references

**Key principle**: "Fusion means understanding WHY techniques work, not blindly copying." The prompt requires selective adoption of ONE technique, not wholesale combination.

### 7.6 Aggregation Agent (`agents/aggregation_agent.py`)

**Purpose**: Creates entirely new drafts by synthesizing insights from all existing branches.

**Trigger conditions**: Root node is at draft limit + `should_trigger_branch_fusion()` returns True (global stagnation, 6-10h window, ≥2 successful branches).

**Process**:
1. Collects the best node from each successful branch
2. Presents them as "Branch Experiences" in the prompt
3. Creates a fresh `fusion_draft` node attached to the virtual root (new branch)
4. Capped at `max_fusion_drafts` (default 2) attempts

### 7.7 Supporting Agents

**Code Review Agent** (`agents/code_review_agent.py`):
- Runs after every agent generates code
- Can modify the code before execution

**Result Parse Agent** (`agents/result_parse_agent.py`):
- Parses execution stdout/stderr
- Extracts the validation metric from `Final Validation Score: {score}`
- Determines `is_buggy` status
- Also handles `determine_metric_direction()` at init time

**Data Leakage Agent** (`agents/data_leakage_agent.py`):
- Triggered when a metric hits an extreme value (1.0 for maximize, 0.0 for minimize)
- Checks whether the solution accidentally used test labels during training

---

## 8. Code Generation Pipeline

Three modes of code generation, used in different contexts:

### 8.1 Single-Shot Mode (`agents/coder/base_coder.py`)

The simplest mode. One LLM call produces:
1. Natural language plan text
2. A markdown code block with the complete solution

Used as the default fallback when other modes fail. Retries up to 3 times if code extraction fails.

### 8.2 Stepwise Mode (`agents/coder/stepwise_coder.py`)

Multi-agent pipeline used primarily for **initial drafts**:

```
StepAgent("data_processing_and_feature_engineering")
    ↓ (code + plan)
StepAgent("model_design")
    ↓ (code + plan, sees previous step)
StepAgent("training_evaluation")
    ↓ (code + plan, sees previous steps)
MetaAgent.merge()
    ↓ (single runnable Python script)
```

Each `StepAgent` is specialized:
- **Data processing**: Load from `./input/`, clean, feature engineer, split. Must NOT build models.
- **Model design**: Architecture, loss, optimizer. Must NOT write training loops.
- **Training/evaluation**: Training loop, validation, metric, submission. Must NOT redefine data/model.

The `MetaAgent` merges all three into a single executable script, resolving variable names, removing duplicate imports, and ensuring logical flow.

Cold-start model recommendations are injected into the model_design step.

### 8.3 Diff Mode (`agents/coder/diff_coder/`)

Two-stage pipeline used for **improve/evolution/fusion**:

**Stage 1 — Planning** (`agents/planner/base_planner.py`):
- Asks the LLM to select 1-3 modules from `[data_processing_and_feature_engineering, model_design, training_evaluation]`
- Output: JSON with `{reason, module, plan}` — e.g., `{"module": ["model_design"], "plan": {"model_design": "Switch from ResNet to EfficientNet..."}}`
- For Gemini, uses JSON schema enforcement for structured output
- Retries up to 3 times

**Stage 2 — Diff Generation** (`agents/coder/diff_coder/diff_generate.py`):
- Takes the plan and the parent code
- Generates SEARCH/REPLACE patches
- Applied by `SearchReplacePatcher` which supports:
  - Exact string match
  - Indent-aware matching (finds the right indentation level)
  - Fuzzy matching (>0.6 similarity via `difflib.SequenceMatcher`)
  - Empty SEARCH blocks (for appending new code)
- Retries with incremental patch accumulation if patches fail

**Memory-enhanced planning** (`agents/planner/planner_with_memory.py`):
When global memory is enabled (disabled by default):
1. Stage 1: `generate_initial_plan()` — free-text plan
2. Stage 2: `refine_plan_to_json()` — retrieves similar success/fail records, then converts to structured JSON informed by history

### Fallback Chain

Every diff-based operation has a fallback:
```
Diff mode (SEARCH/REPLACE) → retry (up to 3x) → full code rewrite (single-shot)
```

---

## 9. LLM Interface and Prompt Engineering

### Gemini API Client (`llm/gemini.py`)

Two main functions:

| Function | Use Case | Thinking Level | Streaming | Retries | Max Tokens |
|---|---|---|---|---|---|
| `query()` | Function calling, structured output | `low` | No | Backoff (5s, 300s max) | 16,384 |
| `generate()` | Code generation, planning | `high` | Yes | 20 retries, 3s delay | 16,384 |

Both strip `<think>` blocks from Gemini's output. The singleton client is initialized once with `api_key` and `base_url` from config.

For JSON structured output, `generate()` passes `response_mime_type="application/json"` and `response_json_schema` to enforce schema compliance.

### Prompt Compilation

`compile_prompt_to_md(prompt, depth)` recursively converts nested Python dicts/lists into markdown:
- `str` → as-is
- `list` → bullet points
- `dict` → headers at depth `depth`, with values recursed at `depth + 1`

This means all agent prompts are authored as Python dicts and rendered to markdown at call time.

### Shared Prompt Fragments (`agents/prompts/`)

**`shared.py`**:
- `ROBUSTNESS_GENERALIZATION_STRATEGY`: Architecture matching, regularization, augmentation tips
- `prompt_leakage_prevention()`: Data isolation rules (fit on train only, no target leakage)
- `prompt_resp_fmt()`: "Plan text followed by a single code block"
- `get_internet_clarification()`: Clarifies that internet is available during development

**`impl_guideline.py`**:
- Dynamic content: time remaining, steps remaining, execution timeout
- Non-negotiable requirements: model inference for ALL predictions, submission format, validation metric
- API compatibility notes (LightGBM early stopping, AdamW import)
- Self-check checklist

**`environment.py`**:
- Lists available packages (randomly shuffled to avoid bias)
- Suggests PyTorch over TensorFlow

---

## 10. Execution and Validation

### Interpreter (`engine/executor.py`)

The `Interpreter` class executes generated Python code in isolated subprocesses:

**Slot Management**:
- Maintains `max_parallel_run` slots (default 3, matching `parallel_search_num`)
- Each slot is tracked via `status_map` (0 = free, 1 = busy)
- Thread-safe slot allocation via `multiprocessing.Lock`

**CPU Pinning**:
- Divides available CPUs evenly: `cpu_set = avail_cpus[slot * cpus_per_slot : (slot+1) * cpus_per_slot]`
- Prepends `os.sched_setaffinity(0, {cpu_set})` to the generated code

**Isolation**:
- Renames `submission.csv` → `submission_{node_id}.csv`
- Renames model checkpoints (e.g., `best_model.pth` → `best_model_{node_id}.pth`)
- Writes code to `runfile_{slot}.py`, deletes after execution

**Execution flow**:
```
1. Allocate slot
2. Pin CPUs, isolate paths
3. Write runfile
4. subprocess.Popen(python runfile.py, cwd=workspace, PYTHONUNBUFFERED=1)
5. proc.communicate(timeout=self.timeout)
6. Parse stdout/stderr, extract exception info
7. Return ExecutionResult(term_out, exec_time, exc_type, exc_info, exc_stack)
8. Cleanup: remove runfile, release slot
```

**Timeout handling**: SIGINT → 2s wait → SIGTERM → SIGKILL.

### Post-Execution Validation

**`engine/execution.py`** — `validate_executed_node()`:
1. Checks `submission_{id}.csv` exists → if not, marks buggy
2. Checks metric != 0.0 when maximizing → if so, marks buggy with detailed analysis
3. Registers successful nodes to `branch_successful_nodes`

**`engine/validation/format_client.py`** — Talks to the grading server:
- `call_validate(exp_id, submission_path)` POSTs the CSV to `/validate`
- Returns `(success, result_dict)` with retry logic

**`engine/validation/quality_check.py`** — Local quality checks:
- Detects constant/placeholder filling (>95% same value)
- Detects empty columns
- Detects common placeholder patterns (all zeros, all ones)
- LLM-based submission column fix: if column names don't match sample, asks LLM to suggest renames

---

## 11. Memory and Cold-Start

### Global Memory Layer (`agents/memory/`)

**Architecture**:
```
GlobalMemoryLayer
├── records: List[MemRecord]           # All saved nodes
├── node_metadata_map: Dict            # Additional per-node metadata
├── retriever: HybridRetriever         # BM25 + FAISS search
└── embedding_model: EmbeddingModel    # BAAI/bge-base-en-v1.5
```

**MemRecord**:
```python
@dataclass
class MemRecord:
    record_id: str      # "node_{uuid}"
    title: str          # "{stage} - {id[:8]}"
    description: str    # Node's plan
    method: str         # Code summary
    label: int          # +1 (success), -1 (failure), 0 (neutral)
    timestamp: str      # ISO timestamp
```

**Label determination**:
- Draft/fusion_draft → always +1
- Debug: fixed bug → +1, still buggy → -1
- Improve/evolution/fusion: metric improved → +1, worsened → -1, same → 0

**Hybrid Retriever** (`agents/memory/retriever.py`):
- **BM25** (keyword search): `rank_bm25.BM25Okapi` with whitespace tokenization
- **FAISS** (semantic search): `faiss.IndexFlatL2` with L2 distance
- **Fusion**: Reciprocal Rank Fusion (RRF): `score = α/(k+rank_bm25) + (1-α)/(k+rank_vector)`
- Default `α=0.5` (balanced), `k=60`
- Supports incremental updates via `add_to_index()`

**Used by**:
- `improve_agent` (when memory enabled): Two-stage planning with similar success/fail records
- `debug_agent`: Retrieves similar past errors for historical debug guidance
- `draft_agent` (indirectly): Memory from virtual root's children

**Disabled by default** (`use_global_memory: False`).

### Cold-Start Knowledge (`engine/coldstart/`)

Provides pretrained model recommendations for known competition types:

**`competition_tag_classified.json`**: Maps task IDs to categories
```json
{"denoising-dirty-documents": "General Image", "toxic-comment": "NLP", ...}
```

**`models_guidance_classified.json`**: Maps categories to recommended models
```json
{
  "General Image": {
    "DINOv3": {"Description": "...", "Code_template": "..."},
    "Siglip2": {"Description": "...", "Code_template": "..."}
  },
  "NLP": {
    "ModernBERT": {...},
    "DeBERTa-v3-large": {...}
  }
}
```

**`build_guidance_description(cfg)`** loads both JSONs, matches the task to a category, and returns a formatted string with model names, descriptions, and code templates. This is injected into the draft agent's prompt as "Pretrained Model Strategy."

---

## 12. Post-Processing and Utilities

### Submission Fusion (`utils/submission_fusion_utils.py`)

Runs after the 12h search completes. Attempts to ensemble the top solutions:

1. Loads top-K solutions from `top_solution/` directory
2. Parses metric files for scores and maximize direction
3. Filters by execution time budget (9h cumulative max)
4. Tries weighted averaging of predictions (weight = metric value)
5. Handles ID columns, passthrough columns, probability normalization
6. Sweeps ensemble sizes [1, 2, 3, 4, 6] and picks the best

### Data Preview (`utils/data_preview.py`)

Generates a textual preview of the workspace for the LLM:
- File tree with sizes
- CSV previews: shape, column names, data types, value ranges
- JSON previews: auto-generated schema via `genson`
- Special handling for `sample_submission.csv` — flags as authoritative format
- Detects validation files and suggests merging into training data

`clean_task_desc()` uses the LLM to strip environment noise (internet restrictions, time limits, URLs) from competition descriptions, then appends the sample submission format.

### Visualization (`utils/visualization.py`)

- `journal_to_rich_tree()`: Rich library tree with colored nodes (red=bug, green=success, bold green=best)
- `journal_to_string_tree()`: Plain-text version for logging

---

## 13. Concurrency Model

MLEvolve uses a **pipelined parallel** execution model:

### Phase 1: Sequential Draft Generation
```python
for i in range(initial_drafts):
    node = agent.step(exec_callback=..., execute_immediately=False)
    pending_draft_nodes.append(node)
```
Generates code only (no execution) to ensure diversity between drafts.

### Phase 2: Parallel Execution
```python
executor = ThreadPoolExecutor(max_workers=interpreter.max_parallel_run)

# Submit pending drafts (staggered by 10s)
for node in pending_draft_nodes:
    futures.add(executor.submit(execute_draft_node, node))
    time.sleep(10)  # Stagger initialization

# Fill remaining slots with step_task
for _ in range(initial_step_tasks):
    futures.add(executor.submit(step_task))

# Event loop
while completed < total_steps:
    done, _ = wait(futures, return_when=FIRST_COMPLETED, timeout=1.0)
    for fut in done:
        cur_node = fut.result()
        save_run(cfg, journal)  # Persist after each completion
        if room_for_more:
            futures.add(executor.submit(step_task, cur_node))
```

### Thread Safety

| Resource | Protection |
|---|---|
| Journal (node list) | `journal_lock` (threading.Lock) |
| Solution saving (disk) | `save_node_lock` (threading.Lock) |
| Execution slot allocation | `multiprocessing.Lock` |
| Child count tracking | Per-node `child_count_lock` (threading.Lock) |
| Active subprocess tracking | `_procs_lock` (threading.Lock) |

---

## 14. Key Design Patterns

### 1. Fallback Chains
Every operation has a graceful degradation path:
- Diff patches → retry with accumulated patches → full code rewrite
- Two-stage memory planning → single-stage direct planning
- UCT selection → Top-K exploitation → fallback to UCT from Top-K node
- Fusion agent → fallback to improve agent (if no candidates)
- Evolution agent → fallback to improve agent (if insufficient trajectory)

### 2. Prompt-as-Data-Structure
Prompts are constructed as nested Python dicts and compiled to markdown at the last moment. This enables:
- Composition (`prompt["Instructions"] |= additional_section`)
- Conditional inclusion (cold-start, memory, plateau detection)
- Consistent formatting via `compile_prompt_to_md()`

### 3. Branch Diversity
Multiple mechanisms ensure the system doesn't get stuck:
- Draft agent requires novelty vs. existing Memory
- Top-K selection caps nodes per branch
- Stagnation detection triggers escalating interventions (evolution → fusion → aggregation)
- Plateau detection forces Tier 2/3 changes after repeated failures

### 4. Isolation for Parallelism
- Submission files: `submission_{node_id}.csv`
- Model checkpoints: `best_model_{node_id}.pth`
- CPU pinning: Each slot gets dedicated CPU cores
- Separate runfiles: `runfile_{slot}.py`

### 5. Progressive Commitment
The system starts with broad exploration and progressively commits to the best approaches:
- Early: Pure UCT exploration, diverse drafts
- Mid: Blended explore-exploit, diff-based improvements
- Late: Mostly Top-K exploitation, focused on best branches
- Final: Ensemble of top solutions

---

## 15. End-to-End Walkthrough

Here is what happens when you run:
```bash
bash run_single_task.sh denoising-dirty-documents /mle-bench/data 1
```

**T=0: Launch**
1. Shell script starts grading server on port 5006
2. Waits for health check
3. Invokes `python run.py exp_id=denoising-dirty-documents ...`

**T=0–30s: Initialization**
1. `load_cfg()` merges YAML config with CLI args
2. `load_task_desc()` reads `description.md` from competition data
3. `build_guidance_description()` looks up "denoising-dirty-documents" → "General Image" → recommends DINOv3, Siglip2
4. `prep_agent_workspace()` creates workspace, symlinks data
5. `clean_task_desc()` asks LLM to strip environment noise from description
6. `AgentSearch.__init__()` creates virtual root, determines metric direction
7. `Interpreter.__init__()` sets up 3 parallel execution slots with CPU pinning

**T=30s–5min: Phase 1 — Sequential Drafting**
1. Draft 1: `draft_agent.run()` → stepwise generation (data prep → model → training → merge) → code generated, NOT executed
2. Draft 2: Same, but Memory now shows Draft 1's plan → diversity enforced
3. Draft 3: Same, Memory shows Drafts 1+2

**T=5min onward: Phase 2 — Parallel Execution**
1. Submit all 3 draft executions (staggered by 10s)
2. Each draft runs in its own subprocess with dedicated CPUs
3. As each completes: `result_parse_agent` extracts metrics, `validate_executed_node` checks submission
4. Evaluation: compare to best, backpropagate rewards
5. New steps are submitted for completed nodes

**T=5min–6h: Exploration Phase**
- Mostly UCT selection picks nodes with high explore potential
- `improve_agent` refines working solutions via diff patches
- `debug_agent` fixes buggy solutions
- Each branch independently evolves
- Plateau detection kicks in after 2+ improvements without progress → demands bigger changes

**T=6h–10h: Mixed Exploitation**
- Top-K exploitation increasingly selects globally best nodes
- `evolution_agent` activates for stagnant branches (uses trajectory history)
- `fusion_agent` activates for stagnant branches (borrows from other branches)
- If global stagnation detected: `aggregation_agent` creates fresh drafts from branch insights

**T=10h–12h: Deep Exploitation**
- 80% Top-K exploitation, focusing on best 3 nodes
- Force-backprop mechanism breaks stagnant improvement chains
- System maximizes remaining time on most promising approaches

**T=12h: Timeout**
- `timeout --foreground` sends SIGTERM
- `interpreter.terminate_all_subprocesses()` cleans up
- `save_run()` persists final journal, config, best solution

**T=12h+: Post-Processing**
- `submission_fusion_utils.py` loads top solutions
- Attempts weighted averaging ensemble
- Produces final `submission.csv`

---

## 16. Glossary

| Term | Definition |
|---|---|
| **Branch** | A lineage of nodes from a draft to its descendants. Each draft starts a new branch (incremental `branch_id`). |
| **Branch stagnation** | When the last N successful nodes in a branch didn't beat the branch's best metric. |
| **Backpropagation** | Walking from a node up to the root, updating visit counts and cumulative rewards at each ancestor. |
| **Cold-start** | Pre-loaded knowledge mapping competition types to recommended pretrained models and code templates. |
| **Diff mode** | Code generation via SEARCH/REPLACE patches applied to the parent's code, rather than generating the entire file from scratch. |
| **Draft** | An initial solution generated from the virtual root with no prior code context. |
| **Evolution** | Intra-branch improvement guided by the full trajectory (history of changes in this branch). |
| **Force backprop** | A mechanism in mid/late search that probabilistically forces reward propagation up the tree to break stagnant chains. |
| **Fusion** | Cross-branch improvement: borrowing techniques from solutions in other branches. |
| **Fusion draft** | A completely new solution (attached to root) synthesized from all branches' best solutions. Created by the aggregation agent. |
| **Global memory** | A searchable store (BM25 + FAISS) of past node experiences (plans, outcomes) for retrieval-augmented planning. Disabled by default. |
| **Journal** | The ordered list of all SearchNodes created during a run. |
| **Local best** | The best metric achieved on the current improvement chain (tracked per-node as `local_best_node`). |
| **Magnitude-Based Reasoning** | A prompt strategy for plateau situations that forces the LLM to propose Tier 2/3 changes (not just hyperparameter tuning). |
| **MetricValue** | A direction-aware metric wrapper. Comparisons respect `maximize` flag: `MetricValue(0.9, maximize=True) > MetricValue(0.8, maximize=True)`. |
| **Patience counter** | Count of nodes since the branch's best, split into `success_patience` (non-buggy only) and `total_patience` (all). |
| **RRF** | Reciprocal Rank Fusion: `score = α/(k+rank₁) + (1-α)/(k+rank₂)`. Combines BM25 keyword and FAISS vector search results. |
| **SearchNode** | A single candidate solution in the tree, carrying code, plan, metric, execution results, and MCTS state. |
| **Stepwise generation** | Three-agent code generation: data prep → model → training, then merged by a meta-agent. |
| **Top-K** | Global selection of the K best nodes (with branch diversity caps) for exploitation. |
| **UCT** | Upper Confidence Bound for Trees: `Q/N + C·√(ln(N_parent)/N)`. Balances exploitation (high average reward) with exploration (rarely visited). |
| **Virtual root** | The root node (step 0) that serves as the anchor for all draft nodes. Has no code of its own. |
