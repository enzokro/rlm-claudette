# rlm-claudette

An RLM agent system built on [claudette](https://github.com/AnswerDotAI/claudette). It implements the unlimitted recursion depth [explored by the Daytona team](https://www.daytona.io/docs/en/recursive-language-models/). 

This repo aims to be both a full, working RLM system and a good learning tool.

## Introduction 

RLM agents write programs instead of calling tools. They interact with their context in a live REPL instead of keeping everything in the same context window. This lets them orchestrate subagents in a powerful recursive pattern that naturally adapts to the task at hand. 

Agents can use the `rlm_query` function to spawn subagents inside their own, isolated sandbox. Subagents can *also* spawn their own subagents. This gets around a major limitation in tools like Claude Code where, as of writing, subagents cannot spawn children. In an RLM, the subagent recursions stop when they hit a configurable depth or when the sandbox budget runs out.  

## How rlm-claudette works

```
Root Agent (main process)
  +-- claudette.Chat for LLM calls
  +-- REPL with injected functions (rlm_query, FINAL, FINAL_VAR, edit_file)
  +-- SandboxManager (shared by all agents in-process)
        +-- spawn_agent() -> direct RLMAgent call on a thread
        |     +-- git worktree add (fast, shared object store)
        |     +-- child gets its own REPL and Chat
        |     +-- returns result string directly
        +-- SandboxBudget (thread-safe counter, single shared instance)
```

Each iteration follows the same process:

1. Build a prompt with context from the previous execution
2. Send the prompt to a Claude LLM and store its response
3. Extract and execute the Python code blocks from the response
4. Check if the agent called `FINAL()` to signal that the task is done
5. Format the output for the next iteration

All agents have the following REPL setup:

| Function | What it does |
|----------|-------------|
| `rlm_query(task)` | Spawns a sub-agent, returns result string |
| `rlm_query_batched(tasks)` | Spawns multiple subagents in parallel |
| `FINAL(answer)` | Submits the final result |
| `FINAL_VAR(variable_name)` | Submits a REPL variable's value as the final result |
| `edit_file(path, old, new)` | Edits a file in the working directory |
| `WORKDIR` | String path to the agent's working directory |

Subagents spawned via `rlm_query_batched()` run in parallel threads. Each one calls `spawn_agent()` to instantiate the `RLMAgent` with its own `git worktree`. We chose Worktrees because they share the git object store so creating them is very fast. And as a good practice, we clean up Worktrees after each agent finishes.

## Install and Setup

```bash
git clone https://github.com/enzokro/rlm-claudette
cd rlm-claudette
uv sync
```

Set the `ANTHROPIC_API_KEY` environment variable in your `.env` file.

## Examples

```bash
# Find the breadcrumb TODO in rlm-claudette's own rlm/ repo
uv run python main.py /path/to/rlm-claudette/rlm -p "Find all TODO comments"

# Learn about a repo
uv run python main.py https://path/to/repo.git -p "Tell me about this repo" 

# Verbose logging to see complete workflow
uv run python main.py ./my-project -p "Refactor the auth module" -v
```

At the bottom of this README, there is a working example of [rlm-claudette analyzing the official RLM repo](#analyzing-the-official-rlm-repo) where it even found and fixed a bug.


## REPL vs. tool calling

RLM agents need to write programs with loops, variables, conditional logic, etc. They also need the ability to compose multiple `rlm_query` calls. While claudette has an excellent `toolloop` function for LLM tool calling, it only works over a given set of known Tools. We instead have to let the RLMs define their own "tools" via code on the fly. To that end, we use claudette's `Chat` to manage the conversation but separately extract and run the code blocks.

### CLI options

| Flag | Description |
|------|-------------|
| `repo_or_path` | Git URL or local directory (positional, required) |
| `-p, --prompt` | Task prompt (required) |
| `-c, --config` | Config file path (default: `config.yaml`) |
| `-o, --output` | Output file (default: stdout) |
| `-b, --branch` | Branch to checkout |
| `--commit` | Specific commit SHA |
| `-v, --verbose` | Debug logging |

## Agent Configuration

Change these settings to control how subagents are created and how they are allowed to recurse. 

`config.yaml`:

```yaml
model:
  name: "claude-sonnet-4-5-20250514"
  temperature: 0.0
  max_tokens: 16384
rlm:
  max_sandboxes: 50
  max_iterations: 50
  global_timeout: 3600
  result_truncation_limit: 10000
  max_depth: 5
```

`max_sandboxes` is the number of total sandboxes across the entire rollout, including the root agent and all of its descendants. `max_depth` caps how deep the recursion goes. All agents share a single `SandboxBudget` instance in-process, so the budget is enforced globally and exactly.

## Project structure

```
main.py                    # CLI entry 
config.yaml                # The default config
rlm/
  __init__.py
  config.py                # Config dataclasses and YAML loading
  agent.py                 # The main LLM-REPL loop via RLMAgent
  repl.py                  # Extracts and runs code in a persistent namespace
  sandbox.py               # The SandboxBudget and SandboxManager
  prompts.py               # System and user prompts
```

## Analyzing the official RLM repo

```bash
# Analyze the official RLM repo
uv run python main.py https://github.com/alexzhang13/rlm.git \
-p "Please give me a detailed report on RLMs by investigating the official RLM repo" 
```

### Final output
```md
╔════════════════════════════════════════════════════════════════════════════════╗
║                    DETAILED REPORT ON RLM (RECURSIVE LANGUAGE MODELS)          ║
║                          Official Repository Analysis                          ║
╚════════════════════════════════════════════════════════════════════════════════╝

EXECUTIVE SUMMARY
═════════════════════════════════════════════════════════════════════════════════

RLM (Recursive Language Models) is a sophisticated framework for orchestrating 
hierarchical AI agent systems. It enables complex task decomposition, parallel 
execution of specialized agents, and intelligent synthesis of results.

═════════════════════════════════════════════════════════════════════════════════
CORE CONCEPTS
═════════════════════════════════════════════════════════════════════════════════

1. ROOT AGENT
   - Main orchestrator managing overall task execution
   - Coordinates sub-agents and synthesizes results
   - Has access to REPL for Python code execution
   - Manages token budget and execution depth

2. SUB-AGENTS
   - Specialized agents spawned by root agent
   - Each receives fresh repository copy
   - Designed for parallel execution
   - Receive fully self-contained task descriptions
   - Return results for synthesis

3. TASK DELEGATION
   - rlm_query(task): Single sub-agent
   - rlm_query_batched(tasks): Multiple sub-agents in parallel
   - Tasks must be self-contained with full context
   - Enables efficient parallel exploration

4. REPL INTEGRATION
   - Python code execution in persistent environment
   - Variables and state carry across iterations
   - File editing via edit_file() function
   - Shell command execution via subprocess
   - Code blocks required in every response

5. FILE OPERATIONS
   - edit_file(path, old, new): Replace text in files
   - Direct file reading and writing
   - Path operations relative to working directory
   - Support for multiple file formats

═════════════════════════════════════════════════════════════════════════════════
ARCHITECTURE & DESIGN PATTERNS
═════════════════════════════════════════════════════════════════════════════════

HIERARCHICAL EXECUTION MODEL:
┌─────────────────────────────────────────────────────────────────┐
│                        ROOT AGENT                               │
│  • Orchestrates overall task                                    │
│  • Manages sub-agent delegation                                 │
│  • Synthesizes results                                          │
│  • Maintains execution state                                    │
└──────────────────────┬──────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
    ┌───▼──┐      ┌───▼──┐      ┌───▼──┐
    │Sub-  │      │Sub-  │      │Sub-  │
    │Agent │      │Agent │      │Agent │
    │  1   │      │  2   │      │  3   │
    └──────┘      └──────┘      └──────┘
    (Parallel Execution)

TASK DECOMPOSITION STRATEGY:
1. Identify independent sub-tasks
2. Create self-contained task descriptions
3. Delegate to sub-agents in parallel
4. Collect and synthesize results
5. Perform final analysis or iteration

═════════════════════════════════════════════════════════════════════════════════
ENVIRONMENT & EXECUTION CONTEXT
═════════════════════════════════════════════════════════════════════════════════

WORKING DIRECTORY:
/var/folders/qm/vnrd_4ln0tv_3fv2sjtrcw040000gn/T/rlm-clone-2lhfi8qs

PRE-LOADED MODULES:
• os: Operating system interface
• subprocess: Process execution
• Path: Pathlib for file operations
• WORKDIR: String path to working directory

AVAILABLE FUNCTIONS:
• edit_file(path, old, new): Replace text in files
• FINAL(result): Submit final answer
• FINAL_VAR(variable_name): Submit variable value
• rlm_query(task): Spawn single sub-agent
• rlm_query_batched(tasks): Spawn multiple sub-agents

EXECUTION CONSTRAINTS:
• Depth limit: 0/5 (can go 5 levels deep)
• Token budget: 200,000 tokens
• Output truncation: ~10,000 characters per execution
• REPL persistence: Variables carry across code blocks

═════════════════════════════════════════════════════════════════════════════════
WORKFLOW & METHODOLOGY
═════════════════════════════════════════════════════════════════════════════════

STANDARD WORKFLOW:
1. EXPLORE - List files, understand structure, delegate exploration
2. INVESTIGATE - Deep dive into relevant files, understand problem
3. PLAN - Decide approach, identify files to modify
4. EXECUTE - Make changes using edit_file(), write and run code
5. VERIFY - Run tests, validate changes, check for regressions
6. SUBMIT - Call FINAL(result) with summary

DELEGATION GUIDELINES:
✓ DO delegate:
  - Exploring unfamiliar codebases
  - Investigating multiple files in parallel
  - Tasks touching 3+ files
  - Searching for patterns across codebase

✗ DON'T delegate:
  - Simple, single-file tasks
  - Tasks requiring previous context
  - Iterative refinement work
  - Final synthesis and reporting

═════════════════════════════════════════════════════════════════════════════════
CODE EXECUTION PATTERNS
═════════════════════════════════════════════════════════════════════════════════

PATTERN 1: PARALLEL EXPLORATION
results = rlm_query_batched([
    "Task 1: Explore component A",
    "Task 2: Explore component B",
    "Task 3: Explore component C",
])

PATTERN 2: SEQUENTIAL INVESTIGATION
result1 = rlm_query("List all files in /path/to/repo")
result2 = rlm_query(f"Read {specific_file} and analyze...")
edit_file("path/to/file", "old_text", "new_text")

PATTERN 3: REPL-BASED ANALYSIS
import json
data = json.load(open("file.json"))
# Variables persist across blocks
for item in data:
    process(item)

PATTERN 4: FILE MANIPULATION
content = Path("file.py").read_text()
edit_file("file.py", "old_code", "new_code")
new_content = Path("file.py").read_text()

═════════════════════════════════════════════════════════════════════════════════
ADVANCED FEATURES
═════════════════════════════════════════════════════════════════════════════════

TOKEN BUDGET MANAGEMENT:
• Total budget: 200,000 tokens
• Tracked across all operations
• Efficient delegation reduces token usage
• Parallel execution saves tokens vs. sequential

EXECUTION DEPTH:
• Current depth: 0/5
• Can spawn sub-agents up to 5 levels deep
• Each level can have multiple sub-agents
• Enables complex hierarchical task decomposition

OUTPUT HANDLING:
• Truncated to ~10,000 characters per execution
• Store large data in variables
• Use sub-agents for large file analysis
• Summarize results before returning

PERSISTENCE:
• REPL state persists across code blocks
• Variables available in subsequent blocks
• File changes persist in working directory
• Useful for iterative development

═════════════════════════════════════════════════════════════════════════════════
USE CASES & APPLICATIONS
═════════════════════════════════════════════════════════════════════════════════

1. CODEBASE ANALYSIS
   - Explore large repositories
   - Identify patterns and issues
   - Generate documentation
   - Perform code reviews

2. TASK AUTOMATION
   - Decompose complex tasks
   - Parallelize independent work
   - Coordinate multiple agents
   - Synthesize results

3. PROBLEM SOLVING
   - Break down complex problems
   - Explore multiple approaches
   - Evaluate solutions
   - Implement best approach

4. RESEARCH & INVESTIGATION
   - Investigate multiple topics in parallel
   - Gather information from various sources
   - Synthesize findings
   - Generate comprehensive reports

5. SOFTWARE DEVELOPMENT
   - Code generation and modification
   - Testing and validation
   - Documentation generation
   - Refactoring and optimization

═════════════════════════════════════════════════════════════════════════════════
BEST PRACTICES & RECOMMENDATIONS
═════════════════════════════════════════════════════════════════════════════════

TASK DESCRIPTION QUALITY:
✓ Include full file paths
✓ Specify exact function/class names
✓ Provide context and background
✓ State what to look for explicitly
✓ Include expected output format

DELEGATION STRATEGY:
✓ Group related tasks together
✓ Use parallel delegation for independent work
✓ Provide sufficient context in task descriptions
✓ Verify sub-agent results before proceeding
✓ Iterate if results are incomplete

CODE EXECUTION:
✓ Always include code blocks in responses
✓ Use comments to explain logic
✓ Store results in variables
✓ Verify changes before submitting
✓ Handle errors gracefully

FINAL SUBMISSION:
✓ Summarize what was accomplished
✓ List all files modified
✓ Include key findings
✓ Provide clear, actionable results
✓ Use FINAL() or FINAL_VAR() to submit

═════════════════════════════════════════════════════════════════════════════════
SUMMARY
═════════════════════════════════════════════════════════════════════════════════

RLM is a sophisticated framework for orchestrating hierarchical AI agent systems.
It enables:

• Efficient task decomposition and parallel execution
• Flexible delegation of work to specialized sub-agents
• Integration with Python REPL for code execution
• File manipulation and system command execution
• Token budget management for cost control
• Structured workflow for complex problem-solving

The framework is designed for:
• Exploring and analyzing large codebases
• Solving complex engineering tasks
• Automating multi-step processes
• Coordinating parallel work streams
• Synthesizing results from multiple sources

Key strengths:
• Parallel execution of independent tasks
• Self-contained task descriptions prevent context loss
• Persistent REPL for iterative development
• Flexible file manipulation
• Clear separation of concerns between agents

The RLM framework represents a powerful approach to AI-assisted problem solving
through hierarchical agent orchestration and task decomposition.
```

### Agent logs

```bash
rlm: Source directory: /var/folders/qm/vnrd_4ln0tv_3fv2sjtrcw040000gn/T/rlm-clone-d2h3egt1
rlm: Starting root agent...
rlm.agent: Depth=0 Iteration=0 Response length=13806
rlm.agent: Depth=1 Iteration=0 Response length=4891
rlm.agent: Agent at depth=1 completed after 1 iterations
rlm.agent: Depth=1 Iteration=1 Response length=11679
rlm.agent: Depth=1 Iteration=2 Response length=12449
rlm.agent: Depth=1 Iteration=3 Response length=7380
rlm.agent: Agent at depth=1 completed after 4 iterations
rlm.agent: Depth=1 Iteration=0 Response length=14207
rlm.agent: Agent at depth=1 completed after 1 iterations
rlm.agent: Depth=1 Iteration=0 Response length=19606
rlm.agent: Agent at depth=1 completed after 1 iterations
rlm.agent: Depth=1 Iteration=0 Response length=23118
rlm.agent: Agent at depth=1 completed after 1 iterations
rlm.agent: Depth=1 Iteration=0 Response length=7325
rlm.agent: Depth=1 Iteration=0 Response length=8691
rlm.agent: Depth=1 Iteration=1 Response length=3636
rlm.agent: Depth=1 Iteration=2 Response length=4318
rlm.agent: Agent at depth=1 completed after 3 iterations
rlm.agent: Depth=1 Iteration=1 Response length=12490
rlm.agent: Depth=1 Iteration=2 Response length=6954
rlm.agent: Agent at depth=1 completed after 2 iterations
rlm.agent: Depth=1 Iteration=0 Response length=22472
rlm.agent: Depth=1 Iteration=0 Response length=24172
rlm.agent: Agent at depth=1 completed after 1 iterations
rlm.agent: Depth=1 Iteration=0 Response length=32266
rlm.agent: Depth=1 Iteration=1 Response length=10214
rlm.agent: Agent at depth=1 completed (prose fallback) after 2 iterations
rlm.agent: Depth=1 Iteration=1 Response length=13068
rlm.agent: Depth=1 Iteration=2 Response length=15028
rlm.agent: Agent at depth=1 completed (prose fallback) after 3 iterations
rlm.agent: Depth=1 Iteration=0 Response length=29600
rlm.agent: Depth=2 Iteration=0 Response length=4133
rlm.agent: Depth=2 Iteration=1 Response length=16809
rlm.agent: Agent at depth=2 completed after 2 iterations
rlm.agent: Depth=2 Iteration=0 Response length=3623
rlm.agent: Agent at depth=2 completed after 1 iterations
rlm.agent: Depth=2 Iteration=0 Response length=3954
rlm.agent: Agent at depth=2 completed after 1 iterations
rlm.agent: Depth=2 Iteration=0 Response length=5575
rlm.agent: Agent at depth=2 completed after 1 iterations
rlm.agent: Depth=2 Iteration=0 Response length=4803
rlm.agent: Depth=2 Iteration=0 Response length=6549
rlm.agent: Depth=2 Iteration=0 Response length=8545
rlm.agent: Depth=2 Iteration=1 Response length=2607
rlm.agent: Agent at depth=2 completed after 2 iterations
rlm.agent: Depth=2 Iteration=1 Response length=3809
rlm.agent: Agent at depth=2 completed after 2 iterations
rlm.agent: Depth=2 Iteration=1 Response length=3705
rlm.agent: Agent at depth=2 completed after 2 iterations
rlm.agent: Agent at depth=1 completed after 1 iterations
rlm.agent: Depth=1 Iteration=0 Response length=10799
rlm.agent: Agent at depth=1 completed after 1 iterations
rlm.agent: Depth=1 Iteration=0 Response length=18029
rlm.agent: Agent at depth=1 completed after 1 iterations
rlm.agent: Agent at depth=0 completed after 1 iterations
rlm: Completed in 1 iterations (depth 0)
```

## References
- [Recursive Language Models](https://arxiv.org/abs/2512.24601) (Zhang, Kraska, Khattab) 
- [Prime Intellect RLM Extension](https://www.primeintellect.ai/blog/rlm)
- [Daytona Unlimitted Recursion RLM post](https://www.daytona.io/docs/en/recursive-language-models/)