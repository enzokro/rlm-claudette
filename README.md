# rlm-claudette

A recursive language model agent system built on [claudette](https://github.com/AnswerDotAI/claudette).

Agents don't call tools. They write programs. Each iteration of the loop, Claude produces Python code in fenced blocks. We extract it, execute it in a persistent namespace, and feed the output back. The namespace includes `rlm_query`---a function that spawns a sub-agent as a child process with its own isolated working directory. Sub-agents can spawn their own sub-agents. The recursion bottoms out at a configurable depth, or when the sandbox budget runs dry.

This is the architecture described in [Recursive Language Models](https://arxiv.org/abs/2512.24601) (Zhang, Kraska, Khattab) and extended by [Prime Intellect](https://www.primeintellect.ai/blog/rlm), adapted here to use claudette for LLM calls and local subprocesses with `git worktree` for isolation.

## Why REPL, not tool-calling

The RLM architecture requires agents to write multi-statement programs---loops, variable assignment, composing multiple `rlm_query` calls, conditional logic. claudette's `toolloop` handles single function invocations per turn. That's the wrong granularity. We use `Chat` for conversation management but extract and exec code blocks ourselves.

## How it works

```
Root Agent (main process)
  +-- claudette.Chat for LLM calls
  +-- REPL with injected functions (rlm_query, FINAL, edit_file)
  +-- SandboxManager
        +-- spawn_agent() -> subprocess (python -m rlm.subprocess_runner)
        |     +-- git worktree add (fast, shared object store)
        |     +-- passes task/config via stdin JSON
        |     +-- child runs its own agent loop
        |     +-- returns result via stdout JSON
        +-- SandboxBudget (thread-safe counter)
```

Each iteration:

1. Build a prompt with context from the previous execution
2. Get an LLM completion from Claude
3. Extract and execute Python code blocks in a persistent namespace
4. Check if the agent called `FINAL()` to submit results
5. Format the output for the next iteration

Inside the REPL, agents have access to:

| Function | What it does |
|----------|-------------|
| `rlm_query(task)` | Spawn a sub-agent, returns result string |
| `rlm_query_batched(tasks)` | Spawn multiple sub-agents in parallel |
| `FINAL(answer)` | Submit final result |
| `edit_file(path, old, new)` | Edit a file in the working directory |

Sub-agents spawned via `rlm_query_batched()` run in parallel threads, each calling `spawn_agent()` which creates a subprocess in its own `git worktree`. Worktrees share the git object store, so creation is fast. They're removed after the subprocess completes. For non-git source directories, we fall back to `shutil.copytree`.

## Setup

```bash
git clone <this-repo>
cd rlm-claudette
uv sync
```

Requires an `ANTHROPIC_API_KEY` environment variable.

## Usage

```bash
# Run against a local repo
uv run python main.py /path/to/repo -p "Find all TODO comments and fix the easiest one"

# Run against a git URL
uv run python main.py https://github.com/org/repo -p "Investigate the test suite" -b main

# Write output to file
uv run python main.py ./my-project -p "List all API endpoints" -o result.txt

# Verbose logging
uv run python main.py ./my-project -p "Refactor the auth module" -v
```

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

## Configuration

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

`max_sandboxes` is the total budget across the entire rollout---root agent plus all descendants. `max_depth` caps how deep the recursion goes. Sub-agents inherit the remaining budget, not the original.

## Project structure

```
main.py                    # CLI entry point
config.yaml                # Default configuration
rlm/
  __init__.py
  config.py                # Config dataclasses + YAML loading
  agent.py                 # RLMAgent: the LLM-REPL iteration loop
  repl.py                  # Code extraction + exec in persistent namespace
  sandbox.py               # SandboxBudget + SandboxManager
  prompts.py               # System/user prompt templates
  subprocess_runner.py     # Child process entry point
```
