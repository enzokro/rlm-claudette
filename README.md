# rlm-claudette

An RLM agent system built on [claudette](https://github.com/AnswerDotAI/claudette). This repo is meant to be a full, working RLM agent and a good learning tool.

RLM agents write programs instead of calling tools. They interact with their prompts and context via a live REPL instead of holding everything in the same, static context window. This also lets them orchestrate sub-agents in a very powerful recursive pattern that naturally adapts to the task at hand. 

Agents have access to the `rlm_query` function that spawns a sub-agent with its own isolated working directory inside a child process. Sub-agents can then also spawn their own sub-agents. This already gets around a major limitation in tools like Claude Code where, as of writing, sub-agents cannot spawn their own agents. In an RLM the sub-agent recursion stops at either a configurable depth, or when the sandbox budget runs out.

The architecture is described in [Recursive Language Models](https://arxiv.org/abs/2512.24601) (Zhang, Kraska, Khattab) and extended by [Prime Intellect](https://www.primeintellect.ai/blog/rlm). We implemented it here using claudette for LLM calls and local subprocesses with `git worktree` for sandboxes.

## Why REPL, not tool-calling 

RLM agents need to write multi-statement programs with loops, conditional logic, etc. They must also be able to compose multiple `rlm_query` calls. While claudette has an excellent `toolloop` function to handle LLM tool calls, it runs from a fixed set of given Tools. We instead have to let the RLMs define their own "tools" via code on the fly. To that end, we use claudette's `Chat` for easy conversation management but extract and run the code blocks separately.

## How rlm-claudette works

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

Each iteration does the following:

1. Builds a prompt with context from the previous execution
2. Sends the prompt to a Claude LLM and stores its response
3. Extracts and executes the Python code blocks from the response
4. Checks if the agent called `FINAL()` to signal that the task is finished
5. Formats the output for the next iteration

Agents are given the following REPL setup:

| Function | What it does |
|----------|-------------|
| `rlm_query(task)` | Spawns a sub-agent, returns result string |
| `rlm_query_batched(tasks)` | Spawns multiple sub-agents in parallel |
| `FINAL(answer)` | Submits the final result |
| `edit_file(path, old, new)` | Edits a file in the working directory |

Sub-agents spawned via `rlm_query_batched()` run in parallel threads, each calling `spawn_agent()` which creates a subprocess in its own `git worktree`. Worktrees share the git object store so creating them is very fast. As a good practice, we remove the worktress after the subprocess completes. 

## Setup

```bash
git clone https://github.com/enzokro/rlm-claudette
cd rlm-claudette
uv sync
```

Set the `ANTHROPIC_API_KEY` environment variable in your `.env` file.

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

## Agent Configuration

Change these settings to control how sub-agents are created and how they recurse. 

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

`max_sandboxes` is the number of total sandboxes across the entire rollout, including the root agent and all of its descendants. `max_depth` caps how deep the recursion goes. Note that sub-agents inherit any remaining budget, not the original amount.

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
  subprocess_runner.py     # Entry point for child processes
```
