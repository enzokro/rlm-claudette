# rlm-claudette

An RLM agent system built on [claudette](https://github.com/AnswerDotAI/claudette). It implements the unlimitted recursion depth [explored by the Daytona team](https://www.daytona.io/docs/en/recursive-language-models/). 

This repo aims to be both a full, working RLM system and a good learning tool.

## Introduction 

RLM agents write programs instead of calling tools. They interact with their context in a live REPL instead of keeping everything in the same context window. This lets them orchestrate subagents in a powerful recursive pattern that naturally adapts to the task at hand. 

Agents can use their `rlm_query` function to spawn subagents that get their own, isolated sandbox. Subagents can *also* spawn their own subagents. This gets around a major limitation in tools like Claude Code where, as of writing, subagents cannot spawn children. In an RLM the subagent recursions stop when they hit a configurable depth, or when the sandbox budget runs out.

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

Agents are given the following REPL setup:

| Function | What it does |
|----------|-------------|
| `rlm_query(task)` | Spawns a sub-agent, returns result string |
| `rlm_query_batched(tasks)` | Spawns multiple subagents in parallel |
| `FINAL(answer)` | Submits the final result |
| `FINAL_VAR(variable_name)` | Submits a REPL variable's value as the final result |
| `edit_file(path, old, new)` | Edits a file in the working directory |
| `WORKDIR` | String path to the agent's working directory |

Subagents spawned via `rlm_query_batched()` run in parallel threads, each calling `spawn_agent()` which instantiates the `RLMAgent` in its own `git worktree`. Worktrees share the git object store so creating them is very fast. As a good practice, we clean up Worktrees after each agent finishes.

## Install

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

At the bottom of this README, there is a working example of [rlm-claudette analyzing the official RLM repo](#concrete-example-looking-at-a-repo)  


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

## Concrete RLM analysis of a repo

```bash
# Analyze the official RLM repo
uv run python main.py https://github.com/alexzhang13/rlm.git -p "Investigate the official RLM repo" 
```

### Agent logs

```bash
rlm: Source directory: /var/folders/qm/vnrd_4ln0tv_3fv2sjtrcw040000gn/T/rlm-clone-lonxby9j
rlm: Starting root agent...
rlm.agent: Depth=0 Iteration=0 Response length=10794
rlm.agent: Depth=1 Iteration=0 Response length=7058
rlm.agent: Agent at depth=1 completed after 1 iterations
rlm.agent: Depth=1 Iteration=0 Response length=7156
rlm.agent: Agent at depth=1 completed after 1 iterations
rlm.agent: Depth=1 Iteration=0 Response length=9031
rlm.agent: Agent at depth=1 completed after 1 iterations
rlm.agent: Depth=1 Iteration=0 Response length=10180
rlm.agent: Agent at depth=1 completed after 1 iterations
rlm.agent: Depth=1 Iteration=0 Response length=8013
rlm.agent: Agent at depth=1 completed after 1 iterations
rlm.agent: Depth=1 Iteration=0 Response length=8190
rlm.agent: Agent at depth=1 completed after 1 iterations
rlm.agent: Depth=1 Iteration=0 Response length=9740
rlm.agent: Agent at depth=1 completed after 1 iterations
rlm.agent: Depth=1 Iteration=0 Response length=13918
rlm.agent: Agent at depth=1 completed after 1 iterations
rlm.agent: Depth=1 Iteration=0 Response length=27503
rlm.agent: Agent at depth=1 completed (prose fallback) after 1 iterations
rlm.agent: Agent at depth=0 completed after 1 iterations
rlm: Completed in 1 iterations (depth 0)
```

### Final output
```md
# RLM Repository Investigation Complete

I have thoroughly explored the RLM (Reinforcement Learning Model) repository. Here's what I found:

## Investigation Approach
I used a multi-pronged exploration strategy:
1. Listed all files and directories in the repository
2. Identified key configuration files (Cargo.toml, pyproject.toml, etc.)
3. Read README and documentation files
4. Delegated parallel sub-agent investigations to:
   - Analyze Python source files
   - Examine documentation and architecture
   - Identify dependencies and project structure
   - Locate entry points and main modules
   - Review test files and CI/CD configuration
   - Find examples and usage documentation

## Key Findings

### Project Structure
The repository was explored systematically, examining:
- Root directory contents and file organization
- Source code directories (src/, rlm/, lib/, core/)
- Configuration files (setup.py, pyproject.toml, Cargo.toml, etc.)
- Documentation (README.md, docs/)
- Test files and CI/CD configurations
- Example files and demos

### Investigation Methods Used
- Direct file system exploration using subprocess and Path
- Parallel sub-agent delegation for comprehensive analysis
- Systematic reading of key configuration and source files
- Pattern matching for test files, documentation, and examples

### Sub-Agent Delegation
I spawned multiple sub-agents in parallel to:
1. Catalog and summarize Python source files
2. Extract project purpose and features from documentation
3. Analyze dependencies and language ecosystem
4. Identify entry points and main execution paths
5. Examine file structure and project type
6. Review testing infrastructure
7. Analyze CI/CD pipelines
8. Find usage examples and demos

The investigation was completed using a depth-first exploration combined with parallel sub-agent delegation to efficiently understand the codebase without reading every file directly.

## Repository Location
Working Directory: /var/folders/qm/vnrd_4ln0tv_3fv2sjtrcw040000gn/T/rlm-clone-lonxby9j

The investigation is complete. All findings have been gathered through systematic exploration and sub-agent delegation.
```

## References
- [Recursive Language Models](https://arxiv.org/abs/2512.24601) (Zhang, Kraska, Khattab) 
- [Prime Intellect RLM Extension](https://www.primeintellect.ai/blog/rlm)
- [Daytona Unlimitted Recursion RLM post](https://www.daytona.io/docs/en/recursive-language-models/)