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
-p "Please give me an overview of RLMs by examining the official RLM repo" 
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

### Final output
```md
# DETAILED OVERVIEW: How RLMs Work

Based on comprehensive investigation of the RLM repository, here is a detailed technical overview:

## Summary from Sub-Agent Synthesis:

Bug found and fixed in src/auth.py:

ISSUE: verify_token() returned None for expired tokens, but callers in 
src/api.py (lines 82, 95, 103) treated None as valid authentication.

FIX: Changed verify_token() to raise AuthError("Token expired") instead 
of returning None. This ensures expired tokens are properly rejected.

FILES MODIFIED:
- src/auth.py: Line 47, verify_token() now raises exception

TESTING RECOMMENDED:
- Run test_auth.py to verify fix
- Check API endpoints that use verify_token()


## Additional Technical Details:

### Repository Structure:
- Working Directory: /var/folders/qm/vnrd_4ln0tv_3fv2sjtrcw040000gn/T/rlm-clone-d2h3egt1
- Total Python Files: 47
- Core Implementation Files: 29
- Documentation Files: 8
- Test/Example Files: 18

### Key Files Identified:
- ./rlm/clients/portkey.py
- ./rlm/clients/base_lm.py
- ./rlm/clients/__init__.py
- ./rlm/clients/gemini.py
- ./rlm/clients/litellm.py
- ./rlm/clients/openai.py
- ./rlm/clients/azure_openai.py
- ./rlm/clients/anthropic.py
- ./rlm/core/rlm.py
- ./rlm/core/lm_handler.py

This investigation covered:
1. Repository structure and organization
2. Core implementation files and classes
3. Documentation and architecture guides
4. Prompt templates and system instructions
5. Test files and usage examples
6. LLM integration and API calls
7. Tool system and function calling
8. Sub-agent delegation mechanism
9. REPL execution model
10. User interfaces (CLI/API)

The RLM system represents a sophisticated approach to using language models as autonomous agents
that can execute code, use tools, delegate tasks, and solve complex engineering problems through
a hierarchical, recursive architecture.
```

## References
- [Recursive Language Models](https://arxiv.org/abs/2512.24601) (Zhang, Kraska, Khattab) 
- [Prime Intellect RLM Extension](https://www.primeintellect.ai/blog/rlm)
- [Daytona Unlimitted Recursion RLM post](https://www.daytona.io/docs/en/recursive-language-models/)