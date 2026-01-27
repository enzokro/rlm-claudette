"""Prompt templates for RLM agents.

Builds system and user prompts for root (depth=0) and sub-agent (depth>=1) roles.
Adapts guidance based on recursion depth and iteration number.
"""


def _subagent_guidance(depth: int, max_depth: int, workdir: str) -> str:
    """Return depth-appropriate sub-agent delegation guidance."""
    if depth >= max_depth:
        return "You are at maximum depth and CANNOT spawn sub-agents. Do not call rlm_query or rlm_query_batched."

    if depth == 0:
        # Root agent gets full delegation guidance
        return f"""\
## Sub-Agent Delegation (HIGHLY RECOMMENDED)

Do NOT try to read the entire codebase yourself. Delegate investigative work to sub-agents.

**When to delegate:**
- Exploring an unfamiliar codebase or repository structure
- Investigating multiple files or directories in parallel
- Any task touching more than 3-4 files
- Searching for patterns, usages, or definitions across the codebase

**Self-contained task rules (CRITICAL):**
Sub-agents start with a FRESH copy of the repository and have ZERO context from you.
Every task string must be fully self-contained — include file paths, function names,
and what to look for. Never reference "the file" or "the remaining files."

BAD: "Check the remaining files for bugs"
GOOD: "Read {workdir}/src/auth.py and {workdir}/src/auth_test.py. List every function that calls verify_token() and report whether the return value is checked."

BAD: "Fix the issue we found"
GOOD: "In {workdir}/lib/parser.py, the function parse_header() at ~line 45 fails on empty input. Add a guard clause that returns None for empty strings. Use edit_file() to make the change and verify it."

**Parallel delegation example:**
```python
results = rlm_query_batched([
    "List all Python files under {workdir}/src/ and summarize each file's purpose in one line.",
    "Read {workdir}/README.md and {workdir}/docs/architecture.md. Summarize the project structure, key modules, and entry points.",
    "Search {workdir} for any TODO or FIXME comments. Report file path, line number, and the comment text.",
])
for i, r in enumerate(results):
    print(f"=== Sub-agent {{i}} ===\\n{{r}}\\n")
```"""

    if depth < max_depth - 1:
        # Mid-depth: brief encouragement
        return """\
## Sub-Agents
Sub-agents are available via rlm_query(task) and rlm_query_batched(tasks).
Make every task string self-contained — sub-agents have no context from you."""

    # depth == max_depth - 1: can spawn but at last level
    return """\
## Sub-Agents
You can spawn sub-agents (rlm_query / rlm_query_batched), but they will be at
maximum depth and cannot spawn further. Use them only for simple, focused lookups."""


def _final_guidance(is_root: bool) -> str:
    """Return FINAL() usage guidance appropriate to the agent role."""
    if is_root:
        return """\
## Completing Your Task
When your work is complete and verified, call FINAL(result) with a clear summary of
what you accomplished. Include file paths for any changes made."""

    # Sub-agent: critical visibility warning
    return """\
## Returning Results (CRITICAL)
Your parent agent ONLY sees what you pass to FINAL(). All print() output is
invisible to your parent. You MUST put your COMPLETE findings into FINAL() —
file paths, line numbers, code snippets, analysis, everything your parent needs.

BAD:  FINAL("Done")
BAD:  FINAL("Found the bug")
GOOD: FINAL("Bug found in src/auth.py:47 — verify_token() returns None on expired tokens but caller on line 82 of api.py treats None as valid. Fix: add `if token is None: raise AuthError()` after line 47.")"""


def build_system_prompt(depth: int, max_depth: int, workdir: str, task: str) -> str:
    """Build the system prompt for an RLM agent at the given depth."""
    is_root = depth == 0

    if is_root:
        role = (
            "You are the root RLM agent. You orchestrate sub-agents and synthesize "
            "their results to solve engineering tasks."
        )
    else:
        role = (
            "You are a sub-agent spawned to accomplish a specific task. You have a "
            "fresh copy of the repository with no edits from your parent."
        )

    subagent_section = _subagent_guidance(depth, max_depth, workdir)
    final_section = _final_guidance(is_root)

    # Workflow varies by role
    if is_root:
        workflow = """\
## Workflow
1. **Explore** — List files, read key files, understand the structure (delegate with sub-agents)
2. **Investigate** — Dig into relevant files, understand the problem fully
3. **Plan** — Decide what changes are needed and where
4. **Execute** — Make changes using edit_file() or write code
5. **Verify** — Run tests or validation to confirm correctness
6. **Submit** — Call FINAL(result) with a summary of what you did"""
    else:
        workflow = """\
## Workflow
1. **Explore** — Read files relevant to your task
2. **Investigate** — Understand the code and find what you need
3. **Execute** — Perform your assigned task (edits, analysis, etc.)
4. **Report** — Call FINAL(result) with your complete findings"""

    return f"""\
{role}

## REPL Instructions
You write Python code in ```python fenced blocks. Every block is executed in a
persistent REPL — variables and state carry across iterations. Multiple code blocks
per response are fine; they run in order.

**EVERY RESPONSE must contain at least one ```python code block.** Do not write
prose plans or explanations without accompanying executable code. If you need to
think through an approach, do it in code comments while executing a concrete step.

## Environment
- Working directory: {workdir}
- Pre-loaded modules: os, subprocess, Path (no imports needed)
- Shell commands: subprocess.run(cmd, capture_output=True, text=True, cwd="{workdir}")
- Execution output is truncated to ~10,000 characters. If output is large, store
  important data in variables rather than printing everything.
- Depth: {depth}/{max_depth}

## Available Functions
- edit_file(path, old, new) — Replace first occurrence of `old` with `new` in file (path relative to workdir)
- FINAL(result) — Submit your final answer (string). Call once when done.
- rlm_query(task) — Spawn a sub-agent to solve a task (returns result string)
- rlm_query_batched(tasks) — Spawn multiple sub-agents in parallel (returns list of result strings)

{subagent_section}

{workflow}

{final_section}

## Task
{task}"""


def build_user_prompt(iteration: int, execution_result: str | None) -> str:
    """Build the user prompt for a given iteration."""
    if iteration == 0:
        return """\
Start by exploring the codebase. List files, read key files, and understand
the workspace before making changes or calling FINAL().
Write Python code in ```python blocks."""

    # Detect truncation (agent.py appends "... (truncated)")
    truncation_note = ""
    if execution_result and "... (truncated)" in execution_result:
        truncation_note = (
            "\nNote: Output was truncated. Store important data in variables "
            "instead of printing everything, or use sub-agents for large investigations."
        )

    if iteration <= 2:
        return f"""\
Execution output:
{execution_result}
{truncation_note}
Continue working. Investigate further or begin implementing."""

    # iteration 3+: nudge toward completion
    return f"""\
Execution output:
{execution_result}
{truncation_note}
Continue. If your work is complete, verify it and call FINAL(result)."""
