"""Prompt templates for RLM agents."""


def build_system_prompt(depth: int, max_depth: int, workdir: str, task: str) -> str:
    """Build the system prompt for an RLM agent at the given depth."""
    can_spawn = "You may spawn sub-agents with rlm_query(task) or rlm_query_batched(tasks)." if depth < max_depth else "You are at maximum depth and CANNOT spawn sub-agents."

    return f"""\
You are an RLM agent. You write Python code in ```python blocks to accomplish tasks.

Available functions:
- rlm_query(task): Spawn a sub-agent to solve a sub-task (returns result string).
- rlm_query_batched(tasks): Spawn multiple sub-agents in parallel (returns list of results).
- FINAL(answer): Call this when you have the final answer to report back.
- edit_file(path, old, new): Replace text in a file.

Your working directory is: {workdir}
Depth: {depth}/{max_depth}. {can_spawn}

Use subprocess.run() for shell commands. Call FINAL(result) when you are done.

Task:
{task}"""


def build_user_prompt(iteration: int, execution_result: str | None) -> str:
    """Build the user prompt for a given iteration."""
    if iteration == 0:
        return "Begin working on your task. Write Python code in ```python blocks."
    return f"""\
REPL output from your code:
{execution_result}

Continue working. Write more code or call FINAL(result) when done."""
