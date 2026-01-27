"""RLM Agent: the LLM-REPL iteration loop."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from claudette import Chat

from rlm.config import Config
from rlm.prompts import build_system_prompt, build_user_prompt
from rlm.repl import REPL
from rlm.sandbox import SandboxManager

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result from an agent run."""
    result: str | None
    iterations: int
    depth: int


class RLMAgent:
    """Recursive Language Model agent with LLM-REPL loop."""

    def __init__(
        self,
        config: Config,
        task: str,
        workdir: str,
        depth: int = 0,
        sandbox_manager: SandboxManager | None = None,
    ):
        self._config = config
        self._task = task
        self._workdir = workdir
        self._depth = depth
        self._sandbox_mgr = sandbox_manager
        self._start_time = time.time()

    def run(self) -> AgentResult:
        """Drive the LLM-REPL loop. Returns AgentResult."""
        sp = build_system_prompt(
            depth=self._depth,
            max_depth=self._config.rlm.max_depth,
            workdir=self._workdir,
            task=self._task,
        )

        chat = Chat(
            model=self._config.model.name,
            sp=sp,
            temp=self._config.model.temperature,
        )

        namespace_extras = self._build_namespace()
        repl = REPL(namespace_extras=namespace_extras, workdir=self._workdir)

        execution_result = None
        iterations = 0

        for iteration in range(self._config.rlm.max_iterations):
            if self._is_timeout():
                logger.warning("Agent timed out at depth=%d, iteration=%d", self._depth, iteration)
                break

            user_prompt = build_user_prompt(iteration, execution_result)
            response = chat(user_prompt, maxtok=self._config.model.max_tokens)

            # Extract text from claudette response
            response_text = self._extract_text(response)
            logger.info("Depth=%d Iteration=%d Response length=%d", self._depth, iteration, len(response_text))

            repl_result = repl.execute_response(response_text)
            iterations = iteration + 1

            if repl_result.final_answer is not None:
                logger.info("Agent at depth=%d completed after %d iterations", self._depth, iterations)
                return AgentResult(
                    result=repl_result.final_answer,
                    iterations=iterations,
                    depth=self._depth,
                )

            # Truncate output for next prompt
            execution_result = repl_result.output
            if len(execution_result) > self._config.rlm.result_truncation_limit:
                execution_result = execution_result[:self._config.rlm.result_truncation_limit] + "\n... (truncated)"

        logger.warning("Agent at depth=%d exhausted iterations (%d)", self._depth, iterations)
        return AgentResult(result=None, iterations=iterations, depth=self._depth)

    def _extract_text(self, response) -> str:
        """Extract text content from claudette response.

        claudette Chat.__call__ returns an anthropic Message object.
        Use the .content attribute which is a list of ContentBlock objects.
        Each ContentBlock has a .text attribute.
        """
        if hasattr(response, 'content'):
            parts = []
            for block in response.content:
                if hasattr(block, 'text'):
                    parts.append(block.text)
            return "\n".join(parts)
        return str(response)

    def _build_namespace(self) -> dict:
        """Build the namespace extras for the REPL."""
        ns = {}
        ns["rlm_query"] = self._make_rlm_query()
        ns["rlm_query_batched"] = self._make_rlm_query_batched()
        ns["edit_file"] = self._make_edit_file()
        return ns

    def _make_rlm_query(self):
        """Return closure that spawns a sub-agent."""
        def rlm_query(task: str) -> str:
            if self._sandbox_mgr is None:
                return "Error: no sandbox manager (sub-agent spawning disabled)"
            if self._depth >= self._config.rlm.max_depth:
                return "Error: maximum recursion depth reached"
            return self._sandbox_mgr.spawn_agent(task, self._depth + 1)
        return rlm_query

    def _make_rlm_query_batched(self):
        """Return closure that spawns multiple sub-agents in parallel."""
        rlm_query = self._make_rlm_query()

        def rlm_query_batched(tasks: list[str]) -> list[str]:
            results = [""] * len(tasks)
            with ThreadPoolExecutor(max_workers=min(10, len(tasks))) as executor:
                future_to_idx = {
                    executor.submit(rlm_query, task): i
                    for i, task in enumerate(tasks)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        results[idx] = f"Error: {e}"
            return results
        return rlm_query_batched

    def _make_edit_file(self):
        """Return closure for editing files in the workdir."""
        workdir = self._workdir

        def edit_file(path: str, old: str, new: str) -> str:
            """Replace 'old' text with 'new' in file at path (relative to workdir)."""
            from pathlib import Path as P
            filepath = P(workdir) / path
            if not filepath.exists():
                return f"Error: file not found: {filepath}"
            content = filepath.read_text()
            if old not in content:
                return f"Error: old text not found in {path}"
            content = content.replace(old, new, 1)
            filepath.write_text(content)
            return f"OK: edited {path}"
        return edit_file

    def _is_timeout(self) -> bool:
        return (time.time() - self._start_time) > self._config.rlm.global_timeout
