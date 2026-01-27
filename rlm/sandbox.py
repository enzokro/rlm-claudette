"""In-process isolation for sub-agents using git worktrees."""

import logging
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from rlm.config import Config

logger = logging.getLogger(__name__)


class SandboxBudget:
    """Thread-safe counter for limiting total sandbox spawns across all agents."""

    def __init__(self, max_sandboxes: int):
        self._max = max_sandboxes
        self._used = 0
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        """Atomically decrement budget. Returns True if acquired, False if exhausted."""
        with self._lock:
            if self._used >= self._max:
                return False
            self._used += 1
            return True

    def can_acquire(self) -> bool:
        with self._lock:
            return self._used < self._max

    @property
    def remaining(self) -> int:
        with self._lock:
            return self._max - self._used


class SandboxManager:
    """Manages spawning sub-agents in isolated git worktrees via in-process threads."""

    def __init__(self, source_dir: str, budget: SandboxBudget, config: Config):
        self._source_dir = source_dir
        self._budget = budget
        self._config = config

    @property
    def budget(self) -> SandboxBudget:
        return self._budget

    def spawn_agent(self, task: str, depth: int) -> str:
        """Spawn a sub-agent in an isolated worktree. Returns result string."""
        from rlm.agent import RLMAgent  # local import to avoid circular

        if not self._budget.acquire():
            return "Error: sandbox budget exhausted"

        workdir = None
        is_worktree = False
        try:
            workdir, is_worktree = self._create_workdir()

            agent = RLMAgent(
                config=self._config,
                task=task,
                workdir=workdir,
                depth=depth,
                sandbox_manager=self,
            )
            result = agent.run()
            return result.result or "No result"

        except Exception as e:
            return f"Error: sub-agent failed: {e}"
        finally:
            if workdir:
                self._cleanup_workdir(workdir, is_worktree)

    def _create_workdir(self) -> tuple[str, bool]:
        """Create isolated working directory. Returns (path, is_worktree)."""
        source = Path(self._source_dir)
        git_dir = source / ".git"

        if git_dir.exists():
            # Use git worktree for fast, lightweight isolation
            tmp = tempfile.mkdtemp(prefix="rlm-worktree-")
            subprocess.run(
                ["git", "worktree", "add", "--detach", tmp],
                cwd=self._source_dir,
                capture_output=True,
                check=True,
            )
            return tmp, True
        else:
            # Fallback: copy the directory
            tmp = tempfile.mkdtemp(prefix="rlm-copy-")
            shutil.copytree(self._source_dir, tmp, dirs_exist_ok=True)
            return tmp, False

    def _cleanup_workdir(self, workdir: str, is_worktree: bool) -> None:
        """Remove the working directory."""
        try:
            if is_worktree:
                subprocess.run(
                    ["git", "worktree", "remove", "--force", workdir],
                    cwd=self._source_dir,
                    capture_output=True,
                )
            else:
                shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            logger.warning("Failed to cleanup workdir: %s", workdir)
