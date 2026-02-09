"""RLM Agent System - CLI entry point."""

from dotenv import load_dotenv
load_dotenv()

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile

from rlm.agent import RLMAgent
from rlm.config import load_config
from rlm.sandbox import SandboxBudget, SandboxManager


def _is_git_url(path: str) -> bool:
    """Check if path looks like a git URL."""
    return path.startswith(("http://", "https://", "git@", "git://"))


def _prepare_source(repo_or_path: str, branch: str | None, commit: str | None) -> str:
    """Clone a git URL or validate a local path. Returns source directory path."""
    if _is_git_url(repo_or_path):
        tmpdir = tempfile.mkdtemp(prefix="rlm-clone-")
        cmd = ["git", "clone"]
        if not commit:
            cmd.append("--depth=1")
        if branch:
            cmd.extend(["--branch", branch])
        cmd.extend([repo_or_path, tmpdir])
        subprocess.run(cmd, check=True)
        if commit:
            subprocess.run(["git", "checkout", commit], cwd=tmpdir, check=True)
        return tmpdir
    else:
        path = os.path.abspath(repo_or_path)
        if not os.path.isdir(path):
            print(f"Error: '{repo_or_path}' is not a valid directory or git URL", file=sys.stderr)
            sys.exit(1)
        return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RLM Agent System - Recursive Language Model agents",
    )
    parser.add_argument("repo_or_path", help="Git URL or local path to repository")
    parser.add_argument("-p", "--prompt", required=True, help="Task prompt for the agent")
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to config file (default: config.yaml)")
    parser.add_argument("-o", "--output", help="Output file path (default: stdout)")
    parser.add_argument("-b", "--branch", help="Branch name to checkout")
    parser.add_argument("--commit", help="Specific commit SHA to checkout")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    logger = logging.getLogger("rlm")

    # Load config
    config = load_config(args.config)
    logger.info("Model: %s", config.model.name)
    logger.info("Max sandboxes: %d, Max iterations: %d, Max depth: %d",
                config.rlm.max_sandboxes, config.rlm.max_iterations, config.rlm.max_depth)

    # Prepare source directory
    source_dir = _prepare_source(args.repo_or_path, args.branch, args.commit)
    is_cloned = _is_git_url(args.repo_or_path)
    logger.info("Source directory: %s", source_dir)

    try:
        # Create budget and sandbox manager
        budget = SandboxBudget(config.rlm.max_sandboxes)
        sandbox_mgr = SandboxManager(source_dir=source_dir, budget=budget, config=config)

        # Create and run root agent
        agent = RLMAgent(
            config=config,
            task=args.prompt,
            workdir=source_dir,
            depth=0,
            sandbox_manager=sandbox_mgr,
        )

        logger.info("Starting root agent...")
        result = agent.run()

        # Output result
        output_text = result.result or "(no result)"
        if args.output:
            with open(args.output, "w") as f:
                f.write(output_text)
            logger.info("Result written to %s", args.output)
        else:
            print(output_text)

        logger.info("Completed in %d iterations (depth %d)", result.iterations, result.depth)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    finally:
        if is_cloned:
            shutil.rmtree(source_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
