"""Child process entry point for sub-agents.

Invoked as: python -m rlm.subprocess_runner
Reads JSON from stdin, runs an agent, writes JSON result to stdout.
All logging goes to stderr.
Environment is inherited from the parent process (main.py loads dotenv).
"""

import json
import logging
import sys

from rlm.agent import RLMAgent
from rlm.config import Config
from rlm.sandbox import SandboxBudget, SandboxManager


def main() -> None:
    # Configure logging to stderr only
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] depth=%(name)s %(message)s",
        stream=sys.stderr,
    )
    logger = logging.getLogger("subprocess_runner")

    try:
        payload = json.loads(sys.stdin.read())
    except json.JSONDecodeError as e:
        json.dump({"result": f"Error: invalid input JSON: {e}", "iterations": 0, "depth": -1}, sys.stdout)
        return

    task = payload["task"]
    workdir = payload["workdir"]
    depth = payload["depth"]
    config = Config.from_dict(payload["config"])
    source_dir = payload["source_dir"]
    remaining_budget = payload.get("remaining_budget", 0)

    budget = SandboxBudget(remaining_budget)
    sandbox_mgr = SandboxManager(source_dir=source_dir, budget=budget, config=config)

    agent = RLMAgent(
        config=config,
        task=task,
        workdir=workdir,
        depth=depth,
        sandbox_manager=sandbox_mgr,
    )

    logger.info("Starting sub-agent: depth=%d task=%s", depth, task[:100])

    try:
        result = agent.run()
        json.dump({
            "result": result.result,
            "iterations": result.iterations,
            "depth": result.depth,
        }, sys.stdout)
    except Exception as e:
        logger.exception("Sub-agent failed")
        json.dump({
            "result": f"Error: agent failed: {e}",
            "iterations": 0,
            "depth": depth,
        }, sys.stdout)

# TODO: buried breadcrumb for the RML to find

if __name__ == "__main__":
    main()
