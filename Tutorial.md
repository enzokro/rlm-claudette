# Recursive Language Models: A Top-Down Tutorial

## The idea

An LLM agent that can spawn copies of itself. That's the whole idea behind Recursive Language Models, introduced by [Zhang, Kraska, and Khattab](https://arxiv.org/abs/2512.24601) and extended by [Prime Intellect](https://www.primeintellect.ai/blog/rlm).

Let's chart the two extremes. At one end: a single agent that gets a task, thinks really hard, and produces an answer. At the other: a tree of agents, where the root decomposes a problem, spawns children to work on the pieces in parallel, those children spawn *their* children for sub-sub-tasks, and results flow back up through the tree. Between those extremes sit most of the agent systems we use today---single agents with tool use, or fixed multi-agent pipelines with hardcoded roles.

RLMs sit at the far end. The agent decides at runtime whether to delegate, what to delegate, and how deep to go. There's no pre-defined graph of agents. The recursion emerges from the task.

This tutorial walks through a complete implementation built on [claudette](https://github.com/AnswerDotAI/claudette). We'll start at the top---what it looks like to run the thing---and peel layers until we've covered every module.

---

## Start at the top

```bash
uv run python main.py ./my-repo -p "Find all TODO comments and fix the easiest one" -v
```

That's it. A root agent gets the prompt, explores the repo by writing Python code, decides which TODOs exist, optionally spawns sub-agents to investigate different directories in parallel, picks the easiest one, edits the file, and calls `FINAL()` with the result.

What happens under the hood:

```
main.py
  load config from config.yaml
  create SandboxBudget(50)
  create SandboxManager(source_dir, budget, config)
  create RLMAgent(config, task, workdir, depth=0, sandbox_manager)
  agent.run()  →  the iteration loop begins
```

Let's trace `agent.run()`.

---

## The iteration loop

Open `rlm/agent.py`. The `run()` method is the entire agent:

```python
def run(self) -> AgentResult:
    sp = build_system_prompt(depth=self._depth, max_depth=..., workdir=..., task=...)
    chat = Chat(model=self._config.model.name, sp=sp, temp=0)

    namespace_extras = self._build_namespace()
    repl = REPL(namespace_extras=namespace_extras, workdir=self._workdir)

    execution_result = None
    for iteration in range(self._config.rlm.max_iterations):
        if self._is_timeout():
            break

        user_prompt = build_user_prompt(iteration, execution_result)
        response = chat(user_prompt, maxtok=self._config.model.max_tokens)
        response_text = self._extract_text(response)

        repl_result = repl.execute_response(response_text)

        if repl_result.final_answer is not None:
            return AgentResult(result=repl_result.final_answer, ...)

        execution_result = repl_result.output
```

Three things to notice.

**First**, we use claudette's `Chat` for conversation management. `Chat` handles message history, system prompts, and API calls. We call it with a user message, get back an Anthropic `Message` object, and extract the text. claudette manages the conversation state---we don't touch the messages list directly.

**Second**, the loop structure is: prompt the LLM → extract code from its response → execute the code → feed output back as the next prompt. This is the REPL pattern. The LLM writes code, sees what happened, writes more code. It keeps going until it calls `FINAL()` or we run out of iterations.

**Third**, we build a `namespace_extras` dict and hand it to the REPL. This is how the agent gets its special functions: `rlm_query`, `rlm_query_batched`, `edit_file`. The LLM doesn't know these are closures that spawn subprocesses. It just sees functions in scope.

---

## Why a REPL, not tool use

This is a design decision worth explaining.

claudette has `toolloop`---a mechanism where the LLM calls functions by name, one at a time, and claudette handles the dispatch. It works well for tool-use agents. But the RLM architecture requires something different.

An RLM agent needs to write *programs*, not make function calls. Consider what a typical agent iteration looks like:

```python
# The LLM writes this
import os
files = []
for root, dirs, filenames in os.walk("."):
    for f in filenames:
        if f.endswith(".py"):
            files.append(os.path.join(root, f))

# Spawn sub-agents for each directory
dirs_with_todos = set(os.path.dirname(f) for f in files)
tasks = [f"Search for TODO comments in {d}/" for d in sorted(dirs_with_todos)[:10]]
results = rlm_query_batched(tasks)

for d, r in zip(sorted(dirs_with_todos)[:10], results):
    print(f"=== {d} ===")
    print(r[:500])
```

That's a multi-statement program with loops, list comprehensions, variable assignment, and two different function calls composed together. A single tool-call interface can't express this. We need `exec`.

So we extract fenced Python code blocks from the LLM's response and run them in a persistent namespace. The namespace persists across blocks within a single iteration *and* across the REPL's lifetime, so variables assigned in one code block are available in the next.

---

## The REPL

Open `rlm/repl.py`. The extraction is a regex:

```python
_CODE_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
```

It pulls every ` ```python ` fenced block from the LLM's response. Then `execute_response` runs them sequentially:

```python
def execute_response(self, text: str) -> REPLResult:
    blocks = extract_code_blocks(text)
    if not blocks:
        return REPLResult(output="(no code blocks found)")

    outputs = []
    for block in blocks:
        out = self._execute_block(block)
        outputs.append(out)
        if self._final_called:
            break

    return REPLResult(
        output="\n".join(outputs),
        final_answer=self._final_value if self._final_called else None,
    )
```

The interesting work is in `_execute_block`. It uses `ast.parse` to handle a subtle problem: if the last statement in a code block is a bare expression (like `len(files)` on its own line), we want to capture its value---just like a real Python REPL prints the result of an expression. So we pop the last AST node if it's an `Expr`, `exec` everything else, then `eval` just that final expression:

```python
tree = ast.parse(code)
if tree.body and isinstance(tree.body[-1], ast.Expr):
    last_expr = tree.body.pop()
    if tree.body:
        exec(compile(ast.Module(body=tree.body, type_ignores=[]), "<repl>", "exec"), self._namespace)
    val = eval(compile(ast.Expression(body=last_expr.value), "<repl>", "eval"), self._namespace)
    if val is not None:
        print(repr(val))
else:
    exec(compile(tree, "<repl>", "exec"), self._namespace)
```

Stdout is redirected to a `StringIO` buffer during execution. Exceptions are caught and their tracebacks written to the same buffer. The LLM sees everything---print output, expression values, error traces---in the next iteration's prompt.

### FINAL

`FINAL` is a closure injected into the namespace at REPL init time:

```python
def _final(answer):
    self._final_called = True
    self._final_value = str(answer)
self._namespace["FINAL"] = _final
```

When the LLM calls `FINAL("here is my answer")`, the flag flips, execution of further blocks stops, and the answer propagates up through `REPLResult.final_answer`.

---

## The namespace: what the agent can see

The REPL pre-loads `os`, `subprocess`, and `Path`. The agent adds three more via closures:

```python
def _build_namespace(self) -> dict:
    ns = {}
    ns["rlm_query"] = self._make_rlm_query()
    ns["rlm_query_batched"] = self._make_rlm_query_batched()
    ns["edit_file"] = self._make_edit_file()
    return ns
```

Together with `FINAL` (injected by the REPL itself), the agent's code has access to: `os`, `subprocess`, `Path`, `rlm_query`, `rlm_query_batched`, `edit_file`, and `FINAL`. Plus anything it imports---the namespace is a real Python namespace, so `import json` works fine.

### `rlm_query`

```python
def _make_rlm_query(self):
    def rlm_query(task: str) -> str:
        if self._sandbox_mgr is None:
            return "Error: no sandbox manager"
        if self._depth >= self._config.rlm.max_depth:
            return "Error: maximum recursion depth reached"
        return self._sandbox_mgr.spawn_agent(task, self._depth + 1)
    return rlm_query
```

A closure over the agent's sandbox manager and depth. It checks the depth limit, then delegates to `SandboxManager.spawn_agent()`. From the LLM's perspective, it's a function that takes a task description and returns a string. What actually happens is a subprocess gets created in an isolated directory, runs its own full agent loop, and returns its `FINAL()` answer.

### `rlm_query_batched`

```python
def rlm_query_batched(tasks: list[str]) -> list[str]:
    results = [""] * len(tasks)
    with ThreadPoolExecutor(max_workers=min(10, len(tasks))) as executor:
        future_to_idx = {
            executor.submit(rlm_query, task): i
            for i, task in enumerate(tasks)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
    return results
```

Fan-out. Each task gets its own thread, each thread calls `rlm_query`, each `rlm_query` spawns a subprocess. The results come back in the original order. This is where the parallelism comes from---25 sub-agents exploring different directories simultaneously, each in their own isolated copy of the repo.

---

## Sandbox isolation

Open `rlm/sandbox.py`. Two classes here: `SandboxBudget` and `SandboxManager`.

### The budget

```python
class SandboxBudget:
    def __init__(self, max_sandboxes: int):
        self._max = max_sandboxes
        self._used = 0
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        with self._lock:
            if self._used >= self._max:
                return False
            self._used += 1
            return True
```

Thread-safe counter. `rlm_query_batched` runs multiple spawns in parallel from different threads, so the lock matters. The budget tracks total sandboxes created over the lifetime of the rollout, not concurrent ones. Once 50 sub-agents have been spawned, no more can be created regardless of how many have finished.

When a child process starts, it gets a `SandboxBudget` initialized with `remaining_budget` from the parent. The child can then spawn its own sub-agents up to that limit.

### Spawning

```python
def spawn_agent(self, task: str, depth: int) -> str:
    if not self._budget.acquire():
        return "Error: sandbox budget exhausted"

    workdir = None
    is_worktree = False
    try:
        workdir, is_worktree = self._create_workdir()

        payload = json.dumps({
            "task": task, "workdir": workdir, "depth": depth,
            "config": self._config.to_dict(),
            "source_dir": self._source_dir,
            "remaining_budget": self._budget.remaining,
        })

        result = subprocess.run(
            [sys.executable, "-m", "rlm.subprocess_runner"],
            input=payload, capture_output=True, text=True,
            timeout=self._config.rlm.global_timeout, cwd=workdir,
        )

        data = json.loads(result.stdout)
        return data.get("result", "No result")
    finally:
        if workdir:
            self._cleanup_workdir(workdir, is_worktree)
```

The flow: acquire budget → create working directory → serialize config as JSON → run `python -m rlm.subprocess_runner` with JSON on stdin → parse JSON result from stdout → clean up.

### Git worktrees

The working directory creation is the cleverest part of the isolation story:

```python
def _create_workdir(self) -> tuple[str, bool]:
    source = Path(self._source_dir)
    if (source / ".git").exists():
        tmp = tempfile.mkdtemp(prefix="rlm-worktree-")
        subprocess.run(
            ["git", "worktree", "add", "--detach", tmp],
            cwd=self._source_dir, capture_output=True, check=True,
        )
        return tmp, True
    else:
        tmp = tempfile.mkdtemp(prefix="rlm-copy-")
        shutil.copytree(self._source_dir, tmp, dirs_exist_ok=True)
        return tmp, False
```

`git worktree add` creates a new working tree that shares the same git object store as the original repo. It's fast---no copying of files in `.git/objects`---and each worktree has its own working directory where the agent can make edits without affecting any other agent's copy. After the subprocess completes, `git worktree remove` cleans it up.

For non-git directories, we fall back to `shutil.copytree`. Slower, but correct.

---

## The subprocess boundary

Open `rlm/subprocess_runner.py`. This is what runs inside each child process:

```python
def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    payload = json.loads(sys.stdin.read())

    config = Config.from_dict(payload["config"])
    budget = SandboxBudget(payload.get("remaining_budget", 0))
    sandbox_mgr = SandboxManager(source_dir=payload["source_dir"], budget=budget, config=config)

    agent = RLMAgent(
        config=config,
        task=payload["task"],
        workdir=payload["workdir"],
        depth=payload["depth"],
        sandbox_manager=sandbox_mgr,
    )

    result = agent.run()
    json.dump({
        "result": result.result,
        "iterations": result.iterations,
        "depth": result.depth,
    }, sys.stdout)
```

Read JSON from stdin. Reconstruct all the objects. Create a full `RLMAgent`. Run it. Write JSON to stdout. All logging goes to stderr so it doesn't contaminate the JSON result channel.

The child gets its own `SandboxBudget` initialized with the parent's remaining budget. If the parent started with 50 and has used 10, the child gets `SandboxBudget(40)`. The child can then spawn its own sub-agents, passing along *its* remaining budget, and so on down the tree.

This is where the recursion happens. The child's `RLMAgent` has a `SandboxManager` that can call `spawn_agent`, which runs `python -m rlm.subprocess_runner` again, which creates another `RLMAgent`, which can spawn more children. Depth is incremented at each level and checked against `max_depth`.

---

## The prompts

Open `rlm/prompts.py`. Two functions.

`build_system_prompt` tells the LLM what it is and what it can do:

```python
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
```

The `can_spawn` line changes based on depth. At maximum depth, it says "You are at maximum depth and CANNOT spawn sub-agents." This prevents the LLM from trying to call `rlm_query` when it would just get an error back.

`build_user_prompt` handles the iteration-to-iteration communication:

```python
def build_user_prompt(iteration: int, execution_result: str | None) -> str:
    if iteration == 0:
        return "Begin working on your task. Write Python code in ```python blocks."
    return f"""\
REPL output from your code:
{execution_result}

Continue working. Write more code or call FINAL(result) when done."""
```

Iteration 0 says "go." Every subsequent iteration shows the LLM what its code produced and tells it to continue. The LLM sees its own stdout, tracebacks, and return values. It can react---fix errors, refine its approach, run more code based on what it learned.

---

## Configuration

Open `rlm/config.py`. Three dataclasses:

```python
@dataclass
class ModelConfig:
    name: str = "claude-sonnet-4-5-20250514"
    temperature: float = 0.0
    max_tokens: int = 16384

@dataclass
class RLMConfig:
    max_sandboxes: int = 50
    max_iterations: int = 50
    global_timeout: int = 3600
    result_truncation_limit: int = 10000
    max_depth: int = 5

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    rlm: RLMConfig = field(default_factory=RLMConfig)
```

`Config` has `to_dict()` and `from_dict()` for serialization across the subprocess boundary. `load_config()` reads a YAML file and falls back to defaults if the file doesn't exist.

The defaults are sensible for moderate-sized tasks. `max_sandboxes=50` means up to 50 sub-agents total across the entire tree. `max_depth=5` caps recursion. `result_truncation_limit=10000` prevents a verbose sub-agent from blowing up the parent's context window---results longer than 10K characters get truncated before being fed back into the next prompt.

---

## The CLI

Open `main.py`. argparse, nothing exotic:

```python
parser.add_argument("repo_or_path", help="Git URL or local path")
parser.add_argument("-p", "--prompt", required=True)
parser.add_argument("-c", "--config", default="config.yaml")
parser.add_argument("-o", "--output")
parser.add_argument("-b", "--branch")
parser.add_argument("--commit")
parser.add_argument("-v", "--verbose")
```

If `repo_or_path` is a URL, we `git clone --depth=1` into a tempdir. Otherwise we use the local path directly. Then:

```python
budget = SandboxBudget(config.rlm.max_sandboxes)
sandbox_mgr = SandboxManager(source_dir=source_dir, budget=budget, config=config)
agent = RLMAgent(config=config, task=args.prompt, workdir=source_dir, depth=0, sandbox_manager=sandbox_mgr)
result = agent.run()
```

Depth 0. The root. Everything else---the tree of sub-agents, the parallel exploration, the recursive delegation---emerges from the LLM's decisions inside the loop.

---

## Tracing an execution

Let's walk through what happens end to end with a concrete example:

```bash
uv run python main.py ./my-project -p "Count the lines of code in each Python file"
```

**Iteration 0** (depth 0): The LLM receives "Begin working on your task." It writes:

```python
import os
py_files = []
for root, dirs, files in os.walk("."):
    for f in files:
        if f.endswith(".py"):
            py_files.append(os.path.join(root, f))
print(f"Found {len(py_files)} Python files")
```

The REPL executes this, captures `Found 12 Python files`, feeds it back.

**Iteration 1** (depth 0): The LLM sees the output. It writes:

```python
results = {}
for f in py_files:
    with open(f) as fh:
        results[f] = len(fh.readlines())

summary = "\n".join(f"{f}: {n} lines" for f, n in sorted(results.items()))
FINAL(summary)
```

The REPL executes this, `FINAL` gets called, the loop ends. `agent.run()` returns an `AgentResult` with the summary string. `main.py` prints it.

Two iterations. No sub-agents needed. The LLM decided it could handle this alone.

For a harder task---"find all TODOs and fix the easiest one"---the LLM might spend iteration 0 exploring the file tree, iteration 1 spawning 10 sub-agents via `rlm_query_batched` to search different directories in parallel, iteration 2 reviewing the results and picking the easiest TODO, and iteration 3 calling `edit_file` and `FINAL`. Each of those 10 sub-agents runs its own iteration loop at depth 1, potentially spawning depth-2 agents for particularly large directories.

---

## The full picture

```
main.py
  → load_config("config.yaml") → Config
  → SandboxBudget(50)
  → SandboxManager(source_dir, budget, config)
  → RLMAgent(config, task, workdir, depth=0, sandbox_manager)
  → agent.run()
      → Chat(model, sp=system_prompt, temp=0)           # claudette
      → REPL(namespace_extras={rlm_query, ...}, workdir) # persistent namespace
      → loop:
          → chat(user_prompt) → response                 # LLM call
          → repl.execute_response(response_text)          # extract + exec
          → if FINAL called: return result
          → else: feed output back as next prompt
              ↓
          (LLM writes code calling rlm_query_batched)
              ↓
          ThreadPoolExecutor → N threads
              ↓ each thread:
          sandbox_mgr.spawn_agent(task, depth+1)
              ↓
          git worktree add → isolated workdir
          subprocess.run("python -m rlm.subprocess_runner")
              ↓ child process:
          stdin JSON → Config, Budget, Manager, Agent
          agent.run() → same loop, deeper depth
          stdout JSON ← result
              ↓
          git worktree remove → cleanup
```

Seven files. No frameworks. The recursion is just function calls, subprocesses, and a shared git object store.

---

## Where this goes

Current language models aren't specifically trained to leverage recursive delegation. RLMs don't necessarily outperform single-agent approaches on benchmarks yet. But the architecture has properties worth paying attention to.

The parallelism is real. Spawning 25 sub-agents that each explore a different module of a large codebase, in parallel, each with isolated file access---that's hard to get without per-agent sandboxes. And the recursive structure means the tree's shape adapts to the problem. A simple task stays flat. A complex task grows deep and wide, because the agents *decide* to make it so.

We're currently closer to single-agent systems being the practical default. But as models get better at decomposition and delegation, architectures like this become more natural. The infrastructure for recursive, parallel, isolated agent execution is the part worth building now.
