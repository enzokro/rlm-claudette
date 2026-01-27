# Recursive Language Models: A Top-Down Tutorial

## The idea

An LLM agent that can spawn copies of itself. That is the whole idea behind Recursive Language Models, introduced by [Zhang, Kraska, and Khattab](https://arxiv.org/abs/2512.24601) and extended by [Prime Intellect](https://www.primeintellect.ai/blog/rlm).

Most agent systems today are either single agents with tool use, or fixed multi-agent pipelines with hardcoded roles. RLMs sit at the other end of the spectrum: a tree of agents where the root decomposes a problem, spawns children to work on the pieces in parallel, and those children spawn *their* children for sub-sub-tasks. Results flow back up through the tree. The agent decides at runtime whether to delegate, what to delegate, and how deep to go. There is no pre-defined graph. The recursion emerges from the task.

This tutorial walks through a complete implementation built on [claudette](https://github.com/AnswerDotAI/claudette). It starts at the top (what it looks like to run the thing) and peels layers until every module is covered.

---

## Starting at the top

```bash
uv run python main.py ./my-repo -p "Find all TODO comments and fix the easiest one" -v
```

A root agent gets the prompt, explores the repo by writing Python code, decides which TODOs exist, optionally spawns sub-agents to investigate different directories in parallel, picks the easiest one, edits the file, and calls `FINAL()` with the result.

Under the hood:

```
main.py
  load config from config.yaml
  create SandboxBudget(50)
  create SandboxManager(source_dir, budget, config)
  create RLMAgent(config, task, workdir, depth=0, sandbox_manager)
  agent.run()  ->  the iteration loop begins
```

---

## The iteration loop

The `run()` method in `rlm/agent.py` is the entire agent:

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

Three things are happening here:

1. **claudette's `Chat` manages conversation state.** It handles message history, system prompts, and API calls. We call it with a user message, get back an Anthropic `Message` object, and extract the text. We never touch the messages list directly.
2. **The loop is a REPL pattern.** Prompt the LLM, extract code from its response, execute the code, feed output back as the next prompt. The LLM writes code, sees what happened, writes more code. It keeps going until it calls `FINAL()` or runs out of iterations.
3. **The agent injects special functions into the REPL namespace.** A `namespace_extras` dict provides `rlm_query`, `rlm_query_batched`, and `edit_file`. The LLM sees functions in scope. It does not know these are closures that spawn sub-agents on threads.

---

## Why a REPL, not tool use

RLM agents need to write programs, not make function calls. A typical agent iteration looks like this:

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

That is a multi-statement program with loops, list comprehensions, variable assignment, and composed function calls. A single tool-call interface cannot express this. We need `exec`.

claudette has `toolloop` for standard tool-use agents, but it works over a fixed set of known tools. RLMs define their own "tools" via code on the fly. We use claudette's `Chat` to manage the conversation but separately extract and run the code blocks in a persistent namespace. Variables assigned in one code block are available in the next.

---

## The REPL

`rlm/repl.py` extracts fenced Python code blocks via regex:

```python
_CODE_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
```

Then `execute_response` runs them sequentially:

```python
def execute_response(self, text: str) -> REPLResult:
    blocks = extract_code_blocks(text)
    if not blocks:
        return REPLResult(output="(no code blocks found)")

    outputs = []
    for block in blocks:
        out = self._execute_block(block)
        outputs.append(out)
        if self._final_answer is not None:
            break

    return REPLResult(
        output="\n".join(outputs),
        final_answer=self._final_answer,
    )
```

The interesting work is in `_execute_block`. It uses `ast.parse` to handle a subtle problem: if the last statement in a code block is a bare expression (like `len(files)` on its own line), we want to capture its value, just like a real Python REPL prints the result of an expression. The code pops the last AST node if it is an `Expr`, `exec`s everything else, then `eval`s just that final expression:

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

A custom `print` closure is injected into the namespace. It writes to a per-block `io.StringIO` buffer instead of `sys.stdout`, so multiple REPLs can execute concurrently without interleaving output. Exceptions are caught and their tracebacks written to the same buffer. The LLM sees everything (print output, expression values, error traces) in the next iteration's prompt.

### FINAL and FINAL_VAR

`FINAL` and `FINAL_VAR` are closures injected into the namespace at REPL init time:

```python
def _final(answer):
    self._final_answer = str(answer)

def _final_var(variable_name):
    self._final_var_name = str(variable_name).strip().strip("\"'")

self._namespace["FINAL"] = _final
self._namespace["FINAL_VAR"] = _final_var
```

When the LLM calls `FINAL("here is my answer")`, execution of further blocks stops and the answer propagates up through `REPLResult.final_answer`.

`FINAL_VAR` is a deferred variant. The LLM calls `FINAL_VAR("result")` to name a variable, and the REPL resolves it *after* the block finishes executing. This lets the agent build up a complex result in a variable and submit it without stringifying inline:

```python
# The LLM writes this
result = "\n".join(f"{f}: {n} lines" for f, n in sorted(counts.items()))
FINAL_VAR("result")
```

After `exec()` completes, the REPL looks up `"result"` in its tracked locals and sets `self._final_answer = str(self._locals["result"])`. The agent loop sees a single `repl_result.final_answer` field regardless of which path was used.

The REPL also tracks user-defined variables in `self._locals` after each block execution. This enables `FINAL_VAR` resolution and also supports a prose-level fallback in the agent loop (see below).

---

## The namespace

The REPL pre-loads `os`, `subprocess`, `Path`, and `WORKDIR` (a string path to the agent's working directory). The agent adds three more via closures:

```python
def _build_namespace(self) -> dict:
    ns = {}
    ns["rlm_query"] = self._make_rlm_query()
    ns["rlm_query_batched"] = self._make_rlm_query_batched()
    ns["edit_file"] = self._make_edit_file()
    return ns
```

Together with `FINAL` and `FINAL_VAR` (injected by the REPL itself), the agent's code has access to everything in this table, plus anything it imports (`import json` works fine since the namespace is a real Python namespace):

| Name | Source | What it does |
|------|--------|-------------|
| `os` | REPL built-in | Standard library `os` module |
| `subprocess` | REPL built-in | Standard library `subprocess` module |
| `Path` | REPL built-in | `pathlib.Path` |
| `WORKDIR` | REPL built-in | String path to the agent's working directory |
| `rlm_query(task)` | Agent closure | Spawns a sub-agent, returns result string |
| `rlm_query_batched(tasks)` | Agent closure | Spawns multiple sub-agents in parallel |
| `edit_file(path, old, new)` | Agent closure | Replaces text in a file |
| `FINAL(answer)` | REPL closure | Submits the final result, ends the loop |
| `FINAL_VAR(variable_name)` | REPL closure | Submits a REPL variable's value as the final result |

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

A closure over the agent's sandbox manager and depth. It checks the depth limit, then delegates to `SandboxManager.spawn_agent()`. From the LLM's perspective, it is a function that takes a task description and returns a string. What actually happens is a new `RLMAgent` gets created on a thread in an isolated directory, runs its own full agent loop, and returns its `FINAL()` answer.

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

Fan-out. Each task gets its own thread, each thread calls `rlm_query`, each `rlm_query` spawns an in-process agent with its own REPL and Chat. Results come back in the original order. This is where the parallelism comes from: 25 sub-agents exploring different directories simultaneously, each in their own isolated copy of the repo.

---

## Sandbox isolation

`rlm/sandbox.py` contains two classes: `SandboxBudget` and `SandboxManager`.

### SandboxBudget

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

A thread-safe counter. `rlm_query_batched` runs multiple spawns in parallel from different threads, so the lock matters. The budget tracks total sandboxes created over the lifetime of the rollout, not concurrent ones. Once 50 sub-agents have been spawned, no more can be created regardless of how many have finished.

All agents run in-process and share a single `SandboxBudget` instance. The `SandboxManager` passes itself (`sandbox_manager=self`) to each child agent, so the budget is enforced globally and exactly. No serialization, no snapshots, no drift.

### SandboxManager.spawn_agent

```python
def spawn_agent(self, task: str, depth: int) -> str:
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
            sandbox_manager=self,  # same manager, same budget
        )
        result = agent.run()
        return result.result or "No result"

    except Exception as e:
        return f"Error: sub-agent failed: {e}"
    finally:
        if workdir:
            self._cleanup_workdir(workdir, is_worktree)
```

The flow:

1. Acquire budget
2. Create working directory (git worktree)
3. Instantiate a new `RLMAgent` directly, passing `self` as the sandbox manager
4. Run the agent's iteration loop in-process
5. Return the result string
6. Clean up the working directory

### Git worktrees

Working directory creation is where the isolation happens:

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

`git worktree add` creates a new working tree that shares the same git object store as the original repo. No copying of files in `.git/objects`, so it is fast. Each worktree has its own working directory where the agent can make edits without affecting any other agent's copy. After the agent completes, `git worktree remove` cleans it up.

For non-git directories, we fall back to `shutil.copytree`.

---

## In-process recursion

Sub-agents run in the same process as their parent. `spawn_agent()` directly instantiates an `RLMAgent` with `sandbox_manager=self`, so the child shares the same `SandboxManager` and `SandboxBudget`. No serialization, no JSON, no child process. The child gets its own `REPL` and `Chat`, runs its iteration loop, and returns a result string.

This is where the recursion happens. The child's `RLMAgent` has the same `SandboxManager`, which can call `spawn_agent` again, which creates another `RLMAgent`, which can spawn more children. Depth is incremented at each level and checked against `max_depth`. The budget is enforced exactly because every agent shares the same counter and lock.

Thread-safety matters here. The REPL avoids process-global mutations: no `os.chdir`, no `sys.stdout` reassignment. Each REPL captures output via a namespace-injected `print` that writes to its own `io.StringIO` buffer. Multiple REPLs executing concurrently on different threads produce isolated output.

---

## The prompts

`rlm/prompts.py` has two functions.

### build_system_prompt

Tells the LLM what it is and what it can do:

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

### build_user_prompt

Handles the iteration-to-iteration communication:

```python
def build_user_prompt(iteration: int, execution_result: str | None) -> str:
    if iteration == 0:
        return "Begin working on your task. Write Python code in ```python blocks."
    return f"""\
REPL output from your code:
{execution_result}

Continue working. Write more code or call FINAL(result) when done."""
```

Iteration 0 says "go." Every subsequent iteration shows the LLM what its code produced and tells it to continue. The LLM sees its own stdout, tracebacks, and return values, and can react: fix errors, refine its approach, run more code based on what it learned.

---

## Configuration

`rlm/config.py` has three dataclasses:

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

`Config` has `to_dict()` and `from_dict()` for serialization. `load_config()` reads a YAML file and falls back to defaults if the file does not exist.

| Setting | Default | What it controls |
|---------|---------|-----------------|
| `max_sandboxes` | 50 | Total sub-agents across the entire tree |
| `max_iterations` | 50 | Max REPL iterations per agent |
| `global_timeout` | 3600 | Agent timeout in seconds |
| `result_truncation_limit` | 10000 | Max characters before truncating sub-agent results |
| `max_depth` | 5 | Maximum recursion depth |

---

## The CLI

`main.py` uses argparse:

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

Depth 0. The root agent. Everything else (the tree of sub-agents, the parallel exploration, the recursive delegation) emerges from the LLM's decisions inside the loop.

---

## Tracing an execution

A concrete example:

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

For a harder task ("find all TODOs and fix the easiest one"), the LLM might spend iteration 0 exploring the file tree, iteration 1 spawning 10 sub-agents via `rlm_query_batched` to search different directories in parallel, iteration 2 reviewing the results and picking the easiest TODO, and iteration 3 calling `edit_file` and `FINAL`. Each of those 10 sub-agents runs its own iteration loop at depth 1, potentially spawning depth-2 agents for particularly large directories.

---

## The full picture

```
main.py
  -> load_config("config.yaml") -> Config
  -> SandboxBudget(50)
  -> SandboxManager(source_dir, budget, config)
  -> RLMAgent(config, task, workdir, depth=0, sandbox_manager)
  -> agent.run()
      +-- Chat(model, sp=system_prompt, temp=0)           # claudette
      +-- REPL(namespace_extras={rlm_query, ...}, workdir) # persistent namespace
      +-- loop:
          |   chat(user_prompt) -> response                 # LLM call
          |   repl.execute_response(response_text)          # extract + exec
          |   if FINAL/FINAL_VAR called: return result
          |   else: check prose fallback, or feed output back
          |
          |   (LLM writes code calling rlm_query_batched)
          |
          +-- ThreadPoolExecutor -> N threads
                +-- sandbox_mgr.spawn_agent(task, depth+1)
                      +-- git worktree add -> isolated workdir
                      +-- RLMAgent(sandbox_manager=self) -> direct call
                      |     +-- own Chat + own REPL
                      |     +-- agent.run() -> same loop, deeper depth
                      |     +-- returns result string
                      +-- git worktree remove -> cleanup
```

Six files. No frameworks. The recursion is function calls, threads, and a shared git object store.

---

## Where this goes

Current language models are not specifically trained to leverage recursive delegation. RLMs do not necessarily outperform single-agent approaches on benchmarks yet. But the architecture has properties worth paying attention to.

The parallelism is real. Spawning 25 sub-agents that each explore a different module of a large codebase, in parallel, each with isolated file access: that is hard to get without per-agent sandboxes. The recursive structure means the tree's shape adapts to the problem. A simple task stays flat. A complex task grows deep and wide, because the agents *decide* to make it so.

Single-agent systems are the practical default today. But as models get better at decomposition and delegation, architectures like this become more natural. The infrastructure for recursive, parallel, isolated agent execution is the part worth building now.
