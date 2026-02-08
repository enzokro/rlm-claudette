# rlm-claudette Tutorial

## The core idea

A [Recursive Language Model](https://arxiv.org/abs/2512.24601) (RLM) is an LLM agent that spawns copies of itself. The root agent decomposes a problem, spawns children to work different pieces in parallel, and those children spawn *their own* children. Results flow back up the tree. The recursion adapts to the task because the agent decides at runtime how deep and wide to go.

Most agent frameworks, including Claude Code as of writing, don't let subagents spawn their own children. RLMs do. The recursion stops when it hits a configurable depth or when the sandbox budget runs out.

This is a complete RLM implementation built on [claudette](https://github.com/AnswerDotAI/claudette). We start at the CLI and work down.

---

## Starting at the top

```bash
uv run python main.py ./my-repo -p "Find all TODO comments and fix the easiest one" -v
```

The root agent gets the prompt, writes Python code to explore the repo, optionally spawns subagents for parallel work, and calls `FINAL()` when done.

The call chain:

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
    chat = Chat(model=self._config.model.name, sp=sp, temp=self._config.model.temperature)

    namespace_extras = self._build_namespace()
    repl = REPL(namespace_extras=namespace_extras, workdir=self._workdir)

    execution_result = None
    iterations = 0

    for iteration in range(self._config.rlm.max_iterations):
        if self._is_timeout():
            break

        user_prompt = build_user_prompt(iteration, execution_result)
        response = chat(user_prompt, maxtok=self._config.model.max_tokens)
        response_text = self._extract_text(response)

        repl_result = repl.execute_response(response_text)
        iterations = iteration + 1

        # Primary: code-level FINAL / FINAL_VAR (handled inside REPL)
        if repl_result.final_answer is not None:
            return AgentResult(result=repl_result.final_answer, iterations=iterations, depth=self._depth)

        # Fallback: prose-level FINAL / FINAL_VAR
        prose_answer = find_final_in_prose(response_text, repl.locals)
        if prose_answer is not None:
            return AgentResult(result=prose_answer, iterations=iterations, depth=self._depth)

        # Truncate output for next prompt
        execution_result = repl_result.output
        if len(execution_result) > self._config.rlm.result_truncation_limit:
            execution_result = execution_result[:self._config.rlm.result_truncation_limit] + "\n... (truncated)"

    return AgentResult(result=None, iterations=iterations, depth=self._depth)
```

claudette's `Chat` manages conversation state. We call it with a user message, get back an Anthropic `Message` object, and extract the text. We never touch the messages list directly.

The loop has a two-stage completion check. After executing the code blocks, the agent checks whether `FINAL` or `FINAL_VAR` was called inside the code (the REPL sets `final_answer`). If not, it falls back to `find_final_in_prose`, which scans the response text outside code blocks for `FINAL(...)` patterns. If neither finds a result, the output is truncated and fed back as the next prompt.

The agent also injects closures into the REPL namespace: `rlm_query`, `rlm_query_batched`, and `edit_file`. The LLM sees functions in scope. It does not know they spawn subagents on threads.

---

## Why a REPL, not tool use

RLM agents need to write programs, not make function calls. An agent iteration:

```python
# The LLM writes this
import os
files = []
for root, dirs, filenames in os.walk("."):
    for f in filenames:
        if f.endswith(".py"):
            files.append(os.path.join(root, f))

# Spawn subagents for each directory
dirs_with_todos = set(os.path.dirname(f) for f in files)
tasks = [f"Search for TODO comments in {d}/" for d in sorted(dirs_with_todos)[:10]]
results = rlm_query_batched(tasks)

for d, r in zip(sorted(dirs_with_todos)[:10], results):
    print(f"=== {d} ===")
    print(r[:500])
```

Loops, list comprehensions, variable assignment, composed function calls — a single tool-call interface cannot express this. We need `exec`.

claudette has an excellent `toolloop` for standard tool-use agents, but it works over a fixed set of known tools. RLMs define their own "tools" via code on the fly. We use claudette's `Chat` to manage the conversation but separately extract and run the code blocks in a persistent namespace. Variables persist across code blocks.

---

## The REPL

`rlm/repl.py` extracts fenced Python code blocks via regex:

```python
_CODE_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
```

`execute_response` runs them sequentially:

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

`_execute_block` parses each block with `ast.parse`. If the last statement is a bare expression (like `len(files)` on its own line), we want to capture its value — so the code pops the last AST node if it is an `Expr`, `exec`s everything else, then `eval`s just that final expression:

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

We inject a custom `print` closure into the namespace. It writes to a per-block `io.StringIO` buffer instead of `sys.stdout`, so multiple REPLs execute concurrently without interleaving output. The REPL catches exceptions and writes tracebacks to the same buffer. The LLM sees everything — print output, expression values, error traces — in the next iteration's prompt.

### FINAL and FINAL_VAR

The REPL injects `FINAL` and `FINAL_VAR` as closures at init time:

```python
def _final(answer):
    self._final_answer = str(answer)

def _final_var(variable_name):
    self._final_var_name = str(variable_name).strip().strip("\"'")

self._namespace["FINAL"] = _final
self._namespace["FINAL_VAR"] = _final_var
```

When the LLM calls `FINAL("here is my answer")`, execution of further blocks stops and the answer propagates up through `REPLResult.final_answer`.

`FINAL_VAR` defers resolution. The LLM calls `FINAL_VAR("result")` to name a variable, and the REPL resolves it *after* the block finishes executing. This lets the agent build up a complex result and submit it without stringifying inline:

```python
# The LLM writes this
result = "\n".join(f"{f}: {n} lines" for f, n in sorted(counts.items()))
FINAL_VAR("result")
```

After `exec()` completes, the REPL looks up `"result"` in its tracked locals and sets `self._final_answer = str(self._locals["result"])`. Either way, the agent loop sees one `repl_result.final_answer` field.

The REPL tracks user-defined variables in `self._locals` after each block. Both `FINAL_VAR` resolution and the prose-level fallback (next section) depend on this.

### Prose fallback: `find_final_in_prose`

Sometimes the LLM writes `FINAL(...)` in prose rather than in a code block. We catch that too. After the REPL runs all code blocks and finds no `final_answer`, the agent calls `find_final_in_prose(response_text, repl.locals)`:

```python
def find_final_in_prose(text: str, repl_locals: dict) -> str | None:
    prose = _CODE_BLOCK_RE.sub("", text)  # strip code blocks

    m = _FINAL_LITERAL_RE.search(prose)   # FINAL("some literal")
    if m:
        return m.group(1)

    m = _FINAL_IDENT_RE.search(prose)     # FINAL(some_variable)
    if m:
        name = m.group(1)
        if name in repl_locals:
            return str(repl_locals[name])

    m = _FINAL_VAR_RE.search(prose)       # FINAL_VAR("varname")
    if m:
        name = m.group(1)
        if name in repl_locals:
            return str(repl_locals[name])

    return None
```

The function strips fenced code blocks, then searches the remaining prose for three patterns: `FINAL("literal")` returns the string directly, `FINAL(identifier)` looks up the identifier in the REPL's tracked locals, and `FINAL_VAR("varname")` does the same lookup. If a variable name is not found in `repl_locals`, the function returns `None` — the agent loop continues. This avoids false completions when the LLM is just talking about calling `FINAL`, not actually calling it.

---

## The namespace

The REPL pre-loads `os`, `subprocess`, `Path`, and `WORKDIR`. The agent adds three more via closures:

```python
def _build_namespace(self) -> dict:
    ns = {}
    ns["rlm_query"] = self._make_rlm_query()
    ns["rlm_query_batched"] = self._make_rlm_query_batched()
    ns["edit_file"] = self._make_edit_file()
    return ns
```

With `FINAL` and `FINAL_VAR` from the REPL, plus standard imports (`import json` works — it is a real Python namespace), the full namespace is:

| Name | Source | What it does |
|------|--------|-------------|
| `os` | REPL built-in | Standard library `os` module |
| `subprocess` | REPL built-in | Standard library `subprocess` module |
| `Path` | REPL built-in | `pathlib.Path` |
| `WORKDIR` | REPL built-in | String path to the agent's working directory |
| `print` | REPL closure | Writes to a per-block `io.StringIO` buffer, not `sys.stdout` |
| `rlm_query(task)` | Agent closure | Spawns a subagent, returns result string |
| `rlm_query_batched(tasks)` | Agent closure | Spawns multiple subagents in parallel |
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

This closure checks the depth limit and delegates to `SandboxManager.spawn_agent()`. The LLM sees a function that takes a task description and returns a string. What actually happens: a new `RLMAgent` gets created on a thread in an isolated directory, runs its own agent loop, and returns its `FINAL()` answer.

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

Fan-out. Each task gets its own thread, each thread calls `rlm_query`, each `rlm_query` spawns an in-process agent with its own REPL and Chat. Results come back in the original order. This is the parallelism: 25 subagents exploring different directories simultaneously, each in an isolated copy of the repo.

---

## Sandbox isolation

`rlm/sandbox.py` has two classes: `SandboxBudget` and `SandboxManager`.

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

Thread-safe counter. `rlm_query_batched` runs multiple spawns in parallel from different threads, so the lock matters. The budget tracks total sandboxes created over the lifetime of the rollout, not concurrent ones. Once 50 subagents have been spawned, no more get created regardless of how many have finished.

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

In sequence:

1. Acquire budget
2. Create working directory via `git worktree add`
3. Instantiate a new `RLMAgent`, passing `self` as the sandbox manager
4. Run the agent's iteration loop in-process
5. Return the result string
6. Clean up the worktree

### Git worktrees

We chose worktrees because they share the git object store — no copying of `.git/objects`, so creating them is fast:

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

Each worktree gives the agent its own working directory. Edits in one worktree do not affect any other agent's copy. And we clean up with `git worktree remove` after each agent finishes.

For non-git directories, we fall back to `shutil.copytree`.

---

## In-process recursion

Subagents run in the same process as their parent. `spawn_agent()` instantiates an `RLMAgent` with `sandbox_manager=self`, so the child shares the same `SandboxManager` and `SandboxBudget`. No serialization, no JSON, no child process. The child gets its own `REPL` and `Chat`, runs its iteration loop, and returns a result string.

The child's `RLMAgent` has the same `SandboxManager`, which can call `spawn_agent` again, which creates another `RLMAgent`, which can spawn more children. Depth is incremented at each level and checked against `max_depth`. The budget is enforced exactly because every agent shares the same counter and lock.

The REPL avoids process-global mutations: no `os.chdir`, no `sys.stdout` reassignment. Each REPL captures output via a namespace-injected `print` that writes to its own `io.StringIO` buffer. Multiple REPLs on different threads produce isolated output.

---

## The prompts

`rlm/prompts.py` has two functions.

### build_system_prompt

Not a flat template. `build_system_prompt` composes a role description (root agents orchestrate, subagents execute a specific task), REPL rules (every response must have code blocks, variables persist), environment info, and the available functions list. Root agents get a 6-step workflow (explore, investigate, plan, execute, verify, submit); subagents get 4 steps. Subagents also get a visibility warning: the parent only sees what goes into `FINAL()`. All `print()` output is invisible to the parent.

`_subagent_guidance` branches on depth:

```python
def _subagent_guidance(depth: int, max_depth: int, workdir: str) -> str:
    if depth >= max_depth:
        return "You are at maximum depth and CANNOT spawn sub-agents."

    if depth == 0:
        # Root agent: full delegation guidance with examples, self-contained
        # task rules, and a parallel delegation code sample.
        return "## Sub-Agent Delegation (HIGHLY RECOMMENDED) ..."

    if depth < max_depth - 1:
        # Mid-depth: brief encouragement to use rlm_query/rlm_query_batched.
        return "## Sub-Agents ..."

    # depth == max_depth - 1: can spawn, but children will be at max depth.
    return "## Sub-Agents\nYou can spawn sub-agents, but they will be at maximum depth ..."
```

The root agent gets full delegation instructions with examples. An agent at max depth cannot spawn at all. We taper the detail because deeper agents have narrower tasks — they don't need the full set of delegation rules.

### build_user_prompt

`build_user_prompt` varies by iteration:

- **Iteration 0:** `"Start by exploring the codebase..."` No execution output yet.
- **Iterations 1-2:** Shows execution output, then `"Continue working. Investigate further or begin implementing."` Early iterations encourage exploration.
- **Iteration 3+:** Shows execution output, then `"Continue. If your work is complete, verify it and call FINAL(result)."` Later iterations nudge toward completion.

When output gets truncated, the prompt adds advice: store data in variables, or delegate to subagents.

```python
def build_user_prompt(iteration: int, execution_result: str | None) -> str:
    if iteration == 0:
        return "Start by exploring the codebase..."

    truncation_note = ""
    if execution_result and "... (truncated)" in execution_result:
        truncation_note = (
            "\nNote: Output was truncated. Store important data in variables "
            "instead of printing everything, or use sub-agents for large investigations."
        )

    if iteration <= 2:
        return f"Execution output:\n{execution_result}\n{truncation_note}\nContinue working. Investigate further or begin implementing."

    return f"Execution output:\n{execution_result}\n{truncation_note}\nContinue. If your work is complete, verify it and call FINAL(result)."
```

The LLM sees its own stdout, tracebacks, and return values, and iterates from there.

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
| `max_sandboxes` | 50 | Total subagents across the entire tree |
| `max_iterations` | 50 | Max REPL iterations per agent |
| `global_timeout` | 3600 | Agent timeout in seconds |
| `result_truncation_limit` | 10000 | Max characters before truncating subagent results |
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

Depth 0. The root agent. Everything else (the tree of subagents, the parallel exploration, the recursive delegation) emerges from the LLM's decisions inside the loop.

---

## Tracing an execution

```bash
uv run python main.py ./my-project -p "Count the lines of code in each Python file"
```

**Iteration 0** (depth 0): The LLM receives "Start by exploring the codebase..." It writes:

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

Two iterations. No subagents needed. The LLM decided it could handle this alone.

For a harder task ("find all TODOs and fix the easiest one"), the LLM might spend iteration 0 exploring the file tree, iteration 1 spawning 10 subagents via `rlm_query_batched` to search different directories in parallel, iteration 2 reviewing the results and picking the easiest TODO, and iteration 3 calling `edit_file` and `FINAL`. Each of those 10 subagents runs its own iteration loop at depth 1, potentially spawning depth-2 agents for particularly large directories.

---

## The full picture

```
main.py
  -> load_config("config.yaml") -> Config
  -> SandboxBudget(50)
  -> SandboxManager(source_dir, budget, config)
  -> RLMAgent(config, task, workdir, depth=0, sandbox_manager)
  -> agent.run()
      +-- Chat(model, sp=system_prompt, temp=config.temp)   # claudette
      +-- REPL(namespace_extras={rlm_query, ...}, workdir) # persistent namespace
      +-- loop:
          |   chat(user_prompt) -> response                 # LLM call
          |   repl.execute_response(response_text)          # extract + exec
          |   if FINAL/FINAL_VAR called: return result      # code-level check
          |   if FINAL in prose: return result               # prose fallback
          |   truncate output, feed back as next prompt      # continue loop
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

Current language models are not trained for recursive delegation. RLMs do not yet outperform single-agent approaches on benchmarks. We think the architecture is worth exploring anyway.

The parallelism is real. Spawning 25 subagents that each explore a different module of a large codebase, in parallel, each with isolated file access — you need per-agent sandboxes for that. The recursive structure means the tree's shape adapts to the problem. A simple task stays flat. A complex task grows deep and wide, because the agents *decide* to make it so.

Single-agent systems are the practical default. As models get better at decomposition and delegation, the recursive pattern becomes more natural. This repo is the infrastructure for when that happens.
