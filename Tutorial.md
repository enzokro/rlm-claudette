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

Three things are happening here:

1. **claudette's `Chat` manages conversation state.** It handles message history, system prompts, and API calls. We call it with a user message, get back an Anthropic `Message` object, and extract the text. We never touch the messages list directly.
2. **The loop has a two-stage completion check.** After executing the code blocks, the agent first checks whether `FINAL` or `FINAL_VAR` was called inside the code (the REPL sets `final_answer`). If not, it falls back to `find_final_in_prose`, which scans the response text outside code blocks for `FINAL(...)` patterns. If neither check finds a result, the execution output is truncated to the configured limit and fed back as the next user prompt.
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

The REPL also tracks user-defined variables in `self._locals` after each block execution. This enables `FINAL_VAR` resolution and also supports a prose-level fallback in the agent loop (next section).

### Prose fallback: `find_final_in_prose`

Sometimes the LLM writes `FINAL(...)` in prose text rather than inside a code block. The agent loop handles this with a fallback in `rlm/agent.py`. After the REPL runs all code blocks and finds no `final_answer`, the agent calls `find_final_in_prose(response_text, repl.locals)`:

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

The function strips all fenced code blocks from the response, then searches the remaining prose for three patterns: `FINAL("literal")` returns the literal string directly, `FINAL(identifier)` looks up the identifier in the REPL's tracked locals, and `FINAL_VAR("varname")` does the same variable lookup. If an identifier or variable name is not found in `repl_locals`, the function returns `None` — the agent loop continues rather than treating a missing variable as a final answer. This avoids false completions when the LLM mentions `FINAL(result_string)` in a sentence as a description of intent rather than an actual call.

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
| `print` | REPL closure | Writes to a per-block `io.StringIO` buffer, not `sys.stdout` |
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

Assembles a structured system prompt from several sections. The final string is not a flat template; `build_system_prompt` composes role text, instructions, and depth-dependent guidance:

- **Role** — differs for root vs sub-agent. Root: "You are the root RLM agent. You orchestrate sub-agents and synthesize their results." Sub-agent: "You are a sub-agent spawned to accomplish a specific task. You have a fresh copy of the repository with no edits from your parent."
- **REPL Instructions** — every response must contain at least one fenced Python code block. No prose-only responses. Variables persist across iterations.
- **Environment** — pre-loaded modules (`os`, `subprocess`, `Path`, `WORKDIR`), current depth, and truncation limit (~10,000 characters).
- **Available Functions** — `edit_file`, `FINAL`, `FINAL_VAR`, `rlm_query`, `rlm_query_batched`.
- **Sub-agent guidance** — via `_subagent_guidance(depth, max_depth, workdir)`, which has four depth-dependent variants.
- **Workflow** — 6-step for root (explore, investigate, plan, execute, verify, submit), 4-step for sub-agents (explore, investigate, execute, report).
- **Final guidance** — via `_final_guidance(is_root)`. Sub-agents get a visibility warning: the parent only sees what is passed to `FINAL()`. All `print()` output is invisible to the parent.

The sub-agent guidance function branches on depth:

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

This means a depth-0 root agent gets detailed delegation instructions with examples, a mid-tree agent gets a short reminder, an agent one level from the max gets a warning that its children cannot recurse further, and an agent at max depth is told it cannot spawn at all.

### build_user_prompt

Handles the iteration-to-iteration communication. Three cases:

- **Iteration 0:** `"Start by exploring the codebase. List files, read key files, and understand the workspace before making changes or calling FINAL()."` No execution output yet.
- **Iterations 1-2:** Shows the execution output, then `"Continue working. Investigate further or begin implementing."` Early iterations encourage exploration before committing to a solution.
- **Iteration 3+:** Shows the execution output, then `"Continue. If your work is complete, verify it and call FINAL(result)."` Later iterations nudge toward completion.

If the execution output contains `"... (truncated)"` (appended by the agent loop when output exceeds the truncation limit), the prompt appends advice: store important data in variables instead of printing everything, or delegate to sub-agents for large investigations.

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

The LLM sees its own stdout, tracebacks, and return values, and can react: fix errors, refine its approach, run more code based on what it learned.

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

Current language models are not specifically trained to leverage recursive delegation. RLMs do not necessarily outperform single-agent approaches on benchmarks yet. But the architecture has properties worth paying attention to.

The parallelism is real. Spawning 25 sub-agents that each explore a different module of a large codebase, in parallel, each with isolated file access: that is hard to get without per-agent sandboxes. The recursive structure means the tree's shape adapts to the problem. A simple task stays flat. A complex task grows deep and wide, because the agents *decide* to make it so.

Single-agent systems are the practical default today. But as models get better at decomposition and delegation, architectures like this become more natural. The infrastructure for recursive, parallel, isolated agent execution is the part worth building now.
