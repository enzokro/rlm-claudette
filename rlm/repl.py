"""Code extraction and sandboxed execution for RLM agents."""

import ast
import builtins
import io
import re
import traceback
from dataclasses import dataclass
from pathlib import Path


_CODE_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)


def extract_code_blocks(text: str) -> list[str]:
    """Return all ```python ... ``` fenced code blocks from *text*."""
    return _CODE_BLOCK_RE.findall(text)


@dataclass
class REPLResult:
    """Result of executing one or more code blocks."""
    output: str
    final_answer: str | None = None
    error: str | None = None


class REPL:
    """Persistent Python REPL with captured stdout.

    Thread-safe: no process-global mutations (no os.chdir, no sys.stdout
    reassignment). Each REPL captures output via a namespace-injected
    ``print`` that writes to a per-block ``io.StringIO`` buffer.
    """

    def __init__(self, namespace_extras: dict, workdir: str) -> None:
        import os
        import subprocess

        self._namespace: dict = {
            "os": os,
            "subprocess": subprocess,
            "Path": Path,
            "WORKDIR": workdir,
        }
        self._namespace.update(namespace_extras)
        self._workdir = workdir

        # FINAL / FINAL_VAR state
        self._final_answer: str | None = None
        self._final_var_name: str | None = None

        def _final(answer):  # noqa: ANN001
            self._final_answer = str(answer)

        def _final_var(variable_name):  # noqa: ANN001
            self._final_var_name = str(variable_name).strip().strip("\"'")

        self._namespace["FINAL"] = _final
        self._namespace["FINAL_VAR"] = _final_var

        # Track the original set of global keys so we can detect user-defined vars
        self._globals_keys: set[str] = set(self._namespace.keys())
        self._locals: dict = {}

        # Per-block buffer; rebound before each execution
        self._buf: io.StringIO = io.StringIO()

        # Inject a custom print that writes to the current buffer
        def _print(*args, **kwargs):  # noqa: ANN002, ANN003
            kwargs.setdefault("file", self._buf)
            builtins.print(*args, **kwargs)

        self._namespace["print"] = _print

    @property
    def locals(self) -> dict:
        """User-defined variables created during execution."""
        return self._locals

    # ------------------------------------------------------------------
    def execute_response(self, text: str) -> REPLResult:
        """Extract code blocks from *text*, run them, return combined result."""
        blocks = extract_code_blocks(text)
        if not blocks:
            return REPLResult(output="(no code blocks found)")

        outputs: list[str] = []
        for block in blocks:
            out = self._execute_block(block)
            outputs.append(out)
            if self._final_answer is not None:
                break

        combined = "\n".join(outputs)
        return REPLResult(
            output=combined,
            final_answer=self._final_answer,
            error=None,
        )

    # ------------------------------------------------------------------
    def _execute_block(self, code: str) -> str:
        """Execute a single code block, returning captured output."""
        buf = io.StringIO()
        self._buf = buf  # rebind so the injected print writes here

        try:
            tree = ast.parse(code)
            # If last statement is an expression, wrap it so its value prints
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last_expr = tree.body.pop()
                # Compile and exec everything except the last expression
                if tree.body:
                    exec(  # noqa: S102
                        compile(ast.Module(body=tree.body, type_ignores=[]), "<repl>", "exec"),
                        self._namespace,
                    )
                # Evaluate the last expression and print its repr
                val = eval(  # noqa: S307
                    compile(ast.Expression(body=last_expr.value), "<repl>", "eval"),
                    self._namespace,
                )
                if val is not None:
                    builtins.print(repr(val), file=buf)
            else:
                exec(compile(tree, "<repl>", "exec"), self._namespace)  # noqa: S102

        except Exception:
            traceback.print_exc(file=buf)

        # Track user-defined variables
        for key, value in self._namespace.items():
            if key not in self._globals_keys and not key.startswith("_"):
                self._locals[key] = value

        # TODO: todo breadcrumb for the RLM to find

        # Resolve FINAL_VAR after execution
        if self._final_var_name is not None and self._final_answer is None:
            name = self._final_var_name
            if name in self._locals:
                self._final_answer = str(self._locals[name])
            else:
                self._final_answer = f"Error: Variable '{name}' not found"

        return buf.getvalue()
