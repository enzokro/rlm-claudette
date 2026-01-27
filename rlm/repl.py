"""Code extraction and sandboxed execution for RLM agents."""

import ast
import io
import os
import re
import subprocess
import sys
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
    """Persistent Python REPL with captured stdout."""

    def __init__(self, namespace_extras: dict, workdir: str) -> None:
        self._namespace: dict = {
            "os": os,
            "subprocess": subprocess,
            "Path": Path,
        }
        self._namespace.update(namespace_extras)
        self._workdir = workdir
        self._final_called = False
        self._final_value: str | None = None

        # Inject FINAL helper into the namespace
        def _final(answer):  # noqa: ANN001
            self._final_called = True
            self._final_value = str(answer)
        self._namespace["FINAL"] = _final

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
            if self._final_called:
                break

        combined = "\n".join(outputs)
        return REPLResult(
            output=combined,
            final_answer=self._final_value if self._final_called else None,
            error=None,
        )

    # ------------------------------------------------------------------
    def _execute_block(self, code: str) -> str:
        """Execute a single code block, returning captured output."""
        old_cwd = os.getcwd()
        buf = io.StringIO()
        old_stdout = sys.stdout
        try:
            os.chdir(self._workdir)
            sys.stdout = buf

            tree = ast.parse(code)
            # If last statement is an expression, wrap it so its value prints
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last_expr = tree.body.pop()
                # Compile and exec everything except the last expression
                if tree.body:
                    exec(compile(ast.Module(body=tree.body, type_ignores=[]), "<repl>", "exec"), self._namespace)  # noqa: S102
                # Evaluate the last expression and print its repr
                val = eval(compile(ast.Expression(body=last_expr.value), "<repl>", "eval"), self._namespace)  # noqa: S307
                if val is not None:
                    print(repr(val))
            else:
                exec(compile(tree, "<repl>", "exec"), self._namespace)  # noqa: S102

        except Exception:
            traceback.print_exc(file=buf)
        finally:
            sys.stdout = old_stdout
            os.chdir(old_cwd)

        return buf.getvalue()
