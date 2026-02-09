"""Tests for rlm.repl: code extraction, execution, FINAL/FINAL_VAR, namespace."""

from rlm.repl import extract_code_blocks, REPL


# -- extract_code_blocks ------------------------------------------------------

def test_extract_no_blocks():
    assert extract_code_blocks("no code here") == []


def test_extract_one_block():
    text = "Here is code:\n```python\nx = 1\n```\nDone."
    assert extract_code_blocks(text) == ["x = 1\n"]


def test_extract_multiple_blocks():
    text = "```python\na = 1\n```\ntext\n```python\nb = 2\n```"
    blocks = extract_code_blocks(text)
    assert len(blocks) == 2
    assert "a = 1" in blocks[0]
    assert "b = 2" in blocks[1]


# -- REPL._execute_block basics -----------------------------------------------

def _make_repl(workdir="/tmp/test"):
    return REPL(namespace_extras={}, workdir=workdir)


def test_execute_block_print():
    repl = _make_repl()
    result = repl.execute_response("```python\nprint('hello')\n```")
    assert "hello" in result.output
    assert result.final_answer is None


def test_execute_block_expression_capture():
    """Trailing expression should be captured like IPython."""
    repl = _make_repl()
    result = repl.execute_response("```python\n1 + 2\n```")
    assert "3" in result.output


def test_execute_block_exception():
    repl = _make_repl()
    result = repl.execute_response("```python\n1 / 0\n```")
    assert "ZeroDivisionError" in result.output
    assert result.final_answer is None


# -- FINAL / FINAL_VAR --------------------------------------------------------

def test_final_literal():
    repl = _make_repl()
    result = repl.execute_response('```python\nFINAL("done")\n```')
    assert result.final_answer == "done"


def test_final_var():
    repl = _make_repl()
    result = repl.execute_response('```python\nanswer = 42\nFINAL_VAR("answer")\n```')
    assert result.final_answer == "42"


def test_final_var_missing():
    """FINAL_VAR with a nonexistent variable should not set final_answer."""
    repl = _make_repl()
    result = repl.execute_response('```python\nFINAL_VAR("nonexistent")\n```')
    assert result.final_answer is None
    assert "not found" in result.output


def test_final_stops_further_blocks():
    """After FINAL is called, subsequent blocks should not execute."""
    repl = _make_repl()
    text = '```python\nFINAL("first")\n```\n```python\nprint("second")\n```'
    result = repl.execute_response(text)
    assert result.final_answer == "first"
    assert "second" not in result.output


# -- Namespace persistence -----------------------------------------------------

def test_namespace_persists_across_calls():
    repl = _make_repl()
    repl.execute_response("```python\nfoo = 123\n```")
    result = repl.execute_response("```python\nprint(foo)\n```")
    assert "123" in result.output


def test_locals_tracking():
    repl = _make_repl()
    repl.execute_response("```python\nmy_var = 'hello'\n```")
    assert "my_var" in repl.locals
    assert repl.locals["my_var"] == "hello"


# -- No code blocks found -----------------------------------------------------

def test_no_code_blocks_output():
    repl = _make_repl()
    result = repl.execute_response("Just some prose, no code.")
    assert "no code blocks found" in result.output
    assert result.final_answer is None
