"""Tests for rlm.agent: find_final_in_prose."""

from rlm.agent import find_final_in_prose


# -- FINAL with string literal -------------------------------------------------

def test_final_literal_in_prose():
    text = 'The answer is FINAL("hello world").'
    assert find_final_in_prose(text, {}) == "hello world"


def test_final_literal_single_quotes():
    text = "Result: FINAL('the result')"
    assert find_final_in_prose(text, {}) == "the result"


# -- FINAL with identifier (variable lookup) -----------------------------------

def test_final_ident_existing_variable():
    text = "Now I call FINAL(my_result)"
    assert find_final_in_prose(text, {"my_result": 42}) == "42"


def test_final_ident_missing_variable():
    text = "Now I call FINAL(nonexistent)"
    assert find_final_in_prose(text, {}) is None


# -- FINAL_VAR -----------------------------------------------------------------

def test_final_var_existing():
    text = 'Submit: FINAL_VAR("answer")'
    assert find_final_in_prose(text, {"answer": "done"}) == "done"


def test_final_var_missing():
    text = 'Submit: FINAL_VAR("nope")'
    assert find_final_in_prose(text, {}) is None


# -- Code blocks should be excluded --------------------------------------------

def test_final_inside_code_block_ignored():
    """FINAL inside a code block should NOT trigger the prose fallback."""
    text = '```python\nFINAL("inside code")\n```\nSome other prose.'
    assert find_final_in_prose(text, {}) is None


def test_final_in_prose_but_not_in_code():
    """FINAL in prose should be found even when code blocks exist."""
    text = '```python\nx = 1\n```\nSo the answer is FINAL("found it").'
    assert find_final_in_prose(text, {}) == "found it"


# -- No FINAL at all -----------------------------------------------------------

def test_no_final_returns_none():
    text = "This is just normal text with no final call."
    assert find_final_in_prose(text, {}) is None


# -- Edge cases ----------------------------------------------------------------

def test_final_with_whitespace():
    text = 'FINAL  (  "spaced out"  )'
    assert find_final_in_prose(text, {}) == "spaced out"
