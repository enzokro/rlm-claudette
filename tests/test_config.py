"""Tests for rlm.config: load_config, from_dict, defaults."""

import tempfile
from pathlib import Path

from rlm.config import Config, ModelConfig, RLMConfig, load_config


# -- Defaults ------------------------------------------------------------------

def test_default_config():
    cfg = Config()
    assert cfg.model.name == "claude-sonnet-4-5-20250514"
    assert cfg.model.temperature == 0.0
    assert cfg.model.max_tokens == 16384
    assert cfg.rlm.max_sandboxes == 50
    assert cfg.rlm.max_iterations == 50
    assert cfg.rlm.agent_timeout == 3600
    assert cfg.rlm.max_depth == 5


# -- from_dict -----------------------------------------------------------------

def test_from_dict_full():
    d = {
        "model": {"name": "test-model", "temperature": 0.5, "max_tokens": 1024},
        "rlm": {"max_sandboxes": 10, "max_iterations": 20, "agent_timeout": 60,
                 "result_truncation_limit": 500, "max_depth": 3},
    }
    cfg = Config.from_dict(d)
    assert cfg.model.name == "test-model"
    assert cfg.model.temperature == 0.5
    assert cfg.rlm.max_sandboxes == 10
    assert cfg.rlm.agent_timeout == 60
    assert cfg.rlm.max_depth == 3


def test_from_dict_partial():
    """Missing keys should fall back to dataclass defaults."""
    d = {"model": {"name": "custom-model"}}
    cfg = Config.from_dict(d)
    assert cfg.model.name == "custom-model"
    assert cfg.model.temperature == 0.0  # default
    assert cfg.rlm.max_sandboxes == 50  # default


def test_from_dict_empty():
    cfg = Config.from_dict({})
    assert cfg.model.name == "claude-sonnet-4-5-20250514"
    assert cfg.rlm.max_depth == 5


# -- to_dict / from_dict roundtrip ---------------------------------------------

def test_roundtrip():
    original = Config(
        model=ModelConfig(name="rt-model", temperature=0.7, max_tokens=2048),
        rlm=RLMConfig(max_sandboxes=5, max_depth=2),
    )
    d = original.to_dict()
    restored = Config.from_dict(d)
    assert restored.model.name == "rt-model"
    assert restored.rlm.max_sandboxes == 5
    assert restored.rlm.max_depth == 2


# -- load_config ---------------------------------------------------------------

def test_load_config_none():
    cfg = load_config(None)
    assert cfg.model.name == "claude-sonnet-4-5-20250514"


def test_load_config_missing_file():
    cfg = load_config("/nonexistent/path/config.yaml")
    assert cfg.model.name == "claude-sonnet-4-5-20250514"


def test_load_config_valid_yaml():
    content = """\
model:
  name: "yaml-model"
  temperature: 0.3
rlm:
  max_depth: 2
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(content)
        f.flush()
        cfg = load_config(f.name)

    assert cfg.model.name == "yaml-model"
    assert cfg.model.temperature == 0.3
    assert cfg.rlm.max_depth == 2
    assert cfg.rlm.max_sandboxes == 50  # default


def test_load_config_empty_yaml():
    """An empty YAML file should return defaults."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("")
        f.flush()
        cfg = load_config(f.name)

    assert cfg.model.name == "claude-sonnet-4-5-20250514"
