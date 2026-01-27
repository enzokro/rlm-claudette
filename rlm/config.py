"""Configuration dataclasses and loader for RLM Agent System."""

from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    """LLM model configuration."""
    name: str = "claude-sonnet-4-5-20250514"
    temperature: float = 0.0
    max_tokens: int = 16384


@dataclass
class RLMConfig:
    """RLM agent runtime configuration."""
    max_sandboxes: int = 50
    max_iterations: int = 50
    global_timeout: int = 3600
    result_truncation_limit: int = 10000
    max_depth: int = 5


@dataclass
class Config:
    """Top-level configuration container."""
    model: ModelConfig = field(default_factory=ModelConfig)
    rlm: RLMConfig = field(default_factory=RLMConfig)

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Deserialize from a plain dict."""
        model = ModelConfig(**d.get("model", {}))
        rlm = RLMConfig(**d.get("rlm", {}))
        return cls(model=model, rlm=rlm)


def load_config(path: str | None = None) -> Config:
    """Load configuration from a YAML file.

    Falls back to defaults if *path* is None or the file does not exist.
    """
    if path is None:
        return Config()
    p = Path(path)
    if not p.exists():
        return Config()
    with open(p, "r") as f:
        data = yaml.safe_load(f) or {}
    return Config.from_dict(data)
