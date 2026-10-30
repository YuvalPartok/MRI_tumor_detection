"""Configuration loading and device selection.

The YAML file in ``configs/`` is the single source of truth for all hyperparameters.
It is loaded into a lightweight attribute-access dictionary so code can write
``cfg.train.lr`` instead of ``cfg["train"]["lr"]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

# Repository root (…/MRI_tumor_identification). Relative paths in the config are resolved against it,
# so scripts work no matter which directory they are launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


class Config(dict):
    """A dict whose keys are also accessible as attributes (recursively)."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as err:
            raise AttributeError(f"Config has no key '{name}'") from err

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        return cls({k: cls.from_dict(v) if isinstance(v, dict) else v for k, v in data.items()})


def load_config(path: str | Path | None = None) -> Config:
    """Load the YAML config and turn every entry of ``paths`` into an absolute ``Path``."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        cfg = Config.from_dict(yaml.safe_load(f))
    for key, value in cfg.paths.items():
        p = Path(value)
        cfg.paths[key] = p if p.is_absolute() else PROJECT_ROOT / p
    return cfg


def get_device(preference: str = "auto") -> torch.device:
    """Pick the best available accelerator: Apple GPU (MPS) > CUDA > CPU."""
    if preference != "auto":
        return torch.device(preference)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed every random number generator we use, for reproducible runs."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
