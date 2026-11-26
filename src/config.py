"""Load project hyperparameters from `config.toml` at the repository root."""
from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any, Dict

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"


def load_config(path: Path | str | None = None) -> Dict[str, Any]:
    """Read the TOML config file fresh every call (so edits apply on rerun)."""
    with open(path or CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def with_overrides(cfg: Dict[str, Any], section: str, **values: Any) -> Dict[str, Any]:
    """Return a copy of `cfg` with some keys of one section replaced (None = keep)."""
    out = copy.deepcopy(cfg)
    for key, val in values.items():
        if val is not None:
            out[section][key] = val
    return out
