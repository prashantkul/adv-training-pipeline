"""Configuration loading for the noise pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from noisy_agentdojo.models import Intensity, NoiseConfig

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
_DEFAULT_CONFIG = _CONFIG_DIR / "default.yaml"


def load_config(path: str | Path | None = None) -> NoiseConfig:
    """Load a NoiseConfig from a YAML file. Falls back to defaults if no file exists."""
    if path is None:
        path = _DEFAULT_CONFIG
    path = Path(path)

    if not path.exists():
        return NoiseConfig()

    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    # Convert intensity_weights keys from strings to Intensity enum
    if "intensity_weights" in raw:
        raw["intensity_weights"] = {
            Intensity(k): v for k, v in raw["intensity_weights"].items()
        }

    return NoiseConfig(**raw)
