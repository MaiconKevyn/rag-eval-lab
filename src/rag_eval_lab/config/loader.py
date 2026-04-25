from __future__ import annotations

from pathlib import Path

import yaml
from dotenv import load_dotenv

from rag_eval_lab.config.schema import ExperimentConfig

load_dotenv()


def load_config(path: str | Path) -> ExperimentConfig:
    """Parse a YAML experiment config and validate via Pydantic.

    Fails fast (before any API call) on schema errors.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ExperimentConfig.model_validate(raw)
