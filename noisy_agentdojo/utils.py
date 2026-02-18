"""Shared utilities: logging, JSON helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from noisy_agentdojo.models import ValidatedExample


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def write_jsonl(examples: list[ValidatedExample], path: str | Path) -> None:
    """Write validated examples to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_jsonl_dict()) + "\n")


def read_jsonl(path: str | Path) -> list[ValidatedExample]:
    """Read validated examples from a JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                examples.append(ValidatedExample(**data))
    return examples


def summarize_examples(examples: list[ValidatedExample]) -> dict[str, Any]:
    """Compute summary stats for a batch of examples."""
    total = len(examples)
    if total == 0:
        return {"total": 0}

    validated = [e for e in examples if e.validation is not None]
    triggered = [e for e in validated if e.validation.triggered]
    benign = [e for e in examples if e.is_benign]
    adversarial = [e for e in examples if not e.is_benign]

    # Suite breakdown
    suites: dict[str, int] = {}
    for e in examples:
        suites[e.suite_name] = suites.get(e.suite_name, 0) + 1

    # Layer breakdown
    layers: dict[str, int] = {}
    for e in examples:
        for nl in e.noise_layers:
            layers[nl.layer_name] = layers.get(nl.layer_name, 0) + 1

    # Attack breakdown
    attacks: dict[str, int] = {}
    for e in adversarial:
        if e.attack_name:
            attacks[e.attack_name] = attacks.get(e.attack_name, 0) + 1

    return {
        "total": total,
        "benign": len(benign),
        "adversarial": len(adversarial),
        "validated": len(validated),
        "triggered": len(triggered),
        "trigger_rate": len(triggered) / len(validated) if validated else 0,
        "suites": suites,
        "layers": layers,
        "attacks": attacks,
    }
