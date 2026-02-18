"""Pydantic models for the noisy_agentdojo pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Intensity(str, Enum):
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"


class ToolCall(BaseModel):
    """A simplified tool call representation."""

    function: str
    args: dict[str, Any]


class ExtractedScenario(BaseModel):
    """A scenario extracted from AgentDojo before noise is applied."""

    suite_name: str
    user_task_id: str
    user_task_prompt: str
    injection_task_id: str | None = None
    injection_task_goal: str | None = None
    attack_name: str | None = None
    injection_vectors: dict[str, str] | None = None
    environment_context: dict[str, Any]
    available_tools: list[str]
    ground_truth_calls: list[ToolCall]
    injection_tool_calls: list[ToolCall] | None = None
    is_benign: bool = False


class NoiseLayerRecord(BaseModel):
    """Record of a noise layer application."""

    layer_name: str
    intensity: Intensity
    params: dict[str, Any] = Field(default_factory=dict)


class NoisyScenario(BaseModel):
    """A scenario with noise layers applied."""

    scenario: ExtractedScenario
    noisy_environment: dict[str, Any]
    noise_layers_applied: list[NoiseLayerRecord]
    noisy_text: str  # The final noisy text that wraps the environment


class ValidationResult(BaseModel):
    """Result from validating whether the injection still triggers."""

    triggered: bool
    matched_calls: list[ToolCall] = Field(default_factory=list)
    agent_response: str = ""
    attempts: int = 1


class ValidatedExample(BaseModel):
    """A fully validated training example ready for JSONL output."""

    suite_name: str
    user_task_id: str
    injection_task_id: str | None = None
    attack_name: str | None = None
    user_prompt: str
    noisy_context: str
    available_tools: list[str]
    ground_truth_calls: list[ToolCall]
    injection_tool_calls: list[ToolCall] | None = None
    noise_layers: list[NoiseLayerRecord]
    validation: ValidationResult | None = None
    is_benign: bool = False

    def to_jsonl_dict(self) -> dict[str, Any]:
        """Serialize for JSONL output."""
        return self.model_dump(mode="json")


class NoiseConfig(BaseModel):
    """Configuration for the noise pipeline."""

    min_layers: int = 1
    max_layers: int = 3
    intensity_weights: dict[Intensity, float] = Field(
        default_factory=lambda: {
            Intensity.LIGHT: 0.3,
            Intensity.MEDIUM: 0.5,
            Intensity.HEAVY: 0.2,
        }
    )
    layer_weights: dict[str, float] = Field(default_factory=dict)
    seed: int | None = None
    suites: list[str] = Field(
        default_factory=lambda: ["workspace", "travel", "banking", "slack"]
    )
    attacks: list[str] = Field(
        default_factory=lambda: ["important_instructions"]
    )
    model_name: str = "gpt-4o-2024-05-13"
    gemini_model: str = "gemini-2.0-flash"
    version: str = "v1.2.2"
