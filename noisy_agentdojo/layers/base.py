"""Abstract base class for noise layers."""

from __future__ import annotations

import abc
from typing import Any

from noisy_agentdojo.models import ExtractedScenario, Intensity, NoisyScenario, NoiseLayerRecord


class NoiseLayer(abc.ABC):
    """Base class for all noise layers."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique identifier for this layer."""
        ...

    @property
    def weight(self) -> float:
        """Default selection weight for this layer."""
        return 1.0

    @abc.abstractmethod
    def applicable_to(self, scenario: ExtractedScenario) -> bool:
        """Whether this layer can be applied to the given scenario."""
        ...

    @abc.abstractmethod
    def apply(
        self,
        scenario: ExtractedScenario,
        intensity: Intensity,
        rng: Any,
        existing_noisy_text: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Apply noise to the scenario.

        Args:
            scenario: The extracted scenario to augment.
            intensity: How much noise to apply.
            rng: A random.Random instance for deterministic behavior.
            existing_noisy_text: Previously generated noisy text to layer on top of.

        Returns:
            Tuple of (noisy_text, params_dict) where params_dict records layer-specific params.
        """
        ...

    def apply_to_scenario(
        self,
        scenario: ExtractedScenario,
        intensity: Intensity,
        rng: Any,
        previous: NoisyScenario | None = None,
    ) -> NoisyScenario:
        """Apply this layer and return a NoisyScenario."""
        existing_text = previous.noisy_text if previous else None
        prior_layers = list(previous.noise_layers_applied) if previous else []

        noisy_text, params = self.apply(scenario, intensity, rng, existing_text)

        record = NoiseLayerRecord(
            layer_name=self.name,
            intensity=intensity,
            params=params,
        )

        return NoisyScenario(
            scenario=scenario,
            noisy_environment=scenario.environment_context,
            noise_layers_applied=prior_layers + [record],
            noisy_text=noisy_text,
        )
