"""Noise pipeline orchestrator."""

from __future__ import annotations

import logging
import random
from typing import Any

from noisy_agentdojo.layers.base import NoiseLayer
from noisy_agentdojo.layers.calendar_contact import CalendarContactLayer
from noisy_agentdojo.layers.forwarded_thread import ForwardedThreadLayer
from noisy_agentdojo.layers.html_email import HTMLEmailLayer
from noisy_agentdojo.layers.voicemail_stt import VoicemailSTTLayer
from noisy_agentdojo.models import (
    ExtractedScenario,
    Intensity,
    NoiseConfig,
    NoisyScenario,
)

logger = logging.getLogger(__name__)

# Registry of all programmatic layers
PROGRAMMATIC_LAYERS: list[NoiseLayer] = [
    ForwardedThreadLayer(),
    VoicemailSTTLayer(),
    HTMLEmailLayer(),
    CalendarContactLayer(),
]


def _get_all_layers(include_llm: bool = False) -> list[NoiseLayer]:
    """Get all available layers, optionally including LLM-powered ones."""
    layers = list(PROGRAMMATIC_LAYERS)
    if include_llm:
        try:
            from noisy_agentdojo.layers.mixed_language import MixedLanguageLayer

            layers.append(MixedLanguageLayer())
        except ImportError:
            logger.debug("MixedLanguageLayer not available.")
        try:
            from noisy_agentdojo.layers.imessage_thread import IMessageThreadLayer

            layers.append(IMessageThreadLayer())
        except ImportError:
            logger.debug("IMessageThreadLayer not available.")
    return layers


class NoisePipeline:
    """Orchestrates noise layer selection and sequential application."""

    def __init__(
        self,
        config: NoiseConfig | None = None,
        include_llm_layers: bool = False,
    ) -> None:
        self.config = config or NoiseConfig()
        self.rng = random.Random(self.config.seed)
        self.layers = _get_all_layers(include_llm_layers)

    def _get_applicable_layers(
        self, scenario: ExtractedScenario
    ) -> list[NoiseLayer]:
        return [l for l in self.layers if l.applicable_to(scenario)]

    def _sample_intensity(self) -> Intensity:
        weights = self.config.intensity_weights
        population = list(weights.keys())
        w = [weights[k] for k in population]
        return self.rng.choices(population, weights=w, k=1)[0]

    def _sample_layers(
        self, applicable: list[NoiseLayer]
    ) -> list[NoiseLayer]:
        if not applicable:
            return []

        num = self.rng.randint(self.config.min_layers, self.config.max_layers)
        num = min(num, len(applicable))

        # Use layer weights from config, falling back to layer defaults
        weights = []
        for layer in applicable:
            w = self.config.layer_weights.get(layer.name, layer.weight)
            weights.append(w)

        selected = self.rng.choices(applicable, weights=weights, k=num)
        return selected

    def apply(self, scenario: ExtractedScenario) -> NoisyScenario:
        """Apply noise layers to a scenario."""
        applicable = self._get_applicable_layers(scenario)
        if not applicable:
            logger.warning(
                "No applicable layers for %s/%s",
                scenario.suite_name,
                scenario.user_task_id,
            )
            return NoisyScenario(
                scenario=scenario,
                noisy_environment=scenario.environment_context,
                noise_layers_applied=[],
                noise_layers_text="",
            )

        selected_layers = self._sample_layers(applicable)
        result: NoisyScenario | None = None

        for layer in selected_layers:
            intensity = self._sample_intensity()
            result = layer.apply_to_scenario(
                scenario, intensity, self.rng, previous=result
            )

        assert result is not None
        return result

    def apply_batch(
        self, scenarios: list[ExtractedScenario]
    ) -> list[NoisyScenario]:
        """Apply noise to a batch of scenarios."""
        return [self.apply(s) for s in scenarios]
