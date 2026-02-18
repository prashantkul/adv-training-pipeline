"""Tests for the noise pipeline orchestrator."""

from __future__ import annotations

import pytest

from noisy_agentdojo.models import ExtractedScenario, Intensity, NoiseConfig, NoisyScenario
from noisy_agentdojo.noise_pipeline import NoisePipeline, PROGRAMMATIC_LAYERS


class TestNoisePipeline:
    def test_default_pipeline_creates(self):
        pipeline = NoisePipeline()
        assert pipeline.config is not None
        assert len(pipeline.layers) == len(PROGRAMMATIC_LAYERS)

    def test_apply_returns_noisy_scenario(self, sample_workspace_scenario):
        cfg = NoiseConfig(seed=42)
        pipeline = NoisePipeline(config=cfg)
        result = pipeline.apply(sample_workspace_scenario)
        assert isinstance(result, NoisyScenario)
        assert len(result.noise_layers_applied) >= 1
        assert len(result.noisy_text) > 0

    def test_deterministic_with_seed(self, sample_workspace_scenario):
        cfg1 = NoiseConfig(seed=42)
        cfg2 = NoiseConfig(seed=42)
        p1 = NoisePipeline(config=cfg1)
        p2 = NoisePipeline(config=cfg2)
        r1 = p1.apply(sample_workspace_scenario)
        r2 = p2.apply(sample_workspace_scenario)
        assert r1.noisy_text == r2.noisy_text
        assert len(r1.noise_layers_applied) == len(r2.noise_layers_applied)

    def test_different_seeds_different_output(self, sample_workspace_scenario):
        p1 = NoisePipeline(config=NoiseConfig(seed=42))
        p2 = NoisePipeline(config=NoiseConfig(seed=99))
        r1 = p1.apply(sample_workspace_scenario)
        r2 = p2.apply(sample_workspace_scenario)
        # Very likely to be different
        assert r1.noisy_text != r2.noisy_text

    def test_apply_batch(self, sample_workspace_scenario, sample_benign_scenario):
        cfg = NoiseConfig(seed=42)
        pipeline = NoisePipeline(config=cfg)
        results = pipeline.apply_batch([sample_workspace_scenario, sample_benign_scenario])
        assert len(results) == 2
        assert all(isinstance(r, NoisyScenario) for r in results)

    def test_max_layers_respected(self, sample_workspace_scenario):
        cfg = NoiseConfig(seed=42, min_layers=1, max_layers=1)
        pipeline = NoisePipeline(config=cfg)
        result = pipeline.apply(sample_workspace_scenario)
        assert len(result.noise_layers_applied) == 1

    def test_layer_weights_affect_selection(self, sample_workspace_scenario):
        # Give all weight to forwarded_thread
        cfg = NoiseConfig(
            seed=42,
            min_layers=1,
            max_layers=1,
            layer_weights={
                "forwarded_thread": 100.0,
                "voicemail_stt": 0.001,
                "html_email": 0.001,
                "calendar_contact": 0.001,
            },
        )
        pipeline = NoisePipeline(config=cfg)
        results = [pipeline.apply(sample_workspace_scenario) for _ in range(10)]
        layer_names = [r.noise_layers_applied[0].layer_name for r in results]
        # Most should be forwarded_thread
        assert layer_names.count("forwarded_thread") >= 7

    def test_intensity_recorded(self, sample_workspace_scenario):
        cfg = NoiseConfig(seed=42)
        pipeline = NoisePipeline(config=cfg)
        result = pipeline.apply(sample_workspace_scenario)
        for record in result.noise_layers_applied:
            assert record.intensity in Intensity
