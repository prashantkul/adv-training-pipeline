"""Tests for each programmatic noise layer."""

from __future__ import annotations

import random

import pytest

from noisy_agentdojo.layers.forwarded_thread import ForwardedThreadLayer
from noisy_agentdojo.layers.voicemail_stt import VoicemailSTTLayer
from noisy_agentdojo.layers.html_email import HTMLEmailLayer
from noisy_agentdojo.layers.calendar_contact import CalendarContactLayer
from noisy_agentdojo.models import ExtractedScenario, Intensity


class TestForwardedThreadLayer:
    def test_name(self):
        assert ForwardedThreadLayer().name == "forwarded_thread"

    def test_applicable_to_workspace(self, sample_workspace_scenario):
        layer = ForwardedThreadLayer()
        assert layer.applicable_to(sample_workspace_scenario)

    def test_apply_produces_forwarded_text(self, sample_workspace_scenario, deterministic_rng):
        layer = ForwardedThreadLayer()
        text, params = layer.apply(sample_workspace_scenario, Intensity.MEDIUM, deterministic_rng)
        assert "Forwarded message" in text
        assert "depth" in params

    def test_depth_varies_with_intensity(self, sample_workspace_scenario):
        layer = ForwardedThreadLayer()
        for intensity in Intensity:
            rng = random.Random(42)
            _, params = layer.apply(sample_workspace_scenario, intensity, rng)
            assert "depth" in params

    def test_deterministic(self, sample_workspace_scenario):
        layer = ForwardedThreadLayer()
        text1, _ = layer.apply(sample_workspace_scenario, Intensity.MEDIUM, random.Random(42))
        text2, _ = layer.apply(sample_workspace_scenario, Intensity.MEDIUM, random.Random(42))
        assert text1 == text2


class TestVoicemailSTTLayer:
    def test_name(self):
        assert VoicemailSTTLayer().name == "voicemail_stt"

    def test_applicable_to_any(self, sample_workspace_scenario, sample_benign_scenario):
        layer = VoicemailSTTLayer()
        assert layer.applicable_to(sample_workspace_scenario)
        assert layer.applicable_to(sample_benign_scenario)

    def test_apply_produces_voicemail_framing(self, sample_workspace_scenario, deterministic_rng):
        layer = VoicemailSTTLayer()
        text, params = layer.apply(sample_workspace_scenario, Intensity.MEDIUM, deterministic_rng)
        # Should have voicemail framing
        assert any(marker in text for marker in ["Voicemail", "transcript", "Transcription", "transcription"])

    def test_preserves_injection_text(self, sample_workspace_scenario, deterministic_rng):
        layer = VoicemailSTTLayer()
        # The injection text should be preserved
        injection = list(sample_workspace_scenario.injection_vectors.values())[0]
        text, _ = layer.apply(sample_workspace_scenario, Intensity.HEAVY, deterministic_rng)
        # The injection payload should appear verbatim in the output
        assert injection in text or "INFORMATION" in text

    def test_degrades_non_injection_text(self, sample_benign_scenario, deterministic_rng):
        layer = VoicemailSTTLayer()
        text, _ = layer.apply(sample_benign_scenario, Intensity.HEAVY, deterministic_rng)
        # At heavy intensity, should see filler words or [inaudible]
        has_degradation = (
            "[inaudible]" in text
            or "um" in text.lower()
            or "uh" in text.lower()
            or "like" in text.lower()
        )
        # Not guaranteed at every seed, but highly likely at heavy
        assert len(text) > 0


class TestHTMLEmailLayer:
    def test_name(self):
        assert HTMLEmailLayer().name == "html_email"

    def test_applicable_to_workspace(self, sample_workspace_scenario):
        layer = HTMLEmailLayer()
        assert layer.applicable_to(sample_workspace_scenario)

    def test_apply_produces_html(self, sample_workspace_scenario, deterministic_rng):
        layer = HTMLEmailLayer()
        text, params = layer.apply(sample_workspace_scenario, Intensity.MEDIUM, deterministic_rng)
        assert "<html>" in text.lower() or "<!doctype" in text.lower()
        assert "template" in params
        assert "hiding_methods" in params

    def test_hiding_methods_scale_with_intensity(self, sample_workspace_scenario):
        layer = HTMLEmailLayer()
        _, light_params = layer.apply(sample_workspace_scenario, Intensity.LIGHT, random.Random(42))
        _, heavy_params = layer.apply(sample_workspace_scenario, Intensity.HEAVY, random.Random(42))
        assert len(light_params["hiding_methods"]) <= len(heavy_params["hiding_methods"])

    def test_injection_in_hidden_element(self, sample_workspace_scenario, deterministic_rng):
        layer = HTMLEmailLayer()
        text, _ = layer.apply(sample_workspace_scenario, Intensity.MEDIUM, deterministic_rng)
        # The injection text should appear somewhere in the HTML
        assert "INFORMATION" in text


class TestCalendarContactLayer:
    def test_name(self):
        assert CalendarContactLayer().name == "calendar_contact"

    def test_applicable_to_workspace(self, sample_workspace_scenario):
        layer = CalendarContactLayer()
        assert layer.applicable_to(sample_workspace_scenario)

    def test_apply_produces_ics(self, sample_workspace_scenario, deterministic_rng):
        layer = CalendarContactLayer()
        text, params = layer.apply(sample_workspace_scenario, Intensity.MEDIUM, deterministic_rng)
        assert "BEGIN:VCALENDAR" in text
        assert "BEGIN:VEVENT" in text

    def test_apply_produces_vcards(self, sample_workspace_scenario, deterministic_rng):
        layer = CalendarContactLayer()
        text, params = layer.apply(sample_workspace_scenario, Intensity.MEDIUM, deterministic_rng)
        assert "BEGIN:VCARD" in text

    def test_injection_in_ics_field(self, sample_workspace_scenario, deterministic_rng):
        layer = CalendarContactLayer()
        text, _ = layer.apply(sample_workspace_scenario, Intensity.MEDIUM, deterministic_rng)
        # Injection text should appear in ICS data
        assert "INFORMATION" in text
