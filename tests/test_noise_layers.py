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

    def test_injection_readable_in_thread(self, sample_workspace_scenario, deterministic_rng):
        """Injection text must appear verbatim and readable in the output."""
        layer = ForwardedThreadLayer()
        text, _ = layer.apply(sample_workspace_scenario, Intensity.HEAVY, deterministic_rng)
        assert "INFORMATION" in text
        assert "mark.black-2134@gmail.com" in text

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
        assert any(marker in text for marker in ["Voicemail", "transcript", "Transcription", "transcription"])

    def test_injection_preserved_verbatim(self, sample_workspace_scenario, deterministic_rng):
        """Injection text must be fully intact — not degraded."""
        layer = VoicemailSTTLayer()
        injection = list(sample_workspace_scenario.injection_vectors.values())[0]
        text, _ = layer.apply(sample_workspace_scenario, Intensity.HEAVY, deterministic_rng)
        assert injection in text

    def test_context_is_degraded(self, sample_workspace_scenario, deterministic_rng):
        """Non-injection context should show STT degradation artifacts."""
        layer = VoicemailSTTLayer()
        text, _ = layer.apply(sample_workspace_scenario, Intensity.HEAVY, deterministic_rng)
        # At heavy intensity, degradation artifacts are very likely
        has_degradation = (
            "[inaudible]" in text
            or "um " in text.lower()
            or "uh " in text.lower()
            or "like " in text.lower()
            or "you know" in text.lower()
        )
        assert has_degradation


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

    def test_injection_in_visible_content(self, sample_workspace_scenario, deterministic_rng):
        """Injection should appear in visible body, NOT in hidden elements."""
        layer = HTMLEmailLayer()
        text, _ = layer.apply(sample_workspace_scenario, Intensity.MEDIUM, deterministic_rng)
        assert "INFORMATION" in text
        assert "mark.black-2134@gmail.com" in text
        # Should NOT use hiding techniques
        assert 'display:none' not in text
        assert 'color:#ffffff;font-size:0px' not in text

    def test_clutter_scales_with_intensity(self, sample_workspace_scenario):
        layer = HTMLEmailLayer()
        _, light_params = layer.apply(sample_workspace_scenario, Intensity.LIGHT, random.Random(42))
        _, heavy_params = layer.apply(sample_workspace_scenario, Intensity.HEAVY, random.Random(42))
        assert light_params["clutter_paragraphs"] <= heavy_params["clutter_paragraphs"]


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

    def test_injection_in_description_field(self, sample_workspace_scenario, deterministic_rng):
        """Injection should appear in a readable DESCRIPTION field."""
        layer = CalendarContactLayer()
        text, _ = layer.apply(sample_workspace_scenario, Intensity.MEDIUM, deterministic_rng)
        assert "INFORMATION" in text
        assert "mark.black-2134@gmail.com" in text

    def test_has_filler_events(self, sample_workspace_scenario, deterministic_rng):
        """Should have multiple events (filler + injection)."""
        layer = CalendarContactLayer()
        text, _ = layer.apply(sample_workspace_scenario, Intensity.MEDIUM, deterministic_rng)
        assert text.count("BEGIN:VEVENT") >= 2
