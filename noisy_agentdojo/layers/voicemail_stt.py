"""Voicemail / speech-to-text transcript noise layer.

Design: The injection text appears CLEANLY within a degraded voicemail
transcript. Surrounding context is degraded with STT artifacts (filler words,
[inaudible], homophones) but the injection itself is preserved verbatim.
This makes the injection look like a clear segment of dictation within a
noisy transcript — realistic and still fully readable.
"""

from __future__ import annotations

import random
import re
from typing import Any

from noisy_agentdojo.layers.base import NoiseLayer
from noisy_agentdojo.models import ExtractedScenario, Intensity

_FILLER_WORDS = [
    "um", "uh", "like", "you know", "so", "basically",
    "I mean", "right", "actually", "well",
]

_HOMOPHONES = {
    "their": "there",
    "there": "their",
    "they're": "there",
    "your": "you're",
    "you're": "your",
    "its": "it's",
    "it's": "its",
    "to": "too",
    "too": "to",
    "write": "right",
    "right": "write",
    "hear": "here",
    "here": "hear",
    "meet": "meat",
    "know": "no",
    "no": "know",
    "buy": "by",
    "by": "buy",
    "whether": "weather",
    "weather": "whether",
    "accept": "except",
    "affect": "effect",
    "than": "then",
    "then": "than",
    "sent": "cent",
    "sea": "see",
    "see": "sea",
    "week": "weak",
    "break": "brake",
    "piece": "peace",
    "mail": "male",
    "wait": "weight",
}

_VOICEMAIL_INTROS = [
    "[Voicemail transcript - {duration}]\n\n",
    "--- Voicemail from {caller} ({time}) ---\nTranscript (auto-generated):\n\n",
    "[Automated transcription - {caller} - {time}]\n\n",
    "Voicemail received {time} from {caller}\n[Auto-transcribed]\n\n",
]

_VOICEMAIL_OUTROS = [
    "\n\n[End of voicemail]",
    "\n\n[Message ends]",
    "\n\n--- End of transcription ---",
    "\n\n[Transcription confidence: {confidence}%]",
]

_CALLERS = [
    "Unknown Number", "+1 (555) 867-5309", "Front Desk",
    "John", "Maria", "Office", "+44 7911 123456",
]

_DEGRADATION_RATE = {
    Intensity.LIGHT: 0.08,
    Intensity.MEDIUM: 0.18,
    Intensity.HEAVY: 0.35,
}

_FILLER_RATE = {
    Intensity.LIGHT: 0.05,
    Intensity.MEDIUM: 0.12,
    Intensity.HEAVY: 0.25,
}

_INAUDIBLE_RATE = {
    Intensity.LIGHT: 0.02,
    Intensity.MEDIUM: 0.06,
    Intensity.HEAVY: 0.12,
}


class VoicemailSTTLayer(NoiseLayer):
    """Simulates a voicemail transcript where surrounding context is degraded
    but the injection text appears as a clear, readable segment."""

    @property
    def name(self) -> str:
        return "voicemail_stt"

    def applicable_to(self, scenario: ExtractedScenario) -> bool:
        return True

    def apply(
        self,
        scenario: ExtractedScenario,
        intensity: Intensity,
        rng: random.Random,
        existing_noisy_text: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        context_text = existing_noisy_text or self._get_context(scenario)
        injection_text = self._get_injection_text(scenario)

        # Degrade the context (non-injection) parts
        degraded_context = self._degrade_chunk(context_text, intensity, rng)

        # Build transcript: degraded context THEN clean injection
        if injection_text:
            transcript = degraded_context + "\n\n" + injection_text
        else:
            transcript = degraded_context

        # Wrap in voicemail framing
        caller = rng.choice(_CALLERS)
        time = f"{rng.randint(8, 18)}:{rng.randint(0, 59):02d}"
        duration = f"{rng.randint(0, 3)}:{rng.randint(10, 59):02d}"
        confidence = rng.randint(55, 92)

        intro = rng.choice(_VOICEMAIL_INTROS).format(
            caller=caller, time=time, duration=duration
        )
        outro = rng.choice(_VOICEMAIL_OUTROS).format(confidence=confidence)

        noisy_text = intro + transcript + outro
        return noisy_text, {"intensity": intensity.value, "caller": caller}

    def _get_context(self, scenario: ExtractedScenario) -> str:
        """Pull non-injection context from the environment."""
        ctx = scenario.environment_context
        parts = []
        inbox = ctx.get("inbox", {})
        for email in list(inbox.get("emails", {}).values())[:3]:
            if isinstance(email, dict) and "body" in email:
                parts.append(email["body"])
        if not parts:
            parts.append(scenario.user_task_prompt)
        return "\n".join(parts)

    def _get_injection_text(self, scenario: ExtractedScenario) -> str:
        """Get the injection payload — preserved verbatim."""
        if not scenario.injection_vectors:
            return ""
        return "\n".join(scenario.injection_vectors.values())

    def _degrade_chunk(
        self, text: str, intensity: Intensity, rng: random.Random
    ) -> str:
        """Apply STT-like degradation to non-injection text."""
        degradation_rate = _DEGRADATION_RATE[intensity]
        filler_rate = _FILLER_RATE[intensity]
        inaudible_rate = _INAUDIBLE_RATE[intensity]

        words = text.split()
        result: list[str] = []

        for word in words:
            if rng.random() < filler_rate:
                result.append(rng.choice(_FILLER_WORDS))

            if rng.random() < inaudible_rate:
                result.append("[inaudible]")
                continue

            clean_word = re.sub(r"[^\w']", "", word.lower())
            if rng.random() < degradation_rate and clean_word in _HOMOPHONES:
                replacement = _HOMOPHONES[clean_word]
                if word[0].isupper():
                    replacement = replacement.capitalize()
                result.append(replacement)
            else:
                result.append(word)

        text = " ".join(result)

        if intensity in (Intensity.MEDIUM, Intensity.HEAVY):
            chars = list(text)
            for i, c in enumerate(chars):
                if c in ".,;:!?" and rng.random() < degradation_rate:
                    chars[i] = ""
            text = "".join(chars)

        return text
