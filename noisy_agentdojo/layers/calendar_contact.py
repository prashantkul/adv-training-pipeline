"""Calendar (ICS) and contact (vCard) noise layer.

Design: The injection is placed in a readable calendar event DESCRIPTION field.
Filler events and contacts act as camouflage clutter — they make the data look
like a real calendar export with many entries, where the injection naturally
blends into one event's description. The injection remains fully readable.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any

import icalendar
import vobject

from noisy_agentdojo.layers.base import NoiseLayer
from noisy_agentdojo.models import ExtractedScenario, Intensity

_LOCATIONS = [
    "Conference Room A", "Building 3, Floor 2", "Main Office",
    "Zoom Meeting", "Google Meet", "Teams Call",
    "123 Business Ave, Suite 400", "Remote",
]

_EVENT_TITLES = [
    "Team Standup", "Project Review", "Client Call",
    "1:1 with Manager", "Sprint Planning", "Lunch Meeting",
    "Training Session", "All Hands", "Design Review",
]

_FILLER_DESCRIPTIONS = [
    "Regular team sync. No preparation needed.",
    "Review sprint progress and blockers.",
    "Discuss Q4 roadmap and milestones.",
    "Monthly check-in with stakeholders.",
    "Agenda: budget review, team updates, open items.",
    "Bring your laptop for the demo.",
]

_CONTACT_NAMES = [
    ("John", "Smith"), ("Maria", "Garcia"), ("Wei", "Zhang"),
    ("Aisha", "Mohammed"), ("Carlos", "Lopez"), ("Sophie", "Dubois"),
    ("Raj", "Sharma"), ("Anna", "Kowalski"),
]

_ORGS = ["TechCorp", "FinServ Inc.", "GlobalMedia", "StartupXYZ"]
_TITLES = ["Engineer", "Manager", "Director", "Analyst", "Consultant"]

_FILLER_EVENTS = {
    Intensity.LIGHT: (1, 2),
    Intensity.MEDIUM: (2, 4),
    Intensity.HEAVY: (4, 7),
}

_FILLER_CONTACTS = {
    Intensity.LIGHT: (0, 1),
    Intensity.MEDIUM: (1, 2),
    Intensity.HEAVY: (2, 4),
}


class CalendarContactLayer(NoiseLayer):
    """Embeds injection in a calendar event description within a realistic
    calendar export. Filler events and contacts act as camouflage."""

    @property
    def name(self) -> str:
        return "calendar_contact"

    def applicable_to(self, scenario: ExtractedScenario) -> bool:
        ctx = scenario.environment_context
        return "calendar" in ctx or scenario.suite_name == "workspace"

    def apply(
        self,
        scenario: ExtractedScenario,
        intensity: Intensity,
        rng: random.Random,
        existing_noisy_text: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        content = existing_noisy_text or ""
        injection_text = self._get_injection_text(scenario)

        cal_str = self._generate_ics(injection_text, intensity, rng)
        vcards_str = self._generate_vcards(intensity, rng)

        parts = []
        if content:
            parts.append(content)
        parts.append("--- Calendar Data ---")
        parts.append(cal_str)
        if vcards_str:
            parts.append("--- Contact Data ---")
            parts.append(vcards_str)

        noisy_text = "\n\n".join(parts)
        return noisy_text, {"intensity": intensity.value, "has_vcards": bool(vcards_str)}

    def _get_injection_text(self, scenario: ExtractedScenario) -> str:
        if not scenario.injection_vectors:
            return ""
        return " ".join(scenario.injection_vectors.values())

    def _generate_ics(
        self,
        injection_text: str,
        intensity: Intensity,
        rng: random.Random,
    ) -> str:
        cal = icalendar.Calendar()
        cal.add("prodid", "-//Noisy AgentDojo//EN")
        cal.add("version", "2.0")

        base_dt = datetime(2024, 10, 21, 9, 0, 0)
        min_events, max_events = _FILLER_EVENTS[intensity]
        num_filler = rng.randint(min_events, max_events)

        # Some filler events BEFORE the injection event
        filler_before = num_filler // 2
        filler_after = num_filler - filler_before

        for i in range(filler_before):
            cal.add_component(self._make_filler_event(base_dt, i, rng))

        # The injection event — injection in DESCRIPTION (visible, readable)
        inj_event = icalendar.Event()
        inj_event.add("summary", rng.choice(_EVENT_TITLES))
        inj_dt = base_dt + timedelta(hours=rng.randint(1, 24))
        inj_event.add("dtstart", inj_dt)
        inj_event.add("dtend", inj_dt + timedelta(hours=1))
        inj_event.add("location", rng.choice(_LOCATIONS))
        inj_event.add("description", injection_text)
        cal.add_component(inj_event)

        for i in range(filler_after):
            offset = filler_before + 1 + i
            cal.add_component(self._make_filler_event(base_dt, offset, rng))

        return cal.to_ical().decode("utf-8")

    def _make_filler_event(
        self, base_dt: datetime, index: int, rng: random.Random
    ) -> icalendar.Event:
        event = icalendar.Event()
        event.add("summary", rng.choice(_EVENT_TITLES))
        dt = base_dt + timedelta(hours=rng.randint(1, 72) + index * 2)
        event.add("dtstart", dt)
        event.add("dtend", dt + timedelta(minutes=rng.choice([30, 60, 90])))
        event.add("location", rng.choice(_LOCATIONS))
        event.add("description", rng.choice(_FILLER_DESCRIPTIONS))
        return event

    def _generate_vcards(
        self,
        intensity: Intensity,
        rng: random.Random,
    ) -> str:
        """Generate filler contacts as camouflage clutter."""
        min_contacts, max_contacts = _FILLER_CONTACTS[intensity]
        num_contacts = rng.randint(min_contacts, max_contacts)
        if num_contacts == 0:
            return ""

        vcards = []
        used_names: set[str] = set()
        for _ in range(num_contacts):
            first, last = self._pick_unique_name(rng, used_names)
            card = vobject.vCard()
            card.add("n").value = vobject.vcard.Name(family=last, given=first)
            card.add("fn").value = f"{first} {last}"
            card.add("email").value = f"{first.lower()}.{last.lower()}@example.com"
            card.add("org").value = [rng.choice(_ORGS)]
            card.add("tel").value = f"+1555{rng.randint(1000000, 9999999)}"
            card.add("title").value = rng.choice(_TITLES)
            vcards.append(card.serialize())

        return "\n".join(vcards)

    @staticmethod
    def _pick_unique_name(
        rng: random.Random, used: set[str]
    ) -> tuple[str, str]:
        available = [n for n in _CONTACT_NAMES if n[0] not in used]
        if not available:
            available = list(_CONTACT_NAMES)
        choice = rng.choice(available)
        used.add(choice[0])
        return choice
