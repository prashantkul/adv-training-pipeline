"""Calendar (ICS) and contact (vCard) noise layer."""

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

_CONTACT_NAMES = [
    ("John", "Smith"), ("Maria", "Garcia"), ("Wei", "Zhang"),
    ("Aisha", "Mohammed"), ("Carlos", "Lopez"), ("Sophie", "Dubois"),
    ("Raj", "Sharma"), ("Anna", "Kowalski"),
]

_ORGS = ["TechCorp", "FinServ Inc.", "GlobalMedia", "StartupXYZ"]
_TITLES = ["Engineer", "Manager", "Director", "Analyst", "Consultant"]

_ICS_INJECTION_FIELDS = ["DESCRIPTION", "LOCATION", "COMMENT"]
_VCARD_INJECTION_FIELDS = ["NOTE"]

_EXTRA_EVENTS = {
    Intensity.LIGHT: (1, 2),
    Intensity.MEDIUM: (2, 4),
    Intensity.HEAVY: (3, 6),
}

_EXTRA_CONTACTS = {
    Intensity.LIGHT: (0, 1),
    Intensity.MEDIUM: (1, 2),
    Intensity.HEAVY: (2, 4),
}


class CalendarContactLayer(NoiseLayer):
    """Generates ICS calendar and vCard data with injections embedded in metadata fields."""

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

        # Generate ICS calendar
        cal_str = self._generate_ics(injection_text, intensity, rng)

        # Generate vCards
        vcards_str = self._generate_vcards(injection_text, intensity, rng)

        # Combine
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
        min_events, max_events = _EXTRA_EVENTS[intensity]
        num_events = rng.randint(min_events, max_events)

        # Main event with injection
        main_event = icalendar.Event()
        main_event.add("summary", rng.choice(_EVENT_TITLES))
        main_event.add("dtstart", base_dt)
        main_event.add("dtend", base_dt + timedelta(hours=1))
        main_event.add("location", rng.choice(_LOCATIONS))

        # Place injection in a random ICS field
        inj_field = rng.choice(_ICS_INJECTION_FIELDS)
        if inj_field == "DESCRIPTION":
            main_event.add("description", injection_text)
        elif inj_field == "LOCATION":
            main_event.add("location", injection_text)
        elif inj_field == "COMMENT":
            main_event.add("comment", injection_text)

        cal.add_component(main_event)

        # Add filler events
        for i in range(num_events):
            event = icalendar.Event()
            event.add("summary", rng.choice(_EVENT_TITLES))
            dt = base_dt + timedelta(hours=rng.randint(1, 72))
            event.add("dtstart", dt)
            event.add("dtend", dt + timedelta(minutes=rng.choice([30, 60, 90])))
            event.add("location", rng.choice(_LOCATIONS))
            event.add("description", "Regular meeting - no action items.")
            cal.add_component(event)

        return cal.to_ical().decode("utf-8")

    def _generate_vcards(
        self,
        injection_text: str,
        intensity: Intensity,
        rng: random.Random,
    ) -> str:
        min_contacts, max_contacts = _EXTRA_CONTACTS[intensity]
        num_contacts = rng.randint(min_contacts, max_contacts)
        if num_contacts == 0 and not injection_text:
            return ""

        vcards = []

        # Main contact with injection
        if injection_text:
            first, last = rng.choice(_CONTACT_NAMES)
            card = vobject.vCard()
            card.add("n").value = vobject.vcard.Name(family=last, given=first)
            card.add("fn").value = f"{first} {last}"
            card.add("email").value = f"{first.lower()}.{last.lower()}@example.com"
            card.add("org").value = [rng.choice(_ORGS)]
            card.add("title").value = rng.choice(_TITLES)
            note = card.add("note")
            note.value = injection_text
            vcards.append(card.serialize())

        # Filler contacts
        used_names = set()
        for _ in range(num_contacts):
            first, last = self._pick_unique_name(rng, used_names)
            card = vobject.vCard()
            card.add("n").value = vobject.vcard.Name(family=last, given=first)
            card.add("fn").value = f"{first} {last}"
            card.add("email").value = f"{first.lower()}.{last.lower()}@example.com"
            card.add("org").value = [rng.choice(_ORGS)]
            card.add("tel").value = f"+1555{rng.randint(1000000, 9999999)}"
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
