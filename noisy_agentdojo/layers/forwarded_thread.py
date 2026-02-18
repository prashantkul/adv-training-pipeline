"""Forwarded email thread noise layer."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import jinja2

from noisy_agentdojo.layers.base import NoiseLayer
from noisy_agentdojo.models import ExtractedScenario, Intensity

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"

_NAMES = [
    ("Sarah Chen", "sarah.chen@techcorp.com"),
    ("Michael Rodriguez", "m.rodriguez@initech.io"),
    ("Priya Patel", "priya.p@globalfin.com"),
    ("James Wilson", "jwilson@acmecorp.net"),
    ("Lisa Nakamura", "l.nakamura@designhub.co"),
    ("David Kim", "dkim@startup.inc"),
    ("Emma Thompson", "ethompson@lawfirm.legal"),
    ("Ahmed Hassan", "a.hassan@consulting.biz"),
    ("Rachel Green", "rgreen@mediagroup.com"),
    ("Tom Baker", "tbaker@engineering.dev"),
]

_SUBJECTS = [
    "Re: Quick update",
    "Re: Follow-up from yesterday",
    "Re: Action items",
    "Re: Fwd: Important notice",
    "Re: Regarding our conversation",
    "Re: Next steps",
    "Re: Meeting notes",
    "Re: Please review",
]

_FILLER_BODIES = [
    "Thanks for sending this over. I'll take a look and get back to you.",
    "Noted, thanks! Let me loop in the rest of the team.",
    "Sounds good. Can we schedule a quick sync to discuss?",
    "Got it. I've added this to our tracking sheet.",
    "Thanks for the heads up. I'll review by EOD.",
    "Forwarding this to you as discussed. See the original message below.",
    "FYI - see below. Let me know if you have questions.",
    "Looping you in on this thread. Key info is in the original message.",
    "Per our earlier discussion, forwarding the relevant details.",
    "Please see the forwarded message below for context.",
]

_SIGNATURE_STYLES = ["corporate", "casual", "legal", "minimal"]

_COMPANIES = ["TechCorp Inc.", "Initech Solutions", "GlobalFin Partners", "ACME Corp"]
_TITLES = ["VP of Engineering", "Senior Analyst", "Project Manager", "Director", "Associate"]
_PHONES = ["+1 (555) 234-5678", "+1 (555) 876-5432", "+44 20 7946 0958"]

_DATES = [
    "Mon, Oct 14, 2024 at 9:32 AM",
    "Tue, Oct 15, 2024 at 2:15 PM",
    "Wed, Oct 16, 2024 at 11:47 AM",
    "Thu, Oct 17, 2024 at 4:03 PM",
    "Fri, Oct 18, 2024 at 8:21 AM",
]

_DEPTH_BY_INTENSITY = {
    Intensity.LIGHT: (1, 2),
    Intensity.MEDIUM: (2, 3),
    Intensity.HEAVY: (3, 5),
}


class ForwardedThreadLayer(NoiseLayer):
    """Wraps content in nested forwarded email threads."""

    @property
    def name(self) -> str:
        return "forwarded_thread"

    def applicable_to(self, scenario: ExtractedScenario) -> bool:
        ctx = scenario.environment_context
        # Applicable to workspace/travel scenarios with inbox
        return "inbox" in ctx or scenario.suite_name in ("workspace", "travel")

    def apply(
        self,
        scenario: ExtractedScenario,
        intensity: Intensity,
        rng: random.Random,
        existing_noisy_text: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
            autoescape=False,
        )
        sig_template = env.get_template("email_signature.jinja2")
        fwd_template = env.get_template("email_forward.jinja2")

        min_depth, max_depth = _DEPTH_BY_INTENSITY[intensity]
        depth = rng.randint(min_depth, max_depth)

        content = existing_noisy_text or self._extract_email_content(scenario)

        forwards = []
        used_names = set()
        for i in range(depth):
            name, email = self._pick_unique(rng, _NAMES, used_names)
            used_names.add(name)
            to_name, to_email = self._pick_unique(rng, _NAMES, used_names)

            sig_style = rng.choice(_SIGNATURE_STYLES)
            signature = sig_template.render(
                style=sig_style,
                name=name,
                title=rng.choice(_TITLES),
                company=rng.choice(_COMPANIES),
                email=email,
                phone=rng.choice(_PHONES),
            )

            body = rng.choice(_FILLER_BODIES) if i < depth - 1 else content
            cc = rng.choice(_NAMES)[1] if rng.random() < 0.3 else None

            forwards.append({
                "from_name": name,
                "from_email": email,
                "date": rng.choice(_DATES),
                "subject": rng.choice(_SUBJECTS),
                "to_email": to_email,
                "cc": cc,
                "body": body,
                "signature": signature.strip(),
            })

        noisy_text = fwd_template.render(forwards=forwards)
        return noisy_text, {"depth": depth, "intensity": intensity.value}

    def _extract_email_content(self, scenario: ExtractedScenario) -> str:
        """Pull representative content from the environment context."""
        ctx = scenario.environment_context
        inbox = ctx.get("inbox", {})
        emails = inbox.get("emails", {})
        if emails:
            first_email = next(iter(emails.values()))
            if isinstance(first_email, dict):
                return first_email.get("body", scenario.user_task_prompt)
        return scenario.user_task_prompt

    @staticmethod
    def _pick_unique(
        rng: random.Random,
        choices: list[tuple[str, str]],
        exclude: set[str],
    ) -> tuple[str, str]:
        available = [c for c in choices if c[0] not in exclude]
        if not available:
            available = choices
        return rng.choice(available)
