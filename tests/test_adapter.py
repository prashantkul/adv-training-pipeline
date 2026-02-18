"""Tests for the AgentDojo adapter."""

from __future__ import annotations

import pytest

from noisy_agentdojo.adapter import (
    DummyPipeline,
    extract_all_scenarios,
    extract_benign_scenarios,
    get_environment_context,
)


class TestDummyPipeline:
    def test_has_valid_name(self):
        p = DummyPipeline()
        assert p.name == "gpt-4o-2024-05-13"

    def test_custom_name(self):
        p = DummyPipeline("claude-3-5-sonnet-20240620")
        assert p.name == "claude-3-5-sonnet-20240620"

    def test_query_raises(self):
        p = DummyPipeline()
        with pytest.raises(NotImplementedError):
            p.query("test", None)


class TestExtractAllScenarios:
    def test_extracts_workspace_scenarios(self):
        scenarios = extract_all_scenarios(
            suite_names=["workspace"],
            attack_names=["important_instructions"],
        )
        assert len(scenarios) > 0
        for s in scenarios:
            assert s.suite_name == "workspace"
            assert s.attack_name == "important_instructions"
            assert s.injection_vectors is not None
            assert len(s.injection_vectors) > 0
            assert s.available_tools
            assert s.ground_truth_calls
            assert s.injection_tool_calls is not None  # May be empty for some tasks
            assert not s.is_benign

    def test_environment_context_is_dict(self):
        scenarios = extract_all_scenarios(
            suite_names=["workspace"],
            attack_names=["important_instructions"],
        )
        for s in scenarios[:3]:
            assert isinstance(s.environment_context, dict)
            assert "inbox" in s.environment_context or "calendar" in s.environment_context

    def test_produces_expected_count(self):
        """All injectable pairs should produce scenarios."""
        scenarios = extract_all_scenarios(
            suite_names=["workspace"],
            attack_names=["important_instructions"],
        )
        # workspace: 40 user tasks * 14 injection tasks = 560 max
        # All user tasks are injectable for important_instructions in workspace
        assert len(scenarios) > 0
        assert len(scenarios) <= 40 * 14


class TestExtractBenignScenarios:
    def test_extracts_benign_workspace(self):
        scenarios = extract_benign_scenarios(suite_names=["workspace"])
        assert len(scenarios) > 0
        for s in scenarios:
            assert s.is_benign
            assert s.injection_vectors is None
            assert s.injection_task_id is None
            assert s.attack_name is None


class TestGetEnvironmentContext:
    def test_returns_serializable_dict(self):
        from agentdojo.task_suite.load_suites import get_suite

        suite = get_suite("v1.2.2", "workspace")
        ctx = get_environment_context(suite, {})
        assert isinstance(ctx, dict)
        # Should be JSON-serializable
        import json
        json.dumps(ctx)
