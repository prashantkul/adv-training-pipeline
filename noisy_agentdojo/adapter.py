"""Adapter bridging AgentDojo API to ExtractedScenario models."""

from __future__ import annotations

import logging
from typing import Any

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
from agentdojo.attacks import load_attack
from agentdojo.attacks.attack_registry import ATTACKS
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.task_suite.task_suite import TaskSuite
from agentdojo.types import ChatMessage

from noisy_agentdojo.models import ExtractedScenario, ToolCall

logger = logging.getLogger(__name__)


class DummyPipeline(BasePipelineElement):
    """A pipeline stub that satisfies attack constructors requiring a named pipeline."""

    def __init__(self, model_name: str = "gpt-4o-2024-05-13") -> None:
        self.name = model_name

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Any = None,
        messages: list[ChatMessage] | tuple[()] = [],
        extra_args: dict[str, Any] | None = None,
    ) -> tuple[str, FunctionsRuntime, Any, list[ChatMessage], dict[str, Any]]:
        raise NotImplementedError("DummyPipeline is not meant to be executed.")


def _function_calls_to_tool_calls(calls: list[Any]) -> list[ToolCall]:
    """Convert AgentDojo FunctionCall objects to our ToolCall model."""
    result = []
    for fc in calls:
        result.append(ToolCall(function=fc.function, args=dict(fc.args)))
    return result


def _get_tool_names(suite: TaskSuite) -> list[str]:
    return [t.name for t in suite.tools]


def get_environment_context(
    suite: TaskSuite, injections: dict[str, str]
) -> dict[str, Any]:
    """Load environment with injections and serialize to dict."""
    env = suite.load_and_inject_default_environment(injections)
    return env.model_dump(mode="json")


def extract_scenario(
    suite: TaskSuite,
    user_task: BaseUserTask,
    injection_task: BaseInjectionTask,
    attack_name: str,
    pipeline: BasePipelineElement,
) -> ExtractedScenario | None:
    """Extract a single adversarial scenario from a (user_task, injection_task, attack) triple.

    Returns None if the user_task is not injectable for this attack.
    """
    attack = load_attack(attack_name, suite, pipeline)

    try:
        _candidates = attack.get_injection_candidates(user_task)
    except ValueError:
        return None

    injections = attack.attack(user_task, injection_task)
    env_context = get_environment_context(suite, injections)

    # Ground truth for the user task (benign path)
    default_env = suite.load_and_inject_default_environment(
        suite.get_injection_vector_defaults()
    )
    user_gt = _function_calls_to_tool_calls(user_task.ground_truth(default_env))

    # Ground truth for the injection task
    inj_gt = _function_calls_to_tool_calls(injection_task.ground_truth(default_env))

    return ExtractedScenario(
        suite_name=suite.name,
        user_task_id=user_task.ID,
        user_task_prompt=user_task.PROMPT,
        injection_task_id=injection_task.ID,
        injection_task_goal=injection_task.GOAL,
        attack_name=attack_name,
        injection_vectors=injections,
        environment_context=env_context,
        available_tools=_get_tool_names(suite),
        ground_truth_calls=user_gt,
        injection_tool_calls=inj_gt,
        is_benign=False,
    )


def extract_all_scenarios(
    suite_names: list[str] | None = None,
    attack_names: list[str] | None = None,
    model_name: str = "gpt-4o-2024-05-13",
    version: str = "v1.2.2",
) -> list[ExtractedScenario]:
    """Extract all adversarial scenarios across suites and attacks."""
    if suite_names is None:
        suite_names = ["workspace", "travel", "banking", "slack"]
    if attack_names is None:
        attack_names = ["important_instructions"]

    pipeline = DummyPipeline(model_name)
    scenarios: list[ExtractedScenario] = []

    for suite_name in suite_names:
        suite = get_suite(version, suite_name)
        for attack_name in attack_names:
            if attack_name not in ATTACKS:
                logger.warning("Attack %r not found in registry, skipping.", attack_name)
                continue
            for user_task in suite.user_tasks.values():
                for inj_task in suite.injection_tasks.values():
                    scenario = extract_scenario(
                        suite, user_task, inj_task, attack_name, pipeline
                    )
                    if scenario is not None:
                        scenarios.append(scenario)

    logger.info("Extracted %d adversarial scenarios.", len(scenarios))
    return scenarios


def extract_benign_scenarios(
    suite_names: list[str] | None = None,
    version: str = "v1.2.2",
) -> list[ExtractedScenario]:
    """Extract benign (no injection) scenarios for FPR calibration."""
    if suite_names is None:
        suite_names = ["workspace", "travel", "banking", "slack"]

    scenarios: list[ExtractedScenario] = []

    for suite_name in suite_names:
        suite = get_suite(version, suite_name)
        env_context = get_environment_context(suite, {})
        default_env = suite.load_and_inject_default_environment({})

        for user_task in suite.user_tasks.values():
            gt = _function_calls_to_tool_calls(user_task.ground_truth(default_env))
            scenarios.append(
                ExtractedScenario(
                    suite_name=suite_name,
                    user_task_id=user_task.ID,
                    user_task_prompt=user_task.PROMPT,
                    environment_context=env_context,
                    available_tools=_get_tool_names(suite),
                    ground_truth_calls=gt,
                    is_benign=True,
                )
            )

    logger.info("Extracted %d benign scenarios.", len(scenarios))
    return scenarios
