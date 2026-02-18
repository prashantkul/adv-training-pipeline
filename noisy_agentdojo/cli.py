"""CLI interface for the noisy_agentdojo pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from noisy_agentdojo.config import load_config
from noisy_agentdojo.utils import read_jsonl, setup_logging, summarize_examples, write_jsonl

console = Console()
logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def cli(verbose: bool) -> None:
    """Noisy AgentDojo: noise-augmented adversarial training data pipeline."""
    setup_logging(verbose)


@cli.command()
@click.option("--suites", "-s", multiple=True, default=["workspace"], help="Suite names.")
@click.option("--attacks", "-a", multiple=True, default=["important_instructions"], help="Attack names.")
@click.option("--count", "-n", type=int, default=None, help="Max scenarios to generate.")
@click.option("--output", "-o", type=click.Path(), default="data/output.jsonl", help="Output JSONL path.")
@click.option("--config", "-c", type=click.Path(exists=True), default=None, help="Config YAML path.")
@click.option("--validate/--no-validate", default=False, help="Run LLM validation.")
@click.option("--include-llm/--no-llm", default=False, help="Include LLM-powered noise layers.")
@click.option("--seed", type=int, default=None, help="Random seed.")
def generate(
    suites: tuple[str, ...],
    attacks: tuple[str, ...],
    count: int | None,
    output: str,
    config: str | None,
    validate: bool,
    include_llm: bool,
    seed: int | None,
) -> None:
    """Generate noisy adversarial training examples."""
    from noisy_agentdojo.adapter import extract_all_scenarios
    from noisy_agentdojo.noise_pipeline import NoisePipeline

    cfg = load_config(config)
    if seed is not None:
        cfg.seed = seed

    console.print(f"[bold]Extracting scenarios from {list(suites)}...[/bold]")
    scenarios = extract_all_scenarios(
        suite_names=list(suites),
        attack_names=list(attacks),
        model_name=cfg.model_name,
        version=cfg.version,
    )
    console.print(f"  Extracted {len(scenarios)} scenarios.")

    if count is not None:
        scenarios = scenarios[:count]
        console.print(f"  Limiting to {count} scenarios.")

    console.print("[bold]Applying noise layers...[/bold]")
    pipeline = NoisePipeline(config=cfg, include_llm_layers=include_llm)
    noisy_scenarios = pipeline.apply_batch(scenarios)

    if validate:
        console.print("[bold]Validating with LLM...[/bold]")
        from noisy_agentdojo.validation import Validator

        validator = Validator(model=cfg.gemini_model)
        examples = [validator.validate(ns) for ns in noisy_scenarios]
    else:
        # Build examples without validation
        examples = []
        for ns in noisy_scenarios:
            s = ns.scenario
            from noisy_agentdojo.models import ValidatedExample

            examples.append(
                ValidatedExample(
                    suite_name=s.suite_name,
                    user_task_id=s.user_task_id,
                    injection_task_id=s.injection_task_id,
                    attack_name=s.attack_name,
                    user_prompt=s.user_task_prompt,
                    noisy_context=ns.noisy_text,
                    available_tools=s.available_tools,
                    tool_descriptions=s.tool_descriptions,
                    ground_truth_calls=s.ground_truth_calls,
                    injection_tool_calls=s.injection_tool_calls,
                    noise_layers=ns.noise_layers_applied,
                    is_benign=s.is_benign,
                )
            )

    write_jsonl(examples, output)
    console.print(f"[green]Wrote {len(examples)} examples to {output}[/green]")

    stats = summarize_examples(examples)
    _print_summary(stats)


@cli.command("generate-benign")
@click.option("--suites", "-s", multiple=True, default=["workspace"], help="Suite names.")
@click.option("--output", "-o", type=click.Path(), default="data/benign.jsonl", help="Output JSONL path.")
@click.option("--config", "-c", type=click.Path(exists=True), default=None, help="Config YAML path.")
@click.option("--include-llm/--no-llm", default=False, help="Include LLM-powered noise layers.")
def generate_benign(
    suites: tuple[str, ...],
    output: str,
    config: str | None,
    include_llm: bool,
) -> None:
    """Generate benign (no injection) training examples for FPR calibration."""
    from noisy_agentdojo.adapter import extract_benign_scenarios
    from noisy_agentdojo.noise_pipeline import NoisePipeline
    from noisy_agentdojo.models import ValidatedExample

    cfg = load_config(config)

    console.print(f"[bold]Extracting benign scenarios from {list(suites)}...[/bold]")
    scenarios = extract_benign_scenarios(suite_names=list(suites), version=cfg.version)
    console.print(f"  Extracted {len(scenarios)} benign scenarios.")

    console.print("[bold]Applying noise layers...[/bold]")
    pipeline = NoisePipeline(config=cfg, include_llm_layers=include_llm)
    noisy_scenarios = pipeline.apply_batch(scenarios)

    examples = []
    for ns in noisy_scenarios:
        s = ns.scenario
        examples.append(
            ValidatedExample(
                suite_name=s.suite_name,
                user_task_id=s.user_task_id,
                user_prompt=s.user_task_prompt,
                noisy_context=ns.noisy_text,
                available_tools=s.available_tools,
                tool_descriptions=s.tool_descriptions,
                ground_truth_calls=s.ground_truth_calls,
                noise_layers=ns.noise_layers_applied,
                is_benign=True,
            )
        )

    write_jsonl(examples, output)
    console.print(f"[green]Wrote {len(examples)} benign examples to {output}[/green]")


@cli.command()
@click.option("--input", "-i", "input_path", type=click.Path(exists=True), required=True, help="Input JSONL.")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output JSONL with validation results.")
@click.option("--model", "-m", default="gemini-2.0-flash", help="Gemini model for validation.")
def validate(input_path: str, output: str | None, model: str) -> None:
    """Validate existing examples against an LLM."""
    from noisy_agentdojo.validation import Validator

    examples = read_jsonl(input_path)
    console.print(f"[bold]Loaded {len(examples)} examples.[/bold]")

    # Re-validate by constructing NoisyScenarios
    # For now, just show stats from existing validation
    stats = summarize_examples(examples)
    _print_summary(stats)

    if output:
        write_jsonl(examples, output)
        console.print(f"[green]Wrote to {output}[/green]")


@cli.command()
@click.option("--input", "-i", "input_path", type=click.Path(exists=True), required=True, help="Input JSONL.")
def stats(input_path: str) -> None:
    """Show coverage statistics for a generated batch."""
    examples = read_jsonl(input_path)
    stats = summarize_examples(examples)
    _print_summary(stats)


@cli.command()
@click.option("--model", "-m", default="gemini-2.0-flash", help="Gemini model.")
@click.option("--suites", "-s", multiple=True, default=["workspace"], help="Suite names.")
@click.option("--attack", "-a", default="important_instructions", help="Attack name.")
@click.option("--count", "-n", type=int, default=10, help="Number of scenarios.")
@click.option("--seed", type=int, default=42, help="Random seed.")
def compare(
    model: str,
    suites: tuple[str, ...],
    attack: str,
    count: int,
    seed: int,
) -> None:
    """Compare clean vs noisy attack success rates (ASR)."""
    from noisy_agentdojo.adapter import extract_all_scenarios
    from noisy_agentdojo.noise_pipeline import NoisePipeline
    from noisy_agentdojo.validation import Validator
    from noisy_agentdojo.models import NoiseConfig, NoisyScenario

    console.print("[bold]Clean vs Noisy ASR Comparison[/bold]")
    console.print(f"Model: {model}, Suites: {list(suites)}, Attack: {attack}\n")

    scenarios = extract_all_scenarios(
        suite_names=list(suites),
        attack_names=[attack],
    )[:count]

    validator = Validator(model=model)

    # Clean ASR: validate scenarios without noise
    console.print("[bold]Testing clean scenarios...[/bold]")
    clean_results = []
    for s in scenarios:
        clean_noisy = NoisyScenario(
            scenario=s,
            noisy_environment=s.environment_context,
            noise_layers_applied=[],
            noise_layers_text="",
        )
        result = validator.validate(clean_noisy)
        clean_results.append(result)

    # Noisy ASR
    console.print("[bold]Testing noisy scenarios...[/bold]")
    cfg = NoiseConfig(seed=seed)
    pipeline = NoisePipeline(config=cfg)
    noisy_scenarios = pipeline.apply_batch(scenarios)
    noisy_results = [validator.validate(ns) for ns in noisy_scenarios]

    # Summary
    clean_triggered = sum(1 for r in clean_results if r.validation and r.validation.triggered)
    noisy_triggered = sum(1 for r in noisy_results if r.validation and r.validation.triggered)

    table = Table(title="ASR Comparison")
    table.add_column("Condition", style="cyan")
    table.add_column("Triggered", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("ASR", justify="right", style="bold")

    n = len(scenarios)
    table.add_row("Clean", str(clean_triggered), str(n), f"{clean_triggered / n:.1%}" if n else "N/A")
    table.add_row("Noisy", str(noisy_triggered), str(n), f"{noisy_triggered / n:.1%}" if n else "N/A")

    console.print(table)


def _print_summary(stats: dict) -> None:
    """Print a Rich summary table."""
    table = Table(title="Batch Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Total examples", str(stats.get("total", 0)))
    table.add_row("Adversarial", str(stats.get("adversarial", 0)))
    table.add_row("Benign", str(stats.get("benign", 0)))
    table.add_row("Validated", str(stats.get("validated", 0)))
    table.add_row("Triggered", str(stats.get("triggered", 0)))
    rate = stats.get("trigger_rate", 0)
    table.add_row("Trigger rate", f"{rate:.1%}")

    console.print(table)

    # Suite breakdown
    if stats.get("suites"):
        st = Table(title="By Suite")
        st.add_column("Suite", style="cyan")
        st.add_column("Count", justify="right")
        for suite, cnt in sorted(stats["suites"].items()):
            st.add_row(suite, str(cnt))
        console.print(st)

    # Layer breakdown
    if stats.get("layers"):
        lt = Table(title="By Noise Layer")
        lt.add_column("Layer", style="cyan")
        lt.add_column("Applications", justify="right")
        for layer, cnt in sorted(stats["layers"].items()):
            lt.add_row(layer, str(cnt))
        console.print(lt)

    # Attack breakdown
    if stats.get("attacks"):
        at = Table(title="By Attack")
        at.add_column("Attack", style="cyan")
        at.add_column("Count", justify="right")
        for attack, cnt in sorted(stats["attacks"].items()):
            at.add_row(attack, str(cnt))
        console.print(at)
