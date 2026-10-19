import asyncio
import json
import logging
import os
from collections import Counter
from pathlib import Path

import typer

from evaluation.dataset import GOLDEN, load_golden, validate_golden
from kb_builder import tracing
from kb_builder.context import SiteContext

app = typer.Typer(help="Evaluate the customer-support assistant with RAGAS.")

EVALS_DIR = Path("evals")


@app.callback()
def main() -> None:
    """Assistant evaluation."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in ("httpx", "openai", "httpcore", "instructor", "ragas"):
        logging.getLogger(name).setLevel(logging.WARNING)
    # RAGAS sends anonymous usage analytics unless told not to.
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")


@app.command()
def run(
    site: str = typer.Argument(..., help="Site name, e.g. art-mat.com.pl"),
    label: str = typer.Option("run", help="Short name of this run, e.g. baseline or top-k-8."),
    limit: int | None = typer.Option(None, help="Only the first N samples (for a cheap smoke test)."),
    ids: str | None = typer.Option(None, help="Comma-separated sample ids to run."),
    concurrency: int = typer.Option(4, help="Parallel judge calls."),
) -> None:
    """Answer the golden questions with the assistant and grade them with RAGAS."""
    from evaluation.runner import EvalConfig, answer_all, push_scores, save_run, score_all
    from support_assistant.assistant import AssistantSettings

    samples = load_golden(EVALS_DIR / site / GOLDEN)
    if ids:
        wanted = set(ids.split(","))
        samples = [s for s in samples if s.id in wanted]
    if limit:
        samples = samples[:limit]

    settings = AssistantSettings(site=site)
    config = EvalConfig(label=label, concurrency=concurrency)
    typer.echo(f"{len(samples)} samples, assistant on {site}, judge {config.judge_model}")
    try:
        results = answer_all(samples, settings)
        asyncio.run(score_all(results, config))
        out_dir = save_run(EVALS_DIR / site / "results", results, config, settings)
        push_scores(results)
    finally:
        tracing.flush()

    summary = json.loads((out_dir / "summary.json").read_text())
    typer.echo(f"\n{out_dir}")
    typer.echo("overall: " + ", ".join(f"{k}={v}" for k, v in summary["overall"].items()))
    for category, values in summary["by_category"].items():
        typer.echo(f"  {category:<13} " + ", ".join(f"{k}={v}" for k, v in values.items()))
    if summary["errors"]:
        typer.secho(f"{summary['errors']} metric errors, see samples.jsonl", fg=typer.colors.YELLOW)


@app.command()
def validate(site: str = typer.Argument(..., help="Site name, e.g. art-mat.com.pl")) -> None:
    """Check the golden dataset: unique ids, quotes verbatim in the site text."""
    path = EVALS_DIR / site / GOLDEN
    samples = load_golden(path)
    counts = Counter(s.category for s in samples)
    typer.echo(f"{path}: {len(samples)} samples  " + ", ".join(f"{k}: {n}" for k, n in sorted(counts.items())))
    problems = validate_golden(samples, SiteContext(site=site))
    for problem in problems:
        typer.secho(f"  {problem}", fg=typer.colors.RED)
    if problems:
        raise typer.Exit(1)
    typer.secho("  all quotes found in the site text", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
