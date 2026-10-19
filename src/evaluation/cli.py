import logging
import os
from collections import Counter
from pathlib import Path

import typer

from evaluation.golden_dataset import GOLDEN_DATASET, load_golden_dataset, validate_golden_dataset
from evaluation.models import EvaluationConfig
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
    # Imported here: RAGAS (with LangChain) takes ~5 s to import and `validate` does not need it.
    from evaluation.evaluation_service import EvaluationService

    config = EvaluationConfig(
        label=label,
        evals_root=EVALS_DIR,
        sample_ids=ids.split(",") if ids else [],
        limit=limit,
        concurrency=concurrency,
    )
    try:
        result = EvaluationService(config).run(SiteContext(site=site))
    finally:
        tracing.flush()

    summary = result.summary
    typer.echo(f"\n{result.out_dir}")
    typer.echo("overall: " + ", ".join(f"{k}={v}" for k, v in summary["overall"].items()))
    for category, values in summary["by_category"].items():
        typer.echo(f"  {category:<13} " + ", ".join(f"{k}={v}" for k, v in values.items()))
    if summary["errors"]:
        typer.secho(f"{summary['errors']} metric errors, see samples.jsonl", fg=typer.colors.YELLOW)


@app.command()
def validate(site: str = typer.Argument(..., help="Site name, e.g. art-mat.com.pl")) -> None:
    """Check the golden dataset: unique ids, quotes verbatim in the site text."""
    path = EVALS_DIR / site / GOLDEN_DATASET
    samples = load_golden_dataset(path)
    counts = Counter(s.category for s in samples)
    typer.echo(f"{path}: {len(samples)} samples  " + ", ".join(f"{k}: {n}" for k, n in sorted(counts.items())))
    problems = validate_golden_dataset(samples, SiteContext(site=site))
    for problem in problems:
        typer.secho(f"  {problem}", fg=typer.colors.RED)
    if problems:
        raise typer.Exit(1)
    typer.secho("  all quotes found in the site text", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
