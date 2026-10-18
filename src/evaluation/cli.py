from collections import Counter
from pathlib import Path

import typer

from evaluation.dataset import GOLDEN, load_golden, validate_golden
from kb_builder.context import SiteContext

app = typer.Typer(help="Evaluate the customer-support assistant with RAGAS.")

EVALS_DIR = Path("evals")


@app.callback()
def main() -> None:
    """Assistant evaluation."""


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
