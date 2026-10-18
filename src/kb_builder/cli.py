import logging
from pathlib import Path

import typer

from kb_builder.models import CrawlConfig
from kb_builder.services import CrawlService

app = typer.Typer(help="Build a customer-support knowledge base from a website.")


@app.callback()
def main() -> None:
    """Knowledge base builder."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)


@app.command()
def crawl(
    url: str = typer.Argument(..., help="Start URL, e.g. https://art-mat.com.pl/"),
    out: Path = typer.Option(Path("data/raw"), help="Root directory for the raw cache."),
    max_pages: int = typer.Option(200, help="Stop after this many fetched URLs."),
    max_depth: int = typer.Option(5, help="Maximum link depth from the start URL."),
    delay: float = typer.Option(0.5, help="Seconds to wait between requests."),
    refresh: bool = typer.Option(False, help="Ignore the cache and fetch everything again."),
) -> None:
    """Discover pages (sitemap + links) and save raw HTML/PDF to the cache."""
    config = CrawlConfig(max_pages=max_pages, max_depth=max_depth, delay=delay, refresh=refresh)
    result = CrawlService(url, out, config).run()

    typer.echo(f"\nSite: {result.site}  ->  {result.out_dir}")
    typer.echo("  " + ", ".join(f"{kind}: {n}" for kind, n in result.counts().items()))
    typer.echo(f"  manifest: {result.out_dir / 'crawl.json'}")


if __name__ == "__main__":
    app()
