from pathlib import Path

import typer

from kb_builder.crawl import CrawlConfig, crawl_site

app = typer.Typer(help="Build a customer-support knowledge base from a website.")


@app.callback()
def main() -> None:
    """Knowledge base builder."""


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
    result = crawl_site(url, out, config)

    counts: dict[str, int] = {}
    for record in result.records:
        counts[record.kind] = counts.get(record.kind, 0) + 1
    typer.echo(f"\nSite: {result.site}  ->  {result.out_dir}")
    typer.echo("  " + ", ".join(f"{kind}: {n}" for kind, n in sorted(counts.items())))
    typer.echo(f"  manifest: {result.out_dir / 'crawl.json'}")


if __name__ == "__main__":
    app()
