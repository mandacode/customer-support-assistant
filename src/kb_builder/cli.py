import logging
from pathlib import Path

import typer

from kb_builder.candidates import extract_candidates
from kb_builder.context import SiteContext
from kb_builder.models import (
    ClassifyConfig,
    CrawlConfig,
    ExtractConfig,
    ExtractManifest,
    MergeConfig,
    NormalizeConfig,
    ProfileConfig,
    SegmentConfig,
)
from kb_builder.pipeline import Pipeline, PipelineError, build_pipeline
from kb_builder.services import (
    BaseService,
    ClassifyService,
    CrawlService,
    ExtractService,
    MergeService,
    NormalizeService,
    ProfileService,
    SegmentService,
)
from kb_builder.services.segment_service import SECTIONS, read_sections

app = typer.Typer(help="Build a customer-support knowledge base from a website.")

DATA_OPTION = typer.Option(Path("data"), help="Root directory for all stage outputs.")


@app.callback()
def main() -> None:
    """Knowledge base builder."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)


@app.command()
def build(
    url: str = typer.Argument(..., help="Start URL, e.g. https://art-mat.com.pl/"),
    start: str | None = typer.Option(None, "--from", help="First stage to run (default: the first)."),
    stop: str | None = typer.Option(None, "--to", help="Last stage to run (default: the last)."),
    data: Path = DATA_OPTION,
    max_pages: int = typer.Option(200, help="Crawl: stop after this many fetched URLs."),
    refresh: bool = typer.Option(False, help="Crawl: ignore the cache and fetch everything again."),
) -> None:
    """Run the whole pipeline (or a slice of it) for one site."""
    pipeline = build_pipeline(crawl=CrawlConfig(max_pages=max_pages, refresh=refresh))
    _run(pipeline, _context(url, data), start, stop)


@app.command()
def crawl(
    url: str = typer.Argument(..., help="Start URL, e.g. https://art-mat.com.pl/"),
    data: Path = DATA_OPTION,
    max_pages: int = typer.Option(200, help="Stop after this many fetched URLs."),
    max_depth: int = typer.Option(5, help="Maximum link depth from the start URL."),
    delay: float = typer.Option(0.5, help="Seconds to wait between requests."),
    refresh: bool = typer.Option(False, help="Ignore the cache and fetch everything again."),
) -> None:
    """Discover pages (sitemap + links) and save raw HTML/PDF to the cache."""
    config = CrawlConfig(max_pages=max_pages, max_depth=max_depth, delay=delay, refresh=refresh)
    _run_stage(CrawlService(config), _context(url, data))


@app.command()
def extract(
    site: str = typer.Argument(..., help="Site name or URL, e.g. art-mat.com.pl"),
    data: Path = DATA_OPTION,
    thin_words: int = typer.Option(40, help="Pages with fewer words are flagged as thin."),
) -> None:
    """Convert cached pages to clean Markdown and flag junk pages."""
    _run_stage(ExtractService(ExtractConfig(thin_words=thin_words)), _context(site, data))


@app.command()
def segment(
    site: str = typer.Argument(..., help="Site name or URL, e.g. art-mat.com.pl"),
    data: Path = DATA_OPTION,
    max_words: int = typer.Option(500, help="Split sections longer than this."),
) -> None:
    """Split extracted pages into heading-based sections."""
    _run_stage(SegmentService(SegmentConfig(max_words=max_words)), _context(site, data))


@app.command()
def profile(
    site: str = typer.Argument(..., help="Site name or URL, e.g. art-mat.com.pl"),
    data: Path = DATA_OPTION,
    snippet_words: int = typer.Option(40, help="Opening words of each section sent to the LLM."),
) -> None:
    """Ask the LLM for the site profile: business, entities, support checklist."""
    _run_stage(ProfileService(ProfileConfig(snippet_words=snippet_words)), _context(site, data))


@app.command()
def classify(
    site: str = typer.Argument(..., help="Site name or URL, e.g. art-mat.com.pl"),
    data: Path = DATA_OPTION,
    batch_size: int = typer.Option(8, help="Sections per LLM call."),
) -> None:
    """Label every section with a topic and entities from the site profile."""
    _run_stage(ClassifyService(ClassifyConfig(batch_size=batch_size)), _context(site, data))


@app.command()
def merge(
    site: str = typer.Argument(..., help="Site name or URL, e.g. art-mat.com.pl"),
    data: Path = DATA_OPTION,
) -> None:
    """Group classified sections into future knowledge-base files (no LLM)."""
    _run_stage(MergeService(MergeConfig()), _context(site, data))


@app.command()
def normalize(
    site: str = typer.Argument(..., help="Site name or URL, e.g. art-mat.com.pl"),
    data: Path = DATA_OPTION,
    min_word_coverage: float = typer.Option(0.9, help="Share of output words that must occur in the source."),
) -> None:
    """Write knowledge-base Markdown files, verifying every LLM rewrite against the source."""
    _run_stage(NormalizeService(NormalizeConfig(min_word_coverage=min_word_coverage)), _context(site, data))


@app.command()
def candidates(
    site: str = typer.Argument(..., help="Site name or URL, e.g. art-mat.com.pl"),
    data: Path = DATA_OPTION,
) -> None:
    """List entity candidates found in the site text (no LLM), for review."""
    context = _context(site, data)
    sections_path = context.segmented_dir / SECTIONS
    if not sections_path.exists():
        typer.secho(f"No {sections_path}; run `kb segment` first.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    extract = ExtractManifest.model_validate_json((context.extracted_dir / "extract.json").read_text())
    for candidate in extract_candidates(read_sections(sections_path), extract):
        kinds = ",".join(candidate.kinds)
        sections = ", ".join(sid.split("#")[-1][:28] for sid in candidate.section_ids)
        typer.echo(f"{candidate.id}  {candidate.text:<42} {kinds:<22} {sections}")


def _context(site_or_url: str, data: Path) -> SiteContext:
    if "://" in site_or_url:
        return SiteContext.from_url(site_or_url, data)
    return SiteContext(site=site_or_url, data_root=data)


def _run_stage(stage: BaseService, context: SiteContext) -> None:
    _run(Pipeline([stage]), context)


def _run(pipeline: Pipeline, context: SiteContext, start: str | None = None, stop: str | None = None) -> None:
    try:
        pipeline.run(context, start, stop)
    except PipelineError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()
