import logging

import typer

from kb_builder import tracing
from support_assistant.assistant import AssistantAnswer, SupportAssistant

app = typer.Typer(help="Customer-support assistant over a site's knowledge base (KB_SITE in .env).")


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Show LLM and retrieval logs.")) -> None:
    """Customer-support assistant."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)


@app.command()
def index() -> None:
    """Build (or reuse) the search index of the knowledge base."""
    assistant = SupportAssistant()
    chunks = assistant.index.chunks
    typer.echo(f"{assistant.settings.site}: {len(chunks)} chunks from {len({c.file for c in chunks})} files")
    for chunk in chunks:
        typer.echo(f"  {chunk.id:<60} {len(chunk.text.split()):>4} words")


@app.command()
def ask(
    question: str = typer.Argument(..., help="The customer's question."),
    show_context: bool = typer.Option(False, "--context", "-c", help="Print the retrieved chunks."),
) -> None:
    """Answer one question."""
    try:
        _print(SupportAssistant().ask(question), show_context)
    finally:
        tracing.flush()


@app.command()
def chat(show_context: bool = typer.Option(False, "--context", "-c", help="Print the retrieved chunks.")) -> None:
    """Interactive chat; an empty line or Ctrl+D ends it."""
    assistant = SupportAssistant()
    typer.echo(f"Asystent {assistant.settings.site}. Pusta linia kończy rozmowę.")
    try:
        while True:
            try:
                question = input("\n> ").strip()
            except EOFError:
                break
            if not question:
                break
            _print(assistant.ask(question), show_context)
    finally:
        tracing.flush()


def _print(result: AssistantAnswer, show_context: bool) -> None:
    typer.echo(result.answer)
    if result.source_ids:
        typer.secho(f"źródła: {', '.join(result.source_ids)}", fg=typer.colors.BRIGHT_BLACK)
    if result.rejected_answer:
        typer.secho(f"(odrzucona odpowiedź: {'; '.join(result.problems)})", fg=typer.colors.YELLOW)
    if show_context:
        for item in result.contexts:
            typer.secho(f"\n[{item.score:.3f}] {item.chunk.id}", fg=typer.colors.CYAN)
            typer.echo(item.chunk.text)


if __name__ == "__main__":
    app()
