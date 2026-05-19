#!/usr/bin/env python3
import shutil
import sys
from pathlib import Path

import click
import anthropic
from sentence_transformers import SentenceTransformer

from config import INDEX_DIR, TOP_K, EMBED_MODEL
from indexer import index_directory, load_index
from retriever import retrieve
from answerer import answer


@click.group(invoke_without_command=True)
@click.argument("question", required=False)
@click.pass_context
def cli(ctx, question):
    """ask-my-docs: Chat with your local docs using Claude + semantic search.

    Usage:
        ask init <directory>   Index a directory of docs
        ask "your question"    Ask a question (default command)
        ask query "question"   Ask a question (explicit)
        ask list               List indexed files
        ask clear              Delete the index
    """
    if ctx.invoked_subcommand is None:
        if question:
            ctx.invoke(query, question=question)
        else:
            click.echo(ctx.get_help())


@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
def init(directory: Path):
    """Index a directory of markdown, text, and PDF files."""
    click.echo(f"Indexing {directory} ...")
    index_directory(directory, INDEX_DIR)


@cli.command()
@click.argument("question")
def query(question: str):
    """Ask a question against the indexed docs."""
    try:
        index, metadata = load_index(INDEX_DIR)
    except FileNotFoundError as e:
        click.echo(str(e), err=True)
        sys.exit(1)

    model = SentenceTransformer(EMBED_MODEL)
    chunks = retrieve(question, index, metadata, TOP_K, model)

    if not chunks:
        click.echo("No relevant documents found.")
        sys.exit(0)

    client = anthropic.Anthropic()
    result = answer(question, chunks, client)
    click.echo(result)


@cli.command("list")
def list_files():
    """Show all indexed files."""
    try:
        _, metadata = load_index(INDEX_DIR)
    except FileNotFoundError as e:
        click.echo(str(e), err=True)
        sys.exit(1)

    sources = sorted(set(m["source"] for m in metadata))
    if not sources:
        click.echo("No files indexed.")
    else:
        click.echo(f"Indexed files ({len(sources)}):")
        for s in sources:
            click.echo(f"  {s}")


@cli.command()
def clear():
    """Delete the index (~/.ask-my-docs/)."""
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
        click.echo(f"Deleted {INDEX_DIR}")
    else:
        click.echo("No index to clear.")


if __name__ == "__main__":
    cli()
