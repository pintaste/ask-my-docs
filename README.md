# ask-my-docs

![Python](https://img.shields.io/badge/python-3.12-blue) ![Claude](https://img.shields.io/badge/Claude-API-orange) ![PyPI](https://img.shields.io/badge/PyPI-ask--my--docs-green)

Chat with your local markdown, text, and PDF files using Claude + semantic search.

## Install

```bash
pip install ask-my-docs
```

## Usage

```bash
# Index a directory of docs
ask init ./docs

# Ask a question (default command)
ask "what is the authentication strategy?"

# Explicit query subcommand
ask query "how do I configure the database?"

# List all indexed files
ask list

# Clear the index
ask clear
```

## How It Works

`ask-my-docs` uses a simple RAG (Retrieval-Augmented Generation) pipeline:

1. **Index** — Scans your docs directory, extracts text from `.md`, `.txt`, and `.pdf` files, splits into overlapping chunks, and generates semantic embeddings using a local sentence-transformer model.
2. **Embed** — Each chunk is embedded with `all-MiniLM-L6-v2` and stored in a FAISS index on disk (`~/.ask-my-docs/`).
3. **Retrieve** — When you ask a question, it is embedded and compared against the FAISS index using inner-product similarity to find the most relevant chunks.
4. **Answer** — The retrieved chunks are sent to Claude (`claude-opus-4-7`) as context, which generates a grounded answer with source citations.

## Supported Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| Markdown | `.md` | Full text extraction |
| Plain text | `.txt` | Full text extraction |
| PDF | `.pdf` | Text extracted via `pdfplumber` |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ask-my-docs RAG Pipeline                 │
└─────────────────────────────────────────────────────────────────┘

  ask init ./docs
       │
       ▼
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  File scan  │────▶│  Text extraction │────▶│  Chunking        │
│  .md .txt   │     │  (pdfplumber for │     │  512-token chunks│
│  .pdf       │     │   PDFs)          │     │  w/ 64-tok overlap│
└─────────────┘     └──────────────────┘     └────────┬─────────┘
                                                       │
                                                       ▼
                                             ┌──────────────────┐
                                             │  SentenceTransf. │
                                             │  all-MiniLM-L6-v2│
                                             │  embeddings       │
                                             └────────┬─────────┘
                                                       │
                                                       ▼
                                             ┌──────────────────┐
                                             │  FAISS IndexFlatIP│
                                             │  ~/.ask-my-docs/  │
                                             │  index.faiss      │
                                             └──────────────────┘

  ask "your question"
       │
       ▼
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Query      │────▶│  Embed query     │────▶│  FAISS search    │
│  (user text)│     │  (same model)    │     │  top-K chunks    │
└─────────────┘     └──────────────────┘     └────────┬─────────┘
                                                       │
                                                       ▼
                                             ┌──────────────────┐
                                             │  Claude API      │
                                             │  claude-opus-4-7 │
                                             │  + context chunks│
                                             └────────┬─────────┘
                                                       │
                                                       ▼
                                             ┌──────────────────┐
                                             │  Answer + Sources│
                                             └──────────────────┘
```

## Environment

Set your Anthropic API key before querying:

```bash
export ANTHROPIC_API_KEY="your-key-here"
```
