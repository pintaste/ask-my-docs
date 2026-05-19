"""Tests for answerer.py — answer function."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from answerer import answer  # noqa: E402


def _make_chunks(sources=None):
    if sources is None:
        sources = ["docs/readme.txt"]
    return [
        {"source": src, "text": f"Some useful content from {src}", "chunk_index": 0}
        for src in sources
    ]


def _make_mock_client(answer_text: str = "The answer is 42."):
    """Return a mock Anthropic client whose messages.create returns a plausible response."""
    client = MagicMock()
    block = MagicMock()
    block.type = "text"
    block.text = answer_text
    client.messages.create.return_value.content = [block]
    return client


def test_answer_calls_claude():
    """answer() must call client.messages.create exactly once."""
    chunks = _make_chunks()
    client = _make_mock_client()

    answer("What is the meaning of life?", chunks, client)

    client.messages.create.assert_called_once()


def test_answer_passes_context_to_claude():
    """The user message sent to Claude must contain the chunk text."""
    chunks = _make_chunks(["docs/guide.md"])
    client = _make_mock_client()

    answer("What is X?", chunks, client)

    call_kwargs = client.messages.create.call_args
    # messages kwarg is a list; the user message content should include chunk text
    messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[3] if len(call_kwargs.args) > 3 else call_kwargs.kwargs["messages"]
    user_content = messages[0]["content"]
    assert "Some useful content from docs/guide.md" in user_content


def test_answer_includes_sources():
    """Returned string must contain a sources section listing the source file."""
    source = "docs/api.txt"
    chunks = _make_chunks([source])
    client = _make_mock_client("Here is the answer.")

    result = answer("What does the API do?", chunks, client)

    assert "**Sources:**" in result
    assert source in result


def test_answer_includes_answer_text():
    """The answer text from Claude must appear in the returned string."""
    client = _make_mock_client("The sky is blue.")
    result = answer("Why is the sky blue?", _make_chunks(), client)
    assert "The sky is blue." in result


def test_answer_deduplicates_sources():
    """Multiple chunks from the same source produce only one source entry."""
    source = "shared.txt"
    chunks = [
        {"source": source, "text": "chunk A", "chunk_index": 0},
        {"source": source, "text": "chunk B", "chunk_index": 1},
    ]
    client = _make_mock_client()

    result = answer("question", chunks, client)

    # Count occurrences of source in the sources section only
    sources_section = result.split("**Sources:**")[-1]
    assert sources_section.count(source) == 1
