"""Tests for indexer.py — chunk_text and extract_text functions."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Stub heavy deps before importing indexer
sys.modules.setdefault("faiss", MagicMock())
sys.modules.setdefault("sentence_transformers", MagicMock())

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from indexer import chunk_text, extract_text  # noqa: E402


# ---------------------------------------------------------------------------
# chunk_text tests
# ---------------------------------------------------------------------------

def test_chunk_text_basic():
    """2000-char text with chunk_size=512 (=2048 chars) and overlap=64 (=256 chars)
    should still produce at least 1 chunk; smaller params produce multiple chunks."""
    text = "a" * 2000
    # Use small chunk_size so we definitely get multiple chunks:
    # chunk_size=10 → char_size=40, overlap=4 → char_overlap=16
    # stride = 40 - 16 = 24; chunks ≈ ceil((2000-40)/24) + 1 = multiple
    chunks = chunk_text(text, chunk_size=10, overlap=4)
    assert len(chunks) > 1


def test_chunk_text_overlap():
    """Consecutive chunks share overlapping content."""
    text = "abcdefghijklmnopqrstuvwxyz" * 10  # 260 chars
    # chunk_size=4 → char_size=16, overlap=2 → char_overlap=8; stride=8
    chunks = chunk_text(text, chunk_size=4, overlap=2)
    assert len(chunks) >= 2
    # The tail of chunk[0] should equal the head of chunk[1]
    # chunk[0] = text[0:16], chunk[1] = text[8:24]
    # overlap region: text[8:16]
    assert chunks[0][8:16] == chunks[1][0:8]


def test_chunk_text_short():
    """Text shorter than char_size returns a single chunk."""
    text = "hello world"
    chunks = chunk_text(text, chunk_size=512, overlap=64)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_empty():
    """Empty string returns empty list (loop condition: start=0, not < 0)."""
    chunks = chunk_text("", chunk_size=512, overlap=64)
    # The while loop condition `while start < len(text)` is False immediately
    # for an empty string, so the result is an empty list.
    assert chunks == []


# ---------------------------------------------------------------------------
# extract_text tests
# ---------------------------------------------------------------------------

def test_extract_text_markdown(tmp_path):
    """extract_text reads a .md file and returns its content."""
    md_file = tmp_path / "doc.md"
    content = "# Hello\n\nThis is markdown."
    md_file.write_text(content, encoding="utf-8")
    assert extract_text(md_file) == content


def test_extract_text_txt(tmp_path):
    """extract_text reads a .txt file and returns its content."""
    txt_file = tmp_path / "notes.txt"
    content = "Plain text content."
    txt_file.write_text(content, encoding="utf-8")
    assert extract_text(txt_file) == content
