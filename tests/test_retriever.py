"""Tests for retriever.py — retrieve function."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from retriever import retrieve  # noqa: E402


def _make_metadata(n: int) -> list[dict]:
    return [
        {"source": f"doc_{i}.txt", "chunk_index": i, "text": f"chunk text {i}"}
        for i in range(n)
    ]


def _make_mock_model(n_items: int, dim: int = 4):
    """Return a mock SentenceTransformer that yields unit vectors."""
    model = MagicMock()
    # encode returns shape (1, dim) array
    embedding = np.ones((1, dim), dtype="float32")
    embedding /= np.linalg.norm(embedding)
    model.encode.return_value = embedding
    return model


def _make_mock_index(n_items: int, top_k: int, dim: int = 4):
    """Return a mock FAISS index whose search returns plausible results."""
    index = MagicMock()
    # FAISS search returns (distances, indices) each shape (1, top_k)
    k = min(top_k, n_items)
    distances = np.ones((1, k), dtype="float32")
    indices = np.arange(k, dtype="int64").reshape(1, k)
    index.search.return_value = (distances, indices)
    return index


def test_retrieve_returns_top_k():
    """With 5 metadata items and top_k=3, retrieve returns exactly 3 results."""
    n_items = 5
    top_k = 3
    dim = 4

    metadata = _make_metadata(n_items)
    model = _make_mock_model(n_items, dim)
    index = _make_mock_index(n_items, top_k, dim)

    results = retrieve("what is X?", index, metadata, top_k, model)

    assert len(results) <= top_k
    assert len(results) == top_k


def test_retrieve_skips_invalid_indices():
    """Indices of -1 (FAISS padding) are excluded from results."""
    metadata = _make_metadata(5)
    model = _make_mock_model(5)
    index = MagicMock()
    # Return two valid indices and one sentinel -1
    indices = np.array([[0, -1, 2]], dtype="int64")
    distances = np.ones((1, 3), dtype="float32")
    index.search.return_value = (distances, indices)

    results = retrieve("query", index, metadata, 3, model)
    assert len(results) == 2
    assert all(r["source"] != "sentinel" for r in results)


def test_retrieve_result_structure():
    """Each result contains expected metadata keys."""
    metadata = _make_metadata(5)
    model = _make_mock_model(5)
    index = _make_mock_index(5, top_k=2)

    results = retrieve("hello", index, metadata, 2, model)
    for r in results:
        assert "source" in r
        assert "text" in r
        assert "chunk_index" in r
