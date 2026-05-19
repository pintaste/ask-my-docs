import json
import numpy as np
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from config import SUPPORTED_EXTS, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks by character count (chunk_size * 4 chars each)."""
    char_size = chunk_size * 4
    char_overlap = overlap * 4
    chunks = []
    start = 0
    while start < len(text):
        end = start + char_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += char_size - char_overlap
    return chunks


def extract_text(path: Path) -> str:
    """Read .md/.txt directly; use pdfplumber for .pdf."""
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".pdf":
        import pdfplumber
        text_parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n".join(text_parts)
    return ""


def index_directory(source_dir: Path, index_dir: Path) -> None:
    """Scan for supported files, chunk, embed, build FAISS index, and save."""
    index_dir.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(EMBED_MODEL)

    all_texts = []
    metadata = []

    files = [p for p in source_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS and p.is_file()]

    for path in files:
        text = extract_text(path)
        if not text.strip():
            continue
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                all_texts.append(chunk)
                metadata.append({
                    "source": str(path),
                    "chunk_index": i,
                    "text": chunk,
                })

    if not all_texts:
        print("No text content found.")
        return

    embeddings = model.encode(all_texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(index_dir / "index.faiss"))
    with open(index_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    file_count = len(set(m["source"] for m in metadata))
    chunk_count = len(metadata)
    print(f"Indexed {file_count} files, {chunk_count} chunks.")


def load_index(index_dir: Path) -> tuple:
    """Load saved FAISS index and metadata."""
    index_path = index_dir / "index.faiss"
    meta_path = index_dir / "metadata.json"
    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"No index found in {index_dir}. Run `ask init <directory>` first.")
    index = faiss.read_index(str(index_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata
