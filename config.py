from pathlib import Path

INDEX_DIR = Path.home() / ".ask-my-docs"
CHUNK_SIZE = 512       # tokens approx (use chars / 4)
CHUNK_OVERLAP = 64
TOP_K = 5
EMBED_MODEL = "all-MiniLM-L6-v2"
CLAUDE_MODEL = "claude-opus-4-7"
SUPPORTED_EXTS = {".md", ".txt", ".pdf"}
