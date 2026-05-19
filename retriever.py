import numpy as np


def retrieve(query: str, index, metadata: list[dict], top_k: int, model) -> list[dict]:
    """Embed query, normalize, search FAISS, return top_k metadata entries."""
    query_embedding = model.encode([query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding, dtype="float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        if idx != -1 and idx < len(metadata):
            results.append(metadata[idx])
    return results
