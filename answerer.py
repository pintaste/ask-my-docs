def answer(query: str, chunks: list[dict], client, max_tokens: int = 1024) -> str:
    """Build context from chunks and call Claude to answer the query."""
    context_parts = []
    sources = []
    for chunk in chunks:
        source = chunk["source"]
        text = chunk["text"]
        context_parts.append(f"[Source: {source}]\n{text}")
        if source not in sources:
            sources.append(source)

    context = "\n\n---\n\n".join(context_parts)

    response = client.messages.create(
        model=__import__("config").CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=(
            "You are a helpful assistant. Answer based only on the provided context. "
            "Cite sources."
        ),
        messages=[
            {
                "role": "user",
                "content": f"Context:\n\n{context}\n\nQuestion: {query}",
            }
        ],
    )

    answer_text = next(
        (block.text for block in response.content if block.type == "text"),
        "No answer generated.",
    )

    sources_section = "\n**Sources:**\n" + "\n".join(f"- {s}" for s in sources)
    return answer_text + "\n\n" + sources_section
