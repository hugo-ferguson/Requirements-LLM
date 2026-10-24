"""Split extracted document text into overlapping chunks."""

from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks for embedding.

    Chunks end on a paragraph break, line break or space where possible.

    Args:
        text: The text to split.
        chunk_size: Maximum chunk length, in characters.
        overlap: Characters repeated from the previous chunk.

    Raises:
        ValueError: If `overlap` isn't smaller than `chunk_size`.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            # Search after start + overlap, or the next start won't move and
            # the loop never ends.
            boundary = _boundary_before(text, start + overlap + 1, end)
            if boundary is not None:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def _boundary_before(text: str, lower: int, upper: int) -> int | None:
    """Return the best break in the window (paragraph, line, then space), or None."""
    for separator in ("\n\n", "\n", " "):
        position = text.rfind(separator, lower, upper)
        if position != -1:
            return position

    return None
