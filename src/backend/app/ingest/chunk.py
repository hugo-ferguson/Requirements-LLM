from __future__ import annotations


def chunk_text(
		text: str, chunk_size: int = 1000, overlap: int = 150
	) -> list[str]:
	"""
	Splits text into overlapping chunks of at most `chunk_size` characters.

	Chunks end on a paragraph break where one is available, and otherwise on a
	space, so that words are not cut in half. The overlap carries context
	across the boundary for retrieval.
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
			boundary = _boundary_before(text, start + overlap, end)
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
	"""
	Finds the last paragraph break, or failing that the last space, in
	`text[lower:upper]`. Returns None when the window holds neither.
	"""
	for separator in ("\n\n", "\n", " "):
		position = text.rfind(separator, lower, upper)
		if position != -1:
			return position

	return None
