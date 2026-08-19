from __future__ import annotations

import base64
import time
from io import BytesIO
from pathlib import Path

import httpx

from app.config import Settings

TEXT_EXTENSIONS = {".txt", ".md"}
PDF_EXTENSIONS = {".pdf"}
IMAGE_MIME_TYPES = {
	".png": "image/png",
	".jpg": "image/jpeg",
	".jpeg": "image/jpeg",
	".webp": "image/webp",
	".gif": "image/gif",
}

IMAGE_PROMPT = (
	"Transcribe every piece of text visible in this image exactly as written. "
	"Then, if the image contains a diagram, chart, screenshot or table, "
	"describe its structure and content in plain prose so the information can "
	"be searched later. Do not add commentary about the image quality."
)

GEMINI_TIMEOUT_SECONDS = 60.0
GEMINI_MAX_ATTEMPTS = 3
GEMINI_RETRY_STATUSES = {429, 500, 502, 503, 504}


class UnsupportedFileError(ValueError):
	"""Raised when a file's extension has no configured extractor."""


class ImageExtractionError(RuntimeError):
	"""Raised when Gemini could not be reached or returned no usable text."""


def extract_text(data: bytes, filename: str, settings: Settings) -> str:
	"""
	Returns the searchable text of an uploaded file, chosen by its extension.
	"""
	extension = Path(filename).suffix.lower()

	if extension in TEXT_EXTENSIONS:
		return data.decode("utf-8", errors="replace")

	if extension in PDF_EXTENSIONS:
		return _extract_pdf(data)

	if extension in IMAGE_MIME_TYPES:
		return _extract_image(data, extension, settings)

	supported = sorted(TEXT_EXTENSIONS | PDF_EXTENSIONS | set(IMAGE_MIME_TYPES))
	raise UnsupportedFileError(
		f"Cannot extract text from {filename!r}. "
		f"Supported extensions: {', '.join(supported)}."
	)


def _extract_pdf(data: bytes) -> str:
	"""Concatenates the embedded text layer of every page."""
	from pypdf import PdfReader

	reader = PdfReader(BytesIO(data))
	pages = [page.extract_text() or "" for page in reader.pages]

	return "\n\n".join(page for page in pages if page.strip())


def _extract_image(data: bytes, extension: str, settings: Settings) -> str:
	"""
	Asks Gemini to transcribe and describe an image so it can be embedded.
	"""
	if not settings.gemini_api_key:
		raise ValueError("GEMINI_API_KEY is required to ingest images")

	url = (
		f"{settings.gemini_base_url}/models/"
		f"{settings.gemini_model}:generateContent"
	)
	payload = {
		"contents": [
			{
				"parts": [
					{"text": IMAGE_PROMPT},
					{
						"inline_data": {
							"mime_type": IMAGE_MIME_TYPES[extension],
							"data": base64.b64encode(data).decode("ascii"),
						}
					},
				]
			}
		]
	}

	response = _post_with_retries(
		url, payload, settings.gemini_api_key
	)

	return _first_candidate_text(response.json())


def _post_with_retries(
		url: str, payload: dict, api_key: str
	) -> httpx.Response:
	"""
	Posts to Gemini, retrying the overload and rate-limit statuses.

	Gemini answers 429/503 whenever the model is busy, which says nothing about
	the request itself, so those are worth a second and third try.
	"""
	last_error: Exception | None = None

	for attempt in range(GEMINI_MAX_ATTEMPTS):
		if attempt:
			time.sleep(2 ** attempt)

		try:
			response = httpx.post(
				url,
				headers={"x-goog-api-key": api_key},
				json=payload,
				timeout=GEMINI_TIMEOUT_SECONDS,
			)
		except httpx.RequestError as error:
			last_error = error
			continue

		if response.status_code in GEMINI_RETRY_STATUSES:
			last_error = httpx.HTTPStatusError(
				f"Gemini returned {response.status_code}",
				request=response.request,
				response=response,
			)
			continue

		if response.is_error:
			raise ImageExtractionError(
				f"Gemini rejected the request with "
				f"{response.status_code}: {response.text[:300]}"
			)

		return response

	raise ImageExtractionError(
		f"Gemini was unavailable after {GEMINI_MAX_ATTEMPTS} attempts: "
		f"{last_error}"
	)


def _first_candidate_text(payload: dict) -> str:
	"""
	Joins the text parts of Gemini's first candidate, ignoring any non-text
	parts the model may return alongside them.
	"""
	candidates = payload.get("candidates") or []
	if not candidates:
		raise ImageExtractionError(f"Gemini returned no candidates: {payload}")

	parts = candidates[0].get("content", {}).get("parts") or []

	return "\n".join(part["text"] for part in parts if "text" in part).strip()
