from __future__ import annotations

import base64
import logging
import time
from io import BytesIO
from pathlib import Path

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

TEXT_EXTENSIONS = {".txt", ".md"}
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
JPEG_EXTENSIONS = {".jpg", ".jpeg"}

IMAGE_PROMPT = (
	"Transcribe every piece of text visible in this image exactly as written. "
	"Then, if the image contains a diagram, chart, screenshot or table, "
	"describe its structure and content in plain prose so the information can "
	"be searched later. Do not add commentary about the image quality."
)

# Vision models tile images into blocks, so anything larger than this costs
# time and tokens without making the text easier to read. 1024 is the floor:
# measured on qwen2.5vl, a 1440px screenshot took 19.3s and a 1024px one
# 14.8s with identical transcripts, while shrinking further saved nothing —
# the model pads back up to the same tile grid, so 768px cost the same 14.4s
# but started dropping text.
MAX_IMAGE_DIMENSION = 1024
JPEG_QUALITY = 90

# A local model loads its weights on the first call and decodes far more
# slowly than a hosted API, so it needs a generous leash.
OLLAMA_TIMEOUT_SECONDS = 300.0
OLLAMA_MAX_ATTEMPTS = 3
OLLAMA_RETRY_STATUSES = {429, 500, 502, 503, 504}

# Ollama evicts an idle model after 5 minutes by default, and reloading this
# one off disk costs ~4s. Uploads come in bursts with long gaps between them,
# which is the worst case for that default.
OLLAMA_KEEP_ALIVE = "30m"


class UnsupportedFileError(ValueError):
	"""Raised when a file's extension has no configured extractor."""


class ImageExtractionError(RuntimeError):
	"""
	Raised when the vision model could not be reached or returned no usable
	text.
	"""


def extract_text(data: bytes, filename: str, settings: Settings) -> str:
	"""
	Returns the searchable text of an uploaded file, chosen by its extension.
	"""
	extension = Path(filename).suffix.lower()

	if extension in TEXT_EXTENSIONS:
		return data.decode("utf-8", errors="replace")

	if extension in PDF_EXTENSIONS:
		return _extract_pdf(data)

	if extension in IMAGE_EXTENSIONS:
		return _extract_image(data, extension, settings)

	supported = sorted(TEXT_EXTENSIONS | PDF_EXTENSIONS | IMAGE_EXTENSIONS)
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
	Transcribes and describes an image with a local Ollama vision model.

	Ollama takes the raw base64 payload on the message itself and infers the
	image format, so no MIME type is sent.
	"""
	payload = {
		"model": settings.ollama_vision_model,
		"stream": False,
		"keep_alive": OLLAMA_KEEP_ALIVE,
		"messages": [
			{
				"role": "user",
				"content": IMAGE_PROMPT,
				"images": [
					base64.b64encode(
						_prepare_image(data, extension)
					).decode("ascii")
				],
			}
		],
	}

	response = _post_with_retries(
		f"{settings.ollama_base_url.rstrip('/')}/api/chat", payload
	)
	content = response.json().get("message", {}).get("content")

	if not isinstance(content, str) or not content.strip():
		raise ImageExtractionError(
			f"{settings.ollama_vision_model} returned no text for the image"
		)

	return content.strip()


def _prepare_image(data: bytes, extension: str) -> bytes:
	"""
	Shrinks oversized images before upload.

	Images already within MAX_IMAGE_DIMENSION are sent untouched, so the common
	case pays nothing and loses no detail. Anything Pillow cannot read falls
	back to the original bytes and lets the model decide.
	"""
	from PIL import Image

	try:
		image = Image.open(BytesIO(data))
		image.load()
	except Exception as error:
		logger.warning("Could not open image for resizing: %s", error)
		return data

	# Resizing costs ~40ms and saves seconds of vision-model time, so there's
	# no tolerance band worth keeping here — anything over the cap gets shrunk.
	if max(image.size) <= MAX_IMAGE_DIMENSION:
		return data

	original_size = image.size
	image.thumbnail(
		(MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.LANCZOS
	)

	buffer = BytesIO()
	if extension in JPEG_EXTENSIONS:
		image.convert("RGB").save(buffer, "JPEG", quality=JPEG_QUALITY)
	else:
		# PNG keeps text edges crisp, which matters more than file size when
		# the point of the upload is reading the text back out.
		image.convert("RGBA").save(buffer, "PNG")

	resized = buffer.getvalue()
	logger.info(
		"Resized image from %sx%s (%s KB) to %sx%s (%s KB)",
		*original_size,
		len(data) // 1024,
		*image.size,
		len(resized) // 1024,
	)

	return resized


def _post_with_retries(url: str, payload: dict) -> httpx.Response:
	"""
	Posts to Ollama, retrying the overload and busy statuses.

	Ollama answers 429/503 while a model is loading or the queue is full,
	which says nothing about the request itself, so those are worth a second
	and third try.
	"""
	last_error: Exception | None = None

	for attempt in range(OLLAMA_MAX_ATTEMPTS):
		if attempt:
			delay = 2 ** attempt
			logger.warning(
				"Retrying in %ss (attempt %s of %s) after: %s",
				delay,
				attempt + 1,
				OLLAMA_MAX_ATTEMPTS,
				last_error,
			)
			time.sleep(delay)

		try:
			response = httpx.post(
				url, json=payload, timeout=OLLAMA_TIMEOUT_SECONDS
			)
		except httpx.RequestError as error:
			last_error = error
			continue

		if response.status_code in OLLAMA_RETRY_STATUSES:
			last_error = httpx.HTTPStatusError(
				f"Ollama returned {response.status_code}",
				request=response.request,
				response=response,
			)
			continue

		if response.is_error:
			raise ImageExtractionError(
				f"Ollama rejected the request with "
				f"{response.status_code}: {response.text[:300]}"
			)

		return response

	raise ImageExtractionError(
		f"Ollama was unavailable after {OLLAMA_MAX_ATTEMPTS} attempts: "
		f"{last_error}"
	)
