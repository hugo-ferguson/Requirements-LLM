from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path

from pydantic_ai import Agent, BinaryContent

from app.config import Settings
from app.llm_config import get_models_config
from llm.spec import build_model

logger = logging.getLogger(__name__)

TEXT_EXTENSIONS = {".txt", ".md"}
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
JPEG_EXTENSIONS = {".jpg", ".jpeg"}

MIME_TYPES = {
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

MAX_IMAGE_DIMENSION = 1024
JPEG_QUALITY = 90


class UnsupportedFileError(ValueError):
    """Raised when a file's extension has no configured extractor."""


class ImageExtractionError(RuntimeError):
    """Raised when the vision model could not be reached or returned no usable text."""


def extract_text(data: bytes, filename: str, settings: Settings) -> str:
    """Returns the searchable text of an uploaded file, chosen by its extension."""
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
    """Transcribes and describes an image via a vision-capable model."""
    vision_model = get_models_config().vision_model
    image_bytes = _prepare_image(data, extension)
    mime = MIME_TYPES.get(extension, "image/png")

    try:
        agent = Agent(
            build_model(vision_model),
            model_settings=vision_model.model_settings(timeout=300.0),
        )
        result = agent.run_sync(
            [IMAGE_PROMPT, BinaryContent(data=image_bytes, media_type=mime)]
        )
    except Exception as error:
        raise ImageExtractionError(
            f"Vision model {vision_model.name} failed: {error}"
        ) from error

    content = result.output

    if not content.strip():
        raise ImageExtractionError(
            f"{vision_model.name} returned no text for the image"
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
