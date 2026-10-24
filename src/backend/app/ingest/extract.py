"""Extract searchable text from uploaded text, PDF and image files."""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path

from pydantic_ai import Agent, BinaryContent

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
    """Raised when a file's extension has no extractor."""


class ImageExtractionError(RuntimeError):
    """Raised when the vision model fails or returns no usable text."""


def extract_text(data: bytes, filename: str) -> str:
    """Return the searchable text of an uploaded file, chosen by its extension.

    Raises:
        UnsupportedFileError: If the extension isn't supported.
        ImageExtractionError: If the vision model fails or returns no text.
    """
    extension = Path(filename).suffix.lower()

    if extension in TEXT_EXTENSIONS:
        return data.decode("utf-8", errors="replace")

    if extension in PDF_EXTENSIONS:
        return _extract_pdf(data)

    if extension in IMAGE_EXTENSIONS:
        return _extract_image(data, extension)

    supported = sorted(TEXT_EXTENSIONS | PDF_EXTENSIONS | IMAGE_EXTENSIONS)
    raise UnsupportedFileError(
        f"Cannot extract text from {filename!r}. "
        f"Supported extensions: {', '.join(supported)}."
    )


def _extract_pdf(data: bytes) -> str:
    """Join the text layer of every page that has one."""
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]

    return "\n\n".join(page for page in pages if page.strip())


def _extract_image(data: bytes, extension: str) -> str:
    """Transcribe and describe an image with the configured vision model."""
    vision_model = get_models_config().vision_model
    image_bytes, media_type = _prepare_image(data, extension)

    try:
        agent = Agent(
            build_model(vision_model),
            model_settings=vision_model.model_settings(timeout=300.0),
        )
        result = agent.run_sync(
            [IMAGE_PROMPT, BinaryContent(data=image_bytes, media_type=media_type)]
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


def _prepare_image(data: bytes, extension: str) -> tuple[bytes, str]:
    """Shrink an oversized image and return its bytes and media type.

    Small or unreadable images are returned as they are. Resized images
    become JPEG if they were JPEG, otherwise PNG.
    """
    from PIL import Image

    original_media_type = MIME_TYPES.get(extension, "image/png")

    try:
        image = Image.open(BytesIO(data))
        image.load()
    except Exception as error:
        logger.warning("Could not open image for resizing: %s", error)
        return data, original_media_type

    if max(image.size) <= MAX_IMAGE_DIMENSION:
        return data, original_media_type

    original_size = image.size
    image.thumbnail(
        (MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS
    )

    buffer = BytesIO()
    if extension in JPEG_EXTENSIONS:
        image.convert("RGB").save(buffer, "JPEG", quality=JPEG_QUALITY)
        media_type = "image/jpeg"
    else:
        image.convert("RGBA").save(buffer, "PNG")
        media_type = "image/png"

    resized = buffer.getvalue()
    logger.info(
        "Resized image from %sx%s (%s KB) to %sx%s (%s KB)",
        *original_size,
        len(data) // 1024,
        *image.size,
        len(resized) // 1024,
    )

    return resized, media_type
