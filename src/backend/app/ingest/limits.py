"""Upload size limits, enforced by IngestService.ingest_file.

Kept in code rather than .env so every developer runs with the same limits.
The frontend mirrors MAX_UPLOAD_BYTES in src/frontend/src/lib/attachments.ts
to reject oversized files before uploading them; this file is the source of
truth.
"""

# Checked before extraction, so a huge file is never parsed. Generous for
# PDFs and screenshots; images are shrunk before the vision model sees them
# anyway (see extract._prepare_image).
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# Checked after extraction. An attachment's full text is pasted into every
# later chat request in its session (services/conversation.py), so this is
# what actually bounds LLM cost: 100k characters is roughly 25k tokens, or
# about a 40-page document.
MAX_EXTRACTED_CHARS = 100_000
