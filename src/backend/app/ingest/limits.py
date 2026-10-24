"""Upload size limits, enforced by IngestService.ingest_file.

The frontend copies MAX_UPLOAD_BYTES in src/frontend/src/lib/attachments.ts.
"""

# Checked before extraction, so a huge file is never parsed.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# Checked after extraction. Attachments are resent with every chat request,
# so this limits LLM cost (about 25k tokens).
MAX_EXTRACTED_CHARS = 100_000
