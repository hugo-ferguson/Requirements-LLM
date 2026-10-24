"""FastAPI backend that turns user stories into reviewed requirements.

Subpackages:

- `generation`: the model ensemble that writes ACs and UAT cases.
- `ingest`: text extraction and chunking for uploads.
- `repositories`: database access.
- `routes`: the HTTP API.
- `services`: business logic.
- `vector_store`: embeddings and document search.

The app is in `app.main`, settings in `app.config`, models in `app.llm_config`.
"""
