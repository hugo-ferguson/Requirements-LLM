"""FastAPI backend that turns user stories into reviewed requirements.

Subpackages:

- `generation`: the multi-model ensemble that writes ACs and UAT cases.
- `ingest`: text extraction and chunking for uploaded files.
- `repositories`: all direct database access.
- `routes`: the HTTP API.
- `services`: business logic between the routes and repositories.
- `vector_store`: embeddings and similarity search over documents.

The application object lives in `app.main`, settings in `app.config`
and the model roster in `app.llm_config`.
"""
