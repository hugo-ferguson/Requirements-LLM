import logging
from contextlib import asynccontextmanager

import litellm
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db import create_db_and_tables
from app.llm_config import get_models_config, warn_about_retired_env_vars
from app.routes import documents, acceptance_criteria, items, sessions, uat_cases
from app.vector_store.embeddings import get_embedding_provider


# LiteLLM logs every call at INFO and prints a "Give Feedback / Get Help"
# banner on every failed attempt, retries included, which buries the app's own
# warnings. Failures still surface through the voting layer's warnings.
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail at startup, not on the first generation, if config/models.json is
    # missing or invalid.
    get_models_config()
    warn_about_retired_env_vars()
    create_db_and_tables()
    # Load the embedding model up front so the first upload isn't slow.
    get_embedding_provider()
    yield


app = FastAPI(title="Requirements-LLM API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(items.router)
app.include_router(sessions.router)
app.include_router(acceptance_criteria.router)
app.include_router(uat_cases.router)
app.include_router(documents.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
