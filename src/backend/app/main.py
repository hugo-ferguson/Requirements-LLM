from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db import create_db_and_tables
from app.routes import documents, items
from app.vector_store.embeddings import get_embedding_provider


@asynccontextmanager
async def lifespan(app: FastAPI):
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
app.include_router(documents.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
