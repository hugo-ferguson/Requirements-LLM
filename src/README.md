# Requirements-LLM: developer guide

How the code is laid out and how to work on it. To run the app as a user, see
the [top-level README](../README.md).

## Stack

| Layer | Choice |
| --- | --- |
| Backend | Python 3.12+, FastAPI, layered as routes → services → repositories |
| Models | PydanticAI agents, configured in `backend/config/models.json` |
| Database | PostgreSQL with pgvector, through SQLModel. One local container per developer |
| Frontend | React + TypeScript (Vite, Tailwind), with an API client typed from the backend's OpenAPI schema |
| Auth | None. Data is scoped to chat sessions, not accounts |
| Infra | Docker Compose (`frontend`, `backend`, `db`) |

## Running it

1. Copy the env template and the model config, then fill in your API keys:
   ```
   cp .env.example .env
   cp backend/config/models.example.json backend/config/models.json
   ```
2. Start everything: `docker compose up --build`
3. Open the frontend at http://localhost:5173, or the API docs at
   http://localhost:8000/docs.

Postgres data lives in the `db_data` volume on your machine only. To wipe it,
run `docker compose down -v`.

### Running without Docker

Backend (needs [uv](https://docs.astral.sh/uv/), and Postgres at the
`DATABASE_URL` in `.env`, for example from `docker compose up -d db`):
```
cd backend
uv sync
uv run uvicorn app.main:app --reload
```

Frontend:
```
cd frontend
npm install
npm run dev
```

## Folder structure

```
src/
├── docker-compose.yml
├── .env.example                # copy to .env (gitignored)
├── openapi.json                # generated, see "Regenerating the typed API client"
├── backend/
│   ├── config/
│   │   └── models.example.json # copy to models.json: chat, vision, generation and judge models
│   ├── ai_prompts/             # rubric prompts the voting judges read
│   ├── llm/spec.py             # turns one models.json entry into a PydanticAI model
│   ├── scripts/export_openapi.py
│   ├── tests/
│   ├── voting/                 # LLM judges that score candidates; also runs as an MCP server
│   └── app/
│       ├── main.py             # FastAPI app, middleware, router registration
│       ├── config.py           # settings from the environment
│       ├── llm_config.py       # loads config/models.json
│       ├── db.py               # engine and session setup
│       ├── dependencies.py     # FastAPI dependency providers shared by routes
│       ├── models_*.py         # SQLModel tables and request/response schemas
│       ├── routes/             # HTTP only: parse the request, call a service, shape the response
│       ├── services/           # business logic, independent of HTTP and SQL
│       ├── repositories/       # database queries for sessions, messages, ACs and UAT cases
│       ├── generation/         # the ensemble: agents, prompts and the parallel orchestrator
│       ├── ingest/             # text extraction (PDF, text, images) and chunking
│       └── vector_store/       # embeddings and pgvector search
└── frontend/
    └── src/
        ├── api/                # one typed module per resource; schema.d.ts is generated
        ├── components/         # reusable UI, grouped by feature (ac, chat, uat, shared)
        ├── hooks/              # data fetching and state per page
        ├── pages/              # one component per route
        └── layouts/            # app shell, session sidebar and tabs
```

### Why this layering?

Keeping HTTP, business rules and database access apart means each can change
without the others, and services can be tested without a web server. New
backend features should follow the same three layers rather than putting
logic in a route handler. On the frontend, components call the typed modules
in `src/api/`, never `fetch` directly.

## Tests and code style

The backend tests need Postgres with pgvector on `localhost:5432` and an
`app_test` database. Create it once:
```
docker compose up -d db
docker compose exec db psql -U postgres -c "CREATE DATABASE app_test"
```
Set `TEST_DATABASE_URL` to use a different database.

| Task | Backend (`src/backend`) | Frontend (`src/frontend`) |
| --- | --- | --- |
| Format | `uv run ruff format` | `npm run format` |
| Lint | `uv run ruff check` | `npm run lint` (type check) |
| Test | `uv run pytest` | (none yet) |

`uv run python tests/run_tests.py` also writes a CSV and Markdown summary of
the test run. Naming, docstring and comment conventions are in
[docs/code-style.md](../docs/code-style.md).

## Regenerating the typed API client

When backend routes or models change, regenerate the frontend's types so
mismatches show up at compile time:
```
cd backend
uv run python scripts/export_openapi.py   # writes ../openapi.json

cd ../frontend
npm run generate-api                      # writes src/api/schema.d.ts
```
Commit both generated files with the backend change that caused them.

## Environment variables

See `.env.example` for the full list. `.env` is gitignored. Model choices
live in `backend/config/models.json` (also gitignored), which names the
environment variable holding each API key but never the key itself.

## Local Prometheus voting

Install and start Ollama, enable the `prometheus` judge in
`backend/config/models.json`, and pull its model (`ggozad/prometheus2:latest`
in the example file). Set its `base_url` to `http://localhost:11434/v1` when
the backend runs on the host, or `http://host.docker.internal:11434/v1` when
the backend runs in Docker.

To use the voting module from an MCP client, launch it over stdio from
`src/backend`:
```
uv run python -m voting.voting
```
The tool is named `evaluate_acceptance_criteria`. Prometheus scores each
output once per rubric (`correctness`, `coverage`, `relevance` and
`understandability`).
