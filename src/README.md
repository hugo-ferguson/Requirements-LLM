# Requirements-LLM — App Skeleton

Base application skeleton for the FIT4002 client project. This is just the
foundation (backend, frontend, DB, containerization) — the AI pipeline
(user story → acceptance criteria → UAT test cases, via Pydantic AI) is not
implemented yet.

## Stack

| Layer    | Choice                                             |
| -------- | --------------------------------------------------- |
| Backend  | Python + FastAPI, layered (routes → services → repositories) |
| ORM      | SQLModel (Pydantic-based, shared model style with the future agent layer) |
| Database | PostgreSQL, one local container per developer (own local data, not shared) |
| Frontend | React + TypeScript (Vite), typed API client generated from OpenAPI |
| Auth     | None for v1 — data is session-scoped, not account-scoped |
| Infra    | Docker Compose (`frontend`, `backend`, `db`) |

## Running it

1. Copy the env template and adjust if needed:
   ```
   cp .env.example .env
   ```
2. Start everything:
   ```
   docker compose up --build
   ```
3. Open:
   - Frontend: http://localhost:5173
   - Backend docs (Swagger UI): http://localhost:8000/docs
   - Backend health check: http://localhost:8000/health

Postgres data persists across restarts in the `db_data` named volume. It's
local to your machine only — nobody else on the team sees your data, and you
don't see theirs. To wipe your local DB: `docker compose down -v`.

### Running without Docker (optional, for faster iteration)

Backend (requires [uv](https://docs.astral.sh/uv/)):
```
cd backend
uv sync
uv run uvicorn app.main:app --reload
```
You'll need a Postgres instance reachable at the `DATABASE_URL` in your
`.env` — either `docker compose up db` in another terminal, or point it at a
local Postgres install.

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
├── .env.example          # copy to .env (gitignored, per-developer)
├── backend/
│   ├── Dockerfile
│   ├── pyproject.toml    # deps, managed with uv
│   ├── scripts/
│   │   └── export_openapi.py   # dumps OpenAPI schema for the frontend codegen step
│   ├── tests/
│   └── app/
│       ├── main.py             # FastAPI app, middleware, router registration
│       ├── config.py           # env-driven settings (pydantic-settings)
│       ├── db.py                # engine + session setup
│       ├── models.py            # SQLModel table + request/response schemas
│       ├── routes/              # HTTP layer — request/response only, no business logic
│       ├── services/            # business logic, independent of HTTP and DB details
│       └── repositories/        # all direct DB access (SQLModel queries) lives here
└── frontend/
    ├── Dockerfile
    ├── package.json
    └── src/
        ├── api/
        │   ├── client.ts         # thin typed fetch wrapper
        │   ├── items.ts          # typed calls for the items resource
        │   └── schema.d.ts       # generated from the backend's OpenAPI schema — don't hand-edit
        ├── components/           # reusable UI (e.g. components/chat/)
        ├── App.tsx
        └── main.tsx
```

### Why this layering?

`routes/` should only ever deal with HTTP concerns (parsing the request,
calling a service, shaping the response). `services/` holds the actual
business rules and doesn't know about FastAPI or SQLModel. `repositories/`
is the only place that talks to the database. It's more structure than this
tiny example needs on its own, but it's the pattern to keep using as the
real feature set (acceptance criteria generation, test case generation)
gets added — new backend features should follow the same three layers
rather than putting logic directly in a route handler.

## The example resource: `items`

A minimal `id / name / created_at` resource used to prove the whole stack
end-to-end:

- `POST /items` — create an item (validated via the `ItemCreate` SQLModel schema)
- `GET /items` — list all items

The backend still exposes `items` as a minimal end-to-end layering
reference, but the frontend no longer has a dedicated Items screen (it's
been superseded by the Story2Spec pages under `frontend/src/pages/`). The
per-resource API module pattern it established — typed calls in
`frontend/src/api/*.ts`, no raw `fetch` calls in components — is still the
one to follow; see `frontend/src/api/conversation.ts` for the current
example.

## Regenerating the typed API client

Whenever backend routes or models change, refresh the frontend's generated
types so type errors show up at compile time instead of at runtime:

```
cd backend
uv run python scripts/export_openapi.py   # writes ../openapi.json

cd ../frontend
npm run generate-api                      # writes src/api/schema.d.ts
```

Commit the updated `schema.d.ts` (and `openapi.json`) alongside the backend
change that caused it.

## Environment variables

See `.env.example` for the full list.  Nothing sensitive is committed —
`.env` is gitignored. Later, the Pydantic AI agent's LLM API key will also
live in `.env` (a placeholder is already there).

## Out of scope for this skeleton

- The AI pipeline itself (acceptance criteria / UAT test case generation).
- Authentication.
- Deployment/hosting beyond local Docker.
