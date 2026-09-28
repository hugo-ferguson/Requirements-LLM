# Story2Spec (Requirements-LLM)

Turns a user story (chat, PDF, text or image) into Gherkin acceptance criteria and UAT test
cases, written by Claude and OpenAI and scored by both. Runs locally in Docker.

## Quick start — this is all you need

**You need:** Docker Desktop (with at least 6 GB of memory), Git, an
[Anthropic API key](https://console.anthropic.com/) and an
[OpenAI API key](https://platform.openai.com/api-keys).

**1. Get the code and go to the `src` folder.** Every command in this README runs from `src`.

```bash
git clone <repository-url>
cd Requirements-LLM/src
```

**2. Copy the two template files.**

```bash
# macOS / Linux
cp .env.example .env
cp backend/config/generation_agents.example.json backend/config/generation_agents.json
```

```powershell
# Windows PowerShell
Copy-Item .env.example .env
Copy-Item backend\config\generation_agents.example.json backend\config\generation_agents.json
```

**3. Put your two API keys in `src/.env`** — they're the first two lines to fill in.
Nothing else in either file needs changing.

```ini
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

**4. Start it.**

```bash
docker compose up --build -d
docker compose logs -f backend
```

The first start takes a few minutes. When the log shows **`Application startup complete.`**,
press Ctrl+C (this only stops the log, not the app) and open **<http://localhost:5173>**.

**Stop:** `docker compose down`  **Start again later:** `docker compose up -d`

---

*Everything below is reference — you only need it if something goes wrong or you want to change
how the app behaves.*

---

## Using the app

1. **+ New Session** in the sidebar.
2. **Input** — describe the feature in chat and/or attach files (PDF, `.txt`, `.md`, images).
   The assistant asks follow-up questions until it has enough context.
3. **Generate** acceptance criteria (takes up to about a minute). Each one shows its scores
   (1–5 on relevance, correctness, understandability and coverage) and which model wrote it.
   - **▾ other version** — see the other model's version and **Use this version** to swap it in
     (you can swap back).
   - Accept / reject each criterion, edit it (✎), or select one and **Add Context &
     Regenerate Selected**.
4. **Generate test cases from N approved →** turns the accepted criteria into UAT test cases,
   which you review the same way on the UAT page. **Regenerate All AC** starts over from the chat.

## Everyday commands

Run from `src`:

| Task | Command |
| --- | --- |
| Start | `docker compose up -d` |
| Stop (data is kept) | `docker compose down` |
| Watch all logs | `docker compose logs -f` (add `backend` for just the backend) |
| Apply a change to `.env` | `docker compose up -d --force-recreate backend` |
| Apply a change to `generation_agents.json` | `docker compose restart backend` |
| After pulling new code | `docker compose up --build -d` |
| **Delete all data** (sessions, criteria, uploads) | `docker compose down -v` |

Other addresses: backend health check <http://localhost:8000/health>, API docs
<http://localhost:8000/docs>.

## Configuration

All settings live in `src/.env`, explained there. The ones you're most likely to change:

- **Models that write criteria and test cases:** `backend/config/generation_agents.json`.
  Every entry with `"enabled": true` writes its own version; the **first** enabled entry
  decides *which* criteria / test cases exist and the others write their version of each.
- **Chat model:** `LLM_MODEL`. **Image reading:** `VISION_MODEL`.
- **Scoring judges:** `VOTING_JUDGES`. By default Claude and OpenAI both score everything and
  the scores are averaged; `VOTING_JUDGES=claude` uses one judge and roughly halves scoring cost.
- **Cost:** each generation calls every enabled model, and every candidate is scored by every
  judge. Fewer enabled models or judges means fewer API calls.

## Troubleshooting

**Nothing loads, and `docker` commands fail with `500 Internal Server Error … dockerDesktopLinuxEngine`**
Docker Desktop's virtual machine has stopped (most often from running out of memory). Quit
Docker Desktop completely from the tray icon, then:
```powershell
wsl --shutdown          # Windows only
```
Start Docker Desktop again, wait for "Engine running", then `docker compose up -d`.

**Memory (Windows).** Docker on Windows runs inside WSL, which by default may use up to half your
RAM and can be killed when the machine runs low. On a 16 GB machine, cap it: create
`C:\Users\<you>\.wslconfig` with

```ini
[wsl2]
memory=6GB
```

then run `wsl --shutdown` and restart Docker Desktop. On macOS, set memory under Docker Desktop
→ Settings → Resources.

**Generation fails with "Generation roster file not found"** — the second copy in step 2 was
skipped. Copy `generation_agents.example.json` to `generation_agents.json`, then
`docker compose restart backend`.

**Every score shows 0.0** — the judges couldn't be reached. Check both API keys in `src/.env`,
then `docker compose up -d --force-recreate backend`. The backend log names the failing judge.

**Only one model's versions appear, or the backend log says "Agent … needs ANTHROPIC_API_KEY /
OPENAI_API_KEY set in the environment"** — that key is missing, or the backend wasn't recreated
after adding it: `docker compose up -d --force-recreate backend`.

**A change to `.env` seems ignored** — `.env` is only read when the container is created, so a
plain restart isn't enough; use `docker compose up -d --force-recreate backend`.

**Upload stays on "Reading file…"** — check `docker compose logs -f backend` for lines starting
`ingest '<file>'`; the last one shows which step it reached. Uploads give up after 6 minutes.

**A port is already in use** — something else is using 5173, 8000 or 5432 (often a local
Postgres). Stop it, or change the left-hand port numbers under `ports:` in
`src/docker-compose.yml`.

## Current limitations

- **Export** is a placeholder page.
- Each model writes at most **8 acceptance criteria** per generation (`GENERATION_MAX_CRITERIA`);
  a story covering many features may not get criteria for all of them in one run.
- **Regenerate Selected** uses a single model (`LLM_MODEL`), so regenerated items have no
  alternative versions.
- No user accounts; all data stays on the machine running Docker.

## For developers

Architecture, tests and the typed API client: [`src/README.md`](src/README.md).
