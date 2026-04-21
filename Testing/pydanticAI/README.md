# PydanticAI Proof-of-Concept

A self-contained spike demonstrating the core PydanticAI concepts that underpin the Project 30 ensemble acceptance criteria generation pipeline.

---

## What this spike covers

| Concept | What it shows |
|---|---|
| **Structured output** | `output_type` forces the LLM to return a validated Pydantic object — not a string. Invalid responses are retried automatically. |
| **Dependency injection** | `RunContext` passes shared state (project name, domain context) into an agent at call time without hardcoding it. |
| **Ensemble agents** | Three agents, three models, one shared output contract. Swapping a model is a one-line change. |
| **Parallel dispatch** | `asyncio.gather` runs all agents concurrently — no framework required. |
| **Voting** | A plain Python function selects the best output from anonymous agent results. |

---

## Setup

**1. Install dependencies**
```bash
pip install pydantic-ai python-dotenv
```

**2. Create a `.env` file** in the same directory as the spike:
```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
```

You do not need all three keys. If you only have one, see [Running with fewer API keys](#running-with-fewer-api-keys) below.

**3. Run the spike**
```bash
python pydanticAI-spike.py
```

---

## Expected output

```
════════════════════════════════════════════════════════════════════
Project 30 — PydanticAI Spike
════════════════════════════════════════════════════════════════════

User story:
  As a learner, I want to access the course dashboard when I select
  the dashboard tab, so I can view relevant course information and updates.

── Dispatching to ensemble (parallel) ──────────────────────────

  [GPT-4o] — 3 criterion/a
    AC1  Given: ...
         When:  ...
         Then:  ...
    ...

  [Gemini 3 Flash] — 3 criterion/a
    AC1  Given: ...
    ...

── Voting ───────────────────────────────────────────────────────
  Criterion counts per agent: [3, 3]. Majority count: 3.
  Winner summary: '...'

── Final accepted output ────────────────────────────────────────
  ...
```

---

## Running with fewer API keys

If you only have access to one or two models, comment out the agents you cannot use in the `ENSEMBLE` dictionary:

```python
ENSEMBLE = {
    "GPT-4o":         agent_gpt,
    # "Gemini 3 Flash": agent_gemini,   ← comment out if no key
}
```

To test the ensemble pattern without multiple API keys, you can point all agents at the same model:

```python
agent_gpt    = create_agent("openai:gpt-4o")
agent_gemini = create_agent("openai:gpt-4o")   # same model, different instance
```

The parallel dispatch and voting behaviour is identical — you just won't see model diversity in the outputs.

---

## Key things to notice when you run it

- `result.output` is an `AcceptanceCriteria` object, not a string. You can access `result.output.criteria[0].given` directly in code.
- The agents run in parallel. Total runtime is roughly the slowest single agent, not the sum of all three.
- The voting function receives a plain `list[AcceptanceCriteria]` with no model labels — it cannot tell which model produced which output. This is intentional.
- Changing the model for any agent requires editing exactly one string.