# MCP Tool Use — Proof of Concept

This folder explores how to build AI agents that use tools via the [Anthropic API](https://docs.anthropic.com/) and the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/). It progresses from a manual, hard-coded tool-use loop up to a fully MCP-connected agent.

---

## What's in here

### `server.py` — MCP Tool Server

A lightweight MCP server built with `FastMCP`. It exposes two tools that any MCP-compatible client can discover and call:

| Tool | Description |
|------|-------------|
| `add_numbers(a, b)` | Adds two integers and returns the result as a string |
| `get_weather(city)` | Returns a fake weather report for a given city |

The server communicates over **stdio** (standard input/output), which is the transport used by `looping_agent.py` to connect to it.

---

### `basic_agent.py` — Manual Tool-Use Loop (no MCP)

Demonstrates the raw Anthropic API tool-use cycle **without** MCP. Useful for understanding what's happening under the hood before introducing the MCP layer.

**Flow:**
1. Sends a user message (`"What is the weather like in Melbourne?"`) along with a tool definition for `get_weather`.
2. The model responds with a `tool_use` block — it's requesting that the tool be called.
3. The script executes the tool locally (a simple fake implementation).
4. The tool result is fed back into the conversation.
5. A second API call is made, and the model produces a natural-language final answer.

This is entirely self-contained — no MCP server needed.

---

### `looping_agent.py` — MCP-Connected Agentic Loop

Builds on the concepts in `basic_agent.py` by replacing the hard-coded tool definitions with **dynamic tool discovery from the MCP server**, and wrapping everything in an **agent loop** that keeps running until the model signals it's done (`end_turn`).

**Flow:**
1. Starts the MCP server as a subprocess (via stdio transport).
2. Asks the server what tools are available (`session.list_tools()`).
3. Converts the MCP tool definitions into the format expected by the Anthropic API.
4. Sends a user message asking two things: `"What is 42 + 58? And what is the weather in Melbourne?"`
5. Enters a loop:
   - If `stop_reason == "tool_use"`: calls the relevant tool via the MCP server and feeds the result back.
   - If `stop_reason == "end_turn"`: prints the final answer and exits.

The model may call multiple tools across multiple turns before finishing.

---

## Local Setup

### Prerequisites

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Mac / Linux:**
```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `anthropic` — Anthropic Python SDK
- `mcp[cli]` — MCP Python SDK (includes the `FastMCP` server framework and client)
- `python-dotenv` — loads environment variables from `.env`

### 3. Set up your API key

Create a `.env` file in this directory (it's already in `.gitignore`):

```
ANTHROPIC_API_KEY=your-api-key-here
```

---

## Running the agents

### `basic_agent.py` — no server needed

Just run it directly:

```bash
python basic_agent.py
```

Expected output: three printed steps showing the tool call request, the fake result, and the model's final answer.

---

### `looping_agent.py` — requires the MCP server

`looping_agent.py` **launches `server.py` automatically** as a subprocess when it runs, so you do **not** need to start the server separately. Just run:

```bash
python looping_agent.py
```

Expected output:
- A list of tools discovered from the MCP server
- Per-turn logs of which tool was called and what it returned
- The model's final answer once it has all the information it needs

> If you ever want to run the server on its own to inspect or debug it, you can:
> ```bash
> python server.py
> ```
> It will start listening on stdio and wait for MCP client messages.

---

## Key concepts

| Concept | Where it appears |
|---------|-----------------|
| Tool definition (Anthropic format) | `basic_agent.py` — defined inline as a dict |
| Tool-use response handling | `basic_agent.py` — manual two-step API call |
| MCP server with `FastMCP` | `server.py` — `@mcp.tool()` decorator |
| Dynamic tool discovery | `looping_agent.py` — `session.list_tools()` |
| Agentic loop (`tool_use` / `end_turn`) | `looping_agent.py` — `while True` loop |
| stdio MCP transport | `looping_agent.py` — `StdioServerParameters` |
