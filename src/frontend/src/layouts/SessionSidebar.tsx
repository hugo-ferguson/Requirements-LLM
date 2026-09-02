import { useState } from "react";
import type { KeyboardEvent } from "react";
import { useNavigate, useParams } from "react-router";
import { useSessions } from "../context/SessionsContext";
import type { SessionSummary } from "../api/sessions";
import { ROUTES } from "../routes";

interface SessionRowProps {
  session: SessionSummary;
  isActive: boolean;
}

function SessionRow({ session, isActive }: SessionRowProps) {
  const navigate = useNavigate();
  const { renameSession, deleteSession } = useSessions();
  const [isRenaming, setIsRenaming] = useState(false);
  const [draftName, setDraftName] = useState(session.name);

  function startRename() {
    setDraftName(session.name);
    setIsRenaming(true);
  }

  async function commitRename() {
    setIsRenaming(false);
    const trimmed = draftName.trim();
    if (trimmed && trimmed !== session.name) {
      await renameSession(String(session.id), trimmed);
    }
  }

  function handleRenameKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "Enter") {
      event.preventDefault();
      commitRename();
    } else if (event.key === "Escape") {
      event.preventDefault();
      // Reset the draft before closing: removing a focused input from the DOM
      // fires a native blur, which would otherwise re-trigger commitRename
      // with the (discarded) edited text still in state.
      setDraftName(session.name);
      setIsRenaming(false);
    }
  }

  async function handleDelete() {
    if (!window.confirm(`Delete "${session.name}"? This can't be undone.`)) return;
    await deleteSession(String(session.id));
    if (isActive) navigate(ROUTES.root);
  }

  if (isRenaming) {
    return (
      <input
        autoFocus
        value={draftName}
        onChange={(event) => setDraftName(event.target.value)}
        onBlur={commitRename}
        onKeyDown={handleRenameKeyDown}
        className="rounded-lg border border-primary px-4 py-2 text-sm outline-none"
      />
    );
  }

  return (
    <div
      className={`group flex items-center justify-between rounded-lg px-4 py-2 text-sm ${
        isActive ? "bg-primary text-white" : "text-gray-700 hover:bg-gray-100"
      }`}
    >
      <button
        type="button"
        onClick={() => navigate(ROUTES.input(session.id))}
        className="min-w-0 flex-1 truncate text-left"
        title={session.name}
      >
        {session.name}
      </button>
      <span className="ml-2 hidden shrink-0 gap-1 group-hover:flex">
        <button
          type="button"
          onClick={startRename}
          aria-label="Rename session"
          className={isActive ? "text-white/80 hover:text-white" : "text-gray-500 hover:text-gray-900"}
        >
          ✎
        </button>
        <button
          type="button"
          onClick={handleDelete}
          aria-label="Delete session"
          className={isActive ? "text-white/80 hover:text-white" : "text-gray-500 hover:text-gray-900"}
        >
          🗑
        </button>
      </span>
    </div>
  );
}

export function SessionSidebar() {
  const navigate = useNavigate();
  const { sessionId: activeSessionId } = useParams<{ sessionId: string }>();
  const { sessions, isLoading, error, createSession } = useSessions();

  async function handleCreateSession() {
    const created = await createSession();
    navigate(ROUTES.input(created.id));
  }

  return (
    <aside className="flex w-64 shrink-0 flex-col gap-4 border-r border-gray-200 p-6">
      <h1 className="text-lg font-bold tracking-wide text-gray-900">STORY2SPEC</h1>
      <button
        type="button"
        onClick={handleCreateSession}
        className="rounded-full bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-hover"
      >
        + New Session
      </button>
      <nav className="flex flex-1 flex-col gap-1 overflow-y-auto">
        {isLoading && <p className="px-4 py-2 text-sm text-gray-500">Loading sessions…</p>}
        {error && <p className="px-4 py-2 text-sm text-red-600">{error}</p>}
        {!isLoading && !error && sessions.length === 0 && (
          <p className="px-4 py-2 text-sm text-gray-500">No sessions yet.</p>
        )}
        {sessions.map((session) => (
          <SessionRow
            key={session.id}
            session={session}
            isActive={String(session.id) === activeSessionId}
          />
        ))}
      </nav>
    </aside>
  );
}
