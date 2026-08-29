import { createContext, useCallback, useContext, useEffect, useState } from "react";
import type { ReactNode } from "react";
import { sessionsApi } from "../api/sessions";
import type { SessionSummary } from "../api/sessions";

interface SessionsContextValue {
  sessions: SessionSummary[];
  isLoading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  createSession: (name?: string) => Promise<SessionSummary>;
  renameSession: (sessionId: string, name: string) => Promise<void>;
  deleteSession: (sessionId: string) => Promise<void>;
}

const SessionsContext = createContext<SessionsContextValue | null>(null);

export function SessionsProvider({ children }: { children: ReactNode }) {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setError(null);
    try {
      const list = await sessionsApi.list();
      setSessions(list);
    } catch {
      setError("Couldn't load sessions. Please try again.");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const createSession = useCallback(
    async (name?: string) => {
      const created = await sessionsApi.create(name);
      await refresh();
      return created;
    },
    [refresh],
  );

  const renameSession = useCallback(
    async (sessionId: string, name: string) => {
      await sessionsApi.rename(sessionId, name);
      await refresh();
    },
    [refresh],
  );

  const deleteSession = useCallback(
    async (sessionId: string) => {
      await sessionsApi.remove(sessionId);
      await refresh();
    },
    [refresh],
  );

  return (
    <SessionsContext.Provider
      value={{ sessions, isLoading, error, refresh, createSession, renameSession, deleteSession }}
    >
      {children}
    </SessionsContext.Provider>
  );
}

export function useSessions(): SessionsContextValue {
  const context = useContext(SessionsContext);
  if (!context) {
    throw new Error("useSessions must be used within a SessionsProvider");
  }
  return context;
}
