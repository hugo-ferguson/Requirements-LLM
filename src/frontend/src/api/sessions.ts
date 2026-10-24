import { request } from "./client";
import type { paths } from "./schema";

/** A session as the sidebar lists it. */
export type SessionSummary =
  paths["/sessions"]["get"]["responses"]["200"]["content"]["application/json"][number];
type SessionDetail =
  paths["/sessions/{session_id}"]["get"]["responses"]["200"]["content"]["application/json"];
/** A saved chat message as the API returns it. */
export type MessageRead =
  paths["/sessions/{session_id}/messages"]["post"]["responses"]["201"]["content"]["application/json"];
type GenerateResult =
  paths["/sessions/{session_id}/generate"]["post"]["responses"]["200"]["content"]["application/json"];
type MessageCreate =
  paths["/sessions/{session_id}/messages"]["post"]["requestBody"]["content"]["application/json"];

/** Calls for sessions, their chat and AC generation. */
export const sessionsApi = {
  list: (): Promise<SessionSummary[]> => request<SessionSummary[]>("/sessions"),

  /** Creates a session. Without a name, the backend calls it "New session". */
  create: (name?: string): Promise<SessionSummary> =>
    request<SessionSummary>("/sessions", {
      method: "POST",
      body: JSON.stringify({ name: name ?? null }),
    }),

  get: (sessionId: string): Promise<SessionDetail> =>
    request<SessionDetail>(`/sessions/${sessionId}`),

  rename: (sessionId: string, name: string): Promise<SessionSummary> =>
    request<SessionSummary>(`/sessions/${sessionId}`, {
      method: "PATCH",
      body: JSON.stringify({ name }),
    }),

  remove: (sessionId: string): Promise<void> =>
    request<void>(`/sessions/${sessionId}`, { method: "DELETE" }),

  /** Saves the user's message and resolves to the assistant's reply. */
  sendMessage: (sessionId: string, message: MessageCreate): Promise<MessageRead> =>
    request<MessageRead>(`/sessions/${sessionId}/messages`, {
      method: "POST",
      body: JSON.stringify(message),
    }),

  /** Generates ACs from the chat. Replaces existing ACs and deletes their UAT cases. */
  generate: (sessionId: string): Promise<GenerateResult> =>
    request<GenerateResult>(`/sessions/${sessionId}/generate`, { method: "POST" }),
};
