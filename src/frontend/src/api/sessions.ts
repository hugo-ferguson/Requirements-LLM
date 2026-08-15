import { request } from "./client";
import type { paths } from "./schema";

export type SessionSummary =
  paths["/sessions"]["get"]["responses"]["200"]["content"]["application/json"][number];
export type SessionDetail =
  paths["/sessions/{session_id}"]["get"]["responses"]["200"]["content"]["application/json"];
export type MessageRead =
  paths["/sessions/{session_id}/messages"]["post"]["responses"]["201"]["content"]["application/json"];
export type GenerateResult =
  paths["/sessions/{session_id}/generate"]["post"]["responses"]["200"]["content"]["application/json"];
type MessageCreate =
  paths["/sessions/{session_id}/messages"]["post"]["requestBody"]["content"]["application/json"];

export const sessionsApi = {
  list: (): Promise<SessionSummary[]> => request<SessionSummary[]>("/sessions"),

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

  sendMessage: (sessionId: string, data: MessageCreate): Promise<MessageRead> =>
    request<MessageRead>(`/sessions/${sessionId}/messages`, {
      method: "POST",
      body: JSON.stringify(data),
    }),

  generate: (sessionId: string): Promise<GenerateResult> =>
    request<GenerateResult>(`/sessions/${sessionId}/generate`, { method: "POST" }),
};
