import { request } from "./client";
import type { paths } from "./schema";

export type ExportResponse =
  paths["/sessions/{session_id}/export"]["get"]["responses"]["200"]["content"]["application/json"];
export type ExportDocument = ExportResponse["document"];

export const exportApi = {
  get: (sessionId: string): Promise<ExportResponse> =>
    request<ExportResponse>(`/sessions/${sessionId}/export`),
};
