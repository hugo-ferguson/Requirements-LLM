import { request } from "./client";
import type { paths } from "./schema";

/** The export file, plus warnings to show before downloading it. */
export type ExportResponse =
  paths["/sessions/{session_id}/export"]["get"]["responses"]["200"]["content"]["application/json"];
/** The downloaded JSON: approved ACs, each with its approved UAT cases. */
export type ExportDocument = ExportResponse["document"];

/** Calls for exporting a session. */
export const exportApi = {
  get: (sessionId: string): Promise<ExportResponse> =>
    request<ExportResponse>(`/sessions/${sessionId}/export`),
};
