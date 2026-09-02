import { request } from "./client";
import type { paths } from "./schema";

export type UatCase =
  paths["/sessions/{session_id}/uat-cases"]["get"]["responses"]["200"]["content"]["application/json"]["groups"][number]["uat_cases"][number];
export type UatCaseGroup =
  paths["/sessions/{session_id}/uat-cases"]["get"]["responses"]["200"]["content"]["application/json"]["groups"][number];
export type UatCaseGroupsResult =
  paths["/sessions/{session_id}/uat-cases"]["get"]["responses"]["200"]["content"]["application/json"];
type UatCaseTextUpdate =
  paths["/sessions/{session_id}/uat-cases/{uat_id}"]["patch"]["requestBody"]["content"]["application/json"];
type UatCaseStatusUpdate =
  paths["/sessions/{session_id}/uat-cases/{uat_id}/status"]["patch"]["requestBody"]["content"]["application/json"];
export type UatCaseStatus = UatCaseStatusUpdate["status"];
type UatRegenerateSelectedRequest =
  paths["/sessions/{session_id}/uat-cases/{uat_id}/regenerate"]["post"]["requestBody"]["content"]["application/json"];
export type UatRegenerateSelectedResponse =
  paths["/sessions/{session_id}/uat-cases/{uat_id}/regenerate"]["post"]["responses"]["200"]["content"]["application/json"];
type UatApplyApprovedRequest =
  paths["/sessions/{session_id}/uat-cases/{uat_id}/apply-approved"]["post"]["requestBody"]["content"]["application/json"];

export const uatCasesApi = {
  list: (sessionId: string): Promise<UatCaseGroupsResult> =>
    request<UatCaseGroupsResult>(`/sessions/${sessionId}/uat-cases`),

  generate: (sessionId: string): Promise<UatCaseGroupsResult> =>
    request<UatCaseGroupsResult>(`/sessions/${sessionId}/uat-cases/generate`, {
      method: "POST",
    }),

  updateText: (sessionId: string, uatId: number, data: UatCaseTextUpdate): Promise<UatCase> =>
    request<UatCase>(`/sessions/${sessionId}/uat-cases/${uatId}`, {
      method: "PATCH",
      body: JSON.stringify(data),
    }),

  updateStatus: (sessionId: string, uatId: number, status: UatCaseStatus): Promise<UatCase> =>
    request<UatCase>(`/sessions/${sessionId}/uat-cases/${uatId}/status`, {
      method: "PATCH",
      body: JSON.stringify({ status } satisfies UatCaseStatusUpdate),
    }),

  regenerateSelected: (
    sessionId: string,
    uatId: number,
    data: UatRegenerateSelectedRequest,
  ): Promise<UatRegenerateSelectedResponse> =>
    request<UatRegenerateSelectedResponse>(`/sessions/${sessionId}/uat-cases/${uatId}/regenerate`, {
      method: "POST",
      body: JSON.stringify(data),
    }),

  applyApproved: (
    sessionId: string,
    uatId: number,
    candidates: UatCase[],
  ): Promise<UatCaseGroup> =>
    request<UatCaseGroup>(`/sessions/${sessionId}/uat-cases/${uatId}/apply-approved`, {
      method: "POST",
      body: JSON.stringify({ candidates } satisfies UatApplyApprovedRequest),
    }),
};
