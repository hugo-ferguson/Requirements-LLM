import { request } from "./client";
import type { paths } from "./schema";

/** One UAT case as the API returns it. */
export type UatCase =
  paths["/sessions/{session_id}/uat-cases"]["get"]["responses"]["200"]["content"]["application/json"]["groups"][number]["uat_cases"][number];
/** An AC and its UAT cases. */
export type UatCaseGroup =
  paths["/sessions/{session_id}/uat-cases"]["get"]["responses"]["200"]["content"]["application/json"]["groups"][number];
type UatCaseGroupsResult =
  paths["/sessions/{session_id}/uat-cases"]["get"]["responses"]["200"]["content"]["application/json"];
/** The parts of a UAT case the user can edit. */
export type UatCaseTextUpdate =
  paths["/sessions/{session_id}/uat-cases/{uat_id}"]["patch"]["requestBody"]["content"]["application/json"];
type UatCaseStatusUpdate =
  paths["/sessions/{session_id}/uat-cases/{uat_id}/status"]["patch"]["requestBody"]["content"]["application/json"];
/** Where a UAT case is in review. */
export type UatCaseStatus = UatCaseStatusUpdate["status"];
type UatRegenerateSelectedRequest =
  paths["/sessions/{session_id}/uat-cases/{uat_id}/regenerate"]["post"]["requestBody"]["content"]["application/json"];
type UatRegenerateSelectedResponse =
  paths["/sessions/{session_id}/uat-cases/{uat_id}/regenerate"]["post"]["responses"]["200"]["content"]["application/json"];
type UatApplyApprovedRequest =
  paths["/sessions/{session_id}/uat-cases/{uat_id}/apply-approved"]["post"]["requestBody"]["content"]["application/json"];

/** Calls for a session's UAT cases, grouped by AC. */
export const uatCasesApi = {
  list: (sessionId: string): Promise<UatCaseGroupsResult> =>
    request<UatCaseGroupsResult>(`/sessions/${sessionId}/uat-cases`),

  /** Generates UAT cases for accepted ACs, replacing all existing cases. */
  generate: (sessionId: string): Promise<UatCaseGroupsResult> =>
    request<UatCaseGroupsResult>(`/sessions/${sessionId}/uat-cases/generate`, {
      method: "POST",
    }),

  updateText: (sessionId: string, uatId: number, fields: UatCaseTextUpdate): Promise<UatCase> =>
    request<UatCase>(`/sessions/${sessionId}/uat-cases/${uatId}`, {
      method: "PATCH",
      body: JSON.stringify(fields),
    }),

  updateStatus: (sessionId: string, uatId: number, status: UatCaseStatus): Promise<UatCase> =>
    request<UatCase>(`/sessions/${sessionId}/uat-cases/${uatId}/status`, {
      method: "PATCH",
      body: JSON.stringify({ status } satisfies UatCaseStatusUpdate),
    }),

  /** Regenerates one UAT case using the mini-chat's context. */
  regenerateSelected: (
    sessionId: string,
    uatId: number,
    conversation: UatRegenerateSelectedRequest,
  ): Promise<UatRegenerateSelectedResponse> =>
    request<UatRegenerateSelectedResponse>(`/sessions/${sessionId}/uat-cases/${uatId}/regenerate`, {
      method: "POST",
      body: JSON.stringify(conversation),
    }),

  /** Replaces one UAT case with the approved candidates and returns its AC's group. */
  applyApproved: (sessionId: string, uatId: number, candidates: UatCase[]): Promise<UatCaseGroup> =>
    request<UatCaseGroup>(`/sessions/${sessionId}/uat-cases/${uatId}/apply-approved`, {
      method: "POST",
      body: JSON.stringify({ candidates } satisfies UatApplyApprovedRequest),
    }),

  /** Swaps in another model's version of a UAT case. */
  selectAlternative: (sessionId: string, uatId: number, candidateId: number): Promise<UatCase> =>
    request<UatCase>(
      `/sessions/${sessionId}/uat-cases/${uatId}/alternatives/${candidateId}/select`,
      { method: "POST" },
    ),
};
