import { request } from "./client";
import type { paths } from "./schema";

export type AcceptanceCriterion =
  paths["/sessions/{session_id}/acceptance-criteria"]["get"]["responses"]["200"]["content"]["application/json"]["acceptance_criteria"][number];
export type AcceptanceCriteriaList =
  paths["/sessions/{session_id}/acceptance-criteria"]["get"]["responses"]["200"]["content"]["application/json"];
type AcceptanceCriterionTextUpdate =
  paths["/sessions/{session_id}/acceptance-criteria/{ac_id}"]["patch"]["requestBody"]["content"]["application/json"];
type AcceptanceCriterionStatusUpdate =
  paths["/sessions/{session_id}/acceptance-criteria/{ac_id}/status"]["patch"]["requestBody"]["content"]["application/json"];
export type AcceptanceCriterionStatus = AcceptanceCriterionStatusUpdate["status"];
type RegenerateSelectedRequest =
  paths["/sessions/{session_id}/acceptance-criteria/{ac_id}/regenerate"]["post"]["requestBody"]["content"]["application/json"];
export type RegenerateSelectedResponse =
  paths["/sessions/{session_id}/acceptance-criteria/{ac_id}/regenerate"]["post"]["responses"]["200"]["content"]["application/json"];
type ApplyApprovedRequest =
  paths["/sessions/{session_id}/acceptance-criteria/{ac_id}/apply-approved"]["post"]["requestBody"]["content"]["application/json"];
export type RegenerateAllKickoffMessage =
  paths["/sessions/{session_id}/acceptance-criteria/regenerate-all-kickoff"]["post"]["responses"]["201"]["content"]["application/json"];

export const acceptanceCriteriaApi = {
  list: (sessionId: string): Promise<AcceptanceCriteriaList> =>
    request<AcceptanceCriteriaList>(`/sessions/${sessionId}/acceptance-criteria`),

  updateText: (
    sessionId: string,
    acId: number,
    data: AcceptanceCriterionTextUpdate,
  ): Promise<AcceptanceCriterion> =>
    request<AcceptanceCriterion>(`/sessions/${sessionId}/acceptance-criteria/${acId}`, {
      method: "PATCH",
      body: JSON.stringify(data),
    }),

  updateStatus: (
    sessionId: string,
    acId: number,
    status: AcceptanceCriterionStatus,
  ): Promise<AcceptanceCriterion> =>
    request<AcceptanceCriterion>(`/sessions/${sessionId}/acceptance-criteria/${acId}/status`, {
      method: "PATCH",
      body: JSON.stringify({ status } satisfies AcceptanceCriterionStatusUpdate),
    }),

  regenerateSelected: (
    sessionId: string,
    acId: number,
    data: RegenerateSelectedRequest,
  ): Promise<RegenerateSelectedResponse> =>
    request<RegenerateSelectedResponse>(
      `/sessions/${sessionId}/acceptance-criteria/${acId}/regenerate`,
      { method: "POST", body: JSON.stringify(data) },
    ),

  applyApproved: (
    sessionId: string,
    acId: number,
    candidates: AcceptanceCriterion[],
  ): Promise<AcceptanceCriteriaList> =>
    request<AcceptanceCriteriaList>(
      `/sessions/${sessionId}/acceptance-criteria/${acId}/apply-approved`,
      { method: "POST", body: JSON.stringify({ candidates } satisfies ApplyApprovedRequest) },
    ),

  regenerateAllKickoff: (sessionId: string): Promise<RegenerateAllKickoffMessage> =>
    request<RegenerateAllKickoffMessage>(
      `/sessions/${sessionId}/acceptance-criteria/regenerate-all-kickoff`,
      { method: "POST" },
    ),
};
