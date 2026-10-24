import { request } from "./client";
import type { paths } from "./schema";

/** One acceptance criterion as the API returns it. */
export type AcceptanceCriterion =
  paths["/sessions/{session_id}/acceptance-criteria"]["get"]["responses"]["200"]["content"]["application/json"]["acceptance_criteria"][number];
type AcceptanceCriteriaList =
  paths["/sessions/{session_id}/acceptance-criteria"]["get"]["responses"]["200"]["content"]["application/json"];
/** The parts of an acceptance criterion the user can edit. */
export type AcceptanceCriterionTextUpdate =
  paths["/sessions/{session_id}/acceptance-criteria/{ac_id}"]["patch"]["requestBody"]["content"]["application/json"];
type AcceptanceCriterionStatusUpdate =
  paths["/sessions/{session_id}/acceptance-criteria/{ac_id}/status"]["patch"]["requestBody"]["content"]["application/json"];
/** Where an acceptance criterion is in review. */
export type AcceptanceCriterionStatus = AcceptanceCriterionStatusUpdate["status"];
type RegenerateSelectedRequest =
  paths["/sessions/{session_id}/acceptance-criteria/{ac_id}/regenerate"]["post"]["requestBody"]["content"]["application/json"];
type RegenerateSelectedResponse =
  paths["/sessions/{session_id}/acceptance-criteria/{ac_id}/regenerate"]["post"]["responses"]["200"]["content"]["application/json"];
type ApplyApprovedRequest =
  paths["/sessions/{session_id}/acceptance-criteria/{ac_id}/apply-approved"]["post"]["requestBody"]["content"]["application/json"];
type RegenerateAllKickoffMessage =
  paths["/sessions/{session_id}/acceptance-criteria/regenerate-all-kickoff"]["post"]["responses"]["201"]["content"]["application/json"];
type SelectAlternativeResponse =
  paths["/sessions/{session_id}/acceptance-criteria/{ac_id}/alternatives/{candidate_id}/select"]["post"]["responses"]["200"]["content"]["application/json"];

/** Calls for a session's acceptance criteria. */
export const acceptanceCriteriaApi = {
  list: (sessionId: string): Promise<AcceptanceCriteriaList> =>
    request<AcceptanceCriteriaList>(`/sessions/${sessionId}/acceptance-criteria`),

  updateText: (
    sessionId: string,
    acId: number,
    fields: AcceptanceCriterionTextUpdate,
  ): Promise<AcceptanceCriterion> =>
    request<AcceptanceCriterion>(`/sessions/${sessionId}/acceptance-criteria/${acId}`, {
      method: "PATCH",
      body: JSON.stringify(fields),
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

  /** Regenerates one AC using the mini-chat's context. */
  regenerateSelected: (
    sessionId: string,
    acId: number,
    conversation: RegenerateSelectedRequest,
  ): Promise<RegenerateSelectedResponse> =>
    request<RegenerateSelectedResponse>(
      `/sessions/${sessionId}/acceptance-criteria/${acId}/regenerate`,
      { method: "POST", body: JSON.stringify(conversation) },
    ),

  /** Replaces one AC with the approved candidates and deletes its UAT cases. */
  applyApproved: (
    sessionId: string,
    acId: number,
    candidates: AcceptanceCriterion[],
  ): Promise<AcceptanceCriteriaList> =>
    request<AcceptanceCriteriaList>(
      `/sessions/${sessionId}/acceptance-criteria/${acId}/apply-approved`,
      { method: "POST", body: JSON.stringify({ candidates } satisfies ApplyApprovedRequest) },
    ),

  /** Swaps in another model's version of an AC. */
  selectAlternative: (
    sessionId: string,
    acId: number,
    candidateId: number,
  ): Promise<SelectAlternativeResponse> =>
    request<SelectAlternativeResponse>(
      `/sessions/${sessionId}/acceptance-criteria/${acId}/alternatives/${candidateId}/select`,
      { method: "POST" },
    ),

  /** Posts an Input chat message asking for context to regenerate all ACs. */
  regenerateAllKickoff: (sessionId: string): Promise<RegenerateAllKickoffMessage> =>
    request<RegenerateAllKickoffMessage>(
      `/sessions/${sessionId}/acceptance-criteria/regenerate-all-kickoff`,
      { method: "POST" },
    ),
};
