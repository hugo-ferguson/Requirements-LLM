import { useCallback, useState } from "react";
import { useNavigate, useParams } from "react-router";
import { AcGroupHeader } from "../components/uat/AcGroupHeader";
import { UatCard } from "../components/uat/UatCard";
import { ChatThread } from "../components/chat/ChatThread";
import { ChatInputBar } from "../components/chat/ChatInputBar";
import { uatCasesApi } from "../api/uatCases";
import type { UatCase } from "../api/uatCases";
import { useUatCases } from "../hooks/useUatCases";
import { useRegenerateSelected } from "../hooks/useRegenerateSelected";
import { ROUTES } from "../routes";
import type { Attachment } from "../types/conversation";

type Mode = "list" | "regenerate";

export function UatReviewPage() {
  const navigate = useNavigate();
  const { sessionId } = useParams<{ sessionId: string }>();
  const { groups, isLoading, loadError, updateText, updateStatus, applyApproved } = useUatCases(
    sessionId!,
  );

  const [expandedAcId, setExpandedAcId] = useState<number | null>(null);
  const [selectedUatId, setSelectedUatId] = useState<number | null>(null);
  const [editingUatId, setEditingUatId] = useState<number | null>(null);
  const [mode, setMode] = useState<Mode>("list");
  const [regenerateTargetId, setRegenerateTargetId] = useState<number | null>(null);
  const [isApplying, setIsApplying] = useState(false);
  const [applyError, setApplyError] = useState<string | null>(null);

  const fetchCandidates = useCallback(
    (
      targetId: number,
      messages: { role: "user" | "assistant"; text: string; attachments: Attachment[] }[],
    ) => uatCasesApi.regenerateSelected(sessionId!, targetId, { messages }),
    [sessionId],
  );
  const regenerate = useRegenerateSelected<UatCase>(regenerateTargetId, fetchCandidates);

  function handleToggleExpand(acId: number) {
    setExpandedAcId((prev) => (prev === acId ? null : acId));
    setSelectedUatId(null);
  }

  function handleSelect(uatId: number) {
    setSelectedUatId((prev) => (prev === uatId ? null : uatId));
  }

  function handleAccept(uatId: number, currentStatus: string) {
    updateStatus(uatId, currentStatus === "accepted" ? "pending" : "accepted");
  }

  function handleReject(uatId: number, currentStatus: string) {
    updateStatus(uatId, currentStatus === "rejected" ? "pending" : "rejected");
  }

  function handleEnterRegenerate() {
    if (selectedUatId == null) return;
    setRegenerateTargetId(selectedUatId);
    setMode("regenerate");
  }

  function handleCancelRegenerate() {
    regenerate.reset();
    setRegenerateTargetId(null);
    setSelectedUatId(null);
    setApplyError(null);
    setMode("list");
  }

  async function handleApplyApproved() {
    if (regenerateTargetId == null || regenerate.approvedCandidates.length === 0) return;
    setApplyError(null);
    setIsApplying(true);
    try {
      await applyApproved(regenerateTargetId, regenerate.approvedCandidates);
      regenerate.reset();
      setRegenerateTargetId(null);
      setSelectedUatId(null);
      setMode("list");
    } catch {
      setApplyError("Couldn't merge the approved UAT cases. Please try again.");
    } finally {
      setIsApplying(false);
    }
  }

  function handleContinue() {
    navigate(ROUTES.export(sessionId!));
  }

  if (isLoading) {
    return <div className="flex h-full items-center justify-center text-gray-500">Loading…</div>;
  }

  if (loadError) {
    return <div className="flex h-full items-center justify-center text-red-600">{loadError}</div>;
  }

  const totalAccepted = groups.reduce(
    (sum, group) => sum + group.uat_cases.filter((c) => c.status === "accepted").length,
    0,
  );

  if (mode === "regenerate" && regenerateTargetId != null) {
    const groupIndex = groups.findIndex((g) =>
      g.uat_cases.some((c) => c.id === regenerateTargetId),
    );
    const targetGroup = groups[groupIndex];
    const targetIndex = targetGroup?.uat_cases.findIndex((c) => c.id === regenerateTargetId) ?? -1;
    const target = targetGroup?.uat_cases[targetIndex];

    return (
      <div className="flex h-full flex-col overflow-y-auto p-6">
        <div className="space-y-3">
          {targetGroup && (
            <AcGroupHeader ac={targetGroup.ac} index={groupIndex} expanded onToggle={() => {}} />
          )}
          {target && (
            <UatCard
              uatCase={{ ...target, status: "rejected" }}
              index={targetIndex}
              onAccept={() => {}}
              onReject={() => {}}
            />
          )}
        </div>

        {regenerate.candidates.length > 0 && <hr className="my-4 border-gray-200" />}

        <div className="space-y-3">
          {regenerate.candidates.map((candidate, i) => {
            const displayIndex = i === 0 ? targetIndex : (targetGroup?.uat_cases.length ?? 0) + (i - 1);
            const approved = regenerate.approvedIndexes.has(i);
            return (
              <UatCard
                key={i}
                uatCase={{ ...candidate, status: approved ? "accepted" : "rejected" }}
                index={displayIndex}
                onAccept={() => regenerate.setApproved(i, true)}
                onReject={() => regenerate.setApproved(i, false)}
              />
            );
          })}
        </div>

        <div className="mt-4 flex-1">
          <ChatThread messages={regenerate.miniChatMessages} pending={regenerate.isSending} />
        </div>

        {regenerate.sendError && <p className="mb-2 text-sm text-red-600">{regenerate.sendError}</p>}
        <ChatInputBar
          pendingAttachments={regenerate.pendingAttachments}
          onAttachFiles={regenerate.attachFiles}
          onRemoveAttachment={regenerate.removeAttachment}
          onSend={regenerate.sendMiniChatMessage}
          disabled={regenerate.isSending}
        />

        {applyError && <p className="mt-2 text-sm text-red-600">{applyError}</p>}
        <div className="mt-4 flex justify-between border-t border-gray-200 pt-4">
          <button
            type="button"
            onClick={handleCancelRegenerate}
            className="rounded-full bg-gray-100 px-6 py-2 font-medium text-gray-600 hover:bg-gray-200"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleApplyApproved}
            disabled={isApplying || regenerate.approvedCandidates.length === 0}
            className="rounded-full bg-primary px-6 py-2 font-medium text-white hover:bg-primary-hover disabled:opacity-50"
          >
            Add Approved UAT Cases to List →
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col overflow-y-auto p-6">
      <div className="flex-1 space-y-3">
        {groups.map((group, index) => (
          <div key={group.ac.id}>
            <AcGroupHeader
              ac={group.ac}
              index={index}
              expanded={expandedAcId === group.ac.id}
              onToggle={() => handleToggleExpand(group.ac.id)}
            />
            {expandedAcId === group.ac.id && (
              <div className="mt-2 max-h-96 space-y-3 overflow-y-auto rounded-xl border border-gray-100 bg-gray-50/50 p-3">
                {group.uat_cases.map((uatCase, uatIndex) => (
                  <UatCard
                    key={uatCase.id}
                    uatCase={uatCase}
                    index={uatIndex}
                    selected={uatCase.id === selectedUatId}
                    editing={uatCase.id === editingUatId}
                    onSelect={() => handleSelect(uatCase.id)}
                    onEditStart={() => setEditingUatId(uatCase.id)}
                    onEditCancel={() => setEditingUatId(null)}
                    onEditSave={async (fields) => {
                      await updateText(uatCase.id, fields);
                      setEditingUatId(null);
                    }}
                    onAccept={() => handleAccept(uatCase.id, uatCase.status)}
                    onReject={() => handleReject(uatCase.id, uatCase.status)}
                  />
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="mt-4 flex items-center justify-between gap-3 border-t border-gray-200 pt-4">
        <button
          type="button"
          onClick={handleEnterRegenerate}
          disabled={selectedUatId == null}
          className="rounded-full bg-primary px-5 py-2 text-sm font-medium text-white hover:bg-primary-hover disabled:opacity-50"
        >
          + Add Context &amp; Regenerate Selected
        </button>
        <button
          type="button"
          onClick={handleContinue}
          disabled={totalAccepted === 0}
          className="rounded-full bg-primary px-5 py-2 text-sm font-medium text-white hover:bg-primary-hover disabled:opacity-50"
        >
          Continue →
        </button>
      </div>
    </div>
  );
}
