import { useCallback, useState } from "react";
import { useNavigate, useParams } from "react-router";
import { AcGroupHeader } from "../components/uat/AcGroupHeader";
import { UatCard } from "../components/uat/UatCard";
import { ChatThread } from "../components/chat/ChatThread";
import { ChatInputBar } from "../components/chat/ChatInputBar";
import { uatCasesApi } from "../api/uatCases";
import type { UatCase, UatCaseStatus, UatCaseTextUpdate } from "../api/uatCases";
import { useUatCases } from "../hooks/useUatCases";
import { useRegenerateSelected } from "../hooks/useRegenerateSelected";
import type { RegenerateMessage } from "../hooks/useRegenerateSelected";
import { ROUTES } from "../routes";

type Mode = "list" | "regenerate";

/** The UAT Review tab: review and regenerate test cases, grouped by AC. */
export function UatReviewPage() {
  const navigate = useNavigate();
  const { sessionId } = useParams<{ sessionId: string }>();
  const {
    groups,
    isLoading,
    loadError,
    updateText,
    updateStatus,
    applyApproved,
    selectAlternative,
  } = useUatCases(sessionId!);

  const [expandedAcId, setExpandedAcId] = useState<number | null>(null);
  const [selectedUatId, setSelectedUatId] = useState<number | null>(null);
  const [editingUatId, setEditingUatId] = useState<number | null>(null);
  const [mode, setMode] = useState<Mode>("list");
  const [regenerateTargetId, setRegenerateTargetId] = useState<number | null>(null);
  const [isApplying, setIsApplying] = useState(false);
  const [applyError, setApplyError] = useState<string | null>(null);
  const [swapError, setSwapError] = useState<string | null>(null);

  const fetchCandidates = useCallback(
    (targetId: number, messages: RegenerateMessage[]) =>
      uatCasesApi.regenerateSelected(sessionId!, targetId, { messages }),
    [sessionId],
  );
  const regenerate = useRegenerateSelected<UatCase>(regenerateTargetId, fetchCandidates);

  function handleToggleExpand(acId: number) {
    setExpandedAcId((previous) => (previous === acId ? null : acId));
    setSelectedUatId(null);
  }

  function handleSelect(uatId: number) {
    setSelectedUatId((previous) => (previous === uatId ? null : uatId));
  }

  function handleAccept(uatId: number, currentStatus: UatCaseStatus) {
    updateStatus(uatId, currentStatus === "accepted" ? "pending" : "accepted");
  }

  function handleReject(uatId: number, currentStatus: UatCaseStatus) {
    updateStatus(uatId, currentStatus === "rejected" ? "pending" : "rejected");
  }

  async function handleEditSave(uatId: number, fields: UatCaseTextUpdate) {
    await updateText(uatId, fields);
    setEditingUatId(null);
  }

  async function handleSelectAlternative(uatId: number, candidateId: number) {
    setSwapError(null);
    try {
      await selectAlternative(uatId, candidateId);
    } catch {
      setSwapError("Couldn't switch to that version. Please try again.");
    }
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

  const acceptedCount = groups.reduce(
    (total, group) =>
      total + group.uat_cases.filter((uatCase) => uatCase.status === "accepted").length,
    0,
  );

  if (mode === "regenerate" && regenerateTargetId != null) {
    const groupIndex = groups.findIndex((group) =>
      group.uat_cases.some((uatCase) => uatCase.id === regenerateTargetId),
    );
    const targetGroup = groups[groupIndex];
    const targetIndex =
      targetGroup?.uat_cases.findIndex((uatCase) => uatCase.id === regenerateTargetId) ?? -1;
    const target = targetGroup?.uat_cases[targetIndex];

    return (
      <div className="flex h-full flex-col overflow-y-auto p-6">
        <div className="space-y-3">
          {targetGroup && (
            <AcGroupHeader
              criterion={targetGroup.ac}
              index={groupIndex}
              isExpanded
              onToggle={() => {}}
            />
          )}
          {target && (
            // Read-only, shown as rejected because the candidates replace it.
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
          {regenerate.candidates.map((candidate, candidateIndex) => {
            const displayIndex =
              candidateIndex === 0
                ? targetIndex
                : (targetGroup?.uat_cases.length ?? 0) + (candidateIndex - 1);
            const isApproved = regenerate.approvedIndexes.has(candidateIndex);
            return (
              <UatCard
                key={candidateIndex}
                uatCase={{ ...candidate, status: isApproved ? "accepted" : "rejected" }}
                index={displayIndex}
                onAccept={() => regenerate.setApproved(candidateIndex, true)}
                onReject={() => regenerate.setApproved(candidateIndex, false)}
              />
            );
          })}
        </div>

        <div className="mt-4 flex-1">
          <ChatThread
            messages={regenerate.miniChatMessages}
            isAwaitingReply={regenerate.isSending}
          />
        </div>

        {regenerate.uploadError && (
          <p className="mb-2 text-sm text-red-600">{regenerate.uploadError}</p>
        )}
        {regenerate.sendError && (
          <p className="mb-2 text-sm text-red-600">{regenerate.sendError}</p>
        )}
        <ChatInputBar
          pendingAttachments={regenerate.pendingAttachments}
          onAttachFiles={regenerate.attachFiles}
          onRemoveAttachment={regenerate.removeAttachment}
          onSend={regenerate.sendMiniChatMessage}
          disabled={regenerate.isSending}
          isUploading={regenerate.isUploading}
          uploadCount={regenerate.uploadCount}
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
      {swapError && <p className="mb-3 text-sm text-red-600">{swapError}</p>}
      <div className="flex-1 space-y-3">
        {groups.map((group, index) => (
          <div key={group.ac.id}>
            <AcGroupHeader
              criterion={group.ac}
              index={index}
              isExpanded={expandedAcId === group.ac.id}
              onToggle={() => handleToggleExpand(group.ac.id)}
            />
            {expandedAcId === group.ac.id && (
              <div className="mt-2 max-h-96 space-y-3 overflow-y-auto rounded-xl border border-gray-100 bg-gray-50/50 p-3">
                {group.uat_cases.map((uatCase, uatIndex) => (
                  <UatCard
                    key={uatCase.id}
                    uatCase={uatCase}
                    index={uatIndex}
                    isSelected={uatCase.id === selectedUatId}
                    isEditing={uatCase.id === editingUatId}
                    onSelect={() => handleSelect(uatCase.id)}
                    onEditStart={() => setEditingUatId(uatCase.id)}
                    onEditCancel={() => setEditingUatId(null)}
                    onEditSave={(fields) => handleEditSave(uatCase.id, fields)}
                    onAccept={() => handleAccept(uatCase.id, uatCase.status)}
                    onReject={() => handleReject(uatCase.id, uatCase.status)}
                    onSelectAlternative={(candidateId) =>
                      handleSelectAlternative(uatCase.id, candidateId)
                    }
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
          disabled={acceptedCount === 0}
          className="rounded-full bg-primary px-5 py-2 text-sm font-medium text-white hover:bg-primary-hover disabled:opacity-50"
        >
          Continue →
        </button>
      </div>
    </div>
  );
}
