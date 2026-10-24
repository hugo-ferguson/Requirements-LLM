import { useCallback, useState } from "react";
import { useNavigate, useParams } from "react-router";
import { AcCard } from "../components/ac/AcCard";
import { ChatThread } from "../components/chat/ChatThread";
import { ChatInputBar } from "../components/chat/ChatInputBar";
import { useConfirm } from "../components/shared/ConfirmDialog";
import { acceptanceCriteriaApi } from "../api/acceptanceCriteria";
import type {
  AcceptanceCriterion,
  AcceptanceCriterionStatus,
  AcceptanceCriterionTextUpdate,
} from "../api/acceptanceCriteria";
import { uatCasesApi } from "../api/uatCases";
import { useAcceptanceCriteria } from "../hooks/useAcceptanceCriteria";
import { useRegenerateSelected } from "../hooks/useRegenerateSelected";
import type { RegenerateMessage } from "../hooks/useRegenerateSelected";
import { ROUTES } from "../routes";

type Mode = "list" | "regenerate";

/** `1` -> "1 test case", `3` -> "3 test cases". */
function formatTestCaseCount(count: number): string {
  return `${count} test case${count === 1 ? "" : "s"}`;
}

/** The AC Review tab: review and regenerate ACs, then generate test cases. */
export function AcReviewPage() {
  const navigate = useNavigate();
  const { sessionId } = useParams<{ sessionId: string }>();
  const {
    criteria,
    isLoading,
    loadError,
    updateText,
    updateStatus,
    applyApproved,
    selectAlternative,
  } = useAcceptanceCriteria(sessionId!);

  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [mode, setMode] = useState<Mode>("list");
  const [regenerateTargetId, setRegenerateTargetId] = useState<number | null>(null);
  const [isApplying, setIsApplying] = useState(false);
  const [applyError, setApplyError] = useState<string | null>(null);
  const [isRegeneratingAll, setIsRegeneratingAll] = useState(false);
  const [regenerateAllError, setRegenerateAllError] = useState<string | null>(null);
  const [isGeneratingUat, setIsGeneratingUat] = useState(false);
  const [generateUatError, setGenerateUatError] = useState<string | null>(null);
  const [swapError, setSwapError] = useState<string | null>(null);
  const { confirm, dialog } = useConfirm();

  const fetchCandidates = useCallback(
    (targetId: number, messages: RegenerateMessage[]) =>
      acceptanceCriteriaApi.regenerateSelected(sessionId!, targetId, { messages }),
    [sessionId],
  );
  const regenerate = useRegenerateSelected<AcceptanceCriterion>(
    regenerateTargetId,
    fetchCandidates,
  );

  function handleSelect(criterionId: number) {
    setSelectedId((previous) => (previous === criterionId ? null : criterionId));
  }

  function handleAccept(criterionId: number, currentStatus: AcceptanceCriterionStatus) {
    updateStatus(criterionId, currentStatus === "accepted" ? "pending" : "accepted");
  }

  function handleReject(criterionId: number, currentStatus: AcceptanceCriterionStatus) {
    updateStatus(criterionId, currentStatus === "rejected" ? "pending" : "rejected");
  }

  // Warns that the AC's test cases will go out of date. True if it has none.
  function confirmStaleTestCases(criterion: AcceptanceCriterion, index: number, action: string) {
    if (criterion.uat_case_count === 0) return Promise.resolve(true);
    const testCasePhrase = formatTestCaseCount(criterion.uat_case_count);
    return confirm({
      title: "This AC has test cases",
      message:
        `AC #${index + 1} has ${testCasePhrase} written against its current ` +
        `wording. ${action} will leave them out of date; you can regenerate them on the UAT page.`,
    });
  }

  async function handleEditSave(
    criterion: AcceptanceCriterion,
    index: number,
    fields: AcceptanceCriterionTextUpdate,
  ) {
    // Declining keeps the card in edit mode with the draft intact.
    if (!(await confirmStaleTestCases(criterion, index, "Saving this edit"))) return;
    await updateText(criterion.id, fields);
    setEditingId(null);
  }

  async function handleSelectAlternative(
    criterion: AcceptanceCriterion,
    index: number,
    candidateId: number,
  ) {
    setSwapError(null);
    if (!(await confirmStaleTestCases(criterion, index, "Switching to this version"))) return;
    try {
      await selectAlternative(criterion.id, candidateId);
    } catch {
      setSwapError("Couldn't switch to that version. Please try again.");
    }
  }

  function handleEnterRegenerate() {
    if (selectedId == null) return;
    setRegenerateTargetId(selectedId);
    setMode("regenerate");
  }

  function handleCancelRegenerate() {
    regenerate.reset();
    setRegenerateTargetId(null);
    setSelectedId(null);
    setApplyError(null);
    setMode("list");
  }

  async function handleApplyApproved() {
    if (regenerateTargetId == null || regenerate.approvedCandidates.length === 0) return;
    const targetIndex = criteria.findIndex((criterion) => criterion.id === regenerateTargetId);
    const target = criteria[targetIndex];
    // The backend deletes the replaced AC's test cases.
    if (target && target.uat_case_count > 0) {
      const testCasePhrase = formatTestCaseCount(target.uat_case_count);
      const isConfirmed = await confirm({
        title: "Delete this AC's test cases?",
        message:
          `Replacing AC #${targetIndex + 1} will delete its ${testCasePhrase}. ` +
          "This can't be undone.",
        confirmLabel: "Delete test cases & replace",
        isDestructive: true,
      });
      if (!isConfirmed) return;
    }
    setApplyError(null);
    setIsApplying(true);
    try {
      await applyApproved(regenerateTargetId, regenerate.approvedCandidates);
      regenerate.reset();
      setRegenerateTargetId(null);
      setSelectedId(null);
      setMode("list");
    } catch {
      setApplyError("Couldn't merge the approved criteria. Please try again.");
    } finally {
      setIsApplying(false);
    }
  }

  async function handleRegenerateAll() {
    setRegenerateAllError(null);
    setIsRegeneratingAll(true);
    try {
      await acceptanceCriteriaApi.regenerateAllKickoff(sessionId!);
      navigate(ROUTES.input(sessionId!));
    } catch {
      setRegenerateAllError("Couldn't start over. Please try again.");
      setIsRegeneratingAll(false);
    }
  }

  async function handleGenerateUatCases() {
    setGenerateUatError(null);
    setIsGeneratingUat(true);
    try {
      await uatCasesApi.generate(sessionId!);
      navigate(ROUTES.uatReview(sessionId!));
    } catch {
      setGenerateUatError("Couldn't generate test cases. Please try again.");
      setIsGeneratingUat(false);
    }
  }

  if (isLoading) {
    return <div className="flex h-full items-center justify-center text-gray-500">Loading…</div>;
  }

  if (loadError) {
    return <div className="flex h-full items-center justify-center text-red-600">{loadError}</div>;
  }

  const acceptedCount = criteria.filter((criterion) => criterion.status === "accepted").length;

  if (mode === "regenerate" && regenerateTargetId != null) {
    const targetIndex = criteria.findIndex((criterion) => criterion.id === regenerateTargetId);
    const target = criteria[targetIndex];

    return (
      <div className="flex h-full flex-col overflow-y-auto p-6">
        <div className="space-y-3">
          {target && (
            // Read-only, shown as rejected because the candidates replace it.
            <AcCard
              criterion={{ ...target, status: "rejected" }}
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
              candidateIndex === 0 ? targetIndex : criteria.length + (candidateIndex - 1);
            const isApproved = regenerate.approvedIndexes.has(candidateIndex);
            return (
              <AcCard
                key={candidateIndex}
                criterion={{ ...candidate, status: isApproved ? "accepted" : "rejected" }}
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
        {dialog}
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
            Add Approved AC to AC review List →
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col overflow-y-auto p-6">
      {dialog}
      {swapError && <p className="mb-3 text-sm text-red-600">{swapError}</p>}
      <div className="flex-1 space-y-3">
        {criteria.map((criterion, index) => (
          <AcCard
            key={criterion.id}
            criterion={criterion}
            index={index}
            isSelected={criterion.id === selectedId}
            isEditing={criterion.id === editingId}
            onSelect={() => handleSelect(criterion.id)}
            onEditStart={() => setEditingId(criterion.id)}
            onEditCancel={() => setEditingId(null)}
            onEditSave={(fields) => handleEditSave(criterion, index, fields)}
            onAccept={() => handleAccept(criterion.id, criterion.status)}
            onReject={() => handleReject(criterion.id, criterion.status)}
            onSelectAlternative={(candidateId) =>
              handleSelectAlternative(criterion, index, candidateId)
            }
          />
        ))}
      </div>

      {regenerateAllError && <p className="mt-2 text-sm text-red-600">{regenerateAllError}</p>}
      <div className="mt-4 flex items-center justify-between gap-3 border-t border-gray-200 pt-4">
        <button
          type="button"
          onClick={handleEnterRegenerate}
          disabled={selectedId == null}
          className="rounded-full bg-primary px-5 py-2 text-sm font-medium text-white hover:bg-primary-hover disabled:opacity-50"
        >
          + Add Context &amp; Regenerate Selected
        </button>
        <button
          type="button"
          onClick={handleRegenerateAll}
          disabled={isRegeneratingAll}
          className="rounded-full bg-gray-100 px-5 py-2 text-sm font-medium text-gray-700 hover:bg-gray-200 disabled:opacity-50"
        >
          Regenerate All AC
        </button>
        <button
          type="button"
          onClick={handleGenerateUatCases}
          disabled={acceptedCount === 0 || isGeneratingUat}
          className="rounded-full bg-primary px-5 py-2 text-sm font-medium text-white hover:bg-primary-hover disabled:opacity-50"
        >
          Generate test cases from {acceptedCount} approved →
        </button>
      </div>
      {generateUatError && <p className="mt-2 text-sm text-red-600">{generateUatError}</p>}
    </div>
  );
}
