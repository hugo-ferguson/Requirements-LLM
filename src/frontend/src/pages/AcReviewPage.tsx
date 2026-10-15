import { useCallback, useState } from "react";
import { useNavigate, useParams } from "react-router";
import { AcCard } from "../components/ac/AcCard";
import { ChatThread } from "../components/chat/ChatThread";
import { ChatInputBar } from "../components/chat/ChatInputBar";
import { useConfirm } from "../components/shared/ConfirmDialog";
import { acceptanceCriteriaApi } from "../api/acceptanceCriteria";
import type { AcceptanceCriterion } from "../api/acceptanceCriteria";
import { uatCasesApi } from "../api/uatCases";
import { useAcceptanceCriteria } from "../hooks/useAcceptanceCriteria";
import { useRegenerateSelected } from "../hooks/useRegenerateSelected";
import { ROUTES } from "../routes";
import type { Attachment } from "../types/conversation";

type Mode = "list" | "regenerate";

function testCases(count: number): string {
  return `${count} test case${count === 1 ? "" : "s"}`;
}

export function AcReviewPage() {
  const navigate = useNavigate();
  const { sessionId } = useParams<{ sessionId: string }>();
  const { items, isLoading, loadError, updateText, updateStatus, applyApproved, selectAlternative } =
    useAcceptanceCriteria(sessionId!);

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
    (
      targetId: number,
      messages: { role: "user" | "assistant"; text: string; attachments: Attachment[] }[],
    ) => acceptanceCriteriaApi.regenerateSelected(sessionId!, targetId, { messages }),
    [sessionId],
  );
  const regenerate = useRegenerateSelected<AcceptanceCriterion>(regenerateTargetId, fetchCandidates);

  function handleSelect(id: number) {
    setSelectedId((prev) => (prev === id ? null : id));
  }

  function handleAccept(id: number, currentStatus: string) {
    updateStatus(id, currentStatus === "accepted" ? "pending" : "accepted");
  }

  function handleReject(id: number, currentStatus: string) {
    updateStatus(id, currentStatus === "rejected" ? "pending" : "rejected");
  }

  /**
   * Asks before a change that leaves an AC's test cases describing wording it
   * no longer has. Resolves true straight away when the AC has none.
   */
  function confirmStaleUat(item: AcceptanceCriterion, index: number, action: string) {
    if (item.uat_case_count === 0) return Promise.resolve(true);
    return confirm({
      title: "This AC has test cases",
      message:
        `AC #${index + 1} has ${testCases(item.uat_case_count)} written against its current ` +
        `wording. ${action} will leave them out of date; you can regenerate them on the UAT page.`,
    });
  }

  async function handleEditSave(
    item: AcceptanceCriterion,
    index: number,
    fields: { title: string; given: string; when: string; then: string },
  ) {
    // Declining keeps the card in edit mode with the draft intact.
    if (!(await confirmStaleUat(item, index, "Saving this edit"))) return;
    await updateText(item.id, fields);
    setEditingId(null);
  }

  async function handleSelectAlternative(
    item: AcceptanceCriterion,
    index: number,
    candidateId: number,
  ) {
    setSwapError(null);
    if (!(await confirmStaleUat(item, index, "Switching to this version"))) return;
    try {
      await selectAlternative(item.id, candidateId);
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
    const targetIndex = items.findIndex((item) => item.id === regenerateTargetId);
    const target = items[targetIndex];
    // The backend deletes the replaced AC's test cases, so this one is not
    // just "out of date".
    if (target && target.uat_case_count > 0) {
      const confirmed = await confirm({
        title: "Delete this AC's test cases?",
        message:
          `Replacing AC #${targetIndex + 1} will delete its ${testCases(target.uat_case_count)}. ` +
          "This can't be undone.",
        confirmLabel: "Delete test cases & replace",
        destructive: true,
      });
      if (!confirmed) return;
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

  const acceptedCount = items.filter((item) => item.status === "accepted").length;

  if (mode === "regenerate" && regenerateTargetId != null) {
    const targetIndex = items.findIndex((item) => item.id === regenerateTargetId);
    const target = items[targetIndex];

    return (
      <div className="flex h-full flex-col overflow-y-auto p-6">
        <div className="space-y-3">
          {target && (
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
          {regenerate.candidates.map((candidate, i) => {
            const displayIndex = i === 0 ? targetIndex : items.length + (i - 1);
            const approved = regenerate.approvedIndexes.has(i);
            return (
              <AcCard
                key={i}
                criterion={{ ...candidate, status: approved ? "accepted" : "rejected" }}
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

        {regenerate.uploadError && (
          <p className="mb-2 text-sm text-red-600">{regenerate.uploadError}</p>
        )}
        {regenerate.sendError && <p className="mb-2 text-sm text-red-600">{regenerate.sendError}</p>}
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
        {items.map((item, index) => (
          <AcCard
            key={item.id}
            criterion={item}
            index={index}
            selected={item.id === selectedId}
            editing={item.id === editingId}
            onSelect={() => handleSelect(item.id)}
            onEditStart={() => setEditingId(item.id)}
            onEditCancel={() => setEditingId(null)}
            onEditSave={(fields) => handleEditSave(item, index, fields)}
            onAccept={() => handleAccept(item.id, item.status)}
            onReject={() => handleReject(item.id, item.status)}
            onSelectAlternative={(candidateId) =>
              handleSelectAlternative(item, index, candidateId)
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
