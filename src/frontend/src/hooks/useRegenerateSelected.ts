import { useCallback, useState } from "react";
import type { Attachment, ChatMessage } from "../types/conversation";
import { useAttachmentUploads } from "./useAttachmentUploads";

function newId(): string {
  return crypto.randomUUID();
}

interface WireMessage {
  role: "user" | "assistant";
  text: string;
  attachments: Attachment[];
}

interface FetchCandidatesResult<T> {
  reply: { text: string };
  candidates: T[];
}

type FetchCandidates<T> = (
  targetId: number,
  messages: WireMessage[],
) => Promise<FetchCandidatesResult<T>>;

/**
 * Owns the ephemeral "regenerate selected" mini-chat + candidate review state.
 * None of this is persisted — it's local to the sub-view and discarded on
 * Cancel (or once the approved candidates are merged into the main list).
 *
 * Generic over the candidate item type `T` so both AC's and UAT's
 * regenerate-selected sub-flows can share this hook — the actual API call is
 * injected via `fetchCandidates` rather than hard-wired to one endpoint.
 */
export function useRegenerateSelected<T>(
  targetId: number | null,
  fetchCandidates: FetchCandidates<T>,
) {
  const [miniChatMessages, setMiniChatMessages] = useState<ChatMessage[]>([]);
  const [candidates, setCandidates] = useState<T[]>([]);
  const [approvedIndexes, setApprovedIndexes] = useState<Set<number>>(new Set());
  const [isSending, setIsSending] = useState(false);
  const [sendError, setSendError] = useState<string | null>(null);
  const uploads = useAttachmentUploads();
  const { pendingAttachments, clearAttachments, discardAttachments } = uploads;

  const reset = useCallback(() => {
    setMiniChatMessages([]);
    // Cancelling the sub-flow throws its attachments away, so the documents
    // they ingested go too.
    discardAttachments();
    setCandidates([]);
    setApprovedIndexes(new Set());
    setSendError(null);
  }, [discardAttachments]);

  const sendMiniChatMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if ((!trimmed && pendingAttachments.length === 0) || targetId == null) return;

      const nextHistory = [
        ...miniChatMessages,
        { id: newId(), role: "user" as const, text: trimmed, attachments: pendingAttachments },
      ];
      setMiniChatMessages(nextHistory);
      clearAttachments();
      setSendError(null);
      setIsSending(true);
      try {
        const response = await fetchCandidates(
          targetId,
          nextHistory.map((m) => ({ role: m.role, text: m.text, attachments: m.attachments })),
        );
        setMiniChatMessages((prev) => [
          ...prev,
          { id: newId(), role: "assistant", text: response.reply.text, attachments: [] },
        ]);
        setCandidates(response.candidates);
        // Pre-tick every candidate, matching the mockup's already-approved appearance.
        setApprovedIndexes(new Set(response.candidates.map((_, index) => index)));
      } catch {
        setSendError("Couldn't reach the server. Please try again.");
      } finally {
        setIsSending(false);
      }
    },
    [targetId, fetchCandidates, miniChatMessages, pendingAttachments, clearAttachments],
  );

  const setApproved = useCallback((index: number, approved: boolean) => {
    setApprovedIndexes((prev) => {
      const next = new Set(prev);
      if (approved) next.add(index);
      else next.delete(index);
      return next;
    });
  }, []);

  const approvedCandidates = candidates.filter((_, index) => approvedIndexes.has(index));

  return {
    ...uploads,
    miniChatMessages,
    candidates,
    approvedIndexes,
    approvedCandidates,
    isSending,
    sendError,
    sendMiniChatMessage,
    setApproved,
    reset,
  };
}
