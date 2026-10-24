import { useCallback, useState } from "react";
import type { ChatMessage } from "../types/conversation";
import { useAttachmentUploads } from "./useAttachmentUploads";

/** A mini-chat message in the form the regenerate endpoints take. */
export type RegenerateMessage = Omit<ChatMessage, "id">;

interface FetchCandidatesResult<Candidate> {
  reply: { text: string };
  candidates: Candidate[];
}

type FetchCandidates<Candidate> = (
  targetId: number,
  messages: RegenerateMessage[],
) => Promise<FetchCandidatesResult<Candidate>>;

/**
 * Holds the "regenerate selected" mini-chat and its candidates. Nothing here is saved.
 *
 * @param targetId - The item being regenerated. Sending does nothing while it's null.
 */
export function useRegenerateSelected<Candidate>(
  targetId: number | null,
  fetchCandidates: FetchCandidates<Candidate>,
) {
  const [miniChatMessages, setMiniChatMessages] = useState<ChatMessage[]>([]);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [approvedIndexes, setApprovedIndexes] = useState<Set<number>>(new Set());
  const [isSending, setIsSending] = useState(false);
  const [sendError, setSendError] = useState<string | null>(null);
  const uploads = useAttachmentUploads();
  const { pendingAttachments, clearAttachments, discardAttachments } = uploads;

  const reset = useCallback(() => {
    setMiniChatMessages([]);
    // Also deletes the attachments' documents.
    discardAttachments();
    setCandidates([]);
    setApprovedIndexes(new Set());
    setSendError(null);
  }, [discardAttachments]);

  const sendMiniChatMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if ((!trimmed && pendingAttachments.length === 0) || targetId == null) return;

      const nextHistory: ChatMessage[] = [
        ...miniChatMessages,
        { id: crypto.randomUUID(), role: "user", text: trimmed, attachments: pendingAttachments },
      ];
      setMiniChatMessages(nextHistory);
      clearAttachments();
      setSendError(null);
      setIsSending(true);
      try {
        const response = await fetchCandidates(
          targetId,
          nextHistory.map((message) => ({
            role: message.role,
            text: message.text,
            attachments: message.attachments,
          })),
        );
        setMiniChatMessages((previous) => [
          ...previous,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            text: response.reply.text,
            attachments: [],
          },
        ]);
        setCandidates(response.candidates);
        // Tick every candidate by default, as in the mockup.
        setApprovedIndexes(new Set(response.candidates.map((_, index) => index)));
      } catch {
        setSendError("Couldn't reach the server. Please try again.");
      } finally {
        setIsSending(false);
      }
    },
    [targetId, fetchCandidates, miniChatMessages, pendingAttachments, clearAttachments],
  );

  const setApproved = useCallback((index: number, isApproved: boolean) => {
    setApprovedIndexes((previous) => {
      const next = new Set(previous);
      if (isApproved) next.add(index);
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
