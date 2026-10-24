import { useCallback, useEffect, useState } from "react";
import { sessionsApi } from "../api/sessions";
import type { MessageRead } from "../api/sessions";
import type { ChatMessage } from "../types/conversation";
import { useAttachmentUploads } from "./useAttachmentUploads";

function toChatMessage(message: MessageRead): ChatMessage {
  return {
    id: String(message.id),
    role: message.role,
    text: message.text,
    attachments: message.attachments,
  };
}

/** Loads a session's chat and sends messages. `generate` resolves to null on failure. */
export function useSessionConversation(sessionId: string) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(true);
  const [isSending, setIsSending] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [sendError, setSendError] = useState<string | null>(null);
  const [generateError, setGenerateError] = useState<string | null>(null);
  const uploads = useAttachmentUploads();
  const { pendingAttachments, clearAttachments } = uploads;

  useEffect(() => {
    let isCancelled = false;
    setIsLoadingHistory(true);
    setMessages([]);
    sessionsApi
      .get(sessionId)
      .then((detail) => {
        if (!isCancelled) setMessages(detail.messages.map(toChatMessage));
      })
      .finally(() => {
        if (!isCancelled) setIsLoadingHistory(false);
      });
    return () => {
      isCancelled = true;
    };
  }, [sessionId]);

  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed && pendingAttachments.length === 0) return;

      const userMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: "user",
        text: trimmed,
        attachments: pendingAttachments,
      };
      setMessages((previous) => [...previous, userMessage]);
      clearAttachments();
      setSendError(null);
      setIsSending(true);
      try {
        const reply = await sessionsApi.sendMessage(sessionId, {
          text: trimmed,
          attachments: pendingAttachments,
        });
        setMessages((previous) => [...previous, toChatMessage(reply)]);
      } catch {
        setSendError("Couldn't reach the server. Please try again.");
      } finally {
        setIsSending(false);
      }
    },
    [sessionId, pendingAttachments, clearAttachments],
  );

  const generate = useCallback(async () => {
    setGenerateError(null);
    setIsGenerating(true);
    try {
      return await sessionsApi.generate(sessionId);
    } catch {
      setGenerateError("Couldn't generate acceptance criteria. Please try again.");
      return null;
    } finally {
      setIsGenerating(false);
    }
  }, [sessionId]);

  return {
    ...uploads,
    messages,
    isLoadingHistory,
    isSending,
    isGenerating,
    sendError,
    generateError,
    sendMessage,
    generate,
  };
}
