import { useCallback, useEffect, useState } from "react";
import { sessionsApi } from "../api/sessions";
import type { MessageRead } from "../api/sessions";
import { readFilesAsAttachments } from "../lib/attachments";
import type { Attachment, ChatMessage } from "../types/conversation";

function newId(): string {
  return crypto.randomUUID();
}

function toChatMessage(message: MessageRead): ChatMessage {
  return {
    id: String(message.id),
    role: message.role,
    text: message.text,
    attachments: message.attachments,
  };
}

export function useSessionConversation(sessionId: string) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [pendingAttachments, setPendingAttachments] = useState<Attachment[]>([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(true);
  const [isSending, setIsSending] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [sendError, setSendError] = useState<string | null>(null);
  const [generateError, setGenerateError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setIsLoadingHistory(true);
    setMessages([]);
    sessionsApi
      .get(sessionId)
      .then((detail) => {
        if (!cancelled) setMessages(detail.messages.map(toChatMessage));
      })
      .finally(() => {
        if (!cancelled) setIsLoadingHistory(false);
      });
    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  const attachFiles = useCallback(async (files: FileList) => {
    const read = await readFilesAsAttachments(files);
    setPendingAttachments((prev) => [...prev, ...read]);
  }, []);

  const removeAttachment = useCallback((index: number) => {
    setPendingAttachments((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed && pendingAttachments.length === 0) return;

      const userMessage: ChatMessage = {
        id: newId(),
        role: "user",
        text: trimmed,
        attachments: pendingAttachments,
      };
      setMessages((prev) => [...prev, userMessage]);
      setPendingAttachments([]);
      setSendError(null);
      setIsSending(true);
      try {
        const reply = await sessionsApi.sendMessage(sessionId, {
          text: trimmed,
          attachments: pendingAttachments,
        });
        setMessages((prev) => [...prev, toChatMessage(reply)]);
      } catch {
        setSendError("Couldn't reach the server. Please try again.");
      } finally {
        setIsSending(false);
      }
    },
    [sessionId, pendingAttachments],
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
    messages,
    pendingAttachments,
    isLoadingHistory,
    isSending,
    isGenerating,
    sendError,
    generateError,
    attachFiles,
    removeAttachment,
    sendMessage,
    generate,
  };
}
