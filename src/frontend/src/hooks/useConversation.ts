import { useCallback, useState } from "react";
import { conversationApi } from "../api/conversation";
import type { GenerateResult } from "../api/conversation";
import { readFilesAsAttachments } from "../lib/attachments";
import type { Attachment, ChatMessage } from "../types/conversation";

function newId(): string {
  return crypto.randomUUID();
}

export function useConversation() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [pendingAttachments, setPendingAttachments] = useState<Attachment[]>([]);
  const [isSending, setIsSending] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [sendError, setSendError] = useState<string | null>(null);
  const [generateError, setGenerateError] = useState<string | null>(null);

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
      const nextMessages = [...messages, userMessage];
      setMessages(nextMessages);
      setPendingAttachments([]);
      setSendError(null);
      setIsSending(true);
      try {
        const reply = await conversationApi.sendMessage(nextMessages);
        setMessages((prev) => [
          ...prev,
          { id: newId(), role: "assistant", text: reply.text, attachments: [] },
        ]);
      } catch {
        setSendError("Couldn't reach the server. Please try again.");
      } finally {
        setIsSending(false);
      }
    },
    [messages, pendingAttachments],
  );

  const generate = useCallback(async (): Promise<GenerateResult | null> => {
    setGenerateError(null);
    setIsGenerating(true);
    try {
      return await conversationApi.generate(messages);
    } catch {
      setGenerateError("Couldn't generate acceptance criteria. Please try again.");
      return null;
    } finally {
      setIsGenerating(false);
    }
  }, [messages]);

  return {
    messages,
    pendingAttachments,
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
