import { useNavigate, useParams } from "react-router";
import { ChatThread } from "../components/chat/ChatThread";
import { ChatInputBar } from "../components/chat/ChatInputBar";
import { useSessionConversation } from "../hooks/useSessionConversation";
import { ROUTES } from "../routes";

export function InputPage() {
  const navigate = useNavigate();
  const { sessionId } = useParams<{ sessionId: string }>();
  const {
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
  } = useSessionConversation(sessionId!);

  const busy = isSending || isGenerating;
  const generateDisabled = busy || messages.length === 0;

  async function handleGenerate() {
    const result = await generate();
    if (result) {
      navigate(ROUTES.acReview(sessionId!));
    }
  }

  return (
    <div className="flex h-full flex-col">
      <ChatThread messages={messages} pending={isSending} />
      <footer className="border-t border-gray-200 p-4">
        {sendError && <p className="mb-2 text-sm text-red-600">{sendError}</p>}
        {generateError && <p className="mb-2 text-sm text-red-600">{generateError}</p>}
        <ChatInputBar
          pendingAttachments={pendingAttachments}
          onAttachFiles={attachFiles}
          onRemoveAttachment={removeAttachment}
          onSend={sendMessage}
          disabled={busy}
          trailingAction={
            <button
              type="button"
              onClick={handleGenerate}
              disabled={generateDisabled}
              className="rounded-full bg-primary px-6 py-2 font-medium text-white hover:bg-primary-hover disabled:opacity-50"
            >
              Generate
            </button>
          }
        />
      </footer>
    </div>
  );
}
