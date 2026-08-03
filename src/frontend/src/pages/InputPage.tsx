import { useNavigate } from "react-router";
import { ChatThread } from "../components/chat/ChatThread";
import { ChatInputBar } from "../components/chat/ChatInputBar";
import { useConversation } from "../hooks/useConversation";
import { ROUTES } from "../routes";

export function InputPage() {
  const navigate = useNavigate();
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
  } = useConversation();

  const busy = isSending || isGenerating;

  async function handleGenerate() {
    const result = await generate();
    if (result) {
      console.log("[Story2Spec] generate result", result);
      navigate(ROUTES.acReview, { state: { generateResult: result } });
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
        />
        <div className="mt-3 flex justify-end">
          <button
            type="button"
            onClick={handleGenerate}
            disabled={busy}
            className="rounded-full bg-primary px-6 py-2 font-medium text-white hover:bg-primary-hover disabled:opacity-50"
          >
            Generate
          </button>
        </div>
      </footer>
    </div>
  );
}
