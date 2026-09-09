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
    isUploading,
    uploadCount,
    uploadError,
    sendError,
    generateError,
    attachFiles,
    dismissUploadError,
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
        {uploadError && (
          <p className="mb-2 flex items-start gap-2 text-sm text-red-600">
            <span>{uploadError}</span>
            <button
              type="button"
              onClick={dismissUploadError}
              aria-label="Dismiss upload error"
              className="text-red-400 hover:text-red-600"
            >
              ×
            </button>
          </p>
        )}
        {sendError && <p className="mb-2 text-sm text-red-600">{sendError}</p>}
        {generateError && <p className="mb-2 text-sm text-red-600">{generateError}</p>}
        <ChatInputBar
          pendingAttachments={pendingAttachments}
          onAttachFiles={attachFiles}
          onRemoveAttachment={removeAttachment}
          onSend={sendMessage}
          disabled={busy}
          isUploading={isUploading}
          uploadCount={uploadCount}
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
