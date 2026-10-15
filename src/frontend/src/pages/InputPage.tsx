import { useNavigate, useParams } from "react-router";
import { ChatThread } from "../components/chat/ChatThread";
import { ChatInputBar } from "../components/chat/ChatInputBar";
import { useConfirm } from "../components/shared/ConfirmDialog";
import { uatCasesApi } from "../api/uatCases";
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

  const { confirm, dialog } = useConfirm();

  const busy = isSending || isGenerating;
  const generateDisabled = busy || messages.length === 0;

  /** Total UAT cases in the session; 0 if they can't be loaded, so Generate never blocks on it. */
  async function existingTestCaseCount(): Promise<number> {
    try {
      const { groups } = await uatCasesApi.list(sessionId!);
      return groups.reduce((total, group) => total + group.uat_cases.length, 0);
    } catch {
      return 0;
    }
  }

  async function handleGenerate() {
    // Generating replaces every AC, and the backend deletes all the session's
    // test cases along with them.
    const testCaseCount = await existingTestCaseCount();
    if (testCaseCount > 0) {
      const confirmed = await confirm({
        title: "Delete all test cases?",
        message:
          "Generating again replaces all acceptance criteria and deletes all " +
          `${testCaseCount} test case${testCaseCount === 1 ? "" : "s"} in this session. ` +
          "This can't be undone.",
        confirmLabel: "Delete test cases & generate",
        destructive: true,
      });
      if (!confirmed) return;
    }
    const result = await generate();
    if (result) {
      navigate(ROUTES.acReview(sessionId!));
    }
  }

  return (
    <div className="flex h-full flex-col">
      {dialog}
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
