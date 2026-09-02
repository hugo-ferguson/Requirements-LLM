import { useRef, useState } from "react";
import type { ChangeEvent, KeyboardEvent, ReactNode } from "react";
import type { Attachment } from "../../types/conversation";
import { ATTACHMENT_ACCEPT, SUPPORTED_EXTENSIONS } from "../../lib/attachments";
import { AttachmentChip } from "./AttachmentChip";

interface ChatInputBarProps {
  pendingAttachments: Attachment[];
  onAttachFiles: (files: FileList) => void;
  onRemoveAttachment: (index: number) => void;
  onSend: (text: string) => void;
  disabled: boolean;
  isUploading?: boolean;
  uploadCount?: number;
  trailingAction?: ReactNode;
}

export function ChatInputBar({
  pendingAttachments,
  onAttachFiles,
  onRemoveAttachment,
  onSend,
  disabled,
  isUploading = false,
  uploadCount = 0,
  trailingAction,
}: ChatInputBarProps) {
  const [text, setText] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    if (event.target.files && event.target.files.length > 0) {
      onAttachFiles(event.target.files);
    }
    event.target.value = "";
  }

  const sendDisabled =
    disabled || isUploading || (!text.trim() && pendingAttachments.length === 0);

  function handleSend() {
    const trimmed = text.trim();
    if (sendDisabled) return;
    onSend(trimmed);
    setText("");
  }

  function handleKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "Enter") {
      event.preventDefault();
      handleSend();
    }
  }

  return (
    <div>
      {(pendingAttachments.length > 0 || isUploading) && (
        <div className="mb-2 flex flex-wrap items-center gap-2">
          {pendingAttachments.map((attachment, i) => (
            <AttachmentChip
              key={`${attachment.filename}-${i}`}
              filename={attachment.filename}
              onRemove={() => onRemoveAttachment(i)}
            />
          ))}
          {isUploading && (
            <span
              role="status"
              className="inline-flex items-center rounded-full bg-white px-3 py-1 text-sm text-gray-500 shadow-sm"
            >
              Reading {uploadCount === 1 ? "file" : `${uploadCount} files`}…
            </span>
          )}
        </div>
      )}
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled || isUploading}
          aria-label="Attach documents or images"
          title={`Attach a document or image (${SUPPORTED_EXTENSIONS.join(", ")})`}
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-gray-100 text-xl text-gray-600 hover:bg-gray-200 disabled:opacity-50"
        >
          +
        </button>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          hidden
          accept={ATTACHMENT_ACCEPT}
          onChange={handleFileChange}
        />
        <input
          type="text"
          value={text}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message here"
          disabled={disabled}
          className="flex-1 rounded-full border border-gray-200 px-4 py-2 outline-none focus:border-primary"
        />
        <button
          type="button"
          onClick={handleSend}
          disabled={sendDisabled}
          className="rounded-full bg-primary px-5 py-2 font-medium text-white hover:bg-primary-hover disabled:opacity-50"
        >
          Send
        </button>
        {trailingAction}
      </div>
    </div>
  );
}
