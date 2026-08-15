import { useRef, useState } from "react";
import type { ChangeEvent, KeyboardEvent, ReactNode } from "react";
import type { Attachment } from "../../types/conversation";
import { AttachmentChip } from "./AttachmentChip";

interface ChatInputBarProps {
  pendingAttachments: Attachment[];
  onAttachFiles: (files: FileList) => void;
  onRemoveAttachment: (index: number) => void;
  onSend: (text: string) => void;
  disabled: boolean;
  trailingAction?: ReactNode;
}

export function ChatInputBar({
  pendingAttachments,
  onAttachFiles,
  onRemoveAttachment,
  onSend,
  disabled,
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

  function handleSend() {
    const trimmed = text.trim();
    if (disabled || (!trimmed && pendingAttachments.length === 0)) return;
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
      {pendingAttachments.length > 0 && (
        <div className="mb-2 flex flex-wrap gap-2">
          {pendingAttachments.map((attachment, i) => (
            <AttachmentChip
              key={`${attachment.filename}-${i}`}
              filename={attachment.filename}
              onRemove={() => onRemoveAttachment(i)}
            />
          ))}
        </div>
      )}
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled}
          aria-label="Attach files"
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-gray-100 text-xl text-gray-600 hover:bg-gray-200 disabled:opacity-50"
        >
          +
        </button>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          hidden
          accept=".txt,.md,text/plain"
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
          disabled={disabled || (!text.trim() && pendingAttachments.length === 0)}
          className="rounded-full bg-primary px-5 py-2 font-medium text-white hover:bg-primary-hover disabled:opacity-50"
        >
          Send
        </button>
        {trailingAction}
      </div>
    </div>
  );
}
