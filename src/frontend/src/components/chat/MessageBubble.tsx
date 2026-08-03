import type { ChatMessage } from "../../types/conversation";
import { AttachmentChip } from "./AttachmentChip";

interface MessageBubbleProps {
  message: ChatMessage;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[70%] rounded-2xl px-4 py-3 ${
          isUser ? "bg-primary text-white" : "bg-gray-100 text-gray-900"
        }`}
      >
        {message.text && <p className="whitespace-pre-wrap">{message.text}</p>}
        {message.attachments.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-2">
            {message.attachments.map((attachment, i) => (
              <AttachmentChip key={`${attachment.filename}-${i}`} filename={attachment.filename} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
