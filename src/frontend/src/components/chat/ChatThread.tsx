import type { ChatMessage } from "../../types/conversation";
import { MessageBubble } from "./MessageBubble";

interface ChatThreadProps {
  messages: ChatMessage[];
  /** Shows a "Typing…" bubble after the messages. */
  isAwaitingReply: boolean;
}

/** A scrolling list of chat messages. */
export function ChatThread({ messages, isAwaitingReply }: ChatThreadProps) {
  return (
    <div className="flex-1 space-y-4 overflow-y-auto px-6 py-6">
      {messages.map((message) => (
        <MessageBubble key={message.id} message={message} />
      ))}
      {isAwaitingReply && (
        <div className="flex justify-start">
          <div className="max-w-[70%] rounded-2xl bg-gray-100 px-4 py-3 text-gray-500">Typing…</div>
        </div>
      )}
    </div>
  );
}
