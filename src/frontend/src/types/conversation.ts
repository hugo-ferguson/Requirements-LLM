export interface Attachment {
  filename: string;
  content: string;
}

export type ChatRole = "user" | "assistant";

export interface ChatMessage {
  id: string;
  role: ChatRole;
  text: string;
  attachments: Attachment[];
}
