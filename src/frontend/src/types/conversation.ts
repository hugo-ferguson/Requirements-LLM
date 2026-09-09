export interface Attachment {
  filename: string;
  content: string;
  /** Set when the file was ingested through /documents/upload. */
  document_id?: number | null;
}

export type ChatRole = "user" | "assistant";

export interface ChatMessage {
  id: string;
  role: ChatRole;
  text: string;
  attachments: Attachment[];
}
