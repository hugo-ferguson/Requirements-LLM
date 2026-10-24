/** A file attached to a chat message, as its extracted text. */
export interface Attachment {
  filename: string;
  content: string;
  /** Stored document ID. Null for images, which aren't chunked or embedded. */
  document_id?: number | null;
}

type ChatRole = "user" | "assistant";

/** One message in a chat thread. */
export interface ChatMessage {
  /** Backend message ID, or a random UUID for messages added in the browser. */
  id: string;
  role: ChatRole;
  text: string;
  attachments: Attachment[];
}
