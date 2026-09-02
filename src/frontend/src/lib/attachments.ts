import { ApiError } from "../api/client";
import { documentsApi } from "../api/documents";
import type { Attachment } from "../types/conversation";

/** Extensions the backend's ingest pipeline can extract text from. */
export const SUPPORTED_EXTENSIONS = [
  ".txt",
  ".md",
  ".pdf",
  ".png",
  ".jpg",
  ".jpeg",
  ".webp",
  ".gif",
] as const;

/** Value for an <input type="file"> accept attribute. */
export const ATTACHMENT_ACCEPT = SUPPORTED_EXTENSIONS.join(",");

export interface AttachmentUploadError {
  filename: string;
  message: string;
}

export interface AttachmentUploadResult {
  attachments: Attachment[];
  errors: AttachmentUploadError[];
}

function messageForError(filename: string, error: unknown): string {
  if (error instanceof ApiError) {
    // FastAPI reports the reason (unsupported type, no text found, vision
    // model unreachable) in a JSON detail field.
    try {
      const detail = JSON.parse(error.message)?.detail;
      if (typeof detail === "string") return detail;
    } catch {
      // Not JSON — fall through to the generic message.
    }
  }
  return `Couldn't upload ${filename}. Please try again.`;
}

/**
 * Sends each file to the backend for ingest and turns the extracted text into
 * chat attachments. Files are uploaded one at a time because image extraction
 * runs a local vision model that does not benefit from concurrency.
 */
export async function uploadFilesAsAttachments(files: FileList): Promise<AttachmentUploadResult> {
  const attachments: Attachment[] = [];
  const errors: AttachmentUploadError[] = [];

  for (const file of Array.from(files)) {
    try {
      const uploaded = await documentsApi.upload(file);
      attachments.push({
        filename: uploaded.filename ?? file.name,
        content: uploaded.text,
        document_id: uploaded.document_id,
      });
    } catch (error) {
      errors.push({ filename: file.name, message: messageForError(file.name, error) });
    }
  }

  return { attachments, errors };
}
