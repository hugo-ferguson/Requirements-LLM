import { ApiError } from "../api/client";
import { documentsApi } from "../api/documents";
import type { Attachment } from "../types/conversation";

/** Extensions the backend can extract text from. */
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

/** Value for an `<input type="file">` accept attribute. */
export const ATTACHMENT_ACCEPT = SUPPORTED_EXTENSIONS.join(",");

/** Upload size limit. Must match MAX_UPLOAD_BYTES in src/backend/app/ingest/limits.py. */
const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;

interface AttachmentUploadResult {
  attachments: Attachment[];
  /** One user-facing message per failed file. */
  errors: string[];
}

function messageForError(filename: string, error: unknown): string {
  if (error instanceof DOMException && error.name === "TimeoutError") {
    return `Reading ${filename} took too long and was stopped. Please try again.`;
  }
  if (error instanceof TypeError) {
    // fetch throws a TypeError when it can't reach the server.
    return `Couldn't reach the server to upload ${filename} — is the backend running?`;
  }
  if (error instanceof ApiError) {
    // The backend puts the reason in a JSON `detail` field.
    try {
      const detail: unknown = JSON.parse(error.message)?.detail;
      if (typeof detail === "string") return detail;
    } catch {
      // Not JSON, so use the generic message below.
    }
  }
  return `Couldn't upload ${filename}. Please try again.`;
}

/** Uploads files one at a time, so the vision model gets one image at once. */
export async function uploadFilesAsAttachments(files: FileList): Promise<AttachmentUploadResult> {
  const attachments: Attachment[] = [];
  const errors: string[] = [];

  for (const file of Array.from(files)) {
    if (file.size > MAX_UPLOAD_BYTES) {
      errors.push(
        `${file.name} is larger than the ${MAX_UPLOAD_BYTES / (1024 * 1024)} MB upload limit.`,
      );
      continue;
    }
    try {
      const uploaded = await documentsApi.upload(file);
      attachments.push({
        filename: uploaded.filename ?? file.name,
        content: uploaded.text,
        document_id: uploaded.document_id,
      });
    } catch (error) {
      errors.push(messageForError(file.name, error));
    }
  }

  return { attachments, errors };
}
