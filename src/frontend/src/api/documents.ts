import { request } from "./client";
import type { paths } from "./schema";

type UploadRead =
  paths["/documents/upload"]["post"]["responses"]["201"]["content"]["application/json"];

/** Upload timeout. Longer than the backend's 300 s vision timeout per attempt. */
const UPLOAD_TIMEOUT_MS = 6 * 60 * 1000;

/** Calls for uploading attachments and deleting their documents. */
export const documentsApi = {
  /** Uploads a file and returns its text. Images get a null `document_id`. */
  upload: (file: File): Promise<UploadRead> => {
    const body = new FormData();
    body.append("file", file);
    return request<UploadRead>("/documents/upload", {
      method: "POST",
      body,
      signal: AbortSignal.timeout(UPLOAD_TIMEOUT_MS),
    });
  },

  /** Deletes a document and its chunks. */
  remove: (documentId: number): Promise<void> =>
    request<void>(`/documents/${documentId}`, { method: "DELETE" }),
};
