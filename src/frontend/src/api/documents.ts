import { request } from "./client";
import type { paths } from "./schema";

export type UploadRead =
  paths["/documents/upload"]["post"]["responses"]["201"]["content"]["application/json"];

/**
 * How long an upload may take before the browser gives up. The backend allows
 * the vision model 300s plus retries for an image, so this sits above that —
 * it exists to end a request that will never answer, not to hurry a slow one.
 */
export const UPLOAD_TIMEOUT_MS = 6 * 60 * 1000;

export const documentsApi = {
  /**
   * Uploads one file for ingest and returns its extracted text.
   *
   * Documents that can outgrow a prompt (PDFs, long text) are also chunked
   * and embedded, and come back with a `document_id`. Images are only
   * transcribed — their text is small enough to send to the model directly —
   * so they come back with a null `document_id` and nothing to clean up.
   */
  upload: (file: File): Promise<UploadRead> => {
    const body = new FormData();
    body.append("file", file);
    return request<UploadRead>("/documents/upload", {
      method: "POST",
      body,
      signal: AbortSignal.timeout(UPLOAD_TIMEOUT_MS),
    });
  },

  /** Deletes an ingested document along with its chunks. */
  remove: (documentId: number): Promise<void> =>
    request<void>(`/documents/${documentId}`, { method: "DELETE" }),
};
