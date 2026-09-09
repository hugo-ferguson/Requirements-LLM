import { request } from "./client";
import type { paths } from "./schema";

export type UploadRead =
  paths["/documents/upload"]["post"]["responses"]["201"]["content"]["application/json"];

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
    return request<UploadRead>("/documents/upload", { method: "POST", body });
  },

  /** Deletes an ingested document along with its chunks. */
  remove: (documentId: number): Promise<void> =>
    request<void>(`/documents/${documentId}`, { method: "DELETE" }),
};
