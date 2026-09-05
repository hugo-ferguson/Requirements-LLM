import { useCallback, useState } from "react";
import { documentsApi } from "../api/documents";
import { uploadFilesAsAttachments } from "../lib/attachments";
import type { Attachment } from "../types/conversation";

/**
 * Deletes an ingested document that never made it into a message.
 *
 * Best-effort and deliberately not awaited: the chip disappears either way,
 * and a failed cleanup is worth a console warning, not an error in the user's
 * face about a file they've already discarded.
 */
function discardDocument(attachment: Attachment): void {
  if (attachment.document_id == null) return;
  documentsApi.remove(attachment.document_id).catch((error) => {
    console.warn(`Could not delete document ${attachment.document_id}`, error);
  });
}

/**
 * Owns the attachments waiting to be sent with the next message.
 *
 * Attaching a file uploads it to the backend's ingest pipeline, which can take
 * a while — a PDF is parsed and a local vision model reads images — so the
 * upload has its own in-flight and error state, separate from sending.
 */
export function useAttachmentUploads() {
  const [pendingAttachments, setPendingAttachments] = useState<Attachment[]>([]);
  const [uploadCount, setUploadCount] = useState(0);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const attachFiles = useCallback(async (files: FileList) => {
    setUploadError(null);
    setUploadCount(files.length);
    try {
      const { attachments, errors } = await uploadFilesAsAttachments(files);
      if (attachments.length > 0) {
        setPendingAttachments((prev) => [...prev, ...attachments]);
      }
      if (errors.length > 0) {
        setUploadError(errors.map((error) => error.message).join(" "));
      }
    } finally {
      setUploadCount(0);
    }
  }, []);

  const removeAttachment = useCallback(
    (index: number) => {
      const removed = pendingAttachments[index];
      setPendingAttachments((prev) => prev.filter((_, i) => i !== index));
      if (removed) discardDocument(removed);
    },
    [pendingAttachments],
  );

  /**
   * Forgets the pending attachments because they've been sent — the message
   * owns them now, so the ingested documents must stay.
   */
  const clearAttachments = useCallback(() => setPendingAttachments([]), []);

  /**
   * Drops the pending attachments because the user backed out, deleting the
   * documents they ingested.
   */
  const discardAttachments = useCallback(() => {
    pendingAttachments.forEach(discardDocument);
    setPendingAttachments([]);
    setUploadError(null);
  }, [pendingAttachments]);

  const dismissUploadError = useCallback(() => setUploadError(null), []);

  return {
    pendingAttachments,
    uploadCount,
    isUploading: uploadCount > 0,
    uploadError,
    attachFiles,
    removeAttachment,
    clearAttachments,
    discardAttachments,
    dismissUploadError,
  };
}
