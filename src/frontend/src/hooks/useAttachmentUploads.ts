import { useCallback, useState } from "react";
import { documentsApi } from "../api/documents";
import { uploadFilesAsAttachments } from "../lib/attachments";
import type { Attachment } from "../types/conversation";

/** Deletes an unsent attachment's document. Failures only log a warning. */
function discardDocument(attachment: Attachment): void {
  if (attachment.document_id == null) return;
  documentsApi.remove(attachment.document_id).catch((error) => {
    console.warn(`Could not delete document ${attachment.document_id}`, error);
  });
}

/** Holds the attachments for the next message, with upload progress and errors. */
export function useAttachmentUploads() {
  const [pendingAttachments, setPendingAttachments] = useState<Attachment[]>([]);
  // Files in the current upload batch, or 0.
  const [uploadCount, setUploadCount] = useState(0);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const attachFiles = useCallback(async (files: FileList) => {
    setUploadError(null);
    setUploadCount(files.length);
    try {
      const { attachments, errors } = await uploadFilesAsAttachments(files);
      if (attachments.length > 0) {
        setPendingAttachments((previous) => [...previous, ...attachments]);
      }
      if (errors.length > 0) {
        setUploadError(errors.join(" "));
      }
    } finally {
      setUploadCount(0);
    }
  }, []);

  const removeAttachment = useCallback(
    (index: number) => {
      const removed = pendingAttachments[index];
      setPendingAttachments((previous) =>
        previous.filter((_, attachmentIndex) => attachmentIndex !== index),
      );
      if (removed) discardDocument(removed);
    },
    [pendingAttachments],
  );

  // After sending. The message owns the documents now, so keep them.
  const clearAttachments = useCallback(() => setPendingAttachments([]), []);

  // When the user backs out. Also deletes the documents.
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
