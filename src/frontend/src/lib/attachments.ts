import type { Attachment } from "../types/conversation";

export async function readFilesAsAttachments(files: FileList): Promise<Attachment[]> {
  return Promise.all(
    Array.from(files).map(async (file) => ({
      filename: file.name,
      content: await file.text(),
    })),
  );
}
