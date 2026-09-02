interface AttachmentChipProps {
  filename: string;
  onRemove?: () => void;
}

export function AttachmentChip({ filename, onRemove }: AttachmentChipProps) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full bg-white px-3 py-1 text-sm text-gray-700 shadow-sm">
      {filename}
      {onRemove && (
        <button
          type="button"
          onClick={onRemove}
          aria-label={`Remove ${filename}`}
          className="text-gray-400 hover:text-gray-600"
        >
          ×
        </button>
      )}
    </span>
  );
}
