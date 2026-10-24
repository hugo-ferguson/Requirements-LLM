import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";

interface ConfirmOptions {
  title: string;
  message: ReactNode;
  /** Label for the go-ahead button. Defaults to "Continue". */
  confirmLabel?: string;
  /** Makes the go-ahead button red, for deletions. */
  isDestructive?: boolean;
}

interface ConfirmDialogProps extends ConfirmOptions {
  onConfirm: () => void;
  onCancel: () => void;
}

/** A modal asking the user to confirm an action. */
function ConfirmDialog({
  title,
  message,
  confirmLabel = "Continue",
  isDestructive = false,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  const cancelButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    // Focus Cancel so Enter can't confirm by accident.
    cancelButtonRef.current?.focus();
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") onCancel();
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onCancel]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      onClick={onCancel}
    >
      <div
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-dialog-title"
        onClick={(event) => event.stopPropagation()}
        className="w-full max-w-md rounded-xl bg-white p-6 shadow-lg"
      >
        <h2 id="confirm-dialog-title" className="text-lg font-semibold text-gray-900">
          {title}
        </h2>
        <div className="mt-2 text-sm text-gray-600">{message}</div>
        <div className="mt-6 flex justify-end gap-2">
          <button
            ref={cancelButtonRef}
            type="button"
            onClick={onCancel}
            className="rounded-full bg-gray-100 px-5 py-2 text-sm font-medium text-gray-700 hover:bg-gray-200"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onConfirm}
            className={`rounded-full px-5 py-2 text-sm font-medium text-white ${
              isDestructive ? "bg-red-600 hover:bg-red-700" : "bg-primary hover:bg-primary-hover"
            }`}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}

/** Promise-based confirmation: `if (!(await confirm({...}))) return;`. Render `dialog` once. */
export function useConfirm() {
  const [pendingConfirmation, setPendingConfirmation] = useState<
    (ConfirmOptions & { resolve: (isConfirmed: boolean) => void }) | null
  >(null);

  const confirm = useCallback(
    (options: ConfirmOptions) =>
      new Promise<boolean>((resolve) => setPendingConfirmation({ ...options, resolve })),
    [],
  );

  const settle = useCallback(
    (isConfirmed: boolean) => {
      pendingConfirmation?.resolve(isConfirmed);
      setPendingConfirmation(null);
    },
    [pendingConfirmation],
  );
  const handleConfirm = useCallback(() => settle(true), [settle]);
  const handleCancel = useCallback(() => settle(false), [settle]);

  const dialog = pendingConfirmation ? (
    <ConfirmDialog
      title={pendingConfirmation.title}
      message={pendingConfirmation.message}
      confirmLabel={pendingConfirmation.confirmLabel}
      isDestructive={pendingConfirmation.isDestructive}
      onConfirm={handleConfirm}
      onCancel={handleCancel}
    />
  ) : null;

  return { confirm, dialog };
}
