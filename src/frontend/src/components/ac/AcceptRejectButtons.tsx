import type { AcceptanceCriterionStatus } from "../../api/acceptanceCriteria";

interface AcceptRejectButtonsProps {
  status: AcceptanceCriterionStatus;
  onAccept: () => void;
  onReject: () => void;
}

/** Tick and cross buttons for a card. The one matching `status` shows as pressed. */
export function AcceptRejectButtons({ status, onAccept, onReject }: AcceptRejectButtonsProps) {
  const isAccepted = status === "accepted";
  const isRejected = status === "rejected";

  return (
    <div className="flex flex-col gap-2">
      <button
        type="button"
        onClick={onAccept}
        aria-label={isAccepted ? "Unaccept" : "Accept"}
        aria-pressed={isAccepted}
        className={`flex h-9 w-9 items-center justify-center rounded-full border text-sm font-bold ${
          isAccepted
            ? "border-green-600 bg-green-600 text-white"
            : "border-gray-300 text-gray-400 hover:bg-gray-100"
        }`}
      >
        ✓
      </button>
      <button
        type="button"
        onClick={onReject}
        aria-label={isRejected ? "Unreject" : "Reject"}
        aria-pressed={isRejected}
        className={`flex h-9 w-9 items-center justify-center rounded-full border text-sm font-bold ${
          isRejected
            ? "border-red-600 bg-red-600 text-white"
            : "border-gray-300 text-gray-400 hover:bg-gray-100"
        }`}
      >
        ✕
      </button>
    </div>
  );
}
