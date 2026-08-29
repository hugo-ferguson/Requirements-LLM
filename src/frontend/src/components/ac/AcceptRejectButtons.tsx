import type { AcceptanceCriterionStatus } from "../../api/acceptanceCriteria";

interface AcceptRejectButtonsProps {
  status: AcceptanceCriterionStatus;
  onAccept: () => void;
  onReject: () => void;
}

export function AcceptRejectButtons({ status, onAccept, onReject }: AcceptRejectButtonsProps) {
  const accepted = status === "accepted";
  const rejected = status === "rejected";

  return (
    <div className="flex flex-col gap-2">
      <button
        type="button"
        onClick={onAccept}
        aria-label={accepted ? "Unaccept" : "Accept"}
        aria-pressed={accepted}
        className={`flex h-9 w-9 items-center justify-center rounded-full border text-sm font-bold ${
          accepted
            ? "border-green-600 bg-green-600 text-white"
            : "border-gray-300 text-gray-400 hover:bg-gray-100"
        }`}
      >
        ✓
      </button>
      <button
        type="button"
        onClick={onReject}
        aria-label={rejected ? "Unreject" : "Reject"}
        aria-pressed={rejected}
        className={`flex h-9 w-9 items-center justify-center rounded-full border text-sm font-bold ${
          rejected
            ? "border-red-600 bg-red-600 text-white"
            : "border-gray-300 text-gray-400 hover:bg-gray-100"
        }`}
      >
        ✕
      </button>
    </div>
  );
}
