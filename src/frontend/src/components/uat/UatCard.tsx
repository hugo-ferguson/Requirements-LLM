import { useEffect, useState } from "react";
import type { UatCase } from "../../api/uatCases";
import { ScorePill } from "../ac/ScorePill";
import { OverallScoreBadge } from "../ac/OverallScoreBadge";
import { AcceptRejectButtons } from "../ac/AcceptRejectButtons";

interface EditFields {
  title: string;
  description: string;
}

interface UatCardProps {
  uatCase: UatCase;
  index: number;
  selected?: boolean;
  editing?: boolean;
  onSelect?: () => void;
  onEditStart?: () => void;
  onEditCancel?: () => void;
  onEditSave?: (fields: EditFields) => void;
  onAccept: () => void;
  onReject: () => void;
}

export function UatCard({
  uatCase,
  index,
  selected = false,
  editing = false,
  onSelect,
  onEditStart,
  onEditCancel,
  onEditSave,
  onAccept,
  onReject,
}: UatCardProps) {
  const [draft, setDraft] = useState<EditFields>(uatCase);

  // Re-seed the draft from the current values each time edit mode is entered,
  // so a previous (cancelled) edit never leaks into a later edit session.
  useEffect(() => {
    if (editing) {
      setDraft({ title: uatCase.title, description: uatCase.description });
    }
  }, [editing, uatCase]);

  const cardClasses = `rounded-xl border bg-white p-4 shadow-sm ${
    selected ? "border-primary ring-2 ring-primary ring-offset-2" : "border-gray-200"
  }`;

  if (editing) {
    return (
      <div className={cardClasses}>
        <div className="mb-3 flex items-center gap-2">
          <span className="text-sm font-semibold text-gray-400">#UAT {index + 1}</span>
          <input
            value={draft.title}
            onChange={(event) => setDraft((prev) => ({ ...prev, title: event.target.value }))}
            className="flex-1 rounded-lg border border-gray-200 px-3 py-1.5 font-bold text-gray-900 outline-none focus:border-primary"
          />
        </div>
        <textarea
          value={draft.description}
          onChange={(event) => setDraft((prev) => ({ ...prev, description: event.target.value }))}
          rows={3}
          className="w-full resize-none rounded-lg border border-gray-200 p-2 text-sm text-gray-900 outline-none focus:border-primary"
        />
        <div className="mt-3 flex gap-2">
          <button
            type="button"
            onClick={() => onEditSave?.(draft)}
            className="rounded-full bg-primary px-5 py-2 text-sm font-medium text-white hover:bg-primary-hover"
          >
            Save
          </button>
          <button
            type="button"
            onClick={onEditCancel}
            className="rounded-full bg-gray-100 px-5 py-2 text-sm font-medium text-gray-600 hover:bg-gray-200"
          >
            Cancel
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={cardClasses}>
      <div className="flex items-start gap-4">
        <div onClick={onSelect} className={`min-w-0 flex-1 ${onSelect ? "cursor-pointer" : ""}`}>
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-gray-400">#UAT {index + 1}</span>
            <span className="font-bold text-gray-900">{uatCase.title}</span>
            {onEditStart && (
              <button
                type="button"
                onClick={(event) => {
                  event.stopPropagation();
                  onEditStart();
                }}
                aria-label="Edit"
                className="text-gray-400 hover:text-gray-700"
              >
                ✎
              </button>
            )}
          </div>
          <p className="mt-1 text-sm text-gray-600">{uatCase.description}</p>
          <div className="mt-2 flex flex-wrap gap-2">
            <ScorePill label="Relevance" value={uatCase.scores.relevance} />
            <ScorePill label="Correctness" value={uatCase.scores.correctness} />
            <ScorePill label="Understandability" value={uatCase.scores.understandability} />
            <ScorePill label="Coverage" value={uatCase.scores.coverage} />
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-3">
          <OverallScoreBadge value={uatCase.overall_score} />
          <AcceptRejectButtons status={uatCase.status} onAccept={onAccept} onReject={onReject} />
        </div>
      </div>
    </div>
  );
}
