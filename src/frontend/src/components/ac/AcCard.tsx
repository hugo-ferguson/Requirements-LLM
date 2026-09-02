import { useEffect, useState } from "react";
import type { AcceptanceCriterion } from "../../api/acceptanceCriteria";
import { ScorePill } from "./ScorePill";
import { OverallScoreBadge } from "./OverallScoreBadge";
import { AcceptRejectButtons } from "./AcceptRejectButtons";

interface EditFields {
  title: string;
  given: string;
  when: string;
  then: string;
}

interface AcCardProps {
  criterion: AcceptanceCriterion;
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

export function AcCard({
  criterion,
  index,
  selected = false,
  editing = false,
  onSelect,
  onEditStart,
  onEditCancel,
  onEditSave,
  onAccept,
  onReject,
}: AcCardProps) {
  const [draft, setDraft] = useState<EditFields>(criterion);

  // Re-seed the draft from the current values each time edit mode is entered,
  // so a previous (cancelled) edit never leaks into a later edit session.
  useEffect(() => {
    if (editing) {
      setDraft({
        title: criterion.title,
        given: criterion.given,
        when: criterion.when,
        then: criterion.then,
      });
    }
  }, [editing, criterion]);

  const cardClasses = `rounded-xl border bg-white p-4 shadow-sm ${
    selected ? "border-primary ring-2 ring-primary ring-offset-2" : "border-gray-200"
  }`;

  if (editing) {
    return (
      <div className={cardClasses}>
        <div className="mb-3 flex items-center gap-2">
          <span className="text-sm font-semibold text-gray-400">#{index + 1}</span>
          <input
            value={draft.title}
            onChange={(event) => setDraft((prev) => ({ ...prev, title: event.target.value }))}
            className="flex-1 rounded-lg border border-gray-200 px-3 py-1.5 font-bold text-gray-900 outline-none focus:border-primary"
          />
        </div>
        <div className="grid gap-3 sm:grid-cols-3">
          {(["given", "when", "then"] as const).map((field) => (
            <label key={field} className="flex flex-col gap-1 text-xs font-medium text-gray-500">
              {field.toUpperCase()}
              <textarea
                value={draft[field]}
                onChange={(event) =>
                  setDraft((prev) => ({ ...prev, [field]: event.target.value }))
                }
                rows={3}
                className="resize-none rounded-lg border border-gray-200 p-2 text-sm text-gray-900 outline-none focus:border-primary"
              />
            </label>
          ))}
        </div>
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
        <div
          onClick={onSelect}
          className={`min-w-0 flex-1 ${onSelect ? "cursor-pointer" : ""}`}
        >
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-gray-400">#{index + 1}</span>
            <span className="font-bold text-gray-900">{criterion.title}</span>
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
          <p className="mt-1 text-sm text-gray-600">
            GIVEN {criterion.given}, WHEN {criterion.when}, THEN {criterion.then}
          </p>
          <div className="mt-2 flex flex-wrap gap-2">
            <ScorePill label="Relevance" value={criterion.scores.relevance} />
            <ScorePill label="Correctness" value={criterion.scores.correctness} />
            <ScorePill label="Understandability" value={criterion.scores.understandability} />
            <ScorePill label="Coverage" value={criterion.scores.coverage} />
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-3">
          <OverallScoreBadge value={criterion.overall_score} />
          <AcceptRejectButtons status={criterion.status} onAccept={onAccept} onReject={onReject} />
        </div>
      </div>
    </div>
  );
}
