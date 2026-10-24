import type { AcceptanceCriterion } from "../../api/acceptanceCriteria";

interface AcGroupHeaderProps {
  criterion: AcceptanceCriterion;
  /** Zero-based; shown as "#AC 1" and so on. */
  index: number;
  /** Expanded adds Given/When/Then below the title. */
  isExpanded: boolean;
  onToggle: () => void;
}

/** The acceptance criterion at the head of a group of UAT cases. */
export function AcGroupHeader({ criterion, index, isExpanded, onToggle }: AcGroupHeaderProps) {
  if (!isExpanded) {
    return (
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full items-center justify-between rounded-xl border border-gray-200 bg-gray-50 px-4 py-3 text-left text-gray-600 hover:bg-gray-100"
      >
        <span>
          <span className="mr-2 text-sm font-semibold text-gray-400">#AC {index + 1}</span>
          <span className="font-semibold">{criterion.title}</span>
        </span>
        <span aria-hidden="true">⌄</span>
      </button>
    );
  }

  return (
    <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-gray-400">#AC {index + 1}</span>
            <span className="font-bold text-gray-900">{criterion.title}</span>
          </div>
          <p className="mt-1 text-sm text-gray-600">
            GIVEN {criterion.given}, WHEN {criterion.when}, THEN {criterion.then}
          </p>
        </div>
        <button
          type="button"
          onClick={onToggle}
          aria-label="Collapse"
          className="shrink-0 text-gray-400 hover:text-gray-700"
        >
          ⌃
        </button>
      </div>
    </div>
  );
}
