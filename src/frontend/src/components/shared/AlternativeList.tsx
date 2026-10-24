import { useState, type ReactNode } from "react";
import { ScorePill } from "../ac/ScorePill";
import { OverallScoreBadge } from "../ac/OverallScoreBadge";

/** Fields shared by AC and UAT case alternatives. */
interface AlternativeItem {
  candidate_id?: number | null;
  source_agent?: string | null;
  scores: {
    relevance: number;
    correctness: number;
    understandability: number;
    coverage: number;
  };
  overall_score: number;
}

interface AlternativeListProps<Alternative extends AlternativeItem> {
  alternatives: Alternative[];
  /** The candidate being swapped in. Disables every button while set. */
  swappingId: number | null;
  onSwap: (candidateId: number) => void;
  /** Given/When/Then for an AC, the description for a UAT case. */
  renderBody: (alternative: Alternative) => ReactNode;
}

/** Other models' versions of one item, each swappable into the main card. */
export function AlternativeList<Alternative extends AlternativeItem>({
  alternatives,
  swappingId,
  onSwap,
  renderBody,
}: AlternativeListProps<Alternative>) {
  return (
    <div className="mt-3 space-y-2 border-t border-gray-100 pt-3">
      {alternatives.map((alternative, index) => {
        const candidateId = alternative.candidate_id;
        return (
          <div
            key={candidateId ?? `${alternative.source_agent}-${index}`}
            className="rounded-lg border border-gray-200 bg-gray-50 p-3"
          >
            <div className="flex items-start gap-4">
              <div className="min-w-0 flex-1">
                <span className="text-xs font-medium text-gray-500">
                  by {alternative.source_agent ?? "unknown model"}
                </span>
                <div className="mt-1 text-sm text-gray-600">{renderBody(alternative)}</div>
                <div className="mt-2 flex flex-wrap gap-2">
                  <ScorePill label="Relevance" value={alternative.scores.relevance} />
                  <ScorePill label="Correctness" value={alternative.scores.correctness} />
                  <ScorePill
                    label="Understandability"
                    value={alternative.scores.understandability}
                  />
                  <ScorePill label="Coverage" value={alternative.scores.coverage} />
                </div>
              </div>
              <div className="flex shrink-0 items-center gap-3">
                <OverallScoreBadge value={alternative.overall_score} />
                <button
                  type="button"
                  onClick={() => candidateId != null && onSwap(candidateId)}
                  disabled={candidateId == null || swappingId != null}
                  className="rounded-full bg-gray-100 px-4 py-1.5 text-sm font-medium text-gray-700 hover:bg-gray-200 disabled:opacity-50"
                >
                  {swappingId === candidateId ? "Swapping…" : "Use this version"}
                </button>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

interface AlternativesToggleProps {
  count: number;
  isExpanded: boolean;
  onToggle: () => void;
}

/** The "▾ N other versions" button. Stops clicks reaching the clickable card. */
export function AlternativesToggle({ count, isExpanded, onToggle }: AlternativesToggleProps) {
  return (
    <button
      type="button"
      onClick={(event) => {
        event.stopPropagation();
        onToggle();
      }}
      aria-expanded={isExpanded}
      className="mt-2 text-sm font-medium text-primary hover:underline"
    >
      {isExpanded ? "▴" : "▾"} {count} other version{count === 1 ? "" : "s"}
    </button>
  );
}

/** Whether a card's alternatives are showing, and which one is being swapped in. */
export function useAlternativeSwap(onSelectAlternative?: (candidateId: number) => Promise<void>) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [swappingId, setSwappingId] = useState<number | null>(null);

  async function swap(candidateId: number) {
    if (!onSelectAlternative) return;
    setSwappingId(candidateId);
    try {
      await onSelectAlternative(candidateId);
    } finally {
      setSwappingId(null);
    }
  }

  return { isExpanded, toggle: () => setIsExpanded((previous) => !previous), swappingId, swap };
}
