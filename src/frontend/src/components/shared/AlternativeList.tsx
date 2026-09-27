import { useState, type ReactNode } from "react";
import { ScorePill } from "../ac/ScorePill";
import { OverallScoreBadge } from "../ac/OverallScoreBadge";

/** The fields every alternative shares, whether it's an AC or a UAT case. */
export interface AlternativeItem {
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

interface AlternativeListProps<T extends AlternativeItem> {
  alternatives: T[];
  swappingId: number | null;
  onUse: (candidateId: number) => void;
  /** The version's own text: Given/When/Then for an AC, the description for a UAT case. */
  renderBody: (alternative: T) => ReactNode;
}

/** Other models' versions of one item, each swappable into the main card. */
export function AlternativeList<T extends AlternativeItem>({
  alternatives,
  swappingId,
  onUse,
  renderBody,
}: AlternativeListProps<T>) {
  return (
    <div className="mt-3 space-y-2 border-t border-gray-100 pt-3">
      {alternatives.map((alt, index) => {
        const candidateId = alt.candidate_id;
        return (
          <div
            key={candidateId ?? `${alt.source_agent}-${index}`}
            className="rounded-lg border border-gray-200 bg-gray-50 p-3"
          >
            <div className="flex items-start gap-4">
              <div className="min-w-0 flex-1">
                <span className="text-xs font-medium text-gray-500">
                  by {alt.source_agent ?? "unknown model"}
                </span>
                <div className="mt-1 text-sm text-gray-600">{renderBody(alt)}</div>
                <div className="mt-2 flex flex-wrap gap-2">
                  <ScorePill label="Relevance" value={alt.scores.relevance} />
                  <ScorePill label="Correctness" value={alt.scores.correctness} />
                  <ScorePill label="Understandability" value={alt.scores.understandability} />
                  <ScorePill label="Coverage" value={alt.scores.coverage} />
                </div>
              </div>
              <div className="flex shrink-0 items-center gap-3">
                <OverallScoreBadge value={alt.overall_score} />
                <button
                  type="button"
                  onClick={() => candidateId != null && onUse(candidateId)}
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
  expanded: boolean;
  onToggle: () => void;
}

/** "▾ N other versions" — sits inside a clickable card, so it stops propagation. */
export function AlternativesToggle({ count, expanded, onToggle }: AlternativesToggleProps) {
  return (
    <button
      type="button"
      onClick={(event) => {
        event.stopPropagation();
        onToggle();
      }}
      aria-expanded={expanded}
      className="mt-2 text-sm font-medium text-primary hover:underline"
    >
      {expanded ? "▴" : "▾"} {count} other version{count === 1 ? "" : "s"}
    </button>
  );
}

/** Expanded/collapsed and in-flight state for one card's alternatives. */
export function useAlternativeSwap(onSelectAlternative?: (candidateId: number) => Promise<void>) {
  const [expanded, setExpanded] = useState(false);
  const [swappingId, setSwappingId] = useState<number | null>(null);

  async function use(candidateId: number) {
    if (!onSelectAlternative) return;
    setSwappingId(candidateId);
    try {
      await onSelectAlternative(candidateId);
    } finally {
      setSwappingId(null);
    }
  }

  return { expanded, toggle: () => setExpanded((prev) => !prev), swappingId, use };
}
