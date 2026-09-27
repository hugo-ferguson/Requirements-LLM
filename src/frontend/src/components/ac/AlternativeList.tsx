import type { AcceptanceCriterionAlternative } from "../../api/acceptanceCriteria";
import { ScorePill } from "./ScorePill";
import { OverallScoreBadge } from "./OverallScoreBadge";

interface AlternativeListProps {
  alternatives: AcceptanceCriterionAlternative[];
  swappingId: number | null;
  onUse: (candidateId: number) => void;
}

/** Other models' versions of one AC, each swappable into the main card. */
export function AlternativeList({ alternatives, swappingId, onUse }: AlternativeListProps) {
  return (
    <div className="mt-3 space-y-2 border-t border-gray-100 pt-3">
      {alternatives.map((alt) => {
        const candidateId = alt.candidate_id;
        return (
          <div
            key={candidateId ?? `${alt.source_agent}-${alt.given}`}
            className="rounded-lg border border-gray-200 bg-gray-50 p-3"
          >
            <div className="flex items-start gap-4">
              <div className="min-w-0 flex-1">
                <span className="text-xs font-medium text-gray-500">
                  by {alt.source_agent ?? "unknown model"}
                </span>
                <p className="mt-1 text-sm text-gray-600">
                  GIVEN {alt.given}, WHEN {alt.when}, THEN {alt.then}
                </p>
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
