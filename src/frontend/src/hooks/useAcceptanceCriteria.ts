import { useCallback, useEffect, useState } from "react";
import { acceptanceCriteriaApi } from "../api/acceptanceCriteria";
import type {
  AcceptanceCriterion,
  AcceptanceCriterionStatus,
  AcceptanceCriterionTextUpdate,
} from "../api/acceptanceCriteria";

/** Loads and updates a session's ACs. Ignores loads for a previous session. */
export function useAcceptanceCriteria(sessionId: string) {
  const [criteria, setCriteria] = useState<AcceptanceCriterion[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    let isCancelled = false;
    setIsLoading(true);
    setLoadError(null);
    acceptanceCriteriaApi
      .list(sessionId)
      .then((result) => {
        if (!isCancelled) setCriteria(result.acceptance_criteria);
      })
      .catch(() => {
        if (!isCancelled) setLoadError("Couldn't load acceptance criteria. Please try again.");
      })
      .finally(() => {
        if (!isCancelled) setIsLoading(false);
      });
    return () => {
      isCancelled = true;
    };
  }, [sessionId]);

  function replaceCriterion(updated: AcceptanceCriterion) {
    setCriteria((previous) =>
      previous.map((criterion) => (criterion.id === updated.id ? updated : criterion)),
    );
  }

  const updateText = useCallback(
    async (acId: number, fields: AcceptanceCriterionTextUpdate) => {
      const updated = await acceptanceCriteriaApi.updateText(sessionId, acId, fields);
      replaceCriterion(updated);
    },
    [sessionId],
  );

  const updateStatus = useCallback(
    async (acId: number, status: AcceptanceCriterionStatus) => {
      const updated = await acceptanceCriteriaApi.updateStatus(sessionId, acId, status);
      replaceCriterion(updated);
    },
    [sessionId],
  );

  const applyApproved = useCallback(
    async (acId: number, candidates: AcceptanceCriterion[]) => {
      const result = await acceptanceCriteriaApi.applyApproved(sessionId, acId, candidates);
      setCriteria(result.acceptance_criteria);
    },
    [sessionId],
  );

  const selectAlternative = useCallback(
    async (acId: number, candidateId: number) => {
      const result = await acceptanceCriteriaApi.selectAlternative(sessionId, acId, candidateId);
      replaceCriterion(result.acceptance_criterion);
    },
    [sessionId],
  );

  return {
    criteria,
    isLoading,
    loadError,
    updateText,
    updateStatus,
    applyApproved,
    selectAlternative,
  };
}
