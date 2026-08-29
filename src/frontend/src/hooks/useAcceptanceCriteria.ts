import { useCallback, useEffect, useState } from "react";
import { acceptanceCriteriaApi } from "../api/acceptanceCriteria";
import type { AcceptanceCriterion, AcceptanceCriterionStatus } from "../api/acceptanceCriteria";

export function useAcceptanceCriteria(sessionId: string) {
  const [items, setItems] = useState<AcceptanceCriterion[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setIsLoading(true);
    setLoadError(null);
    acceptanceCriteriaApi
      .list(sessionId)
      .then((result) => {
        if (!cancelled) setItems(result.acceptance_criteria);
      })
      .catch(() => {
        if (!cancelled) setLoadError("Couldn't load acceptance criteria. Please try again.");
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  const updateText = useCallback(
    async (acId: number, fields: { title: string; given: string; when: string; then: string }) => {
      const updated = await acceptanceCriteriaApi.updateText(sessionId, acId, fields);
      setItems((prev) => prev.map((item) => (item.id === updated.id ? updated : item)));
    },
    [sessionId],
  );

  const updateStatus = useCallback(
    async (acId: number, status: AcceptanceCriterionStatus) => {
      const updated = await acceptanceCriteriaApi.updateStatus(sessionId, acId, status);
      setItems((prev) => prev.map((item) => (item.id === updated.id ? updated : item)));
    },
    [sessionId],
  );

  const applyApproved = useCallback(
    async (targetId: number, candidates: AcceptanceCriterion[]) => {
      const result = await acceptanceCriteriaApi.applyApproved(sessionId, targetId, candidates);
      setItems(result.acceptance_criteria);
    },
    [sessionId],
  );

  return { items, isLoading, loadError, updateText, updateStatus, applyApproved };
}
