import { useCallback, useEffect, useState } from "react";
import { uatCasesApi } from "../api/uatCases";
import type { UatCase, UatCaseGroup, UatCaseStatus, UatCaseTextUpdate } from "../api/uatCases";

/** Loads and updates a session's UAT cases. Ignores loads for a previous session. */
export function useUatCases(sessionId: string) {
  const [groups, setGroups] = useState<UatCaseGroup[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    let isCancelled = false;
    setIsLoading(true);
    setLoadError(null);
    uatCasesApi
      .list(sessionId)
      .then((result) => {
        if (!isCancelled) setGroups(result.groups);
      })
      .catch(() => {
        if (!isCancelled) setLoadError("Couldn't load UAT cases. Please try again.");
      })
      .finally(() => {
        if (!isCancelled) setIsLoading(false);
      });
    return () => {
      isCancelled = true;
    };
  }, [sessionId]);

  function replaceUatCase(updated: UatCase) {
    setGroups((previous) =>
      previous.map((group) =>
        group.ac.id === updated.ac_id
          ? {
              ...group,
              uat_cases: group.uat_cases.map((uatCase) =>
                uatCase.id === updated.id ? updated : uatCase,
              ),
            }
          : group,
      ),
    );
  }

  const updateText = useCallback(
    async (uatId: number, fields: UatCaseTextUpdate) => {
      const updated = await uatCasesApi.updateText(sessionId, uatId, fields);
      replaceUatCase(updated);
    },
    [sessionId],
  );

  const updateStatus = useCallback(
    async (uatId: number, status: UatCaseStatus) => {
      const updated = await uatCasesApi.updateStatus(sessionId, uatId, status);
      replaceUatCase(updated);
    },
    [sessionId],
  );

  const applyApproved = useCallback(
    async (uatId: number, candidates: UatCase[]) => {
      const updatedGroup = await uatCasesApi.applyApproved(sessionId, uatId, candidates);
      setGroups((previous) =>
        previous.map((group) => (group.ac.id === updatedGroup.ac.id ? updatedGroup : group)),
      );
    },
    [sessionId],
  );

  const selectAlternative = useCallback(
    async (uatId: number, candidateId: number) => {
      const updated = await uatCasesApi.selectAlternative(sessionId, uatId, candidateId);
      replaceUatCase(updated);
    },
    [sessionId],
  );

  return {
    groups,
    isLoading,
    loadError,
    updateText,
    updateStatus,
    applyApproved,
    selectAlternative,
  };
}
