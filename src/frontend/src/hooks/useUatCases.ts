import { useCallback, useEffect, useState } from "react";
import { uatCasesApi } from "../api/uatCases";
import type { UatCase, UatCaseGroup, UatCaseStatus } from "../api/uatCases";

export function useUatCases(sessionId: string) {
  const [groups, setGroups] = useState<UatCaseGroup[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setIsLoading(true);
    setLoadError(null);
    uatCasesApi
      .list(sessionId)
      .then((result) => {
        if (!cancelled) setGroups(result.groups);
      })
      .catch(() => {
        if (!cancelled) setLoadError("Couldn't load UAT cases. Please try again.");
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  function spliceCase(updated: UatCase) {
    setGroups((prev) =>
      prev.map((group) =>
        group.ac.id === updated.ac_id
          ? {
              ...group,
              uat_cases: group.uat_cases.map((c) => (c.id === updated.id ? updated : c)),
            }
          : group,
      ),
    );
  }

  const updateText = useCallback(
    async (uatId: number, fields: { title: string; description: string }) => {
      const updated = await uatCasesApi.updateText(sessionId, uatId, fields);
      spliceCase(updated);
    },
    [sessionId],
  );

  const updateStatus = useCallback(
    async (uatId: number, status: UatCaseStatus) => {
      const updated = await uatCasesApi.updateStatus(sessionId, uatId, status);
      spliceCase(updated);
    },
    [sessionId],
  );

  const applyApproved = useCallback(
    async (uatId: number, candidates: UatCase[]) => {
      const updatedGroup = await uatCasesApi.applyApproved(sessionId, uatId, candidates);
      setGroups((prev) =>
        prev.map((group) => (group.ac.id === updatedGroup.ac.id ? updatedGroup : group)),
      );
    },
    [sessionId],
  );

  return { groups, isLoading, loadError, updateText, updateStatus, applyApproved };
}
