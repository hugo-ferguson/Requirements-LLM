import { useEffect, useState } from "react";
import { useParams } from "react-router";
import { exportApi, type ExportDocument, type ExportResponse } from "../api/export";
import { useConfirm } from "../components/shared/ConfirmDialog";

function countUatCases(exportDocument: ExportDocument): number {
  return exportDocument.acceptance_criteria.reduce(
    (total, criterion) => total + criterion.uat_cases.length,
    0,
  );
}

/** `formatCount(2, "file")` -> "2 files". */
function formatCount(count: number, singular: string, plural = `${singular}s`): string {
  return `${count} ${count === 1 ? singular : plural}`;
}

/** `formatCriteriaCount(2, "approved ")` -> "2 approved acceptance criteria". */
function formatCriteriaCount(count: number, qualifier = ""): string {
  return formatCount(count, `${qualifier}acceptance criterion`, `${qualifier}acceptance criteria`);
}

/** `Login & Signup!` -> `login-signup-export.json`. */
function exportFilename(sessionName: string): string {
  const slug = sessionName
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return `${slug || "session"}-export.json`;
}

function downloadJson(exportDocument: ExportDocument): void {
  const blob = new Blob([JSON.stringify(exportDocument, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = exportFilename(exportDocument.session_name);
  link.click();
  URL.revokeObjectURL(url);
}

/** The Export tab. Warns first if ACs are unapproved or have no approved test cases. */
export function ExportPage() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const { confirm, dialog } = useConfirm();
  const [summary, setSummary] = useState<ExportResponse | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [isExporting, setIsExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);

  useEffect(() => {
    let isCancelled = false;
    setSummary(null);
    setLoadError(null);
    exportApi
      .get(sessionId!)
      .then((result) => {
        if (!isCancelled) setSummary(result);
      })
      .catch(() => {
        if (!isCancelled) setLoadError("Couldn't load what's ready to export. Please try again.");
      });
    return () => {
      isCancelled = true;
    };
  }, [sessionId]);

  async function handleDownload() {
    setExportError(null);
    setIsExporting(true);
    try {
      // Refetch to include approvals made since the page loaded.
      const { document: exportDocument, warnings } = await exportApi.get(sessionId!);
      setSummary({ document: exportDocument, warnings });

      const problems: string[] = [];
      if (warnings.unapproved_ac_count > 0) {
        problems.push(
          `${formatCriteriaCount(warnings.unapproved_ac_count)} ` +
            `${warnings.unapproved_ac_count === 1 ? "isn't" : "aren't"} approved and won't be exported.`,
        );
      }
      if (warnings.approved_ac_without_uat.length > 0) {
        problems.push(
          "These approved acceptance criteria have no approved test cases: " +
            warnings.approved_ac_without_uat.map((title) => `"${title}"`).join(", ") +
            ".",
        );
      }

      if (problems.length > 0) {
        const isConfirmed = await confirm({
          title: "Export anyway?",
          message: (
            <ul className="list-disc space-y-1 pl-5">
              {problems.map((problem) => (
                <li key={problem}>{problem}</li>
              ))}
            </ul>
          ),
          confirmLabel: "Export anyway",
        });
        if (!isConfirmed) return;
      }

      downloadJson(exportDocument);
    } catch {
      setExportError("Couldn't export. Please try again.");
    } finally {
      setIsExporting(false);
    }
  }

  if (loadError) {
    return <div className="flex h-full items-center justify-center text-red-600">{loadError}</div>;
  }

  if (summary == null) {
    return <div className="flex h-full items-center justify-center text-gray-500">Loading…</div>;
  }

  const approvedAcCount = summary.document.acceptance_criteria.length;

  return (
    <div className="flex h-full flex-col items-center justify-center gap-4 p-6 text-center">
      {dialog}
      <h2 className="text-xl font-semibold text-gray-900">Export</h2>
      <p className="max-w-md text-sm text-gray-600">
        Downloads a JSON file of the approved acceptance criteria, each with its approved test cases
        nested under it.
      </p>
      <p className="text-sm font-medium text-gray-700">
        {formatCriteriaCount(approvedAcCount, "approved ")}
        {" · "}
        {formatCount(countUatCases(summary.document), "approved test case")}
      </p>
      <button
        type="button"
        onClick={handleDownload}
        disabled={approvedAcCount === 0 || isExporting}
        className="rounded-full bg-primary px-6 py-2 font-medium text-white hover:bg-primary-hover disabled:opacity-50"
      >
        {isExporting ? "Preparing…" : "Download JSON"}
      </button>
      {approvedAcCount === 0 && (
        <p className="text-sm text-gray-500">Approve at least one acceptance criterion first.</p>
      )}
      {exportError && <p className="text-sm text-red-600">{exportError}</p>}
    </div>
  );
}
