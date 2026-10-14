"""Runs the backend test suite and writes a results CSV and a Markdown summary.

Pytest records each test in a JUnit XML report, which is flattened into
`test_results.csv`; `test_summary.md` is then built from that CSV. Both files
are written next to this script.

	python tests/run_tests.py                    # whole suite
	python tests/run_tests.py tests -k voting    # any arguments go to pytest

Exits with pytest's exit code.
"""

import csv
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
RESULTS_CSV = TESTS_DIR / "test_results.csv"
SUMMARY_MD = TESTS_DIR / "test_summary.md"

FIELDS = ["test_id", "file", "outcome", "duration_s", "message"]
OUTCOMES = ["passed", "failed", "error", "skipped"]
# The JUnit element a testcase holds for each outcome other than passed.
JUNIT_OUTCOMES = {"failure": "failed", "error": "error", "skipped": "skipped"}


def run_pytest(args: list[str], report: Path) -> int:
    """Run pytest with `args`, writing a JUnit XML report to `report`."""
    # xunit1 is the JUnit flavour that records each test's file path.
    return pytest.main([*args, f"--junitxml={report}", "-o", "junit_family=xunit1"])


def read_junit(report: Path) -> list[dict]:
    """Turn each <testcase> in a JUnit XML report into a CSV row."""
    rows = []
    for case in ET.parse(report).iter("testcase"):
        result = next(
            (child for child in case if child.tag in JUNIT_OUTCOMES), None)
        message = result.get("message", "") if result is not None else ""
        rows.append({
            "test_id": f"{case.get('file')}::{case.get('name')}",
            "file": case.get("file"),
            "outcome": "passed" if result is None else JUNIT_OUTCOMES[result.tag],
            "duration_s": case.get("time"),
            "message": message.splitlines()[0] if message else "",
        })
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    """Write one row per test to `path`."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def success_rate(counts: Counter) -> str:
    """Passed tests as a percentage of the tests that ran, so skips don't count."""
    ran = counts.total() - counts["skipped"]
    return f"{counts['passed'] / ran:.1%}" if ran else "n/a"


def table(headers: list[str], rows: list[list]) -> list[str]:
    """Format `rows` as the lines of a Markdown table."""
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
        *("| " + " | ".join(str(cell)
                            for cell in row) + " |" for row in rows),
    ]


def write_summary(csv_path: Path, path: Path) -> None:
    """Build a Markdown summary of the results in `csv_path` and write it to `path`."""
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    totals = Counter(row["outcome"] for row in rows)
    by_file = defaultdict(Counter)
    for row in rows:
        by_file[row["file"]][row["outcome"]] += 1
    duration = sum(float(row["duration_s"]) for row in rows)

    lines = [
        "# Test Summary",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} from `{csv_path.name}`.",
        "",
        f"**Success rate: {success_rate(totals)}** "
        f"({totals['passed']} of {totals.total() - totals['skipped']} tests passed, skipped tests not counted)",
        "",
        *table(
            ["Total", *(o.capitalize() for o in OUTCOMES), "Duration"],
            [[totals.total(), *(totals[o]
                                for o in OUTCOMES), f"{duration:.1f}s"]],
        ),
        "",
        "## By file",
        "",
        *table(
            ["File", "Tests", *(o.capitalize()
                                for o in OUTCOMES), "Success rate"],
            [
                [f"`{file}`", counts.total(), *(counts[o]
                                                for o in OUTCOMES), success_rate(counts)]
                for file, counts in sorted(by_file.items())
            ],
        ),
    ]

    not_passed = [row for row in rows if row["outcome"] != "passed"]
    if not_passed:
        lines += [
            "",
            "## Tests that did not pass",
            "",
            *table(
                ["Outcome", "Test", "Message"],
                [[row["outcome"], f"`{row['test_id']}`", row["message"].replace(
                    "|", "\\|")] for row in not_passed],
            ),
        ]

    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    """Run the tests, then write the CSV and the summary built from it."""
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "junit.xml"
        exit_code = run_pytest(sys.argv[1:] or [str(TESTS_DIR)], report)
        write_csv(read_junit(report), RESULTS_CSV)

    write_summary(RESULTS_CSV, SUMMARY_MD)
    print(f"Wrote {RESULTS_CSV} and {SUMMARY_MD}")
    return int(exit_code)


if __name__ == "__main__":
    sys.exit(main())
