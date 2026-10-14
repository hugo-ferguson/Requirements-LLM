"""Runs the backend test suite and records the results.

Writes one row per test to `test_results.csv`, then rebuilds
`test_summary.md` from that CSV: the success rate, a per-file breakdown and
the reason for every test that did not pass. Both files land next to this
script.

	python tests/run_tests.py                    # whole suite
	python tests/run_tests.py tests -k voting    # extra args go to pytest
	python tests/run_tests.py --summary-only     # rebuild the summary from the last CSV

Exits with pytest's exit code, so it can stand in for `pytest` in CI.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
RESULTS_CSV = TESTS_DIR / "test_results.csv"
SUMMARY_MD = TESTS_DIR / "test_summary.md"

FIELDS = ["test_id", "file", "test", "outcome", "duration_s", "message"]
# Outcomes that count towards the success rate. Skipped and xfailed tests
# were never expected to pass, so they are reported but left out of it.
COUNTED = ("passed", "xpassed", "failed", "error")
OUTCOMES = ("passed", "failed", "error", "skipped", "xfailed", "xpassed")


class ResultCollector:
	"""Pytest plugin that folds each test's setup/call/teardown into one row."""

	def __init__(self) -> None:
		self.rows: dict[str, dict] = {}

	def pytest_collectreport(self, report: pytest.CollectReport) -> None:
		if report.failed:
			row = self._row(report.nodeid)
			row["outcome"] = "error"
			row["message"] = _message(report)

	def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
		row = self._row(report.nodeid)
		row["duration_s"] += report.duration
		if report.when == "call":
			if hasattr(report, "wasxfail"):
				row["outcome"] = "xfailed" if report.skipped else "xpassed"
			else:
				row["outcome"] = report.outcome
		elif report.failed:
			row["outcome"] = "error"
		elif report.skipped:
			row["outcome"] = "skipped"
		if not report.passed and not row["message"]:
			row["message"] = _message(report)

	def _row(self, nodeid: str) -> dict:
		if nodeid not in self.rows:
			file, _, test = nodeid.partition("::")
			self.rows[nodeid] = {
				"test_id": nodeid,
				"file": file,
				"test": test,
				"outcome": "passed",
				"duration_s": 0.0,
				"message": "",
			}
		return self.rows[nodeid]


def _message(report: pytest.CollectReport | pytest.TestReport) -> str:
	if hasattr(report, "wasxfail"):
		return report.wasxfail
	longrepr = report.longrepr
	if isinstance(longrepr, tuple):
		return longrepr[2].removeprefix("Skipped: ")
	crash = getattr(longrepr, "reprcrash", None)
	if crash is not None:
		return crash.message.splitlines()[0]
	text = report.longreprtext.strip()
	return text.splitlines()[-1] if text else ""


def write_results(rows: list[dict], path: Path = RESULTS_CSV) -> None:
	with path.open("w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=FIELDS)
		writer.writeheader()
		for row in rows:
			writer.writerow({**row, "duration_s": f"{row['duration_s']:.3f}"})


def read_results(path: Path = RESULTS_CSV) -> list[dict]:
	with path.open(newline="") as f:
		return list(csv.DictReader(f))


def success_rate(counts: Counter) -> str:
	counted = sum(counts[o] for o in COUNTED)
	if not counted:
		return "n/a"
	return f"{(counts['passed'] + counts['xpassed']) / counted:.1%}"


def write_summary(rows: list[dict], path: Path = SUMMARY_MD) -> None:
	counts = Counter(row["outcome"] for row in rows)
	duration = sum(float(row["duration_s"]) for row in rows)
	by_file: dict[str, Counter] = defaultdict(Counter)
	for row in rows:
		by_file[row["file"]][row["outcome"]] += 1
	present = [o for o in OUTCOMES if counts[o]]

	lines = [
		"# Test Summary",
		"",
		f"Generated {datetime.now():%Y-%m-%d %H:%M} from `{RESULTS_CSV.name}`.",
		"",
		f"**Success rate: {success_rate(counts)}** "
		f"({counts['passed'] + counts['xpassed']} of {sum(counts[o] for o in COUNTED)} tests passed; "
		"skipped and xfailed tests are not counted)",
		"",
		"| Total | " + " | ".join(o.capitalize() for o in present) + " | Duration |",
		"|---:|" + "---:|" * len(present) + "---:|",
		f"| {len(rows)} | " + " | ".join(str(counts[o]) for o in present) + f" | {duration:.1f}s |",
		"",
		"## By file",
		"",
		"| File | Tests | " + " | ".join(o.capitalize() for o in present) + " | Success rate |",
		"|---|---:|" + "---:|" * len(present) + "---:|",
	]
	for file, file_counts in sorted(by_file.items()):
		lines.append(
			f"| `{file}` | {file_counts.total()} | "
			+ " | ".join(str(file_counts[o]) for o in present)
			+ f" | {success_rate(file_counts)} |"
		)

	not_passed = [row for row in rows if row["outcome"] not in ("passed", "xpassed")]
	if not_passed:
		lines += ["", "## Tests that did not pass", "", "| Outcome | Test | Message |", "|---|---|---|"]
		for row in sorted(not_passed, key=lambda r: (OUTCOMES.index(r["outcome"]), r["test_id"])):
			message = row["message"].replace("|", "\\|")
			lines.append(f"| {row['outcome']} | `{row['test_id']}` | {message} |")

	path.write_text("\n".join(lines) + "\n")


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--summary-only", action="store_true", help=f"rebuild {SUMMARY_MD.name} from the existing CSV")
	args, pytest_args = parser.parse_known_args()

	exit_code = 0
	if not args.summary_only:
		collector = ResultCollector()
		exit_code = int(pytest.main(pytest_args or [str(TESTS_DIR)], plugins=[collector]))
		write_results(list(collector.rows.values()))
		print(f"Wrote {RESULTS_CSV}")

	write_summary(read_results())
	print(f"Wrote {SUMMARY_MD}")
	return exit_code


if __name__ == "__main__":
	sys.exit(main())
