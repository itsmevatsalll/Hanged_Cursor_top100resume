"""
Runs honeypot detection on the entire candidate pool.

Outputs:
    artifacts/
        clean_candidates.jsonl
        flagged_honeypots.jsonl
        honeypot_summary.csv
"""

import csv
import json
import time
from pathlib import Path

from tqdm import tqdm

from preprocessing.honeypot import is_honeypot


# --------------------------------------------------
# Project Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

ARTIFACTS_DIR.mkdir(exist_ok=True)

INPUT_FILE = DATA_DIR / "candidates.jsonl"

CLEAN_FILE = ARTIFACTS_DIR / "clean_candidates.jsonl"
HONEYPOT_FILE = ARTIFACTS_DIR / "flagged_honeypots.jsonl"
SUMMARY_FILE = ARTIFACTS_DIR / "honeypot_summary.csv"


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    start_time = time.time()

    total = 0
    flagged = 0
    skipped = 0

    reason_counts = {}

    with (
        open(INPUT_FILE, "r", encoding="utf-8") as infile,
        open(CLEAN_FILE, "w", encoding="utf-8") as clean_out,
        open(HONEYPOT_FILE, "w", encoding="utf-8") as honeypot_out,
        open(SUMMARY_FILE, "w", newline="", encoding="utf-8") as csv_out,
    ):

        csv_writer = csv.writer(csv_out)
        csv_writer.writerow(
            [
                "candidate_id",
                "reason",
            ]
        )

        for line in tqdm(
            infile,
            desc="Scanning Candidates",
            unit="candidate",
        ):

            line = line.strip()

            if not line:
                continue

            try:
                candidate = json.loads(line)

            except json.JSONDecodeError:

                skipped += 1
                continue

            total += 1

            hp, reason = is_honeypot(candidate)

            if hp:

                flagged += 1

                reason_counts[reason] = (
                    reason_counts.get(reason, 0) + 1
                )

                candidate["_honeypot_reason"] = reason

                honeypot_out.write(
                    json.dumps(
                        candidate,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                csv_writer.writerow(
                    [
                        candidate.get(
                            "candidate_id",
                            "",
                        ),
                        reason,
                    ]
                )

            else:

                clean_out.write(
                    json.dumps(
                        candidate,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("HONEYPOT DETECTION COMPLETE")
    print("=" * 60)

    print(f"Total Candidates : {total:,}")
    print(f"Flagged          : {flagged:,}")
    print(f"Clean            : {total - flagged:,}")
    print(f"Skipped          : {skipped:,}")

    if total:
        print(
            f"Flag Rate        : "
            f"{100 * flagged / total:.4f}%"
        )

    print(f"\nExecution Time  : {elapsed:.2f} sec")

    print("\nGenerated Files")

    print(f"  {CLEAN_FILE}")
    print(f"  {HONEYPOT_FILE}")
    print(f"  {SUMMARY_FILE}")

    print("\nTop Reasons")

    print("-" * 60)

    for reason, count in sorted(
        reason_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):

        print(f"{count:6d}  {reason}")

    print("=" * 60)


if __name__ == "__main__":
    main()