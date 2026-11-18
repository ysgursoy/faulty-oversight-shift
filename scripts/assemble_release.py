"""Copy a dataset release and sealed runs into one release directory and validate it."""

import argparse
from pathlib import Path

from oversight_study.validation.release import assemble_release

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", required=True, dest="runs")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = assemble_release(args.dataset, args.runs, args.out)
    print(f"release {args.out}: dataset {report.dataset_version}, {len(report.runs)} run(s)")
