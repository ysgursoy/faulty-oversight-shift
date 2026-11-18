"""Run a stage (trial, pilot, main). Same as `oversight-study run`."""

import sys

from oversight_study.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["run", *sys.argv[1:]]))
