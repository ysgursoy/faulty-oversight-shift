"""Analyze a validated release. Same as `oversight-study analyze`."""

import sys

from oversight_study.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["analyze", *sys.argv[1:]]))
