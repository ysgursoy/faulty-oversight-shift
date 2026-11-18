"""Generate and write a dataset release. Same as `oversight-study generate`."""

import sys

from oversight_study.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["generate", *sys.argv[1:]]))
