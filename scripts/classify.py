"""Second-model classification of a run. Same as `oversight-study classify`."""

import sys

from oversight_study.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["classify", *sys.argv[1:]]))
