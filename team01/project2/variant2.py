"""Standalone runner for Project 2 variant 2."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


from team01.project2.variant_runner import main_for_variant


def main(argv=None):
    return main_for_variant(2, argv)


if __name__ == "__main__":
    main()