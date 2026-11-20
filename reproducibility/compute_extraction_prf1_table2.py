#!/usr/bin/env python3
"""Score the four Table 2 systems at T=601 and export only flags and metrics."""

import argparse
from pathlib import Path

from table2_scoring import score_table2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="Local private evaluation inputs")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    total_scorable, _ = score_table2(args.source_dir.resolve(), args.output_dir.resolve())
    print(f"Table 2 scored at T={total_scorable}; flag-only outputs saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
