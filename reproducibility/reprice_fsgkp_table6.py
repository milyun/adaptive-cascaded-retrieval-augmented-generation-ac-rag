#!/usr/bin/env python3
"""Reprice combined FS-GKP extraction and summary token logs at Table 6 rates."""

import argparse
import csv
import re
from decimal import Decimal
from pathlib import Path


ASRS = ("deepgram", "whisper")
INPUT_RATE = Decimal("2.50")
OUTPUT_RATE = Decimal("10.00")
MILLION = Decimal(1_000_000)


def read_stage1(path: Path) -> dict[str, int]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        expected = ["visit_id", "input_tokens", "output_tokens", "cost_usd"]
        if reader.fieldnames != expected:
            raise ValueError(f"Unexpected extraction log columns: {path.name}")
        rows = list(reader)
    visits = [row["visit_id"] for row in rows]
    if len(rows) != 89 or len(set(visits)) != len(rows):
        raise ValueError(f"Expected 89 unique pseudonymous visits in {path.name}")
    return {
        "input_tokens": sum(int(row["input_tokens"]) for row in rows),
        "output_tokens": sum(int(row["output_tokens"]) for row in rows),
    }


def read_stage2(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    patterns = {
        "input_tokens": r"Total input tokens:\s*([0-9,]+)",
        "output_tokens": r"Total output tokens:\s*([0-9,]+)",
    }
    result = {}
    for field, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            raise ValueError(f"Missing {field} in aggregate log {path.name}")
        result[field] = int(match.group(1).replace(",", ""))
    return result


def reprice(logs_dir: Path, output: Path) -> None:
    rows = []
    for asr in ASRS:
        stage1 = read_stage1(logs_dir / f"fsgkp_stage1_extraction_{asr}.csv")
        stage2 = read_stage2(logs_dir / f"fsgkp_stage2_summary_{asr}.txt")
        input_tokens = stage1["input_tokens"] + stage2["input_tokens"]
        output_tokens = stage1["output_tokens"] + stage2["output_tokens"]
        cost = (
            Decimal(input_tokens) * INPUT_RATE
            + Decimal(output_tokens) * OUTPUT_RATE
        ) / MILLION
        rows.append({
            "asr": asr.title(),
            "stage1_input_tokens": stage1["input_tokens"],
            "stage1_output_tokens": stage1["output_tokens"],
            "stage2_input_tokens": stage2["input_tokens"],
            "stage2_output_tokens": stage2["output_tokens"],
            "total_input_tokens": input_tokens,
            "total_output_tokens": output_tokens,
            "repriced_cost_usd": f"{cost:.6f}",
            "repriced_cost_usd_rounded": f"{cost:.2f}",
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", type=Path, default=Path(__file__).parent / "table6_logs")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "table6_logs" / "fsgkp_repriced_costs.csv",
    )
    args = parser.parse_args()
    reprice(args.logs_dir.resolve(), args.output.resolve())
    print(f"Saved repriced FS-GKP costs to {args.output.resolve()}")


if __name__ == "__main__":
    main()
