#!/usr/bin/env python3
"""Compute de-identified Friedman/Wilcoxon statistics for Table 4 nurse ratings."""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


SYSTEMS = ["Zero-Shot", "FS-GKP", "Original-RAG", "AC-RAG"]
SYSTEM_LETTERS = dict(zip("ABCD", SYSTEMS))
DIMS = ["accuracy", "completeness", "hallucination", "safety", "format", "usefulness"]


def _holm_adjust(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def compute(input_xlsx: Path, output_dir: Path) -> None:
    responses = pd.read_excel(input_xlsx, sheet_name="Responses")
    required = {
        "display_id",
        *[f"system_{letter}" for letter in "ABCD"],
        *[f"{dimension}_{letter}" for dimension in DIMS for letter in "ABCD"],
    }
    if not required.issubset(responses.columns):
        raise ValueError("Nurse-rating workbook has an unexpected schema")
    responses = responses.loc[responses["display_id"].notna()].copy()
    if responses.empty:
        raise ValueError("Nurse-rating workbook contains no responses")
    for letter, expected in SYSTEM_LETTERS.items():
        labels = set(responses[f"system_{letter}"].dropna().astype(str))
        if labels != {expected}:
            raise ValueError("Blinded system mapping is inconsistent across responses")

    long_rows = []
    for _, row in responses.iterrows():
        for letter, system in SYSTEM_LETTERS.items():
            for dimension in DIMS:
                value = pd.to_numeric(row[f"{dimension}_{letter}"], errors="coerce")
                if pd.notna(value):
                    long_rows.append({
                        "case": row["display_id"],
                        "system": system,
                        "dimension": dimension,
                        "rating": float(value),
                    })
    long = pd.DataFrame(long_rows)
    if long.empty:
        raise ValueError("No nurse ratings found")
    case_means = long.groupby(["case", "system", "dimension"], as_index=False)["rating"].mean()

    summary_rows = []
    test_rows = []
    for dimension in DIMS:
        subset = case_means.loc[case_means["dimension"].eq(dimension)]
        wide = subset.pivot(index="case", columns="system", values="rating")
        wide = wide.reindex(columns=SYSTEMS).dropna()
        if len(wide) < 2:
            raise ValueError(f"Too few complete cases for {dimension}")
        for system in SYSTEMS:
            values = wide[system]
            summary_rows.append({
                "dimension": dimension,
                "system": system,
                "complete_case_n": int(len(wide)),
                "mean_rating": float(values.mean()),
                "sd_sample": float(values.std(ddof=1)),
            })
        omnibus = stats.friedmanchisquare(*(wide[system].to_numpy(float) for system in SYSTEMS))
        test_rows.append({
            "dimension": dimension,
            "test": "Friedman",
            "comparison": "all four systems",
            "n": int(len(wide)),
            "statistic": float(omnibus.statistic),
            "p_raw": float(omnibus.pvalue),
            "p_adjusted": "",
            "effect_size": float(omnibus.statistic / (len(wide) * (len(SYSTEMS) - 1))),
            "effect_size_name": "Kendall W",
        })
        pairwise = []
        for first, second in itertools.combinations(SYSTEMS, 2):
            a, b = wide[first].to_numpy(float), wide[second].to_numpy(float)
            test = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
            differences = a - b
            nonzero = differences[differences != 0]
            rank_biserial = (
                float((np.sum(nonzero > 0) - np.sum(nonzero < 0)) / len(nonzero))
                if len(nonzero) else 0.0
            )
            pairwise.append({
                "dimension": dimension,
                "test": "Wilcoxon signed-rank",
                "comparison": f"{first} vs {second}",
                "n": int(len(a)),
                "statistic": float(test.statistic),
                "p_raw": float(test.pvalue),
                "p_adjusted": "",
                "effect_size": rank_biserial,
                "effect_size_name": "matched-pairs rank-biserial correlation",
            })
        for result, adjusted in zip(pairwise, _holm_adjust([row["p_raw"] for row in pairwise])):
            result["p_adjusted"] = adjusted
        test_rows.extend(pairwise)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(output_dir / "table4_nurse_rating_summary.csv", index=False)
    pd.DataFrame(test_rows).to_csv(output_dir / "table4_nurse_rating_tests.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-xlsx", type=Path, required=True, help="Local nurse-rating workbook")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    compute(args.input_xlsx.resolve(), args.output_dir.resolve())
    print(f"Saved de-identified Table 4 statistics to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
