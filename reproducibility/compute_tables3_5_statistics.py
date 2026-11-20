#!/usr/bin/env python3
"""Compute aggregate Tables 3-5 statistics from de-identified score files."""

import argparse
import csv
import itertools
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


STRATEGIES = ["Hybrid", "History-Only", "Cross-Patient-Only", "No-RAG"]
METRICS = [
    "entity_recall", "numerical_recall", "section_completeness",
    "semantic_similarity", "tfidf_cosine", "jaccard_similarity",
    "cosine_similarity", "vocab_overlap", "rouge1_f1", "rouge2_f1",
    "rougeL_f1_word", "rougeL_f1_char", "bleu_word", "bleu_char",
    "bertscore_f1",
]
CONFIGS = [
    ("AC-RAG", "Deepgram", "AC_RAG_Deepgram"),
    ("AC-RAG", "Whisper", "AC_RAG_Whisper"),
    ("FS-GKP", "Deepgram", "FS_GKP_Deepgram"),
    ("FS-GKP", "Whisper", "FS_GKP_Whisper"),
    ("Original-RAG", "Deepgram", "Original_RAG_Deepgram"),
    ("Original-RAG", "Whisper", "Original_RAG_Whisper"),
    ("Zero-Shot", "Deepgram", "Zero_Shot_Deepgram"),
    ("Zero-Shot", "Whisper", "Zero_Shot_Whisper"),
]


def load_inputs(detailed_path: Path, assignments_path: Path):
    detailed = pd.read_csv(detailed_path)
    assignments = pd.read_csv(assignments_path)
    if "visit_id" not in detailed or not detailed["visit_id"].is_unique or len(detailed) != 89:
        raise ValueError("Detailed results must contain 89 unique pseudonymous visits")
    required = {"visit_id", "asr", "strategy"}
    if not required.issubset(assignments.columns) or len(assignments) != 178:
        raise ValueError("Strategy assignments must contain 178 pseudonymous summaries")
    if assignments.duplicated(["visit_id", "asr"]).any():
        raise ValueError("Each visit and ASR pair must have exactly one strategy assignment")
    if set(assignments["strategy"]) != set(STRATEGIES):
        raise ValueError("Unexpected retrieval strategy labels")
    if set(assignments["visit_id"]) != set(detailed["visit_id"]):
        raise ValueError("Strategy and score visit sets do not match")
    return detailed, assignments


def write_table3(detailed: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for strategy, asr, prefix in CONFIGS:
        for metric in METRICS:
            column = f"{prefix}_{metric}"
            if column not in detailed:
                continue
            values = pd.to_numeric(detailed[column], errors="coerce").dropna()
            rows.append({
                "configuration": strategy,
                "asr": asr,
                "metric": metric,
                "n": int(values.size),
                "mean": float(values.mean()),
                "sd_population": float(values.std(ddof=0)),
            })
    if not rows:
        raise ValueError("No per-summary evaluation metrics found")
    pd.DataFrame(rows).to_csv(output_dir / "table3_system_summary.csv", index=False)


def _holm_adjust(p_values: List[float]) -> List[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (len(p_values) - rank) * p_values[index]
        running = max(running, candidate)
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def _paired_effect(first: np.ndarray, second: np.ndarray) -> Tuple[float, float]:
    difference = first - second
    standard_deviation = difference.std(ddof=1)
    cohens_dz = float(difference.mean() / standard_deviation) if standard_deviation else 0.0
    nonzero = difference[difference != 0]
    rank_biserial = (
        float((np.sum(nonzero > 0) - np.sum(nonzero < 0)) / len(nonzero))
        if len(nonzero) else 0.0
    )
    return cohens_dz, rank_biserial


def write_table3_pairwise_effect_sizes(detailed: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for metric in METRICS:
        available = [
            (strategy, asr, f"{prefix}_{metric}")
            for strategy, asr, prefix in CONFIGS
            if f"{prefix}_{metric}" in detailed
        ]
        pair_rows = []
        for first, second in itertools.combinations(available, 2):
            first_strategy, first_asr, first_column = first
            second_strategy, second_asr, second_column = second
            first_values = pd.to_numeric(detailed[first_column], errors="coerce").to_numpy(float)
            second_values = pd.to_numeric(detailed[second_column], errors="coerce").to_numpy(float)
            if not np.isfinite(first_values).all() or not np.isfinite(second_values).all():
                raise ValueError(f"Missing scores for Table 3 metric {metric}")
            paired_t = stats.ttest_rel(first_values, second_values)
            wilcoxon = stats.wilcoxon(first_values, second_values, alternative="two-sided", zero_method="wilcox")
            dz, rank_biserial = _paired_effect(first_values, second_values)
            pooled_variance = (
                (len(first_values) - 1) * first_values.var(ddof=1)
                + (len(second_values) - 1) * second_values.var(ddof=1)
            ) / (len(first_values) + len(second_values) - 2)
            cohens_d = (
                float((first_values.mean() - second_values.mean()) / np.sqrt(pooled_variance))
                if pooled_variance else 0.0
            )
            pair_rows.append({
                "metric": metric,
                "configuration_1": f"{first_strategy}-{first_asr}",
                "configuration_2": f"{second_strategy}-{second_asr}",
                "n": len(first_values),
                "mean_difference": float((first_values - second_values).mean()),
                "paired_t": float(paired_t.statistic),
                "p_paired_t": float(paired_t.pvalue),
                "wilcoxon_statistic": float(wilcoxon.statistic),
                "p_wilcoxon": float(wilcoxon.pvalue),
                "cohens_d": cohens_d,
                "cohens_dz": dz,
                "rank_biserial": rank_biserial,
            })
        adjusted_t = _holm_adjust([row["p_paired_t"] for row in pair_rows])
        adjusted_w = _holm_adjust([row["p_wilcoxon"] for row in pair_rows])
        for row, p_t, p_w in zip(pair_rows, adjusted_t, adjusted_w):
            row["p_paired_t_holm"] = p_t
            row["p_wilcoxon_holm"] = p_w
        rows.extend(pair_rows)
    if not rows:
        raise ValueError("No complete Table 3 comparisons found")
    pd.DataFrame(rows).to_csv(output_dir / "table3_pairwise_effect_sizes.csv", index=False)


def write_table4_friedman_wilcoxon(detailed: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    systems = ["AC-RAG", "FS-GKP", "Original-RAG", "Zero-Shot"]
    for metric in METRICS:
        for asr in ("Deepgram", "Whisper"):
            columns = {
                strategy: f"{prefix}_{metric}"
                for strategy, configured_asr, prefix in CONFIGS
                if configured_asr == asr
            }
            if set(columns) != set(systems):
                raise ValueError(f"Missing Table 4 configurations for {metric}, {asr}")
            scores = {
                system: pd.to_numeric(detailed[columns[system]], errors="coerce").to_numpy(float)
                for system in systems
            }
            if any(not np.isfinite(values).all() for values in scores.values()):
                raise ValueError(f"Missing Table 4 scores for {metric}, {asr}")
            omnibus = stats.friedmanchisquare(*(scores[system] for system in systems))
            rows.append({
                "metric": metric,
                "asr": asr,
                "test": "Friedman",
                "comparison": "AC-RAG vs FS-GKP vs Original-RAG vs Zero-Shot",
                "n": len(scores[systems[0]]),
                "statistic": float(omnibus.statistic),
                "p_raw": float(omnibus.pvalue),
                "p_adjusted": "",
                "cohens_dz": "",
                "rank_biserial": "",
            })
            pair_rows = []
            for first, second in itertools.combinations(systems, 2):
                first_values, second_values = scores[first], scores[second]
                if len(first_values) != len(second_values):
                    raise ValueError(f"Unpaired Table 4 scores for {metric}, {asr}")
                test = stats.wilcoxon(
                    first_values,
                    second_values,
                    alternative="two-sided",
                    zero_method="wilcox",
                )
                dz, rank_biserial = _paired_effect(first_values, second_values)
                pair_rows.append({
                    "metric": metric,
                    "asr": asr,
                    "test": "Wilcoxon signed-rank",
                    "comparison": f"{first} vs {second}",
                    "n": len(first_values),
                    "statistic": float(test.statistic),
                    "p_raw": float(test.pvalue),
                    "p_adjusted": "",
                    "cohens_dz": dz,
                    "rank_biserial": rank_biserial,
                })
            for row, adjusted in zip(
                pair_rows, _holm_adjust([row["p_raw"] for row in pair_rows])
            ):
                row["p_adjusted"] = adjusted
            rows.extend(pair_rows)
    if not rows:
        raise ValueError("No complete Table 4 comparisons found")
    pd.DataFrame(rows).to_csv(output_dir / "table4_friedman_wilcoxon.csv", index=False)


def write_table5(detailed: pd.DataFrame, assignments: pd.DataFrame, output_dir: Path) -> None:
    summary_rows = []
    pair_rows = []
    for asr in ("Deepgram", "Whisper"):
        joined = assignments.loc[assignments["asr"].eq(asr)].merge(
            detailed[["visit_id", f"AC_RAG_{asr}_bertscore_f1"]],
            on="visit_id", how="left", validate="one_to_one",
        )
        values_by_strategy = {}
        for strategy in STRATEGIES:
            values = pd.to_numeric(
                joined.loc[joined["strategy"].eq(strategy), f"AC_RAG_{asr}_bertscore_f1"],
                errors="coerce",
            ).dropna()
            if values.empty:
                raise ValueError(f"No BERTScore values for {strategy}, {asr}")
            values_by_strategy[strategy] = values.to_numpy(float)
            summary_rows.append({
                "strategy": strategy,
                "asr": asr,
                "n": int(values.size),
                "mean": float(values.mean()),
                "sd_sample": float(values.std(ddof=1)),
            })
        pair_asr = []
        for first_name, second_name in itertools.combinations(STRATEGIES, 2):
            a = values_by_strategy[first_name]
            b = values_by_strategy[second_name]
            if min(len(a), len(b)) < 2:
                continue
            result = stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
            pooled_variance = (
                (len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)
            ) / (len(a) + len(b) - 2)
            cohens_d = float((a.mean() - b.mean()) / np.sqrt(pooled_variance)) if pooled_variance else 0.0
            pair_asr.append({
                "strategy_1": first_name, "strategy_2": second_name, "asr": asr,
                "test": "Mann-Whitney U", "n_1": len(a), "n_2": len(b),
                "statistic": float(result.statistic), "p_raw": float(result.pvalue),
                "cohens_d": cohens_d,
            })
        adjusted = _holm_adjust([row["p_raw"] for row in pair_asr])
        for row, p_adjusted in zip(pair_asr, adjusted):
            row["p_adjusted"] = p_adjusted
        pair_rows.extend(pair_asr)
    pd.DataFrame(summary_rows).to_csv(output_dir / "table5_bertscore_summary.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(output_dir / "table5_bertscore_mann_whitney.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detailed", type=Path, required=True, help="De-identified detailed_per_sample_results.csv")
    parser.add_argument("--strategy-assignments", type=Path, required=True, help="De-identified visit_id/asr/strategy CSV")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed, assignments = load_inputs(args.detailed, args.strategy_assignments)
    write_table3(detailed, output_dir)
    write_table3_pairwise_effect_sizes(detailed, output_dir)
    write_table4_friedman_wilcoxon(detailed, output_dir)
    write_table5(detailed, assignments, output_dir)
    print(f"Saved de-identified Tables 3-5 statistics to {output_dir}")


if __name__ == "__main__":
    main()
