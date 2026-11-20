#!/usr/bin/env python3
"""Export aggregate sensitivity results without patient identifiers or text."""

import argparse
import csv
from pathlib import Path

from export_deidentified_outputs import read_csv
from visit_ids import build_visit_id_map, extract_audio_key


def write_selected(source: Path, target: Path, fields, transform=None) -> None:
    headers, rows = read_csv(source)
    if not set(fields).issubset(headers):
        raise ValueError(f"Unexpected sensitivity output schema: {source.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = {field: row[field] for field in fields}
            writer.writerow(transform(record) if transform else record)


def export(source_dir: Path, output_dir: Path) -> None:
    source = source_dir / "sensitivity_results"
    output = output_dir / "sensitivity_results"
    write_selected(
        source / "sensitivity_tier1_weights.csv",
        output / "sensitivity_tier1_weights.csv",
        ["w_age", "w_gender", "mean_kendall_tau", "mean_jaccard_top3", "patients_context_changed"],
    )
    age_labels = ["tight", "default", "loose"]
    _, age_rows = read_csv(source / "sensitivity_tier1_age_thresholds.csv")
    if len(age_rows) != len(age_labels):
        raise ValueError("Unexpected age-threshold sensitivity row count")
    age_path = output / "sensitivity_tier1_age_thresholds.csv"
    age_path.parent.mkdir(parents=True, exist_ok=True)
    with age_path.open("w", encoding="utf-8", newline="") as stream:
        fields = ["configuration", "mean_kendall_tau", "mean_jaccard_top3", "patients_context_changed"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for label, row in zip(age_labels, age_rows):
            writer.writerow({
                "configuration": label,
                "mean_kendall_tau": row["mean_kendall_tau"],
                "mean_jaccard_top3": row["mean_jaccard_top3"],
                "patients_context_changed": row["patients_context_changed"],
            })
    write_selected(
        source / "sensitivity_tier2_volume_embedding.csv",
        output / "sensitivity_tier2_volume_embedding.csv",
        ["tau", "mean_contexts_passing", "pct_cases_with_context"],
    )
    write_selected(
        source / "sensitivity_tier2_generation.csv",
        output / "sensitivity_tier2_generation.csv",
        ["tau", "n_cases", "mean_rouge_l", "std_rouge_l", "mean_tfidf_cosine", "std_tfidf_cosine"],
    )

    case_path = source / "sensitivity_tier2_case_study_embedding.csv"
    _, case_rows = read_csv(case_path)
    visit_path = source_dir / "visiting_forms_ground_truth_ACRAG_deepgram_v1.csv"
    _, visits = read_csv(visit_path)
    if len(case_rows) != 89 or len(visits) != 89:
        raise ValueError("Expected 89 sensitivity cases in source files")
    keys = [extract_audio_key(row.get("File Name", "")) for row in visits]
    if any(key is None for key in keys) or len(set(keys)) != 89:
        raise ValueError("Sensitivity input visit keys must be unique")
    visit_map = build_visit_id_map(keys)
    fields = [
        "visit_id",
        "max_embedding_cosine_same_disease",
        "context_threshold_bucket",
        "main_eval_rouge_l",
        "gets_context_at_tau_0.55",
        "gets_context_at_tau_0.75",
    ]
    case_rows_out = []
    for visit, result in zip(visits, case_rows):
        key = extract_audio_key(visit.get("File Name", ""))
        case_rows_out.append({
            "visit_id": visit_map[key],
            **{field: result[field] for field in fields[1:]},
        })
    output.mkdir(parents=True, exist_ok=True)
    with (output / "sensitivity_tier2_case_study_embedding.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(case_rows_out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="Local directory with sensitivity outputs and visit inputs")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    export(args.source_dir.resolve(), args.output_dir.resolve())
    print(f"Exported de-identified sensitivity summaries to {args.output_dir.resolve() / 'sensitivity_results'}")


if __name__ == "__main__":
    main()
