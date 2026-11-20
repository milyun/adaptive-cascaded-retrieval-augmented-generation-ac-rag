#!/usr/bin/env python3
"""Compute the Table 2 constant-mode baseline without exporting any values."""

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List

from table2_scoring import (
    EXPECTED_T,
    EXPECTED_VISITS,
    PARAMETERS,
    blank,
    normalize_ground_truth,
    read_csv,
)
from visit_ids import extract_audio_key


GROUND_TRUTH = "Experiment Result - Information extraction and Summarization - Ground Truth (2).csv"
SYSTEM_FILES = [
    ("AC-RAG", "Deepgram", "visiting_forms_ground_truth_ACRAG_deepgram_v1.csv", "File Name"),
    ("AC-RAG", "Whisper", "visiting_forms_ground_truth_ACRAG_whisper_v1.csv", "File Name"),
    ("FS-GKP", "Deepgram", "extracted_health_info_results_fs-gkp_deepgram.csv", "file"),
    ("FS-GKP", "Whisper", "extracted_health_info_results_fs-gkp_whisper.csv", "file"),
]


def matches(parameter_kind: str, reference: str, mode: str) -> bool:
    if parameter_kind in ("temperature", "numeric"):
        try:
            return abs(float(reference) - float(mode)) <= 0.5
        except (ValueError, TypeError):
            return reference == mode
    return reference == mode


def score_baseline(source: Path) -> List[dict]:
    gt_rows = read_csv(source / GROUND_TRUTH)
    gt_by_key = {}
    for row in gt_rows:
        key = extract_audio_key(row.get("Filename", ""))
        if not key or key in gt_by_key:
            raise ValueError("Ground-truth visit keys must be valid and unique")
        gt_by_key[key] = row

    reference_keys = None
    for _, _, filename, file_column in SYSTEM_FILES:
        rows = read_csv(source / filename)
        keys = [extract_audio_key(row.get(file_column, "")) for row in rows]
        if len(rows) != EXPECTED_VISITS or any(key is None for key in keys) or len(set(keys)) != EXPECTED_VISITS:
            raise ValueError(f"Expected {EXPECTED_VISITS} valid unique consultations in {filename}")
        current = set(keys)
        if reference_keys is None:
            reference_keys = current
        elif current != reference_keys:
            raise ValueError(f"Consultation set differs across configurations: {filename}")
    if not reference_keys.issubset(set(gt_by_key)):
        raise ValueError("Ground-truth consultation set differs from Table 2 systems")
    gt_by_key = {key: gt_by_key[key] for key in reference_keys}

    normalized = {
        key: {
            display: normalize_ground_truth(row.get(jp), kind)
            for display, (jp, kind) in PARAMETERS.items()
        }
        for key, row in gt_by_key.items()
    }
    results = []
    for display, (jp, kind) in PARAMETERS.items():
        references = [row[display] for row in normalized.values() if row[display] is not None]
        if not references:
            raise ValueError(f"No scorable references for {display}")
        mode, _ = Counter(references).most_common(1)[0]
        tp = fp = fn = blank_count = excluded = 0
        for key, row in gt_by_key.items():
            reference = normalized[key][display]
            raw = row.get(jp)
            if reference is None and not blank(raw):
                excluded += 1
                continue
            if blank(raw):
                blank_count += 1
            if reference is not None and matches(kind, reference, mode):
                tp += 1
            else:
                fp += 1
                if reference is not None:
                    fn += 1
        extracted = tp + fp
        precision = tp / extracted if extracted else 0
        recall = tp / len(references) if references else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        results.append({
            "parameter": display,
            "T": len(references),
            "blank_references": blank_count,
            "excluded_references": excluded,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "predicted_fields": extracted,
            "precision_pct": round(precision * 100, 1),
            "recall_pct": round(recall * 100, 1),
            "f1_pct": round(f1 * 100, 1),
        })

    total_tp = sum(row["TP"] for row in results)
    total_fp = sum(row["FP"] for row in results)
    total_fn = sum(row["FN"] for row in results)
    total_predicted = total_tp + total_fp
    total_precision = total_tp / total_predicted if total_predicted else 0
    total_recall = total_tp / EXPECTED_T
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if total_precision + total_recall else 0
    results.append({
        "parameter": "Overall (micro)",
        "T": EXPECTED_T,
        "blank_references": sum(row["blank_references"] for row in results),
        "excluded_references": sum(row["excluded_references"] for row in results),
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "predicted_fields": total_predicted,
        "precision_pct": round(total_precision * 100, 1),
        "recall_pct": round(total_recall * 100, 1),
        "f1_pct": round(total_f1 * 100, 1),
    })
    if sum(row["T"] for row in results[:-1]) != EXPECTED_T:
        raise ValueError(f"Expected T={EXPECTED_T}")
    if results[-1]["predicted_fields"] != 622:
        raise ValueError("Expected 622 predicted fields")
    temperature = next(row for row in results if row["parameter"] == "Body Temperature (°C)")
    if temperature["TP"] != 81:
        raise ValueError("Expected temperature TP=81")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="Local private ground-truth inputs")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    results = score_baseline(args.source_dir.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "constant_mode_baseline_table2.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    total = results[-1]
    lines = [
        "# Constant-Mode Baseline for Table 2",
        "",
        "The modal reference is used internally for scoring and is deliberately not exported. "
        "Temperature matches within ±0.5 °C; other numeric values use the same Table 2 tolerance, "
        "and blood pressure and binary fields require exact normalized matches. Blank reference "
        "slots count as false positives; the incomplete blood-pressure reference is excluded.",
        "",
        "| Parameter | T | Blank | Excluded | TP | FP | FN | Predicted fields | Precision (%) | Recall (%) | F1 (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            "| {parameter} | {T} | {blank_references} | {excluded_references} | {TP} | {FP} | {FN} | "
            "{predicted_fields} | {precision_pct} | {recall_pct} | {f1_pct} |".format(**row)
        )
    lines.extend([
        "",
        f"Checks: visits = {EXPECTED_VISITS}; T = {total['T']}; "
        f"predicted fields = {total['predicted_fields']}; temperature TP = 81.",
        "",
    ])
    (output_dir / "constant_mode_baseline_table2.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved de-identified baseline outputs to {output_dir}")


if __name__ == "__main__":
    main()
