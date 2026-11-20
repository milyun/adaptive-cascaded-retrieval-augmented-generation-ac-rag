#!/usr/bin/env python3
"""Export an aggregate, de-identified audit of Table 3 entity-recall terms."""

import argparse
import re
from pathlib import Path

import pandas as pd

FILES = {
    "AC-RAG-Deepgram": "visiting_forms_ground_truth_ACRAG_deepgram_v1.csv",
    "AC-RAG-Whisper": "visiting_forms_ground_truth_ACRAG_whisper_v1.csv",
    "FS-GKP-Deepgram": "visiting_forms_ground_truth_FS-GKP_deepgram.csv",
    "FS-GKP-Whisper": "visiting_forms_ground_truth_FS-GKP_whisper.csv",
    "Original-RAG-Deepgram": "visiting_forms_ground_truth_originalRAG_deepgram.csv",
    "Original-RAG-Whisper": "visiting_forms_ground_truth_originalRAG_whisper.csv",
    "Zero-Shot-Deepgram": "visiting_forms_ground_truth_zeroShot_deepgram.csv",
    "Zero-Shot-Whisper": "visiting_forms_ground_truth_zeroShot_whisper.csv",
}
METHODS = ["Zero-Shot", "FS-GKP", "Original-RAG", "AC-RAG"]


def normalize(text: str) -> str:
    return re.sub(r"[、。！？]", "", re.sub(r"\s+", "", text)).lower()


def export(source_dir: Path, output_dir: Path, dictionary_csv: Path) -> None:
    frames = {label: pd.read_csv(source_dir / name) for label, name in FILES.items()}
    reference = frames["AC-RAG-Deepgram"]["Ground Truth"].tolist()
    if len(reference) != 89:
        raise ValueError("Expected 89 reference summaries")
    for label, frame in frames.items():
        if len(frame) != 89 or frame["Ground Truth"].tolist() != reference:
            raise ValueError(f"Ground truth differs across systems: {label}")
    reference_norm = [normalize(str(text)) for text in reference]
    dictionary = pd.read_csv(dictionary_csv, keep_default_na=False)
    if not {"category", "term"}.issubset(dictionary.columns) or len(dictionary) != 80:
        raise ValueError("Entity dictionary must contain 80 category/term rows")
    if dictionary.duplicated(["category", "term"]).any():
        raise ValueError("Entity dictionary contains duplicate terms")
    all_terms = [
        (str(row.category).lower(), str(row.term))
        for row in dictionary.itertuples(index=False)
    ]
    rows = []
    for category, term in all_terms:
        present = sum(normalize(term) in text for text in reference_norm)
        row = {
            "category": category,
            "term": term,
            "reference_count_of_89": present,
            "reference_pct": round(100 * present / 89, 1),
            "generic_term": str(present / 89 >= 0.9).lower(),
        }
        for method in METHODS:
            row[f"{method}_summaries_containing_of_178"] = sum(
                normalize(term) in normalize(str(text))
                for asr in ("Deepgram", "Whisper")
                for text in frames[f"{method}-{asr}"]["LLM based summarization"]
            )
        rows.append(row)
    if len(rows) != 80:
        raise ValueError(f"Expected 80 entity terms, found {len(rows)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        output_dir / "entity_recall_term_list.csv", index=False, encoding="utf-8-sig"
    )
    print(f"Exported aggregate audit for {len(rows)} generic terms to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="Local private evaluation inputs")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "entity_recall_audit")
    parser.add_argument(
        "--dictionary-csv",
        type=Path,
        default=Path(__file__).parent / "entity_recall_audit" / "entity_dictionary.csv",
        help="De-identified 80-term dictionary CSV",
    )
    args = parser.parse_args()
    export(args.source_dir.resolve(), args.output_dir.resolve(), args.dictionary_csv.resolve())


if __name__ == "__main__":
    main()
