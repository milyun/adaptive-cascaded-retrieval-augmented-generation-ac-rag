#!/usr/bin/env python3
"""Evaluate all eight system/ASR combinations from local source CSVs."""

import argparse
import subprocess
import sys
from pathlib import Path


FILES_AND_LABELS = [
    ("visiting_forms_ground_truth_ACRAG_deepgram_v1.csv", "AC-RAG-Deepgram"),
    ("visiting_forms_ground_truth_ACRAG_whisper_v1.csv", "AC-RAG-Whisper"),
    ("visiting_forms_ground_truth_FS-GKP_deepgram.csv", "FS-GKP-Deepgram"),
    ("visiting_forms_ground_truth_FS-GKP_whisper.csv", "FS-GKP-Whisper"),
    ("visiting_forms_ground_truth_originalRAG_deepgram.csv", "Original-RAG-Deepgram"),
    ("visiting_forms_ground_truth_originalRAG_whisper.csv", "Original-RAG-Whisper"),
    ("visiting_forms_ground_truth_zeroShot_deepgram.csv", "Zero-Shot-Deepgram"),
    ("visiting_forms_ground_truth_zeroShot_whisper.csv", "Zero-Shot-Whisper"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="Local private evaluation inputs")
    parser.add_argument("--output-dir", type=Path, help="Local raw evaluation output directory")
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else source_dir / "batch_evaluation_results_all_systems"
    )
    input_files = [source_dir / filename for filename, _ in FILES_AND_LABELS]
    missing = [path for path in input_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing evaluation inputs: " + ", ".join(str(path) for path in missing)
        )

    evaluation_script = Path(__file__).with_name("evaluation_pipeline_multi.py")
    command = [
        sys.executable,
        str(evaluation_script),
        "--files",
        *(str(path) for path in input_files),
        "--labels",
        *(label for _, label in FILES_AND_LABELS),
        "--output_dir",
        str(output_dir),
    ]
    subprocess.run(command, check=True, cwd=source_dir)
    print(f"Evaluation outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
