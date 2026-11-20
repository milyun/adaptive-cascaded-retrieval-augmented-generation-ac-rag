#!/usr/bin/env python3
"""Export safe, visit-pseudonymized result and usage files from private inputs."""

import argparse
import csv
import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from visit_ids import build_visit_id_map, extract_audio_key
from table2_scoring import score_table2


VISITS = 89
TABLE2_GROUND_TRUTH = "Experiment Result - Information extraction and Summarization - Ground Truth (2).csv"
TABLE3_RESULTS = "batch_evaluation_results_all_systems/detailed_per_sample_results.csv"
FS_GKP_FILES = {
    "Deepgram": "extracted_health_info_results_fs-gkp_deepgram.csv",
    "Whisper": "extracted_health_info_results_fs-gkp_whisper.csv",
}
TABLE2_CONFIGS = [
    ("AC-RAG", "Deepgram", "visiting_forms_ground_truth_ACRAG_deepgram_v1.csv"),
    ("AC-RAG", "Whisper", "visiting_forms_ground_truth_ACRAG_whisper_v1.csv"),
    ("FS-GKP", "Deepgram", FS_GKP_FILES["Deepgram"]),
    ("FS-GKP", "Whisper", FS_GKP_FILES["Whisper"]),
]
WHISPER_RUN = "cross_patient_rag_output_whisper_meter_20261108_run2"
TABLE6_RATES = {"input_per_million": Decimal("2.50"), "output_per_million": Decimal("10.00")}
STRATEGY_LABELS = {
    "hybrid": "Hybrid",
    "history_only": "History-Only",
    "cross_patient_only": "Cross-Patient-Only",
    "no_rag": "No-RAG",
}
REPORTED_TIMINGS = {
    ("AC-RAG", "Deepgram"): (Decimal("4.8"), Decimal("7.1")),
    ("AC-RAG", "Whisper"): (Decimal("3.3"), Decimal("4.9")),
    ("FS-GKP", "Deepgram"): (Decimal("53.4"), Decimal("79.2")),
    ("FS-GKP", "Whisper"): (Decimal("31.6"), Decimal("46.9")),
}


def read_csv(path: Path) -> tuple[List[str], List[dict]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path.name}")
        return list(reader.fieldnames), list(reader)


def keys_from_csv(path: Path, column: str) -> set[str]:
    headers, rows = read_csv(path)
    if column not in headers:
        raise ValueError(f"Missing identifier column {column!r} in {path.name}")
    keys = [extract_audio_key(row.get(column, "")) for row in rows]
    if len(rows) != VISITS or any(key is None for key in keys) or len(set(keys)) != VISITS:
        raise ValueError(f"Expected {VISITS} unique visit keys in {path.name}")
    return set(keys)


def collect_visit_map(source: Path) -> Dict[str, str]:
    _, gt_rows = read_csv(source / TABLE2_GROUND_TRUTH)
    gt_keys = [extract_audio_key(row.get("Filename", "")) for row in gt_rows]
    if any(key is None for key in gt_keys) or len(set(gt_keys)) != len(gt_keys):
        raise ValueError("Ground-truth visit keys must be valid and unique")
    reference = keys_from_csv(source / TABLE3_RESULTS, "file_name")
    if not reference.issubset(set(gt_keys)):
        raise ValueError("Detailed results contain visits without ground truth")
    candidate_files = [
        (source / FS_GKP_FILES["Deepgram"], "file"),
        (source / FS_GKP_FILES["Whisper"], "file"),
    ]
    for _, _, filename in TABLE2_CONFIGS:
        candidate_files.append((source / filename, "File Name" if filename.startswith("visiting_forms") else "file"))
    for path, column in candidate_files:
        if keys_from_csv(path, column) != reference:
            raise ValueError(f"Visit set does not match the ground truth: {path.name}")
    return build_visit_id_map(reference)


def is_number(value: str) -> bool:
    if value is None or value.strip() == "":
        return True
    try:
        float(value)
        return True
    except ValueError:
        return False


def export_detailed_results(source: Path, output: Path, visit_map: Mapping[str, str]) -> None:
    headers, rows = read_csv(source / TABLE3_RESULTS)
    if "file_name" not in headers or len(rows) != VISITS:
        raise ValueError("Unexpected detailed per-summary input schema")
    metric_columns = [
        name for name in headers
        if name != "file_name"
        and not name.endswith("_missing_sections")
        and all(is_number(row.get(name, "")) for row in rows)
    ]
    if not metric_columns:
        raise ValueError("No numeric per-summary metrics found")
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["visit_id", *metric_columns])
        writer.writeheader()
        for row in rows:
            key = extract_audio_key(row["file_name"])
            if key not in visit_map:
                raise ValueError("Detailed results contain an unknown visit")
            writer.writerow({"visit_id": visit_map[key], **{name: row[name] for name in metric_columns}})


def export_strategy_assignments(source: Path, output: Path, visit_map: Mapping[str, str]) -> None:
    headers, rows = read_csv(source / "ablation_recomputed_per_visit.csv")
    required = {"file_key", "asr_system", "strategy"}
    if not required.issubset(headers) or len(rows) != 178:
        raise ValueError("Unexpected AC-RAG strategy assignment schema")
    normalized = []
    for row in rows:
        key = extract_audio_key(row["file_key"])
        if key not in visit_map:
            raise ValueError("Strategy assignment contains an unknown visit")
        asr = row["asr_system"].strip().lower()
        if asr not in ("deepgram", "whisper"):
            raise ValueError("Unexpected ASR label in strategy assignments")
        strategy = STRATEGY_LABELS.get(row["strategy"].strip().lower())
        if strategy is None:
            raise ValueError("Unexpected retrieval strategy label")
        normalized.append({
            "visit_id": visit_map[key],
            "asr": asr.title(),
            "strategy": strategy,
        })
    if len({(row["visit_id"], row["asr"]) for row in normalized}) != 178:
        raise ValueError("Strategy assignments must be unique per visit and ASR system")
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["visit_id", "asr", "strategy"])
        writer.writeheader()
        writer.writerows(normalized)


def export_entity_audit(source: Path, output_dir: Path) -> None:
    source_path = source / "entity_recall_audit" / "entity_recall_term_list.csv"
    headers, rows = read_csv(source_path)
    needed = {"category", "term", "reference_count_of_89", "reference_pct"}
    if not needed.issubset(headers) or len(rows) != 80:
        raise ValueError("Unexpected entity recall audit input")
    output_dir.mkdir(parents=True, exist_ok=True)
    method_columns = [
        "Zero-Shot_summaries_containing_of_178",
        "FS-GKP_summaries_containing_of_178",
        "Original-RAG_summaries_containing_of_178",
        "AC-RAG_summaries_containing_of_178",
    ]
    fields = ["category", "term", "reference_count_of_89", "reference_pct", "generic_term", *method_columns]
    with (output_dir / "entity_recall_term_list.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = {
                "category": row["category"],
                "term": row["term"],
                "reference_count_of_89": row["reference_count_of_89"],
                "reference_pct": row["reference_pct"],
                "generic_term": str(float(row["reference_pct"]) >= 90).lower(),
            }
            record.update({column: row[column] for column in method_columns if column in headers})
            writer.writerow(record)


def export_fsgkp_stage1(source: Path, output_dir: Path, visit_map: Mapping[str, str]) -> Dict[str, dict]:
    summaries = {}
    logs_dir = output_dir / "table6_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    for asr, filename in FS_GKP_FILES.items():
        headers, rows = read_csv(source / filename)
        required = {"file", "input_tokens", "output_tokens", "cost_usd"}
        if not required.issubset(headers) or len(rows) != VISITS:
            raise ValueError(f"Unexpected FS-GKP extraction log schema: {filename}")
        out_path = logs_dir / f"fsgkp_stage1_extraction_{asr.lower()}.csv"
        with out_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["visit_id", "input_tokens", "output_tokens", "cost_usd"],
            )
            writer.writeheader()
            for row in rows:
                key = extract_audio_key(row["file"])
                if key not in visit_map:
                    raise ValueError("FS-GKP extraction log contains an unknown visit")
                writer.writerow({
                    "visit_id": visit_map[key],
                    "input_tokens": row["input_tokens"],
                    "output_tokens": row["output_tokens"],
                    "cost_usd": row["cost_usd"],
                })
        summaries[asr] = {
            "input_tokens": sum(int(row["input_tokens"]) for row in rows),
            "output_tokens": sum(int(row["output_tokens"]) for row in rows),
            "source_cost_usd": sum((Decimal(row["cost_usd"]) for row in rows), Decimal(0)),
        }
    return summaries


def extract_meter_text(notebook: Path) -> str:
    document = json.loads(notebook.read_text(encoding="utf-8"))
    for cell in document.get("cells", []):
        for output in cell.get("outputs", []):
            raw = output.get("text", "")
            text = "".join(raw) if isinstance(raw, list) else str(raw)
            raw = output.get("data", {}).get("text/plain", "")
            text += "".join(raw) if isinstance(raw, list) else str(raw)
            if "PROCESSING SUMMARY" in text.upper():
                return text
    raise ValueError("Saved Deepgram ProcessingMeter output not found")


def export_deepgram_meter(source: Path, output: Path) -> dict:
    text = extract_meter_text(source / "enhanced-RAG_extractionSummarization_v1.ipynb")
    logs_dir = output.parent / "table6_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    marker = text.upper().find("PROCESSING SUMMARY")
    if marker < 0:
        raise ValueError("Deepgram ProcessingMeter summary marker not found")
    meter_text = text[marker:]
    (logs_dir / "acrag_deepgram_meter_20260107.txt").write_text(
        meter_text, encoding="utf-8"
    )
    patterns = {
        "input_tokens": r"Input tokens\s*[:=]\s*([0-9,]+)",
        "output_tokens": r"Output tokens\s*[:=]\s*([0-9,]+)",
        "total_tokens": r"Total tokens\s*[:=]\s*([0-9,]+)",
        "total_cost_usd": r"Total Cost\s*[:=]\s*\$([0-9.]+)",
    }
    result = {"model": "gpt-4o"}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            raise ValueError(f"Deepgram ProcessingMeter field is missing: {key}")
        value = match.group(1).replace(",", "")
        result[key] = float(value) if key.endswith("_usd") else int(value)
    duration = re.search(
        r"Total duration\s*[:=]\s*(?:(\d+)h\s*)?(?:(\d+)m\s*)?([0-9.]+)s",
        text,
        re.IGNORECASE,
    )
    if duration:
        result["total_duration_seconds"] = (
            int(duration.group(1) or 0) * 3600
            + int(duration.group(2) or 0) * 60
            + float(duration.group(3))
        )
    average_call = re.search(
        r"Average time per API call\s*[:=]\s*([0-9.]+)",
        text,
        re.IGNORECASE,
    )
    if average_call:
        result["average_api_call_seconds"] = float(average_call.group(1))
    if result["input_tokens"] + result["output_tokens"] != result["total_tokens"]:
        raise ValueError("Deepgram ProcessingMeter token totals do not reconcile")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def export_whisper_logs(source: Path, output_dir: Path) -> dict:
    run_dir = source / WHISPER_RUN
    headers, rows = read_csv(run_dir / "api_call_log_whisper.csv")
    expected = ["call_type", "model", "input_tokens", "output_tokens", "cost_usd"]
    if headers != expected:
        raise ValueError("Unexpected Whisper API call log schema")
    embedding_tokens = 0
    with (output_dir / "api_call_log_whisper.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=expected)
        writer.writeheader()
        for row in rows:
            if row["call_type"] == "embedding":
                embedding_tokens += int(row["input_tokens"])
            row["model"] = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", row["model"])
            writer.writerow(row)
    meter = json.loads((run_dir / "processing_meter_whisper.json").read_text(encoding="utf-8"))
    meter.pop("run_finished", None)
    meter["model"] = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", meter.get("model", ""))
    (output_dir / "processing_meter_whisper.json").write_text(
        json.dumps(meter, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "input_tokens": int(meter["tokens_used"]["input_tokens"]),
        "output_tokens": int(meter["tokens_used"]["output_tokens"]),
        "total_cost_usd": float(meter["costs"]["total_cost"]),
        "embedding_tokens": embedding_tokens,
    }


def read_stage2_summary(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    fields = {
        "input_tokens": r"Total input tokens\s*[:=]\s*([0-9,]+)",
        "output_tokens": r"Total output tokens\s*[:=]\s*([0-9,]+)",
        "source_estimated_cost_usd": r"Total estimated cost\s*[:=]\s*\$([0-9.]+)",
        "processing_time_sec": r"Total processing time\s*[:=]\s*([0-9.]+)",
    }
    result = {}
    for key, pattern in fields.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            raise ValueError(f"FS-GKP stage 2 field is missing: {key}")
        value = match.group(1).replace(",", "")
        result[key] = float(value) if key.endswith(("_usd", "_sec")) else int(value)
    return result


def money(value: Decimal, places: int) -> str:
    return f"{value:.{places}f}"


def export_table6(source: Path, output_dir: Path, stage1: Mapping[str, dict],
                  deepgram_meter: Mapping[str, object], whisper_meter: Mapping[str, object]) -> None:
    stage2 = {}
    logs_dir = output_dir / "table6_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    for asr in ("Deepgram", "Whisper"):
        source_log = source / f"fs-gkp_output_{asr.lower()}" / "summary_time_cost"
        summary = read_stage2_summary(source_log)
        stage2[asr] = summary
        (logs_dir / f"fsgkp_stage2_summary_{asr.lower()}.txt").write_bytes(
            source_log.read_bytes()
        )
    with (logs_dir / "fsgkp_stage2_summary_totals.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fields = [
            "asr",
            "input_tokens",
            "output_tokens",
            "cost_usd",
            "processing_time_sec",
            "scope",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for asr, summary in stage2.items():
            writer.writerow({
                "asr": asr,
                "input_tokens": summary["input_tokens"],
                "output_tokens": summary["output_tokens"],
                "cost_usd": summary["source_estimated_cost_usd"],
                "processing_time_sec": summary["processing_time_sec"],
                "scope": "aggregate_only",
            })

    fsgkp_costs = {}
    fsgkp_totals = {}
    table6_costs = {}
    for asr in ("Deepgram", "Whisper"):
        total_input = stage1[asr]["input_tokens"] + stage2[asr]["input_tokens"]
        total_output = stage1[asr]["output_tokens"] + stage2[asr]["output_tokens"]
        cost = (
            Decimal(total_input) * TABLE6_RATES["input_per_million"]
            + Decimal(total_output) * TABLE6_RATES["output_per_million"]
        ) / Decimal(1_000_000)
        fsgkp_costs[asr] = cost
        fsgkp_totals[asr] = (total_input, total_output)
        table6_costs[("FS-GKP", asr)] = cost

    acrag_meter = {
        "Deepgram": deepgram_meter,
        "Whisper": whisper_meter,
    }
    rows = []
    configs = [
        ("AC-RAG", "Deepgram"),
        ("AC-RAG", "Whisper"),
        ("FS-GKP", "Deepgram"),
        ("FS-GKP", "Whisper"),
    ]
    for strategy, asr in configs:
        config = f"{strategy}_{asr}"
        if strategy == "AC-RAG":
            meter = acrag_meter[asr]
            input_total = int(meter["input_tokens"])
            output_total = int(meter["output_tokens"])
            total_cost = Decimal(str(meter["total_cost_usd"]))
            embedding_per_case = 2651 if asr == "Deepgram" else round(meter["embedding_tokens"] / VISITS)
            input_per_case = round((input_total - embedding_per_case * VISITS) / VISITS)
            output_per_case = round(output_total / VISITS)
            table6_costs[(strategy, asr)] = total_cost
            token_basis = "meter input minus estimated embedding tokens; divided by 89"
        else:
            input_total, output_total = fsgkp_totals[asr]
            total_cost = fsgkp_costs[asr]
            embedding_per_case = 0
            input_per_case = round(input_total / VISITS)
            output_per_case = round(output_total / VISITS)
            token_basis = "stage 1 per-visit logs plus stage 2 aggregate logs; divided by 89"
        processing_per_case, processing_total_min = REPORTED_TIMINGS[(strategy, asr)]
        rows.extend([
            {"configuration": config, "metric": "processing_time_per_consultation", "value": processing_per_case, "unit": "sec", "basis": "reported Table 6; saved timing logs do not fully reconcile"},
            {"configuration": config, "metric": "processing_time_total", "value": processing_total_min, "unit": "min", "basis": "reported Table 6; saved timing logs do not fully reconcile"},
            {"configuration": config, "metric": "api_cost_per_consultation", "value": money(total_cost / VISITS, 3), "unit": "USD", "basis": "meter or token totals at Table 6 rates"},
            {"configuration": config, "metric": "api_cost_total_89_cases", "value": money(total_cost, 2), "unit": "USD", "basis": "meter or token totals at Table 6 rates"},
            {"configuration": config, "metric": "input_tokens_per_case", "value": input_per_case, "unit": "tokens", "basis": token_basis},
            {"configuration": config, "metric": "embedding_tokens_per_case", "value": embedding_per_case, "unit": "tokens", "basis": (
                "estimated from logged embedding cost; allocated per consultation"
                if strategy == "AC-RAG" and asr == "Deepgram"
                else "summed from Whisper API call log"
                if strategy == "AC-RAG"
                else "not applicable to FS-GKP"
            )},
            {"configuration": config, "metric": "output_tokens_per_case", "value": output_per_case, "unit": "tokens", "basis": "meter or combined stage totals"},
        ])
    for asr in ("Deepgram", "Whisper"):
        acrag = table6_costs[("AC-RAG", asr)]
        fsgkp = table6_costs[("FS-GKP", asr)]
        rows.append({
            "configuration": f"AC-RAG_{asr}",
            "metric": "api_cost_relative_to_fsgkp",
            "value": money((acrag - fsgkp) / fsgkp * 100, 1),
            "unit": "%",
            "basis": "(AC-RAG - FS-GKP) / FS-GKP",
        })
    for asr in ("Deepgram", "Whisper"):
        rows.append({
            "configuration": f"FS-GKP_{asr}",
            "metric": "api_cost_relative_to_fsgkp",
            "value": "",
            "unit": "%",
            "basis": "reference",
        })
    with (output_dir / "table6_usage_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["configuration", "metric", "value", "unit", "basis"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="Local directory with private evaluation inputs and logs")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    visit_map = collect_visit_map(source)
    export_detailed_results(source, output / "detailed_per_sample_results.csv", visit_map)
    export_strategy_assignments(source, output / "acrag_strategy_assignments.csv", visit_map)
    score_table2(source, output)
    stage1 = export_fsgkp_stage1(source, output, visit_map)
    deepgram_meter = export_deepgram_meter(
        source, output / "processing_meter_deepgram.json"
    )
    whisper_meter = export_whisper_logs(source, output)
    export_table6(source, output, stage1, deepgram_meter, whisper_meter)
    export_entity_audit(source, output / "entity_recall_audit")
    print(f"Exported de-identified outputs for {len(visit_map)} visits to {output}")


if __name__ == "__main__":
    main()
