"""Table 2 extraction scoring with release-safe aggregate and flag-only outputs."""

import csv
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from visit_ids import build_visit_id_map, extract_audio_key


EXPECTED_VISITS = 89
EXPECTED_T = 601
PARAMETERS = {
    "Body Temperature (°C)": ("体温", "temperature"),
    "Blood Pressure (mmHg)": ("血圧", "blood_pressure"),
    "Pulse (bpm)": ("脈拍", "numeric"),
    "Respiratory Rate (bpm)": ("呼吸", "numeric"),
    "SpO2 (%)": ("SPO2", "numeric"),
    "Defecation": ("排便", "binary"),
    "Appetite": ("食欲", "binary"),
}
SYSTEMS = [
    ("AC-RAG", "Deepgram", "visiting_forms_ground_truth_ACRAG_deepgram_v1.csv", "File Name", "LLM based summarization", "acrag"),
    ("AC-RAG", "Whisper", "visiting_forms_ground_truth_ACRAG_whisper_v1.csv", "File Name", "LLM based summarization", "acrag"),
    ("FS-GKP", "Deepgram", "extracted_health_info_results_fs-gkp_deepgram.csv", "file", "key_info", "fsgkp"),
    ("FS-GKP", "Whisper", "extracted_health_info_results_fs-gkp_whisper.csv", "file", "key_info", "fsgkp"),
]


def read_csv(path: Path) -> List[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def blank(value: Optional[str]) -> bool:
    return value is None or value.strip().lower() in ("", "nan", "none")


def normalize_ground_truth(value: Optional[str], kind: str) -> Optional[str]:
    if blank(value):
        return None
    text = value.strip()
    if kind == "temperature":
        match = re.search(r"([\d.]+)", text)
        return match.group(1) if match else None
    if kind == "blood_pressure":
        match = re.search(r"(\d+)\s*[/／]\s*(\d+)", text)
        return f"{match.group(1)}/{match.group(2)}" if match else None
    if kind == "numeric":
        match = re.search(r"(\d+)", text)
        return match.group(1) if match else None
    if "有" in text:
        return "有"
    if "無" in text:
        return "無"
    return None


def parse_fsgkp(text: str) -> Dict[str, Optional[str]]:
    output = {jp: None for jp, _ in PARAMETERS.values()}
    text = text.replace("*", "").replace("_", "").replace("`", "")
    patterns = {
        "体温": r"Body Temperature:\s*([\d.]+)\s*°C",
        "血圧": r"Blood Pressure:\s*(\d+/\d+)\s*mmHg",
        "脈拍": r"Pulse:\s*(\d+)\s*回/分",
        "呼吸": r"(?:Breath|Respiratory Rate):\s*(\d+)\s*回/分",
        "SPO2": r"SPO2:\s*(\d+)\s*%",
        "排便": r"Defecation:\s*(有|無)",
        "食欲": r"Appetite:\s*(有|無)",
    }
    for parameter, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE if parameter == "SPO2" else 0)
        if match:
            output[parameter] = match.group(1)
    return output


def parse_acrag(text: str) -> Dict[str, Optional[str]]:
    output = {jp: None for jp, _ in PARAMETERS.values()}
    text = text.replace("*", "").replace("_", "").replace("`", "")
    patterns = {
        "体温": r"体温[^\d]*([\d.]+)\s*[℃度]",
        "血圧": r"血圧[^\d]*([\d]+)\s*[/／]\s*([\d]+)",
        "脈拍": r"脈拍[^\d]*([\d]+)",
        "呼吸": r"呼吸[^\d]*([\d]+)\s*回",
        "SPO2": r"[Ss][Pp][Oo]2[^\d]*([\d]+)\s*[%％]?",
    }
    for parameter, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            output[parameter] = (
                f"{match.group(1)}/{match.group(2)}"
                if parameter == "血圧" else match.group(1)
            )
    if output["脈拍"] is None:
        match = re.search(r"脈[^\d拍]*([\d]+)\s*回", text)
        if match:
            output["脈拍"] = match.group(1)

    def binary(label: str, positive: Iterable[str], negative: Iterable[str]) -> Optional[str]:
        match = re.search(rf"{label}[：:]?\s*(有|無)", text)
        if match:
            return match.group(1)
        if any(re.search(pattern, text) for pattern in positive):
            return "有"
        if any(re.search(pattern, text) for pattern in negative):
            return "無"
        return None

    output["排便"] = binary(
        "排便",
        (r"排便[はが]?(?:毎日)?(?:あり|ある|出て|順調)", r"お通じ[はが]?(?:あり|ある|毎日|順調|出)"),
        (r"排便[はが]?(?:なし|ない|なかった|見られな)", r"お通じ[はが]?(?:なし|ない|なかった)", r"便秘"),
    )
    output["食欲"] = binary(
        "食欲",
        (r"食欲(?:良好|あり|問題な)", r"食事[はが]?(?:完食|全量|三食|摂取|食べ)"),
        (r"食欲(?:不振|低下|なし|ない)", r"(?:食べ(?:られ)?な[いし]|摂取できな)"),
    )
    return output


def value_matches(kind: str, reference: str, prediction: str) -> bool:
    if kind in ("temperature", "numeric"):
        try:
            return abs(float(reference) - float(prediction)) <= 0.5
        except (ValueError, TypeError):
            return reference == prediction
    return reference == prediction


def _read_inputs(source: Path):
    ground_truth = read_csv(source / "Experiment Result - Information extraction and Summarization - Ground Truth (2).csv")
    gt_by_key = {}
    for row in ground_truth:
        key = extract_audio_key(row.get("Filename", ""))
        if not key or key in gt_by_key:
            raise ValueError("Ground-truth visit keys must be valid and unique")
        gt_by_key[key] = row
    inputs = {}
    for strategy, asr, filename, id_column, text_column, kind in SYSTEMS:
        system_rows = read_csv(source / filename)
        result = {}
        for row in system_rows:
            key = extract_audio_key(row.get(id_column, ""))
            if not key or key in result:
                raise ValueError(f"Visit keys must be valid and unique in {filename}")
            result[key] = row.get(text_column, "") or ""
        if len(result) != EXPECTED_VISITS:
            raise ValueError(f"Expected {EXPECTED_VISITS} system outputs in {filename}")
        inputs[(strategy, asr)] = (result, kind)
    reference_keys = set(next(iter(inputs.values()))[0])
    if len(reference_keys) != EXPECTED_VISITS or set(gt_by_key) & reference_keys != reference_keys:
        raise ValueError("Ground truth and system visit sets do not match")
    for predictions, _ in inputs.values():
        if set(predictions) != reference_keys:
            raise ValueError("System visit sets do not match")
    return {key: gt_by_key[key] for key in reference_keys}, inputs, reference_keys


def score_table2(source: Path, output_dir: Path) -> Tuple[int, Dict[str, str]]:
    ground_truth, systems, visit_keys = _read_inputs(source)
    visit_map = build_visit_id_map(visit_keys)
    field_rows = []
    summary = {}
    total_scorable = 0

    parsed = {
        config: {
            key: (parse_acrag(text) if parser_kind == "acrag" else parse_fsgkp(text))
            for key, text in predictions.items()
        }
        for config, (predictions, parser_kind) in systems.items()
    }

    for display_name, (jp, kind) in PARAMETERS.items():
        scorable = sum(
            normalize_ground_truth(row.get(jp), kind) is not None
            for row in ground_truth.values()
        )
        total_scorable += scorable
        for config, predictions in parsed.items():
            tp = fp = fn = 0
            strategy, asr = config
            for key in visit_keys:
                raw_reference = ground_truth[key].get(jp)
                reference = normalize_ground_truth(raw_reference, kind)
                prediction = predictions[key][jp]
                excluded = reference is None and not blank(raw_reference)
                current_tp = current_fp = current_fn = 0
                if not excluded:
                    if reference is not None and prediction is not None and value_matches(kind, reference, prediction):
                        current_tp = 1
                    else:
                        if prediction is not None:
                            current_fp = 1
                        if reference is not None:
                            current_fn = 1
                tp += current_tp
                fp += current_fp
                fn += current_fn
                field_rows.append({
                    "visit_id": visit_map[key],
                    "parameter": display_name,
                    "strategy": strategy,
                    "asr": asr,
                    "TP": current_tp,
                    "FP": current_fp,
                    "FN": current_fn,
                })
            summary[(display_name, strategy, asr)] = {
                "TP": tp, "FP": fp, "FN": fn, "T": scorable,
            }

    if total_scorable != EXPECTED_T:
        raise ValueError(f"Expected T={EXPECTED_T}, found T={total_scorable}")
    if len(field_rows) != EXPECTED_VISITS * len(PARAMETERS) * len(SYSTEMS):
        raise ValueError("Unexpected per-field result count")

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "table2_field_results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["visit_id", "parameter", "strategy", "asr", "TP", "FP", "FN"])
        writer.writeheader()
        writer.writerows(field_rows)
    aggregate_rows = []
    totals = {
        config: {"TP": 0, "FP": 0, "FN": 0}
        for config in systems
    }
    for (parameter, strategy, asr), counts in summary.items():
        tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
        for metric, value in (("TP", tp), ("FP", fp), ("FN", fn)):
            totals[(strategy, asr)][metric] += value
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        aggregate_rows.append({
            "parameter": parameter, "strategy": strategy, "asr": asr, "T": counts["T"],
            "TP": tp, "FP": fp, "FN": fn,
            "precision_pct": round(precision * 100, 1),
            "recall_pct": round(recall * 100, 1),
            "f1_pct": round(f1 * 100, 1),
        })
    for strategy, asr in systems:
        counts = totals[(strategy, asr)]
        tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        aggregate_rows.append({
            "parameter": "Overall (micro)", "strategy": strategy, "asr": asr, "T": total_scorable,
            "TP": tp, "FP": fp, "FN": fn,
            "precision_pct": round(precision * 100, 1),
            "recall_pct": round(recall * 100, 1),
            "f1_pct": round(f1 * 100, 1),
        })
    with (output_dir / "table2_precision_recall_f1.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    return total_scorable, visit_map
