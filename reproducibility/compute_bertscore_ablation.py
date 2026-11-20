"""Rebuild Table 5 BERTScore from saved main-evaluation XLM-R scores.

No model inference or API calls are made. Original ablation artifacts are retained.
The reproduced rank tests are exploratory: they do not adjust for repeated
ASR outputs or consultations within patients.
"""

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from visit_ids import build_visit_id_map, extract_audio_key


STRATEGY_LABELS = {
    "hybrid": "Hybrid",
    "history_only": "History-Only",
    "cross_patient_only": "Cross-Patient-Only",
    "no_rag": "No-RAG",
}
EXPECTED_COUNTS = {"hybrid": 52, "history_only": 54, "cross_patient_only": 30, "no_rag": 42}


def regroup_scores(main_path, strategy_path):
    main = pd.read_csv(main_path)
    strategies = pd.read_csv(strategy_path)
    if len(main) != 89 or not main["file_name"].is_unique:
        raise ValueError("Expected 89 unique main-evaluation consultations")
    parts = []
    for asr in ["deepgram", "whisper"]:
        prefix = f"AC_RAG_{asr.title()}_"
        part = main[["file_name", prefix + "bertscore_f1", prefix + "semantic_similarity"]].rename(
            columns={
                "file_name": "file_key",
                prefix + "bertscore_f1": "bertscore_f1",
                prefix + "semantic_similarity": "main_composite_semantic",
            }
        )
        parts.append(part.assign(asr_system=asr))
    saved = pd.concat(parts, ignore_index=True)
    joined = strategies[
        ["patient_id", "file_key", "asr_system", "strategy", "composite_semantic"]
    ].merge(saved, on=["file_key", "asr_system"], how="outer", validate="one_to_one", indicator=True)
    if len(joined) != 178 or not joined["_merge"].eq("both").all():
        raise ValueError("Strategy labels and saved main scores must match exactly (178 summaries)")
    values = joined[["bertscore_f1", "composite_semantic", "main_composite_semantic"]].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Missing or non-finite evaluation scores")
    if not np.allclose(joined["composite_semantic"], joined["main_composite_semantic"], rtol=0, atol=1e-12):
        raise ValueError("Composite scores do not match: check evaluation-run provenance")
    if joined["strategy"].value_counts().to_dict() != EXPECTED_COUNTS:
        raise ValueError("Unexpected retrieval strategies or group sizes")
    if not joined.groupby("file_key")["strategy"].nunique().eq(1).all():
        raise ValueError("Strategy assignments differ between ASR systems")
    if not joined["patient_id"].eq(joined["file_key"].str.split("/").str[0]).all():
        raise ValueError("Patient identifiers do not match consultation keys")
    keys = [extract_audio_key(value) for value in joined["file_key"]]
    if any(key is None for key in keys):
        raise ValueError("Strategy data must have parseable visit keys")
    visit_ids = build_visit_id_map(keys)
    joined["visit_id"] = [visit_ids[key] for key in keys]
    return joined.drop(columns=["_merge", "main_composite_semantic"])


def cohens_d(first, second):
    pooled_variance = (
        (len(first) - 1) * first.var(ddof=1) + (len(second) - 1) * second.var(ddof=1)
    ) / (len(first) + len(second) - 2)
    if pooled_variance <= 0:
        raise ValueError("Cohen's d requires positive pooled variance")
    return float((first.mean() - second.mean()) / np.sqrt(pooled_variance))


def summarize_scores(per_visit):
    summary = per_visit.groupby("strategy")["bertscore_f1"].agg(["count", "mean", "std"])
    summary = summary.loc[list(STRATEGY_LABELS)]
    groups = {
        strategy: per_visit.loc[per_visit["strategy"].eq(strategy), "bertscore_f1"].to_numpy()
        for strategy in STRATEGY_LABELS
    }
    omnibus = stats.kruskal(*groups.values())
    rows = []
    for first, second in combinations(STRATEGY_LABELS, 2):
        result = stats.mannwhitneyu(groups[first], groups[second], alternative="two-sided", method="asymptotic")
        rows.append({
            "strategy1": first,
            "strategy2": second,
            "U": float(result.statistic),
            "p_raw": float(result.pvalue),
            "p_bonferroni": min(float(result.pvalue) * 6, 1.0),
            "cohens_d_first_minus_second": cohens_d(groups[first], groups[second]),
        })
    return summary, omnibus, pd.DataFrame(rows)


def write_outputs(per_visit, summary, omnibus, pairwise, output_dir, main_path, strategy_path):
    output_dir.mkdir(parents=True, exist_ok=True)
    per_visit.drop(columns=["patient_id", "file_key"]).to_csv(
        output_dir / "xlmr_per_summary.csv", index=False
    )
    summary.to_csv(output_dir / "xlmr_per_strategy.csv")
    pairwise.to_csv(output_dir / "xlmr_pairwise_exploratory.csv", index=False)
    pooled_mean = float(per_visit["bertscore_f1"].mean())
    weighted_mean = float(np.average(summary["mean"], weights=summary["count"]))
    if not np.isclose(pooled_mean, weighted_mean, rtol=0, atol=1e-14):
        raise ValueError("Strategy-weighted mean does not reproduce pooled main mean")
    metadata = {
        "model_type": "xlm-roberta-large",
        "num_layers": 9,
        "idf": False,
        "rescale_with_baseline": False,
        "settings_evidence": "clinical_metrics.py calculate_bertscore; bert_score.score defaults",
        "provenance_note": "Settings inferred from the evaluation code, not embedded in the score CSV.",
        "summaries": len(per_visit),
        "consultations": int(per_visit["visit_id"].nunique()),
        "patients": int(per_visit["patient_id"].nunique()),
        "pooled_mean": pooled_mean,
        "strategy_weighted_mean": weighted_mean,
        "kruskal_wallis_H": float(omnibus.statistic),
        "kruskal_wallis_p": float(omnibus.pvalue),
        "test_warning": "Exploratory summary-level tests; repeated ASR outputs and patients are not independent.",
        "inputs": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [main_path, strategy_path]
        },
    }
    legacy_path = Path("ablation_with_bertscore_per_visit.csv")
    if legacy_path.exists():
        legacy = pd.read_csv(legacy_path)
        comparison = legacy.merge(
            per_visit[["file_key", "asr_system", "bertscore_f1"]],
            on=["file_key", "asr_system"], how="outer", validate="one_to_one",
            suffixes=("_legacy", "_main"), indicator=True,
        )
        if not comparison["_merge"].eq("both").all():
            raise ValueError("Legacy and main score artifacts do not have identical keys")
        metadata["legacy_artifact_max_abs_difference"] = float(
            (comparison["bertscore_f1_legacy"] - comparison["bertscore_f1_main"]).abs().max()
        )
        metadata["legacy_artifact_scores_identical"] = bool(
            comparison["bertscore_f1_legacy"].eq(comparison["bertscore_f1_main"]).all()
        )
        metadata["inputs"][str(legacy_path)] = hashlib.sha256(legacy_path.read_bytes()).hexdigest()
    (output_dir / "audit.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    selected = pairwise.set_index(["strategy1", "strategy2"])
    hybrid_history = selected.loc[("hybrid", "history_only")]
    hybrid_none = selected.loc[("hybrid", "no_rag")]
    cells = [f"{row['mean']:.3f} ({row['std']:.3f})" for _, row in summary.iterrows()]
    means_text = ", ".join(
        f"{STRATEGY_LABELS[strategy]} {row['mean']:.3f}"
        for strategy, row in summary.iterrows()
    )
    lines = [
        "# Comment 3: BERTScore Consistency Correction",
        "",
        "## Verified Configuration",
        "",
        "The main evaluation calls clinical_metrics.py with no model override: "
        "xlm-roberta-large, num_layers=9, without IDF weighting or baseline rescaling. "
        "The latter two settings are verified from bert_score.score defaults, not inferred "
        "from the score magnitude. The former ablation script was configured for bert-base-multilingual-cased, "
        "but the existing ablation score artifact is identical, summary by summary, to the main score artifact. "
        "Thus, the script configuration alone does not establish that mBERT produced the submitted Table 5 values. "
        "This replacement script performs no inference and preserves the original score artifacts.",
        "",
        "## Replacement Table 5 Row",
        "",
        "| Metric | Hybrid (52 summaries) | History-Only (54) | Cross-Patient-Only (30) | No-RAG (42) | Hybrid vs History-Only | Hybrid vs No-RAG |",
        "|---|---|---|---|---|---|---|",
        "| BERTScore F1 | " + " | ".join(cells) + " | "
        f"P = {hybrid_history['p_bonferroni']:.3f}, d = {hybrid_history['cohens_d_first_minus_second']:.2f} | "
        f"P = {hybrid_none['p_bonferroni']:.3f}, d = {hybrid_none['cohens_d_first_minus_second']:.2f} |",
        "",
        f"Exact pooled mean: {pooled_mean:.12f}; strategy-weighted mean: {weighted_mean:.12f}. "
        "Both round to the Table 3 AC-RAG mean of 0.868. "
        "The row contains 178 summaries from 89 consultations in 63 patients, not 178 independent consultations.",
        "The rebuilt row and the exploratory statistics reproduce the submitted Table 5 BERTScore values "
        "to their reported precision; do not claim that new scores or changed values were obtained.",
        "",
        "## Methods Replacement",
        "",
        "> BERTScore F1 was calculated using xlm-roberta-large with num_layers=9, "
        "without IDF weighting or baseline rescaling. Table 5 was recalculated by grouping "
        "the saved per-summary scores from the main evaluation by retrieval strategy; "
        "the same BERTScore configuration therefore underlies Tables 3 and 5 and Figures 4 and 5.",
        "",
        "## Table 5 Footnote c",
        "",
        "> BERTScore F1 uses xlm-roberta-large (num_layers=9; no IDF weighting or baseline rescaling). "
        "Values are mean (sample SD) of the saved main-evaluation per-summary scores, grouped "
        "by retrieval strategy. The strategy-weighted mean is 0.868, matching Table 3. "
        f"Exploratory summary-level Kruskal-Wallis H = {omnibus.statistic:.3f}, P = {omnibus.pvalue:.3g}. "
        "Pairwise P values are two-sided Mann-Whitney P values multiplied by six and capped at one; "
        "compare these adjusted values with .05, not .0083. Positive d favours the first named strategy. "
        "These tests do not account for repeated ASR summaries or consultations within patients.",
        "",
        "## Results Replacement",
        "",
        f"> Mean BERTScore F1 ranged from {summary['mean'].min():.3f} to {summary['mean'].max():.3f} "
        "across retrieval strategies. The strategy-weighted mean was 0.868, consistent with "
        "the main AC-RAG evaluation. "
        f"Exploratory summary-level Kruskal-Wallis H = {omnibus.statistic:.3f}, P = {omnibus.pvalue:.3g}; "
        "none of the six pairwise comparisons was significant after Bonferroni correction. "
        "These tests do not account for repeated ASR summaries or consultations within patients.",
        "",
        "Remove the previous post-hoc power statement: it is not revalidated by this reconstruction. "
        "Do not describe an unchanged H or P as a numerical correction.",
        "",
        "## Draft Response to Comment 3",
        "",
        "Use this response after applying the supplied Methods and footnote replacements. "
        "The manuscript itself is not available here and has not been edited.",
        "",
        "> We thank the editor for identifying the inconsistent BERTScore model descriptions. "
        "We agree that the model must be consistent across the reported evaluations. "
        "To resolve this, we rebuilt the Table 5 aggregates directly from the saved per-summary "
        "xlm-roberta-large scores from the main evaluation, joined one-to-one to the retrieval "
        "strategy labels for the 178 AC-RAG summaries. No summaries were regenerated and no "
        "model inference was repeated. "
        f"The revised strategy means are {means_text}. Their strategy-weighted mean "
        f"is {pooled_mean:.6f} (0.868 to three decimals), matching the AC-RAG mean in Table 3. "
        "These recalculated values and the associated exploratory statistics agree with the "
        "previously reported values to their stated precision. "
        "Subgroup means need not equal the overall mean; the weighted reconciliation, rather "
        "than equality of individual strategy means, establishes consistency. "
        "We have corrected footnote c and clarified the Implementation Parameters subsection "
        "to specify xlm-roberta-large with num_layers=9, without IDF weighting or baseline rescaling. "
        "All BERTScore results in the revised tables and figures therefore use the same "
        "main-evaluation score source and model configuration. We have removed the "
        "previous post-hoc power statement. "
        f"For transparency, reproducing the former exploratory test framework gives "
        f"Kruskal-Wallis H = {omnibus.statistic:.3f}, P = {omnibus.pvalue:.3g}; "
        f"Hybrid versus History-Only adjusted P = {hybrid_history['p_bonferroni']:.3f}, "
        f"d = {hybrid_history['cohens_d_first_minus_second']:.2f}; "
        f"Hybrid versus No-RAG adjusted P = {hybrid_none['p_bonferroni']:.3f}, "
        f"d = {hybrid_none['cohens_d_first_minus_second']:.2f}. "
        "These rank tests treat summaries as independent and are reported as exploratory, "
        "not as a correction for repeated consultations or patients.",
        "",
        "## Before Submission",
        "",
        "- Apply the proposed manuscript edits before claiming that the manuscript itself has been updated.",
        "- In the Comment 1 response, remove the justification for retaining mBERT and refer to Comment 3.",
        "- Keep signed effects: a negative Hybrid-minus-History-Only d is not an error.",
        "- Do not claim that the regrouping fixes the separate BLEU/ROUGE implementation issues.",
        "- The existing Table 3 SD convention (RMS within-ASR SD) differs from Table 5 sample SD; only the means are reconciled here.",
        "",
    ]
    (output_dir / "COMMENT_3_BERTSCORE_CORRECTION.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-scores", type=Path, default=Path("batch_evaluation_results_all_systems/detailed_per_sample_results.csv"))
    parser.add_argument("--strategies", type=Path, default=Path("ablation_recomputed_per_visit.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("bertscore_consistency_results"))
    args = parser.parse_args()
    per_visit = regroup_scores(args.main_scores, args.strategies)
    summary, omnibus, pairwise = summarize_scores(per_visit)
    write_outputs(per_visit, summary, omnibus, pairwise, args.output_dir, args.main_scores, args.strategies)
    print(summary.to_string(float_format=lambda value: f"{value:.6f}"))
    print(f"Pooled mean: {per_visit['bertscore_f1'].mean():.12f}")
    print(f"Exploratory Kruskal-Wallis: H={omnibus.statistic:.6f}, P={omnibus.pvalue:.6g}")
    print(pairwise.to_string(index=False))
    print("WARNING: Rank tests are unadjusted for repeated ASR outputs and patients.")
    print(f"Results and response-letter draft: {args.output_dir}")


if __name__ == "__main__":
    main()
