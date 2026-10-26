#!/usr/bin/env python3
"""
Multi-System Evaluation Pipeline for Clinical Summarization
Supports comparing 2+ systems simultaneously (e.g., RAG-Deepgram, RAG-Whisper, Multi-Deepgram, Multi-Whisper)

Usage:
python evaluation_pipeline_multi.py \
    --files file1.csv file2.csv file3.csv file4.csv \
    --labels "RAG-Deepgram" "RAG-Whisper" "Multi-Deepgram" "Multi-Whisper" \
    --output_dir multi_system_results
"""

import argparse
import pandas as pd
import numpy as np
import json
import importlib
import clinical_metrics
# Reload module to pick up any changes made after initial import
importlib.reload(clinical_metrics)
from clinical_metrics import ClinicalMetrics, StatisticalTests
from pathlib import Path
from itertools import combinations
import warnings
from collections import Counter

def run_multi_system_evaluation(dataframes, labels, output_dir="multi_system_results"):
    """
    Run evaluation comparing multiple systems

    Args:
        dataframes: List of DataFrames, each with ['File Name', 'Ground Truth', 'LLM based summarization']
        labels: List of labels for each system (e.g., ["RAG-Deepgram", "RAG-Whisper", ...])
        output_dir: Directory to save results
    """

    Path(output_dir).mkdir(exist_ok=True)

    n_systems = len(dataframes)
    n_samples = len(dataframes[0])

    # Verify all dataframes have same length
    for i, df in enumerate(dataframes):
        assert len(df) == n_samples, f"System {labels[i]} has different number of samples"

    print("=" * 80)
    print("MULTI-SYSTEM CLINICAL SUMMARIZATION EVALUATION")
    print("=" * 80)
    print(f"\n📊 Dataset: {n_samples} samples")
    print(f"🔬 Systems: {n_systems}")
    for i, label in enumerate(labels):
        print(f"   {i+1}. {label}")

    # =========================================================================
    # GROUND TRUTH VALIDATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("VALIDATING GROUND TRUTH ACROSS FILES")
    print("=" * 80)

    ground_truths = dataframes[0]['Ground Truth'].tolist()
    ground_truth_matches = True

    for i in range(1, len(dataframes)):
        other_gt = dataframes[i]['Ground Truth'].tolist()
        if ground_truths != other_gt:
            print(f"⚠️  WARNING: Ground truth mismatch between file 0 and file {i} ({labels[i]})")
            # Check how many samples differ
            mismatches = sum(1 for gt1, gt2 in zip(ground_truths, other_gt) if gt1 != gt2)
            print(f"   Mismatches: {mismatches}/{n_samples} samples")
            ground_truth_matches = False

    if ground_truth_matches:
        print("✓ Ground truth validated: All files have identical ground truth")
    else:
        print("⚠️  WARNING: Using ground truth from first file. Results may be inconsistent.")

    # Extract summaries from each system
    all_summaries = []
    for df in dataframes:
        all_summaries.append(df['LLM based summarization'].tolist())

    # Initialize metrics calculator
    metrics_calc = ClinicalMetrics()

    # =========================================================================
    # COMPUTE ALL METRICS FOR ALL SYSTEMS
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPUTING METRICS FOR ALL SYSTEMS")
    print("=" * 80)

    results_by_system = {}

    for sys_idx, (summaries, label) in enumerate(zip(all_summaries, labels)):
        print(f"\nProcessing: {label}...")

        results = {
            'entity_recall': [],
            'numerical_recall': [],
            'semantic_similarity': [],
            'section_completeness': [],
            'rouge_scores': [],
            'bleu_scores': [],
            'bertscore': [],
            'cosine_similarity': [],
            'lengths': [],
            'gt_lengths': []  # Add ground truth lengths
        }

        for i, (gt, summ) in enumerate(zip(ground_truths, summaries)):
            if i % 20 == 0:
                print(f"  Progress: {i}/{n_samples}", end='\r')

            # Clinical metrics
            results['entity_recall'].append(metrics_calc.calculate_entity_recall(gt, summ))
            results['numerical_recall'].append(metrics_calc.calculate_numerical_accuracy(gt, summ))
            results['section_completeness'].append(metrics_calc.check_section_completeness(summ))

            # Semantic similarity
            results['semantic_similarity'].append(metrics_calc.calculate_semantic_similarity(gt, summ))

            # NLP metrics
            results['rouge_scores'].append(metrics_calc.calculate_rouge_scores(gt, summ))
            results['bleu_scores'].append(metrics_calc.calculate_bleu_score(gt, summ))
            results['bertscore'].append(metrics_calc.calculate_bertscore(gt, summ))
            results['cosine_similarity'].append(metrics_calc.calculate_cosine_similarity_simple(gt, summ))

            # Length
            results['lengths'].append(len(summ))
            results['gt_lengths'].append(len(gt))

        print(f"  Progress: {n_samples}/{n_samples} - Done!")
        results_by_system[label] = results

    # =========================================================================
    # AGGREGATE METRICS
    # =========================================================================
    print("\n" + "=" * 80)
    print("AGGREGATED METRICS COMPARISON")
    print("=" * 80)

    aggregated_metrics = {}
    aggregated_metrics_with_std = {}  # Store means and std together

    for label, results in results_by_system.items():
        # Extract all metric values
        entity_recall_vals = [r['overall_entity_recall'] for r in results['entity_recall']]
        numerical_recall_vals = [r['overall_numerical_recall'] for r in results['numerical_recall']]
        section_completeness_vals = [r['completeness_score'] for r in results['section_completeness']]
        semantic_similarity_vals = [r['combined_semantic_score'] for r in results['semantic_similarity']]
        tfidf_cosine_vals = [r['tfidf_cosine_similarity'] for r in results['semantic_similarity']]
        jaccard_vals = [r['jaccard_similarity'] for r in results['semantic_similarity']]
        cosine_vals = [r['cosine_similarity'] for r in results['cosine_similarity']]
        vocab_overlap_vals = [r['vocab_overlap_ratio'] for r in results['cosine_similarity']]

        rouge1_f1_vals = [r['rouge1_f1'] for r in results['rouge_scores'] if r.get('rouge1_f1') is not None]
        rouge2_f1_vals = [r['rouge2_f1'] for r in results['rouge_scores'] if r.get('rouge2_f1') is not None]
        rougeL_f1_vals = [r['rougeL_f1'] for r in results['rouge_scores'] if r.get('rougeL_f1') is not None]
        rougeL_f1_char_vals = [r['rougeL_f1_char'] for r in results['rouge_scores'] if r.get('rougeL_f1_char') is not None]

        bleu_vals = [r['bleu'] for r in results['bleu_scores'] if r.get('bleu') is not None]
        bleu_char_vals = [r['bleu_char'] for r in results['bleu_scores'] if r.get('bleu_char') is not None]

        bertscore_f1_vals = [r['bertscore_f1'] for r in results['bertscore'] if r.get('bertscore_f1') is not None]

        gt_length_vals = results['gt_lengths']
        length_vals = results['lengths']

        aggregated_metrics[label] = {
            'entity_recall': np.mean(entity_recall_vals) if len(entity_recall_vals) > 0 else 0,
            'numerical_recall': np.mean(numerical_recall_vals) if len(numerical_recall_vals) > 0 else 0,
            'section_completeness': np.mean(section_completeness_vals) if len(section_completeness_vals) > 0 else 0,
            'semantic_similarity': np.mean(semantic_similarity_vals) if len(semantic_similarity_vals) > 0 else 0,
            'tfidf_cosine': np.mean(tfidf_cosine_vals) if len(tfidf_cosine_vals) > 0 else 0,
            'jaccard_similarity': np.mean(jaccard_vals) if len(jaccard_vals) > 0 else 0,
            'vocab_overlap': np.mean(vocab_overlap_vals) if len(vocab_overlap_vals) > 0 else 0,
            'rouge1_f1': np.mean(rouge1_f1_vals) if len(rouge1_f1_vals) > 0 else 0,
            'rouge2_f1': np.mean(rouge2_f1_vals) if len(rouge2_f1_vals) > 0 else 0,
            'rougeL_f1_word': np.mean(rougeL_f1_vals) if len(rougeL_f1_vals) > 0 else 0,
            'rougeL_f1_char': np.mean(rougeL_f1_char_vals) if len(rougeL_f1_char_vals) > 0 else 0,
            'bleu_word': np.mean(bleu_vals) if len(bleu_vals) > 0 else 0,
            'bleu_char': np.mean(bleu_char_vals) if len(bleu_char_vals) > 0 else 0,
            'bertscore_f1': np.mean(bertscore_f1_vals) if len(bertscore_f1_vals) > 0 else 0,
            'cosine_similarity': np.mean(cosine_vals) if len(cosine_vals) > 0 else 0,
            'avg_gt_length': np.mean(gt_length_vals) if len(gt_length_vals) > 0 else 0,
            'avg_length': np.mean(length_vals) if len(length_vals) > 0 else 0,
            'length_pct_of_gt': (np.mean(length_vals) / np.mean(gt_length_vals) * 100) if np.mean(gt_length_vals) > 0 else 0
        }

        # Store with standard deviations for detailed reporting
        aggregated_metrics_with_std[label] = {
            'entity_recall': (np.mean(entity_recall_vals), np.std(entity_recall_vals)),
            'numerical_recall': (np.mean(numerical_recall_vals), np.std(numerical_recall_vals)),
            'section_completeness': (np.mean(section_completeness_vals), np.std(section_completeness_vals)),
            'semantic_similarity': (np.mean(semantic_similarity_vals), np.std(semantic_similarity_vals)),
            'tfidf_cosine': (np.mean(tfidf_cosine_vals), np.std(tfidf_cosine_vals)),
            'jaccard_similarity': (np.mean(jaccard_vals), np.std(jaccard_vals)),
            'vocab_overlap': (np.mean(vocab_overlap_vals), np.std(vocab_overlap_vals)),
            'rouge1_f1': (np.mean(rouge1_f1_vals), np.std(rouge1_f1_vals)) if len(rouge1_f1_vals) > 0 else (0, 0),
            'rouge2_f1': (np.mean(rouge2_f1_vals), np.std(rouge2_f1_vals)) if len(rouge2_f1_vals) > 0 else (0, 0),
            'rougeL_f1_word': (np.mean(rougeL_f1_vals), np.std(rougeL_f1_vals)) if len(rougeL_f1_vals) > 0 else (0, 0),
            'rougeL_f1_char': (np.mean(rougeL_f1_char_vals), np.std(rougeL_f1_char_vals)) if len(rougeL_f1_char_vals) > 0 else (0, 0),
            'bleu_word': (np.mean(bleu_vals), np.std(bleu_vals)) if len(bleu_vals) > 0 else (0, 0),
            'bleu_char': (np.mean(bleu_char_vals), np.std(bleu_char_vals)) if len(bleu_char_vals) > 0 else (0, 0),
            'bertscore_f1': (np.mean(bertscore_f1_vals), np.std(bertscore_f1_vals)) if len(bertscore_f1_vals) > 0 else (0, 0),
            'cosine_similarity': (np.mean(cosine_vals), np.std(cosine_vals)),
            'avg_gt_length': (np.mean(gt_length_vals), np.std(gt_length_vals)),
            'avg_length': (np.mean(length_vals), np.std(length_vals))
        }

    # Print comparison table with standard deviations
    print("\n" + "=" * 80)
    print("DETAILED METRIC COMPARISON (Mean ± Std)")
    print("=" * 80)

    # Print clinical metrics
    print("\n📋 CLINICAL METRICS:")
    print("-" * 80)
    for label in labels:
        print(f"\n{label}:")
        mean, std = aggregated_metrics_with_std[label]['entity_recall']
        print(f"  Entity Recall:         {mean:.3f} ± {std:.3f}")
        mean, std = aggregated_metrics_with_std[label]['numerical_recall']
        print(f"  Numerical Recall:      {mean:.3f} ± {std:.3f}")
        mean, std = aggregated_metrics_with_std[label]['section_completeness']
        print(f"  Section Completeness:  {mean:.3f} ± {std:.3f}")

    # Print semantic similarity metrics
    print("\n" + "=" * 80)
    print("🔍 SEMANTIC SIMILARITY METRICS:")
    print("-" * 80)
    for label in labels:
        print(f"\n{label}:")
        mean, std = aggregated_metrics_with_std[label]['semantic_similarity']
        print(f"  Combined Semantic:     {mean:.3f} ± {std:.3f}")
        mean, std = aggregated_metrics_with_std[label]['tfidf_cosine']
        print(f"  TF-IDF Cosine:         {mean:.3f} ± {std:.3f}")
        mean, std = aggregated_metrics_with_std[label]['jaccard_similarity']
        print(f"  Jaccard Similarity:    {mean:.3f} ± {std:.3f}")
        mean, std = aggregated_metrics_with_std[label]['vocab_overlap']
        print(f"  Vocabulary Overlap:    {mean:.3f} ± {std:.3f}")
        mean, std = aggregated_metrics_with_std[label]['cosine_similarity']
        print(f"  Cosine Similarity:     {mean:.3f} ± {std:.3f}")

    # Print NLP metrics
    print("\n" + "=" * 80)
    print("📊 NLP METRICS (ROUGE, BLEU, BERTScore):")
    print("-" * 80)
    for label in labels:
        print(f"\n{label}:")
        mean, std = aggregated_metrics_with_std[label]['rouge1_f1']
        print(f"  ROUGE-1 F1:            {mean:.3f} ± {std:.3f}")
        mean, std = aggregated_metrics_with_std[label]['rouge2_f1']
        print(f"  ROUGE-2 F1:            {mean:.3f} ± {std:.3f}")
        mean, std = aggregated_metrics_with_std[label]['rougeL_f1_word']
        print(f"  ROUGE-L F1 (Word):     {mean:.3f} ± {std:.3f}")
        mean, std = aggregated_metrics_with_std[label]['rougeL_f1_char']
        if mean > 0:
            print(f"  ROUGE-L F1 (Char):     {mean:.3f} ± {std:.3f}  ← Character-level tokenization")
        mean, std = aggregated_metrics_with_std[label]['bleu_word']
        print(f"  BLEU (Word):           {mean:.3f} ± {std:.3f}")
        mean, std = aggregated_metrics_with_std[label]['bleu_char']
        if mean > 0:
            print(f"  BLEU (Char):           {mean:.3f} ± {std:.3f}  ← Character-level tokenization")
        mean, std = aggregated_metrics_with_std[label]['bertscore_f1']
        print(f"  BERTScore F1:          {mean:.3f} ± {std:.3f}")

    # Print length analysis
    print("\n" + "=" * 80)
    print("📏 OUTPUT LENGTH ANALYSIS:")
    print("-" * 80)
    gt_mean, gt_std = aggregated_metrics_with_std[labels[0]]['avg_gt_length']
    print(f"\nGround Truth:  {gt_mean:.0f} ± {gt_std:.0f} chars")
    for label in labels:
        mean, std = aggregated_metrics_with_std[label]['avg_length']
        pct_of_gt = aggregated_metrics[label]['length_pct_of_gt']
        print(f"{label:15s}: {mean:.0f} ± {std:.0f} chars ({pct_of_gt:.1f}% of GT)")

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(aggregated_metrics).T
    comparison_df = comparison_df.round(4)

    # Save comparison table
    comparison_df.to_csv(f"{output_dir}/system_comparison.csv")
    print(f"\n✓ Comparison table saved to {output_dir}/system_comparison.csv")

    # =========================================================================
    # PAIRWISE STATISTICAL TESTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("PAIRWISE STATISTICAL COMPARISONS")
    print("=" * 80)

    # For each metric, run pairwise comparisons
    metric_names = ['entity_recall', 'numerical_recall', 'section_completeness',
                   'semantic_similarity', 'rouge1_f1', 'rouge2_f1', 'rougeL_f1_word', 'rougeL_f1_char',
                   'bleu_word', 'bleu_char', 'bertscore_f1', 'cosine_similarity']

    pairwise_results = []

    for metric_name in metric_names:
        print(f"\n{metric_name.upper().replace('_', ' ')}")
        print("-" * 80)

        # Get data for this metric from each system
        metric_data = {}
        for label, results in results_by_system.items():
            if metric_name == 'entity_recall':
                data = [r['overall_entity_recall'] for r in results['entity_recall']]
            elif metric_name == 'numerical_recall':
                data = [r['overall_numerical_recall'] for r in results['numerical_recall']]
            elif metric_name == 'section_completeness':
                data = [r['completeness_score'] for r in results['section_completeness']]
            elif metric_name == 'semantic_similarity':
                data = [r['combined_semantic_score'] for r in results['semantic_similarity']]
            elif metric_name == 'rouge1_f1':
                data = [r.get('rouge1_f1') or 0 for r in results['rouge_scores']]
            elif metric_name == 'rouge2_f1':
                data = [r.get('rouge2_f1') or 0 for r in results['rouge_scores']]
            elif metric_name == 'rougeL_f1_word':
                data = [r.get('rougeL_f1') or 0 for r in results['rouge_scores']]
            elif metric_name == 'rougeL_f1_char':
                data = [r.get('rougeL_f1_char') or 0 for r in results['rouge_scores']]
            elif metric_name == 'bleu_word':
                data = [r.get('bleu') or 0 for r in results['bleu_scores']]
            elif metric_name == 'bleu_char':
                data = [r.get('bleu_char') or 0 for r in results['bleu_scores']]
            elif metric_name == 'bertscore_f1':
                data = [r.get('bertscore_f1') or 0 for r in results['bertscore']]
            elif metric_name == 'cosine_similarity':
                data = [r['cosine_similarity'] for r in results['cosine_similarity']]

            metric_data[label] = data

        # Pairwise comparisons
        for (label1, label2) in combinations(labels, 2):
            data1 = metric_data[label1]
            data2 = metric_data[label2]

            if len(data1) > 0 and len(data2) > 0:
                test_result = StatisticalTests.paired_t_test(
                    data1, data2,
                    f"{metric_name}: {label1} vs {label2}"
                )

                # Calculate relative improvement with division by zero handling
                mean_2 = test_result['mean_system_b']
                if mean_2 != 0 and not np.isnan(mean_2):
                    relative_improvement = ((test_result['mean_system_a'] / mean_2) - 1) * 100
                else:
                    relative_improvement = None

                pairwise_results.append({
                    'metric': metric_name,
                    'system_1': label1,
                    'system_2': label2,
                    'mean_1': test_result['mean_system_a'],
                    'mean_2': test_result['mean_system_b'],
                    'difference': test_result['mean_difference'],
                    'relative_improvement_pct': relative_improvement,
                    'p_value': test_result['p_value_t'],
                    'cohens_d': test_result['cohens_d'],
                    'effect_size': test_result['effect_size'],
                    'significance': test_result['significance']
                })

                print(f"  {label1} vs {label2}:")
                print(f"    {test_result['mean_system_a']:.3f} vs {test_result['mean_system_b']:.3f} "
                      f"(Δ={test_result['mean_difference']:.3f}, ", end="")
                if relative_improvement is not None:
                    print(f"Rel.Imp={relative_improvement:+.1f}%, ", end="")
                else:
                    print(f"Rel.Imp=N/A, ", end="")
                print(f"p={test_result['p_value_t']:.4f} {test_result['significance']}, d={test_result['cohens_d']:.3f} [{test_result['effect_size']}])")

    # Save pairwise results
    pairwise_df = pd.DataFrame(pairwise_results)
    pairwise_df.to_csv(f"{output_dir}/pairwise_comparisons.csv", index=False)
    print(f"\n✓ Pairwise comparisons saved to {output_dir}/pairwise_comparisons.csv")

    # =========================================================================
    # BONFERRONI CORRECTION
    # =========================================================================
    print("\n" + "=" * 80)
    print("BONFERRONI CORRECTION FOR MULTIPLE COMPARISONS")
    print("=" * 80)

    all_p_values = pairwise_df['p_value'].tolist()
    bonferroni_results = StatisticalTests.bonferroni_correction(all_p_values, alpha=0.05)

    print(f"\nTotal comparisons: {bonferroni_results['n_comparisons']}")
    print(f"Corrected α: {bonferroni_results['corrected_alpha']:.6f}")
    print(f"Significant after correction: {bonferroni_results['n_significant']}/{bonferroni_results['n_comparisons']}")

    # Add Bonferroni results to pairwise DataFrame
    pairwise_df['bonferroni_significant'] = [r['is_significant'] for r in bonferroni_results['results']]
    pairwise_df['bonferroni_label'] = [r['significance_label'] for r in bonferroni_results['results']]

    # Show significant comparisons
    significant_comparisons = pairwise_df[pairwise_df['bonferroni_significant'] == True]
    if len(significant_comparisons) > 0:
        print("\nSignificant comparisons after Bonferroni correction:")
        print("-" * 80)
        for _, row in significant_comparisons.iterrows():
            print(f"  {row['metric']}: {row['system_1']} vs {row['system_2']}")
            print(f"    {row['mean_1']:.3f} vs {row['mean_2']:.3f} (p={row['p_value']:.6f} {row['bonferroni_label']})")
    else:
        print("\n⚠️  No comparisons remain significant after Bonferroni correction")

    # Save updated pairwise results
    pairwise_df.to_csv(f"{output_dir}/pairwise_comparisons.csv", index=False)

    # =========================================================================
    # MISSING SECTIONS ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("MISSING SECTIONS ANALYSIS")
    print("=" * 80)

    missing_sections_by_system = {}
    for label, results in results_by_system.items():
        all_missing = [s for score in results['section_completeness'] for s in score['missing_sections']]
        missing_counts = Counter(all_missing)
        missing_sections_by_system[label] = missing_counts

        print(f"\n{label}:")
        if len(missing_counts) > 0:
            print(f"  Total missing sections: {len(all_missing)} across {n_samples} samples")
            print(f"  Most commonly missing:")
            for section, count in missing_counts.most_common(5):
                print(f"    • {section}: {count} times ({count/n_samples*100:.1f}% of samples)")
        else:
            print("  ✓ No missing sections!")

    # =========================================================================
    # DETAILED PER-SAMPLE RESULTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("SAVING DETAILED PER-SAMPLE RESULTS")
    print("=" * 80)

    detailed_results = []

    for i in range(n_samples):
        result = {
            'file_name': dataframes[0]['File Name'].iloc[i],
            'gt_length': results_by_system[labels[0]]['gt_lengths'][i]  # GT length is same for all
        }

        for label, results in results_by_system.items():
            prefix = label.replace(' ', '_').replace('-', '_')

            # Clinical metrics
            result[f'{prefix}_entity_recall'] = results['entity_recall'][i]['overall_entity_recall']
            result[f'{prefix}_numerical_recall'] = results['numerical_recall'][i]['overall_numerical_recall']
            result[f'{prefix}_section_completeness'] = results['section_completeness'][i]['completeness_score']

            # Semantic similarity metrics
            result[f'{prefix}_semantic_similarity'] = results['semantic_similarity'][i]['combined_semantic_score']
            result[f'{prefix}_tfidf_cosine'] = results['semantic_similarity'][i]['tfidf_cosine_similarity']
            result[f'{prefix}_jaccard_similarity'] = results['semantic_similarity'][i]['jaccard_similarity']

            # Cosine similarity and vocabulary overlap
            result[f'{prefix}_cosine_similarity'] = results['cosine_similarity'][i]['cosine_similarity']
            result[f'{prefix}_vocab_overlap'] = results['cosine_similarity'][i]['vocab_overlap_ratio']

            # Length
            result[f'{prefix}_length'] = results['lengths'][i]

            # ROUGE scores (all variants)
            if results['rouge_scores'][i].get('rouge1_f1') is not None:
                result[f'{prefix}_rouge1_f1'] = results['rouge_scores'][i]['rouge1_f1']
            if results['rouge_scores'][i].get('rouge2_f1') is not None:
                result[f'{prefix}_rouge2_f1'] = results['rouge_scores'][i]['rouge2_f1']
            if results['rouge_scores'][i].get('rougeL_f1') is not None:
                result[f'{prefix}_rougeL_f1_word'] = results['rouge_scores'][i]['rougeL_f1']
            if results['rouge_scores'][i].get('rougeL_f1_char') is not None:
                result[f'{prefix}_rougeL_f1_char'] = results['rouge_scores'][i]['rougeL_f1_char']

            # BLEU scores (word and character level)
            if results['bleu_scores'][i].get('bleu') is not None:
                result[f'{prefix}_bleu_word'] = results['bleu_scores'][i]['bleu']
            if results['bleu_scores'][i].get('bleu_char') is not None:
                result[f'{prefix}_bleu_char'] = results['bleu_scores'][i]['bleu_char']

            # BERTScore
            if results['bertscore'][i].get('bertscore_f1') is not None:
                result[f'{prefix}_bertscore_f1'] = results['bertscore'][i]['bertscore_f1']

            # Missing sections
            result[f'{prefix}_missing_sections'] = ','.join(results['section_completeness'][i]['missing_sections'])

        detailed_results.append(result)

    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv(f"{output_dir}/detailed_per_sample_results.csv", index=False)
    print(f"✓ Detailed results saved to {output_dir}/detailed_per_sample_results.csv")

    # =========================================================================
    # SUMMARY REPORT
    # =========================================================================
    print("\n" + "=" * 80)
    print("GENERATING SUMMARY REPORT")
    print("=" * 80)

    summary_report = {
        'dataset_info': {
            'n_samples': n_samples,
            'n_systems': n_systems,
            'system_labels': labels,
            'ground_truth_validated': ground_truth_matches
        },
        'aggregated_metrics': aggregated_metrics,
        'aggregated_metrics_with_std': {
            label: {
                metric: {'mean': float(mean), 'std': float(std)}
                for metric, (mean, std) in metrics.items()
            }
            for label, metrics in aggregated_metrics_with_std.items()
        },
        'missing_sections_analysis': {
            label: dict(counter.most_common())
            for label, counter in missing_sections_by_system.items()
        },
        'bonferroni_correction': {
            'n_comparisons': bonferroni_results['n_comparisons'],
            'corrected_alpha': bonferroni_results['corrected_alpha'],
            'n_significant': bonferroni_results['n_significant']
        },
        'significant_comparisons': significant_comparisons.to_dict('records') if len(significant_comparisons) > 0 else []
    }

    with open(f"{output_dir}/multi_system_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)

    print(f"✓ Summary report saved to {output_dir}/multi_system_summary.json")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    print(f"\n📊 Best Performing System by Metric:")
    for metric in ['entity_recall', 'numerical_recall', 'section_completeness',
                   'rougeL_f1_word', 'bleu_word', 'bertscore_f1']:
        if metric in comparison_df.columns:
            best_system = comparison_df[metric].idxmax()
            best_score = comparison_df[metric].max()
            print(f"  {metric:25s}: {best_system} ({best_score:.3f})")

    print(f"\n📁 Results saved to: {output_dir}/")
    print(f"  - system_comparison.csv")
    print(f"  - pairwise_comparisons.csv")
    print(f"  - detailed_per_sample_results.csv")
    print(f"  - multi_system_summary.json")

    return summary_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate multiple clinical summarization systems')
    parser.add_argument('--files', type=str, nargs='+', required=True,
                       help='CSV files for each system')
    parser.add_argument('--labels', type=str, nargs='+', required=True,
                       help='Labels for each system (must match number of files)')
    parser.add_argument('--output_dir', type=str, default='multi_system_results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Verify same number of files and labels
    if len(args.files) != len(args.labels):
        print("❌ Error: Number of files must match number of labels")
        print(f"   Files: {len(args.files)}, Labels: {len(args.labels)}")
        exit(1)

    if len(args.files) < 2:
        print("❌ Error: Need at least 2 systems to compare")
        exit(1)

    print("Loading data...")
    dataframes = []
    for i, (filepath, label) in enumerate(zip(args.files, args.labels)):
        print(f"  {i+1}. {label}: {filepath}")
        df = pd.read_csv(filepath)
        dataframes.append(df)
        print(f"     → {len(df)} samples")

    # Run evaluation
    summary = run_multi_system_evaluation(dataframes, args.labels, args.output_dir)

    print("\n✅ Multi-system evaluation completed successfully!")
