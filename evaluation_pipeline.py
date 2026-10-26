#!/usr/bin/env python3
"""
Complete Evaluation Pipeline for Clinical Summarization
Usage: python evaluation_pipeline.py --rag_file visiting_forms_ground_truth_RAG_deepgram.csv --multi_file visiting_forms_ground_truth_multi-stage_deepgram.csv
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

def run_complete_evaluation(rag_df, multi_df, output_dir="evaluation_results"):
    """
    Run complete evaluation comparing RAG and Multi-stage approaches

    Args:
        rag_df: DataFrame with columns ['File Name', 'Ground Truth', 'LLM based summarization']
        multi_df: Same structure as rag_df
        output_dir: Directory to save results
    """

    Path(output_dir).mkdir(exist_ok=True)

    print("=" * 80)
    print("CLINICAL SUMMARIZATION EVALUATION PIPELINE")
    print("=" * 80)

    # Verify data alignment
    assert len(rag_df) == len(multi_df), "DataFrames must have same length"
    n_samples = len(rag_df)
    print(f"\n📊 Dataset: {n_samples} samples")

    # Extract texts
    ground_truths = rag_df['Ground Truth'].tolist()
    rag_summaries = rag_df['LLM based summarization'].tolist()
    multi_summaries = multi_df['LLM based summarization'].tolist()

    # Initialize metrics calculator
    metrics_calc = ClinicalMetrics()

    # =========================================================================
    # 1. CLINICAL ENTITY RECALL
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. CLINICAL ENTITY RECALL")
    print("=" * 80)

    entity_recalls_rag = []
    entity_recalls_multi = []

    for gt, rag, multi in zip(ground_truths, rag_summaries, multi_summaries):
        entity_recalls_rag.append(metrics_calc.calculate_entity_recall(gt, rag))
        entity_recalls_multi.append(metrics_calc.calculate_entity_recall(gt, multi))

    # Aggregate results
    entity_recall_rag_overall = [r['overall_entity_recall'] for r in entity_recalls_rag]
    entity_recall_multi_overall = [r['overall_entity_recall'] for r in entity_recalls_multi]
    print("entity_recall_multi_overall",entity_recall_multi_overall)
    print(f"\nRAG Entity Recall:        {np.mean(entity_recall_rag_overall):.3f} ± {np.std(entity_recall_rag_overall):.3f}")
    print(f"Multi-stage Entity Recall: {np.mean(entity_recall_multi_overall):.3f} ± {np.std(entity_recall_multi_overall):.3f}")
    multi_mean = np.mean(entity_recall_multi_overall)

    print("multi_mean",multi_mean)
    if multi_mean == 0 or np.isnan(multi_mean):
        print("Relative Improvement:      N/A (division by zero or NaN)")
    else:
        print(f"Relative Improvement:      {((np.mean(entity_recall_rag_overall) / multi_mean) - 1) * 100:.1f}%")

    # Statistical test
    entity_test = StatisticalTests.paired_t_test(
        entity_recall_rag_overall,
        entity_recall_multi_overall,
        "Entity Recall"
    )
    print(f"Statistical Significance:  p = {entity_test['p_value_t']:.6f} {entity_test['significance']}")
    print(f"Effect Size (Cohen's d):   {entity_test['cohens_d']:.3f} ({entity_test['effect_size']})")

    # =========================================================================
    # 2. NUMERICAL INFORMATION ACCURACY
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. NUMERICAL INFORMATION ACCURACY")
    print("=" * 80)

    num_accuracies_rag = []
    num_accuracies_multi = []

    for gt, rag, multi in zip(ground_truths, rag_summaries, multi_summaries):
        num_accuracies_rag.append(metrics_calc.calculate_numerical_accuracy(gt, rag))
        num_accuracies_multi.append(metrics_calc.calculate_numerical_accuracy(gt, multi))

    # Aggregate
    num_recall_rag = [r['overall_numerical_recall'] for r in num_accuracies_rag]
    num_recall_multi = [r['overall_numerical_recall'] for r in num_accuracies_multi]

    print(f"\nRAG Numerical Recall:        {np.mean(num_recall_rag):.3f} ± {np.std(num_recall_rag):.3f}")
    print(f"Multi-stage Numerical Recall: {np.mean(num_recall_multi):.3f} ± {np.std(num_recall_multi):.3f}")
    num_multi_mean = np.mean(num_recall_multi)
    if num_multi_mean == 0 or np.isnan(num_multi_mean):
        print("Relative Improvement:         N/A (division by zero or NaN)")
    else:
        print(f"Relative Improvement:         {((np.mean(num_recall_rag) / num_multi_mean) - 1) * 100:.1f}%")

    # Statistical test
    num_test = StatisticalTests.paired_t_test(
        num_recall_rag,
        num_recall_multi,
        "Numerical Recall"
    )
    print(f"Statistical Significance:         p = {num_test['p_value_t']:.6f} {num_test['significance']}")
    print(f"Effect Size (Cohen's d):      {num_test['cohens_d']:.3f} ({num_test['effect_size']})")

    # =========================================================================
    # 3. SEMANTIC SIMILARITY
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. SEMANTIC SIMILARITY")
    print("=" * 80)

    semantic_scores_rag = []
    semantic_scores_multi = []

    for gt, rag, multi in zip(ground_truths, rag_summaries, multi_summaries):
        semantic_scores_rag.append(metrics_calc.calculate_semantic_similarity(gt, rag))
        semantic_scores_multi.append(metrics_calc.calculate_semantic_similarity(gt, multi))

    # Extract combined semantic scores
    combined_semantic_rag = [s['combined_semantic_score'] for s in semantic_scores_rag]
    combined_semantic_multi = [s['combined_semantic_score'] for s in semantic_scores_multi]

    # Also extract individual components for detailed analysis
    cosine_rag = [s['tfidf_cosine_similarity'] for s in semantic_scores_rag]
    cosine_multi = [s['tfidf_cosine_similarity'] for s in semantic_scores_multi]

    jaccard_rag = [s['jaccard_similarity'] for s in semantic_scores_rag]
    jaccard_multi = [s['jaccard_similarity'] for s in semantic_scores_multi]

    print(f"\nCombined Semantic Score:")
    print(f"  RAG:        {np.mean(combined_semantic_rag):.3f} ± {np.std(combined_semantic_rag):.3f}")
    print(f"  Multi-stage: {np.mean(combined_semantic_multi):.3f} ± {np.std(combined_semantic_multi):.3f}")

    print(f"\nTF-IDF Cosine Similarity:")
    print(f"  RAG:        {np.mean(cosine_rag):.3f} ± {np.std(cosine_rag):.3f}")
    print(f"  Multi-stage: {np.mean(cosine_multi):.3f} ± {np.std(cosine_multi):.3f}")

    print(f"\nJaccard Token Similarity:")
    print(f"  RAG:        {np.mean(jaccard_rag):.3f} ± {np.std(jaccard_rag):.3f}")
    print(f"  Multi-stage: {np.mean(jaccard_multi):.3f} ± {np.std(jaccard_multi):.3f}")

    # Statistical test for semantic similarity
    semantic_test = StatisticalTests.paired_t_test(
        combined_semantic_rag,
        combined_semantic_multi,
        "Semantic Similarity"
    )
    print(f"\nStatistical Significance (Combined): p = {semantic_test['p_value_t']:.6f} {semantic_test['significance']}")
    print(f"Effect Size (Cohen's d):             {semantic_test['cohens_d']:.3f} ({semantic_test['effect_size']})")

    # =========================================================================
    # 4. ROUGE SCORES
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. ROUGE SCORES")
    print("=" * 80)

    rouge_scores_rag = []
    rouge_scores_multi = []

    for gt, rag, multi in zip(ground_truths, rag_summaries, multi_summaries):
        rouge_scores_rag.append(metrics_calc.calculate_rouge_scores(gt, rag))
        rouge_scores_multi.append(metrics_calc.calculate_rouge_scores(gt, multi))

    # Extract ROUGE scores (word-level by default)
    rouge1_f1_rag = [r['rouge1_f1'] for r in rouge_scores_rag if r['rouge1_f1'] is not None]
    rouge1_f1_multi = [r['rouge1_f1'] for r in rouge_scores_multi if r['rouge1_f1'] is not None]

    rouge2_f1_rag = [r['rouge2_f1'] for r in rouge_scores_rag if r['rouge2_f1'] is not None]
    rouge2_f1_multi = [r['rouge2_f1'] for r in rouge_scores_multi if r['rouge2_f1'] is not None]

    rougeL_f1_rag = [r['rougeL_f1'] for r in rouge_scores_rag if r['rougeL_f1'] is not None]
    rougeL_f1_multi = [r['rougeL_f1'] for r in rouge_scores_multi if r['rougeL_f1'] is not None]

    # Extract character-level ROUGE scores
    rougeL_f1_char_rag = [r['rougeL_f1_char'] for r in rouge_scores_rag if r.get('rougeL_f1_char') is not None]
    rougeL_f1_char_multi = [r['rougeL_f1_char'] for r in rouge_scores_multi if r.get('rougeL_f1_char') is not None]

    # Extract word-level ROUGE scores (explicit)
    rougeL_f1_word_rag = [r['rougeL_f1_word'] for r in rouge_scores_rag if r.get('rougeL_f1_word') is not None]
    rougeL_f1_word_multi = [r['rougeL_f1_word'] for r in rouge_scores_multi if r.get('rougeL_f1_word') is not None]

    if len(rouge1_f1_rag) > 0:
        print(f"\n📊 ROUGE Scores (Word-level tokenization):")
        print(f"\nROUGE-1 F1:")
        print(f"  RAG:        {np.mean(rouge1_f1_rag):.3f} ± {np.std(rouge1_f1_rag):.3f}")
        print(f"  Multi-stage: {np.mean(rouge1_f1_multi):.3f} ± {np.std(rouge1_f1_multi):.3f}")

        print(f"\nROUGE-2 F1:")
        print(f"  RAG:        {np.mean(rouge2_f1_rag):.3f} ± {np.std(rouge2_f1_rag):.3f}")
        print(f"  Multi-stage: {np.mean(rouge2_f1_multi):.3f} ± {np.std(rouge2_f1_multi):.3f}")

        print(f"\nROUGE-L F1 (Word-level):")
        print(f"  RAG:        {np.mean(rougeL_f1_rag):.3f} ± {np.std(rougeL_f1_rag):.3f}")
        print(f"  Multi-stage: {np.mean(rougeL_f1_multi):.3f} ± {np.std(rougeL_f1_multi):.3f}")

        # Character-level ROUGE-L
        if len(rougeL_f1_char_rag) > 0:
            print(f"\nROUGE-L F1 (Character-level):")
            print(f"  RAG:        {np.mean(rougeL_f1_char_rag):.3f} ± {np.std(rougeL_f1_char_rag):.3f}")
            print(f"  Multi-stage: {np.mean(rougeL_f1_char_multi):.3f} ± {np.std(rougeL_f1_char_multi):.3f}")
            print(f"  → Character-level captures exact surface form similarity")

        # Statistical test for ROUGE-L (word-level, most comprehensive)
        rougeL_test = StatisticalTests.paired_t_test(
            rougeL_f1_rag,
            rougeL_f1_multi,
            "ROUGE-L F1 (Word)"
        )
        print(f"\nStatistical Significance (ROUGE-L Word): p = {rougeL_test['p_value_t']:.6f} {rougeL_test['significance']}")
        print(f"Effect Size (Cohen's d):                 {rougeL_test['cohens_d']:.3f} ({rougeL_test['effect_size']})")

        # Statistical test for character-level ROUGE-L
        if len(rougeL_f1_char_rag) > 0:
            rougeL_char_test = StatisticalTests.paired_t_test(
                rougeL_f1_char_rag,
                rougeL_f1_char_multi,
                "ROUGE-L F1 (Char)"
            )
            print(f"Statistical Significance (ROUGE-L Char): p = {rougeL_char_test['p_value_t']:.6f} {rougeL_char_test['significance']}")
        else:
            rougeL_char_test = None
    else:
        print("\n⚠️  ROUGE scores not available. Install with: pip install rouge-score")
        rougeL_test = None
        rougeL_char_test = None

    # =========================================================================
    # 5. BLEU SCORES (DUAL TOKENIZATION)
    # =========================================================================
    print("\n" + "=" * 80)
    print("5. BLEU SCORES (Character + Word Tokenization)")
    print("=" * 80)

    bleu_scores_rag = []
    bleu_scores_multi = []

    for gt, rag, multi in zip(ground_truths, rag_summaries, multi_summaries):
        bleu_scores_rag.append(metrics_calc.calculate_bleu_score(gt, rag))
        bleu_scores_multi.append(metrics_calc.calculate_bleu_score(gt, multi))

    # Extract BLEU scores (word-level by default)
    bleu_rag = [r['bleu'] for r in bleu_scores_rag if r['bleu'] is not None]
    bleu_multi = [r['bleu'] for r in bleu_scores_multi if r['bleu'] is not None]

    # Extract character-level BLEU
    bleu_char_rag = [r['bleu_char'] for r in bleu_scores_rag if r.get('bleu_char') is not None]
    bleu_char_multi = [r['bleu_char'] for r in bleu_scores_multi if r.get('bleu_char') is not None]

    # Extract word-level BLEU (explicit)
    bleu_word_rag = [r['bleu_word'] for r in bleu_scores_rag if r.get('bleu_word') is not None]
    bleu_word_multi = [r['bleu_word'] for r in bleu_scores_multi if r.get('bleu_word') is not None]

    if len(bleu_rag) > 0:
        print(f"\n📊 BLEU Scores (Word-level tokenization with Janome):")
        print(f"  RAG:        {np.mean(bleu_rag):.3f} ± {np.std(bleu_rag):.3f}")
        print(f"  Multi-stage: {np.mean(bleu_multi):.3f} ± {np.std(bleu_multi):.3f}")

        # Character-level BLEU
        if len(bleu_char_rag) > 0:
            print(f"\n📊 BLEU Scores (Character-level tokenization):")
            print(f"  RAG:        {np.mean(bleu_char_rag):.3f} ± {np.std(bleu_char_rag):.3f}")
            print(f"  Multi-stage: {np.mean(bleu_char_multi):.3f} ± {np.std(bleu_char_multi):.3f}")
            print(f"  → Character-level captures exact text similarity")

        # Statistical test for BLEU (word-level)
        bleu_test = StatisticalTests.paired_t_test(
            bleu_rag,
            bleu_multi,
            "BLEU (Word)"
        )
        print(f"\nStatistical Significance (BLEU Word): p = {bleu_test['p_value_t']:.6f} {bleu_test['significance']}")
        print(f"Effect Size (Cohen's d):              {bleu_test['cohens_d']:.3f} ({bleu_test['effect_size']})")

        # Statistical test for character-level BLEU
        if len(bleu_char_rag) > 0:
            bleu_char_test = StatisticalTests.paired_t_test(
                bleu_char_rag,
                bleu_char_multi,
                "BLEU (Char)"
            )
            print(f"Statistical Significance (BLEU Char): p = {bleu_char_test['p_value_t']:.6f} {bleu_char_test['significance']}")
        else:
            bleu_char_test = None
    else:
        print("\n⚠️  BLEU scores not available. Install with: pip install sacrebleu")
        bleu_test = None
        bleu_char_test = None

    # =========================================================================
    # 6. BERTSCORE
    # =========================================================================
    print("\n" + "=" * 80)
    print("6. BERTSCORE (Contextual Embeddings)")
    print("=" * 80)

    bertscore_rag = []
    bertscore_multi = []

    print("Computing BERTScores (this may take a moment)...")
    for i, (gt, rag, multi) in enumerate(zip(ground_truths, rag_summaries, multi_summaries)):
        if i % 10 == 0:
            print(f"  Processing {i}/{len(ground_truths)}...", end='\r')
        bertscore_rag.append(metrics_calc.calculate_bertscore(gt, rag))
        bertscore_multi.append(metrics_calc.calculate_bertscore(gt, multi))
    print(f"  Processing {len(ground_truths)}/{len(ground_truths)}... Done!")

    # Extract BERTScore F1
    bert_f1_rag = [r['bertscore_f1'] for r in bertscore_rag if r['bertscore_f1'] is not None]
    bert_f1_multi = [r['bertscore_f1'] for r in bertscore_multi if r['bertscore_f1'] is not None]

    if len(bert_f1_rag) > 0:
        print(f"\nBERTScore F1:")
        print(f"  RAG:        {np.mean(bert_f1_rag):.3f} ± {np.std(bert_f1_rag):.3f}")
        print(f"  Multi-stage: {np.mean(bert_f1_multi):.3f} ± {np.std(bert_f1_multi):.3f}")

        # Statistical test for BERTScore
        bertscore_test = StatisticalTests.paired_t_test(
            bert_f1_rag,
            bert_f1_multi,
            "BERTScore F1"
        )
        print(f"\nStatistical Significance: p = {bertscore_test['p_value_t']:.6f} {bertscore_test['significance']}")
        print(f"Effect Size (Cohen's d):  {bertscore_test['cohens_d']:.3f} ({bertscore_test['effect_size']})")
        print(f"Model used: {bertscore_rag[0].get('model_type', 'default')}")
    else:
        print("\n⚠️  BERTScore not available. Install with: pip install bert-score")
        bertscore_test = None

    # =========================================================================
    # 7. TF-IDF COSINE SIMILARITY
    # =========================================================================
    print("\n" + "=" * 80)
    print("7. TF-IDF COSINE SIMILARITY")
    print("=" * 80)

    cosine_scores_rag = []
    cosine_scores_multi = []

    for gt, rag, multi in zip(ground_truths, rag_summaries, multi_summaries):
        cosine_scores_rag.append(metrics_calc.calculate_cosine_similarity_simple(gt, rag))
        cosine_scores_multi.append(metrics_calc.calculate_cosine_similarity_simple(gt, multi))

    # Extract cosine similarity
    cosine_sim_rag = [r['cosine_similarity'] for r in cosine_scores_rag]
    cosine_sim_multi = [r['cosine_similarity'] for r in cosine_scores_multi]

    vocab_overlap_rag = [r['vocab_overlap_ratio'] for r in cosine_scores_rag]
    vocab_overlap_multi = [r['vocab_overlap_ratio'] for r in cosine_scores_multi]

    print(f"\nCosine Similarity:")
    print(f"  RAG:        {np.mean(cosine_sim_rag):.3f} ± {np.std(cosine_sim_rag):.3f}")
    print(f"  Multi-stage: {np.mean(cosine_sim_multi):.3f} ± {np.std(cosine_sim_multi):.3f}")

    print(f"\nVocabulary Overlap Ratio:")
    print(f"  RAG:        {np.mean(vocab_overlap_rag):.3f} ± {np.std(vocab_overlap_rag):.3f}")
    print(f"  Multi-stage: {np.mean(vocab_overlap_multi):.3f} ± {np.std(vocab_overlap_multi):.3f}")

    # Statistical test for Cosine Similarity
    cosine_test = StatisticalTests.paired_t_test(
        cosine_sim_rag,
        cosine_sim_multi,
        "Cosine Similarity"
    )
    print(f"\nStatistical Significance: p = {cosine_test['p_value_t']:.6f} {cosine_test['significance']}")
    print(f"Effect Size (Cohen's d):  {cosine_test['cohens_d']:.3f} ({cosine_test['effect_size']})")

    # =========================================================================
    # 8. SECTION COMPLETENESS
    # =========================================================================
    print("\n" + "=" * 80)
    print("8. SECTION COMPLETENESS")
    print("=" * 80)

    section_scores_rag = []
    section_scores_multi = []

    for rag, multi in zip(rag_summaries, multi_summaries):
        section_scores_rag.append(metrics_calc.check_section_completeness(rag))
        section_scores_multi.append(metrics_calc.check_section_completeness(multi))

    # Aggregate
    completeness_rag = [s['completeness_score'] for s in section_scores_rag]
    completeness_multi = [s['completeness_score'] for s in section_scores_multi]

    presence_rag = [s['presence_score'] for s in section_scores_rag]
    presence_multi = [s['presence_score'] for s in section_scores_multi]

    print(f"\nRAG Section Completeness:        {np.mean(completeness_rag):.3f} ± {np.std(completeness_rag):.3f}")
    print(f"Multi-stage Section Completeness: {np.mean(completeness_multi):.3f} ± {np.std(completeness_multi):.3f}")
    completeness_multi_mean = np.mean(completeness_multi)
    if completeness_multi_mean == 0 or np.isnan(completeness_multi_mean):
        print("Relative Improvement:             N/A (division by zero or NaN)")
    else:
        print(f"Relative Improvement:             {((np.mean(completeness_rag) / completeness_multi_mean) - 1) * 100:.1f}%")

    # Statistical test
    section_test = StatisticalTests.paired_t_test(
        completeness_rag,
        completeness_multi,
        "Section Completeness"
    )
    print(f"Statistical Significance:         p = {section_test['p_value_t']:.6f} {section_test['significance']}")
    print(f"Effect Size (Cohen's d):          {section_test['cohens_d']:.3f} ({section_test['effect_size']})")

    # Most commonly missing sections
    all_missing_rag = [s for score in section_scores_rag for s in score['missing_sections']]
    all_missing_multi = [s for score in section_scores_multi for s in score['missing_sections']]

    from collections import Counter
    missing_counts_rag = Counter(all_missing_rag)
    missing_counts_multi = Counter(all_missing_multi)

    print(f"\nMost commonly missing sections:")
    print(f"  RAG:        {missing_counts_rag.most_common(3)}")
    print(f"  Multi-stage: {missing_counts_multi.most_common(3)}")

    # =========================================================================
    # 9. OUTPUT LENGTH ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("9. OUTPUT LENGTH ANALYSIS")
    print("=" * 80)

    gt_lengths = [len(t) for t in ground_truths]
    rag_lengths = [len(t) for t in rag_summaries]
    multi_lengths = [len(t) for t in multi_summaries]

    print(f"\nGround Truth:  {np.mean(gt_lengths):.0f} ± {np.std(gt_lengths):.0f} chars")
    print(f"RAG:           {np.mean(rag_lengths):.0f} ± {np.std(rag_lengths):.0f} chars ({np.mean(rag_lengths)/np.mean(gt_lengths)*100:.1f}% of GT)")
    print(f"Multi-stage:   {np.mean(multi_lengths):.0f} ± {np.std(multi_lengths):.0f} chars ({np.mean(multi_lengths)/np.mean(gt_lengths)*100:.1f}% of GT)")

    length_test = StatisticalTests.paired_t_test(
        rag_lengths,
        multi_lengths,
        "Output Length"
    )
    print(f"\nStatistical Significance: p = {length_test['p_value_t']:.6f} {length_test['significance']}")
    print(f"Effect Size (Cohen's d):  {length_test['cohens_d']:.3f} ({length_test['effect_size']})")

    # =========================================================================
    # 10. COMPREHENSIVE STATISTICAL SUMMARY WITH BONFERRONI CORRECTION
    # =========================================================================
    print("\n" + "=" * 80)
    print("10. COMPREHENSIVE STATISTICAL SUMMARY")
    print("=" * 80)

    # Collect all tests (including optional ones)
    all_tests = [entity_test, num_test, semantic_test, section_test, length_test, cosine_test]
    test_names = []

    # Add optional tests if available
    if rougeL_test is not None:
        all_tests.append(rougeL_test)
    if rougeL_char_test is not None:
        all_tests.append(rougeL_char_test)
    if bleu_test is not None:
        all_tests.append(bleu_test)
    if bleu_char_test is not None:
        all_tests.append(bleu_char_test)
    if bertscore_test is not None:
        all_tests.append(bertscore_test)

    test_summary_df = StatisticalTests.format_results_table(all_tests)

    print("\n" + test_summary_df.to_string(index=False))

    # Apply Bonferroni correction for multiple comparisons
    print("\n" + "=" * 80)
    print("BONFERRONI CORRECTION FOR MULTIPLE COMPARISONS")
    print("=" * 80)

    p_values = [test['p_value_t'] for test in all_tests]
    bonferroni_results = StatisticalTests.bonferroni_correction(p_values, alpha=0.05)

    print(f"\nNumber of comparisons: {bonferroni_results['n_comparisons']}")
    print(f"Original significance level (α): {bonferroni_results['original_alpha']}")
    print(f"Corrected significance level: {bonferroni_results['corrected_alpha']:.6f}")
    print(f"Significant tests after correction: {bonferroni_results['n_significant']}/{bonferroni_results['n_comparisons']}")

    print("\nCorrected Results:")
    print("-" * 80)
    for i, (test, corrected) in enumerate(zip(all_tests, bonferroni_results['results'])):
        metric_name = test['metric']
        orig_sig = test['significance']
        new_sig = corrected['significance_label']
        p_val = corrected['original_p_value']

        status = "✓ Still significant" if corrected['is_significant'] else "✗ Not significant"
        print(f"{metric_name:25s}: p={p_val:.6f}, Original: {orig_sig:3s} → Corrected: {new_sig:3s} {status}")

    # Add Bonferroni results to the summary DataFrame
    test_summary_df['bonferroni_significant'] = [r['is_significant'] for r in bonferroni_results['results']]
    test_summary_df['bonferroni_label'] = [r['significance_label'] for r in bonferroni_results['results']]

    # Save results
    test_summary_df.to_csv(f"{output_dir}/statistical_tests.csv", index=False)
    print(f"\n✓ Statistical tests saved to {output_dir}/statistical_tests.csv")

    # =========================================================================
    # 11. DETAILED PER-SAMPLE RESULTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("11. SAVING DETAILED RESULTS")
    print("=" * 80)

    # Create detailed results DataFrame
    detailed_results = []

    for i, (fname, gt, rag, multi) in enumerate(zip(
        rag_df['File Name'], ground_truths, rag_summaries, multi_summaries
    )):
        result = {
            'file_name': fname,
            'gt_length': len(gt),
            'rag_length': len(rag),
            'multi_length': len(multi),
            'rag_entity_recall': entity_recalls_rag[i]['overall_entity_recall'],
            'multi_entity_recall': entity_recalls_multi[i]['overall_entity_recall'],
            'rag_numerical_recall': num_accuracies_rag[i]['overall_numerical_recall'],
            'multi_numerical_recall': num_accuracies_multi[i]['overall_numerical_recall'],
            'rag_section_completeness': section_scores_rag[i]['completeness_score'],
            'multi_section_completeness': section_scores_multi[i]['completeness_score'],
            'rag_semantic_similarity': semantic_scores_rag[i]['combined_semantic_score'],
            'multi_semantic_similarity': semantic_scores_multi[i]['combined_semantic_score'],
            'rag_cosine_similarity': semantic_scores_rag[i]['tfidf_cosine_similarity'],
            'multi_cosine_similarity': semantic_scores_multi[i]['tfidf_cosine_similarity'],
            'rag_jaccard_similarity': semantic_scores_rag[i]['jaccard_similarity'],
            'multi_jaccard_similarity': semantic_scores_multi[i]['jaccard_similarity'],
            'rag_tfidf_cosine': cosine_scores_rag[i]['cosine_similarity'],
            'multi_tfidf_cosine': cosine_scores_multi[i]['cosine_similarity'],
            'rag_vocab_overlap': cosine_scores_rag[i]['vocab_overlap_ratio'],
            'multi_vocab_overlap': cosine_scores_multi[i]['vocab_overlap_ratio'],
            'rag_missing_sections': ','.join(section_scores_rag[i]['missing_sections']),
            'multi_missing_sections': ','.join(section_scores_multi[i]['missing_sections'])
        }

        # Add ROUGE scores if available
        if len(rouge1_f1_rag) > 0 and i < len(rouge_scores_rag):
            result['rag_rouge1_f1'] = rouge_scores_rag[i]['rouge1_f1']
            result['multi_rouge1_f1'] = rouge_scores_multi[i]['rouge1_f1']
            result['rag_rouge2_f1'] = rouge_scores_rag[i]['rouge2_f1']
            result['multi_rouge2_f1'] = rouge_scores_multi[i]['rouge2_f1']
            result['rag_rougeL_f1'] = rouge_scores_rag[i]['rougeL_f1']
            result['multi_rougeL_f1'] = rouge_scores_multi[i]['rougeL_f1']

        # Add BLEU scores if available
        if len(bleu_rag) > 0 and i < len(bleu_scores_rag):
            result['rag_bleu'] = bleu_scores_rag[i]['bleu']
            result['multi_bleu'] = bleu_scores_multi[i]['bleu']

        # Add BERTScore if available
        if len(bert_f1_rag) > 0 and i < len(bertscore_rag):
            result['rag_bertscore_f1'] = bertscore_rag[i]['bertscore_f1']
            result['multi_bertscore_f1'] = bertscore_multi[i]['bertscore_f1']

        detailed_results.append(result)

    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv(f"{output_dir}/detailed_clinical_metrics.csv", index=False)
    print(f"✓ Detailed results saved to {output_dir}/detailed_clinical_metrics.csv")

    # =========================================================================
    # 12. SUMMARY REPORT
    # =========================================================================
    print("\n" + "=" * 80)
    print("12. GENERATING SUMMARY REPORT")
    print("=" * 80)

    summary_report = {
        'dataset_info': {
            'n_samples': n_samples,
            'avg_gt_length': float(np.mean(gt_lengths)),
            'avg_rag_length': float(np.mean(rag_lengths)),
            'avg_multi_length': float(np.mean(multi_lengths))
        },
        'clinical_metrics': {
            'entity_recall': {
                'rag_mean': float(np.mean(entity_recall_rag_overall)),
                'multi_mean': float(np.mean(entity_recall_multi_overall)),
                'p_value': float(entity_test['p_value_t']),
                'cohens_d': float(entity_test['cohens_d']),
                'relative_improvement': float(((np.mean(entity_recall_rag_overall) / np.mean(entity_recall_multi_overall)) - 1) * 100) if np.mean(entity_recall_multi_overall) not in [0, np.nan] else None
            },
            'numerical_recall': {
                'rag_mean': float(np.mean(num_recall_rag)),
                'multi_mean': float(np.mean(num_recall_multi)),
                'p_value': float(num_test['p_value_t']),
                'cohens_d': float(num_test['cohens_d']),
                'relative_improvement': float(((np.mean(num_recall_rag) / np.mean(num_recall_multi)) - 1) * 100) if np.mean(num_recall_multi) not in [0, np.nan] else None
            },
            'section_completeness': {
                'rag_mean': float(np.mean(completeness_rag)),
                'multi_mean': float(np.mean(completeness_multi)),
                'p_value': float(section_test['p_value_t']),
                'cohens_d': float(section_test['cohens_d']),
                'relative_improvement': float(((np.mean(completeness_rag) / np.mean(completeness_multi)) - 1) * 100) if np.mean(completeness_multi) not in [0, np.nan] else None
            },
            'semantic_similarity': {
                'rag_mean': float(np.mean(combined_semantic_rag)),
                'multi_mean': float(np.mean(combined_semantic_multi)),
                'p_value': float(semantic_test['p_value_t']),
                'cohens_d': float(semantic_test['cohens_d']),
                'relative_improvement': float(((np.mean(combined_semantic_rag) / np.mean(combined_semantic_multi)) - 1) * 100) if np.mean(combined_semantic_multi) not in [0, np.nan] else None
            },
            'cosine_similarity': {
                'rag_mean': float(np.mean(cosine_sim_rag)),
                'multi_mean': float(np.mean(cosine_sim_multi)),
                'p_value': float(cosine_test['p_value_t']),
                'cohens_d': float(cosine_test['cohens_d']),
                'relative_improvement': float(((np.mean(cosine_sim_rag) / np.mean(cosine_sim_multi)) - 1) * 100) if np.mean(cosine_sim_multi) not in [0, np.nan] else None
            }
        },
        'nlp_metrics': {},
        'bonferroni_correction': {
            'n_comparisons': int(bonferroni_results['n_comparisons']),
            'corrected_alpha': float(bonferroni_results['corrected_alpha']),
            'n_significant': int(bonferroni_results['n_significant']),
            'significant_metrics': [test['metric'] for test, result in zip(all_tests, bonferroni_results['results'])
                                   if result['is_significant']]
        }
    }

    # Add NLP metrics if available
    if len(rouge1_f1_rag) > 0:
        summary_report['nlp_metrics']['rouge'] = {
            'rouge1_f1_rag': float(np.mean(rouge1_f1_rag)),
            'rouge1_f1_multi': float(np.mean(rouge1_f1_multi)),
            'rouge2_f1_rag': float(np.mean(rouge2_f1_rag)),
            'rouge2_f1_multi': float(np.mean(rouge2_f1_multi)),
            'rougeL_f1_rag': float(np.mean(rougeL_f1_rag)),
            'rougeL_f1_multi': float(np.mean(rougeL_f1_multi)),
            'p_value': float(rougeL_test['p_value_t']) if rougeL_test else None
        }

    if len(bleu_rag) > 0:
        summary_report['nlp_metrics']['bleu'] = {
            'rag_mean': float(np.mean(bleu_rag)),
            'multi_mean': float(np.mean(bleu_multi)),
            'p_value': float(bleu_test['p_value_t']) if bleu_test else None
        }

    if len(bert_f1_rag) > 0:
        summary_report['nlp_metrics']['bertscore'] = {
            'rag_mean': float(np.mean(bert_f1_rag)),
            'multi_mean': float(np.mean(bert_f1_multi)),
            'p_value': float(bertscore_test['p_value_t']) if bertscore_test else None
        }

    with open(f"{output_dir}/clinical_evaluation_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)

    print(f"✓ Summary report saved to {output_dir}/clinical_evaluation_summary.json")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    print(f"\n📊 Key Findings:")
    print(f"  Clinical Metrics:")
    print(f"    • Entity Recall:       RAG {np.mean(entity_recall_rag_overall):.1%} vs Multi {np.mean(entity_recall_multi_overall):.1%} (p={entity_test['p_value_t']:.4f})")
    print(f"    • Numerical Accuracy:  RAG {np.mean(num_recall_rag):.1%} vs Multi {np.mean(num_recall_multi):.1%} (p={num_test['p_value_t']:.4f})")
    print(f"    • Section Complete:    RAG {np.mean(completeness_rag):.1%} vs Multi {np.mean(completeness_multi):.1%} (p={section_test['p_value_t']:.4f})")

    print(f"\n  Semantic/Similarity Metrics:")
    print(f"    • Semantic Similarity: RAG {np.mean(combined_semantic_rag):.1%} vs Multi {np.mean(combined_semantic_multi):.1%} (p={semantic_test['p_value_t']:.4f})")
    print(f"    • Cosine Similarity:   RAG {np.mean(cosine_sim_rag):.1%} vs Multi {np.mean(cosine_sim_multi):.1%} (p={cosine_test['p_value_t']:.4f})")

    if len(rouge1_f1_rag) > 0:
        print(f"\n  NLP Standard Metrics:")
        print(f"    • ROUGE-L F1:          RAG {np.mean(rougeL_f1_rag):.1%} vs Multi {np.mean(rougeL_f1_multi):.1%} (p={rougeL_test['p_value_t']:.4f})")
    if len(bleu_rag) > 0:
        print(f"    • BLEU:                RAG {np.mean(bleu_rag):.1%} vs Multi {np.mean(bleu_multi):.1%} (p={bleu_test['p_value_t']:.4f})")
    if len(bert_f1_rag) > 0:
        print(f"    • BERTScore F1:        RAG {np.mean(bert_f1_rag):.1%} vs Multi {np.mean(bert_f1_multi):.1%} (p={bertscore_test['p_value_t']:.4f})")

    print(f"\n  Other:")
    print(f"    • Output Length:       RAG {np.mean(rag_lengths):.0f} vs Multi {np.mean(multi_lengths):.0f} chars")

    print(f"\n🔬 Multiple Comparison Correction:")
    print(f"  • Tests conducted: {bonferroni_results['n_comparisons']}")
    print(f"  • Corrected α level: {bonferroni_results['corrected_alpha']:.6f}")
    print(f"  • Significant after correction: {bonferroni_results['n_significant']}/{bonferroni_results['n_comparisons']}")
    if bonferroni_results['n_significant'] > 0:
        significant_metrics = [test['metric'] for test, result in zip(all_tests, bonferroni_results['results'])
                             if result['is_significant']]
        print(f"  • Significant metrics: {', '.join(significant_metrics)}")
    else:
        print(f"  • ⚠️  No metrics remain significant after Bonferroni correction")

    print(f"\n📁 Results saved to: {output_dir}/")
    print(f"  - statistical_tests.csv")
    print(f"  - detailed_clinical_metrics.csv")
    print(f"  - clinical_evaluation_summary.json")

    return summary_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate clinical summarization systems')
    parser.add_argument('--rag_file', type=str, required=True,
                       help='CSV file with RAG summaries')
    parser.add_argument('--multi_file', type=str, required=True,
                       help='CSV file with multi-stage summaries')
    parser.add_argument('--output_dir', type=str, default='clinical_evaluation_results',
                       help='Output directory for results')

    args = parser.parse_args()

    print("Loading data...")
    rag_df = pd.read_csv(args.rag_file)
    multi_df = pd.read_csv(args.multi_file)

    print(f"RAG data: {len(rag_df)} samples")
    print(f"Multi-stage data: {len(multi_df)} samples")

    # Run evaluation
    summary = run_complete_evaluation(rag_df, multi_df, args.output_dir)

    print("\n✅ Evaluation pipeline completed successfully!")
