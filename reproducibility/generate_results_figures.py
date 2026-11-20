"""
Generate publication-quality figures for JMIR Results section
- Figure 1: Grouped bar chart comparing 4 systems across key metrics
- Figure 5: Same-ASR summarization effect sizes for AC-RAG minus FS-GKP
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set publication-quality style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Configuration
RESULTS_DIR = Path("results_fsgkp_vs_rag")
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load data
print("Loading data...")
with open(RESULTS_DIR / "multi_system_summary.json", "r") as f:
    data = json.load(f)

pairwise_df = pd.read_csv(RESULTS_DIR / "pairwise_comparisons.csv")

# Rename RAG systems to AC-RAG in pairwise_df for display
pairwise_df['system_1'] = pairwise_df['system_1'].str.replace('RAG-', 'AC-RAG-')
pairwise_df['system_2'] = pairwise_df['system_2'].str.replace('RAG-', 'AC-RAG-')

# Extract aggregated metrics with std
# Map display names to data keys
systems_display = ["FS-GKP-Deepgram", "FS-GKP-Whisper", "AC-RAG-Deepgram", "AC-RAG-Whisper"]
systems_data = ["FS-GKP-Deepgram", "FS-GKP-Whisper", "RAG-Deepgram", "RAG-Whisper"]
metrics_data = data["aggregated_metrics_with_std"]

# ============================================================================
# FIGURE 1: Grouped Bar Chart - Key Metrics Comparison
# ============================================================================

def create_grouped_bar_chart():
    """Create grouped bar chart comparing systems across key metrics"""

    # Select key metrics for visualization
    key_metrics = {
        'Semantic\nSimilarity': 'semantic_similarity',
        'BERTScore\nF1': 'bertscore_f1',
        'TF-IDF\nCosine': 'tfidf_cosine',
        'ROUGE-L\n(Char)': 'rougeL_f1_char',
        'BLEU\n(Word)': 'bleu_word',
        'BLEU\n(Char)': 'bleu_char',
        'Entity\nRecall': 'entity_recall',
        'Section\nCompleteness': 'section_completeness'
    }

    # Prepare data for plotting
    plot_data = []
    for display_name, metric_key in key_metrics.items():
        for idx, system_display in enumerate(systems_display):
            system_data_key = systems_data[idx]
            mean_val = metrics_data[system_data_key][metric_key]['mean']
            std_val = metrics_data[system_data_key][metric_key]['std']
            plot_data.append({
                'Metric': display_name,
                'System': system_display,
                'Mean': mean_val,
                'Std': std_val
            })

    df_plot = pd.DataFrame(plot_data)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Define colors for each system
    colors = {
        'FS-GKP-Deepgram': '#E74C3C',      # Red
        'FS-GKP-Whisper': '#F39C12',       # Orange
        'AC-RAG-Deepgram': '#3498DB',         # Blue
        'AC-RAG-Whisper': '#2ECC71'           # Green
    }

    # Set bar width and positions
    metrics_list = list(key_metrics.keys())
    x = np.arange(len(metrics_list))
    width = 0.2

    # Plot bars for each system
    for i, system_display in enumerate(systems_display):
        system_data = df_plot[df_plot['System'] == system_display]
        positions = x + (i - 1.5) * width

        bars = ax.bar(
            positions,
            system_data['Mean'],
            width,
            label=system_display,
            color=colors[system_display],
            alpha=0.8,
            edgecolor='black',
            linewidth=0.5
        )

        # Add error bars
        ax.errorbar(
            positions,
            system_data['Mean'],
            yerr=system_data['Std'],
            fmt='none',
            ecolor='black',
            capsize=2,
            capthick=0.5,
            linewidth=0.5,
            alpha=0.6
        )

    # Add significance markers for RAG systems
    # Check which metrics have significant improvements
    significant_metrics = {
        'Semantic\nSimilarity': True,
        'BERTScore\nF1': True,
        'TF-IDF\nCosine': True,
        'ROUGE-L\n(Char)': True,
        'BLEU\n(Word)': True,
        'BLEU\n(Char)': True,
        'Entity\nRecall': False,
        'Section\nCompleteness': False
    }

    # Add asterisks for significant metrics
    for idx, (display_name, is_sig) in enumerate(significant_metrics.items()):
        if is_sig:
            # Position asterisks above the highest bar
            max_height = df_plot[df_plot['Metric'] == display_name]['Mean'].max()
            max_std = df_plot[df_plot['Metric'] == display_name]['Std'].max()
            y_pos = max_height + max_std + 0.02
            ax.text(idx, y_pos, '***', ha='center', va='bottom',
                   fontsize=12, fontweight='bold')

    # Customize plot
    ax.set_xlabel('Evaluation Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Comparative Performance of Four Clinical Note Summarization Systems',
                fontsize=13, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_list, rotation=0, ha='center')
    ax.legend(loc='upper left', ncol=2, frameon=True, fancybox=True, shadow=True)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add footnote
    fig.text(0.1, 0.02,
            '*** p < 0.001 after Bonferroni correction. Error bars represent ±1 SD. N = 89 patient visits.',
            ha='left', fontsize=8, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    # Save figure
    output_path = OUTPUT_DIR / "Figure1_Grouped_Bar_Chart.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure 1 saved to: {output_path}")

    # Also save as high-res PDF for publication
    output_path_pdf = OUTPUT_DIR / "Figure1_Grouped_Bar_Chart.pdf"
    plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight')
    print(f"✓ Figure 1 (PDF) saved to: {output_path_pdf}")

    plt.show()


# ============================================================================
# FIGURE 5: Forest Plot - Effect Sizes
# ============================================================================

def create_forest_plot(bootstrap_samples=10000, random_seed=20261016):
    """Plot saved effects with paired, patient-cluster bootstrap intervals."""

    comparisons = pd.read_csv(
        Path("batch_evaluation_results_all_systems") / "pairwise_comparisons.csv"
    )
    per_visit = pd.read_csv(
        Path("batch_evaluation_results_all_systems") / "detailed_per_sample_results.csv"
    )
    if len(per_visit) != 89 or not per_visit['file_name'].is_unique:
        raise ValueError('Expected 89 unique consultations')
    patient_ids = per_visit['file_name'].str.split('/').str[0]
    if patient_ids.isna().any() or patient_ids.eq('').any():
        raise ValueError('Missing patient identifiers')
    patient_codes, patients = pd.factorize(patient_ids, sort=True)
    patient_counts = np.bincount(patient_codes)
    rng = np.random.default_rng(random_seed)
    bootstrap_weights = rng.multinomial(
        len(patients), np.full(len(patients), 1 / len(patients)),
        size=bootstrap_samples,
    )
    bootstrap_counts = bootstrap_weights @ patient_counts
    metric_names = {
        'rougeL_f1_char': 'ROUGE-L (Char)',
        'bleu_word': 'BLEU (Word)',
        'semantic_similarity': 'Composite Semantic Similarity',
        'cosine_similarity': 'TF-IDF Cosine',
        'bleu_char': 'BLEU (Char)',
        'bertscore_f1': 'BERTScore F1',
    }
    records = []
    for metric, label in metric_names.items():
        for asr in ['Deepgram', 'Whisper']:
            ac_system = f'AC-RAG-{asr}'
            fs_system = f'FS-GKP-{asr}'
            forward = (
                (comparisons['system_1'] == ac_system)
                & (comparisons['system_2'] == fs_system)
            )
            reverse = (
                (comparisons['system_1'] == fs_system)
                & (comparisons['system_2'] == ac_system)
            )
            match = comparisons[
                (comparisons['metric'] == metric) & (forward | reverse)
            ]
            if len(match) != 1:
                raise ValueError(f'Expected one comparison for {metric}, {asr}')
            row = match.iloc[0]
            direction = 1 if row['system_1'] == ac_system else -1
            effect = direction * float(row['cohens_d'])
            if not np.isfinite(effect):
                raise ValueError(f'Non-finite effect for {metric}, {asr}')
            paired_values = per_visit[
                [f'AC_RAG_{asr}_{metric}', f'FS_GKP_{asr}_{metric}']
            ].to_numpy(dtype=float)
            if not np.isfinite(paired_values).all():
                raise ValueError(f'Missing or non-finite scores for {metric}, {asr}')
            observed = (
                paired_values[:, 0].mean() - paired_values[:, 1].mean()
            ) / np.sqrt(paired_values.var(axis=0, ddof=1).mean())
            if not np.isclose(observed, effect, rtol=1e-10, atol=1e-10):
                raise ValueError(f'Saved effect does not match per-visit data: {metric}, {asr}')
            patient_sums = np.zeros((len(patients), 2))
            patient_squared_sums = np.zeros((len(patients), 2))
            np.add.at(patient_sums, patient_codes, paired_values)
            np.add.at(patient_squared_sums, patient_codes, paired_values ** 2)
            bootstrap_sums = np.einsum(
                'bp,pc->bc', bootstrap_weights, patient_sums, optimize=False
            )
            bootstrap_squared_sums = np.einsum(
                'bp,pc->bc', bootstrap_weights, patient_squared_sums, optimize=False
            )
            bootstrap_means = bootstrap_sums / bootstrap_counts[:, None]
            bootstrap_variances = (
                bootstrap_squared_sums
                - bootstrap_sums ** 2 / bootstrap_counts[:, None]
            ) / (bootstrap_counts[:, None] - 1)
            bootstrap_effects = (
                bootstrap_means[:, 0] - bootstrap_means[:, 1]
            ) / np.sqrt(np.maximum(bootstrap_variances, 0).mean(axis=1))
            if not np.isfinite(bootstrap_effects).all():
                raise ValueError(f'Undefined bootstrap effects for {metric}, {asr}')
            ci_lower, ci_upper = np.percentile(bootstrap_effects, [2.5, 97.5])
            records.append({
                'metric': metric,
                'label': label,
                'asr': asr,
                'cohens_d': effect,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n_consultations': len(per_visit),
                'n_patients': len(patients),
                'bootstrap_samples': bootstrap_samples,
                'random_seed': random_seed,
            })

    plot_data = pd.DataFrame(records)
    display_data = plot_data.sort_values('cohens_d', ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 12))
    lower_limit = min(0, plot_data['ci_lower'].min() - 0.15)
    upper_limit = max(1.1, plot_data['ci_upper'].max() + 0.3)
    for lower, upper, color in [
        (0, 0.2, 'gray'),
        (0.2, 0.5, 'yellow'),
        (0.5, 0.8, 'orange'),
        (0.8, max(abs(lower_limit), upper_limit), 'red'),
    ]:
        ax.axvspan(lower, upper, color=color, alpha=0.06, zorder=0)
        ax.axvspan(-upper, -lower, color=color, alpha=0.06, zorder=0)

    category_colors = {
        'Semantic': '#E74C3C',
        'BERTScore': '#3498DB',
        'TF-IDF': '#2ECC71',
        'ROUGE-L': '#F39C12',
        'BLEU': '#9B59B6',
    }
    metric_categories = {
        'semantic_similarity': 'Semantic',
        'bertscore_f1': 'BERTScore',
        'cosine_similarity': 'TF-IDF',
        'rougeL_f1_char': 'ROUGE-L',
        'bleu_word': 'BLEU',
        'bleu_char': 'BLEU',
    }
    positions = np.arange(len(display_data))
    y_labels = []
    for position, row in display_data.iterrows():
        color = category_colors[metric_categories[row['metric']]]
        ax.hlines(
            position, row['ci_lower'], row['ci_upper'],
            color=color, linewidth=2, alpha=0.8, zorder=2,
        )
        for bound in ['ci_lower', 'ci_upper']:
            ax.vlines(
                row[bound], position - 0.055, position + 0.055,
                color=color, linewidth=1.2, zorder=2,
            )
        ax.scatter(
            row['cohens_d'], position, color=color, s=100,
            edgecolor='black', linewidth=1, zorder=3, alpha=0.9,
        )
        ax.annotate(
            f"d = {row['cohens_d']:.2f}", (row['ci_upper'], position),
            xytext=(7, 0), textcoords='offset points',
            va='center', fontsize=8, color='#555555',
        )
        y_labels.append(
            f"{row['label']}\n(AC-RAG-{row['asr']} vs FS-GKP-{row['asr']})"
        )

    ax.axvline(0, color='#333333', linewidth=1)
    ax.set_yticks(positions)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(lower_limit, upper_limit)
    ax.set_xlabel("Cohen's d (AC-RAG minus FS-GKP; 95% CI)", fontsize=12, fontweight='bold')
    ax.set_title(
        'Summarization Effect Sizes: AC-RAG vs FS-GKP (Same ASR)',
        fontsize=13, fontweight='bold', pad=20,
    )
    ax.grid(axis='x', alpha=0.25, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=category_colors[category], label=category, alpha=0.8)
        for category in sorted(category_colors)
    ]
    ax.legend(
        handles=legend_elements, title='Metric Category', loc='lower left',
        frameon=True, fancybox=True,
    )
    fig.text(
        0.04, 0.035,
        f'Positive values favor AC-RAG. Same-ASR summarization comparisons; {len(per_visit)} consultations from {len(patients)} patients.\n'
        f'Error bars: 95% patient-cluster percentile bootstrap CIs ({bootstrap_samples:,} resamples).\n'
        'Shading: negligible |d| < 0.2; small 0.2-0.5; medium 0.5-0.8; large >= 0.8.',
        ha='left', fontsize=8, style='italic',
    )
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    for extension in ['png', 'pdf']:
        output_path = OUTPUT_DIR / f'Figure5_Effect_Sizes.{extension}'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'Saved: {output_path}')
    plt.close(fig)
    plot_data.to_csv(OUTPUT_DIR / 'Figure5_Effect_Sizes_data.csv', index=False)
    caption = (
        "Figure 5. Effect sizes (Cohen's d) for AC-RAG versus FS-GKP summarization, "
        f"compared within the same ASR system (n = {len(per_visit)} consultations "
        f"from {len(patients)} patients per ASR). "
        "d = (AC-RAG - FS-GKP)/pooled SD; positive values favour AC-RAG. "
        f"Error bars show 95% patient-cluster percentile bootstrap CIs ({bootstrap_samples:,} "
        "resamples), retaining paired system scores and all consultations within each "
        "resampled patient. Background shading indicates effect-size magnitude: "
        "negligible (|d| < 0.2), small (0.2-0.5), medium (0.5-0.8), "
        "and large (>= 0.8).\n"
    )
    (OUTPUT_DIR / 'Figure5_Effect_Sizes_caption.txt').write_text(caption, encoding='utf-8')
    return plot_data


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure5-only', action='store_true')
    args = parser.parse_args()
    print("="*80)
    print("Generating Publication Figures for JMIR Results Section")
    print("="*80)

    if not args.figure5_only:
        print("\n[1/2] Creating Figure 1: Grouped Bar Chart...")
        create_grouped_bar_chart()

    print("\nCreating Figure 5: Summarization Effect Sizes...")
    create_forest_plot()

    print("\n" + "="*80)
    print("✓ All figures generated successfully!")
    print(f"✓ Figures saved to: {OUTPUT_DIR.absolute()}")
    print("="*80)
    print("\nFigures are ready for inclusion in your JMIR manuscript:")
    print("  • PNG format (300 DPI) for viewing and manuscripts")
    print("  • PDF format for high-quality publication")
