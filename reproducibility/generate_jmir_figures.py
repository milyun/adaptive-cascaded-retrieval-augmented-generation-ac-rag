"""
Generate Publication-Quality Figures for JMIR Journal
Based on batch evaluation results of all 8 systems
Excludes section completeness metric as requested
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set publication-quality style for JMIR
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3

# Configuration
RESULTS_DIR = Path("batch_evaluation_results_all_systems")
OUTPUT_DIR = Path("jmir_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load data
print("Loading evaluation results...")
with open(RESULTS_DIR / "multi_system_summary.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# System labels and grouping
all_systems = data["dataset_info"]["system_labels"]
system_groups = {
    'Zero-Shot': ['Zero-Shot-Deepgram', 'Zero-Shot-Whisper'],
    'FS-GKP': ['FS-GKP-Deepgram', 'FS-GKP-Whisper'],
    'Original-RAG': ['Original-RAG-Deepgram', 'Original-RAG-Whisper'],
    'AC-RAG': ['AC-RAG-Deepgram', 'AC-RAG-Whisper']
}

# Color scheme (JMIR-friendly, colorblind-safe)
system_colors = {
    'AC-RAG-Deepgram': '#2E86AB',      # Blue
    'AC-RAG-Whisper': '#A23B72',       # Purple
    'Original-RAG-Deepgram': '#F18F01', # Orange
    'Original-RAG-Whisper': '#C73E1D',  # Red-Orange
    'FS-GKP-Deepgram': '#6A994E',      # Green
    'FS-GKP-Whisper': '#BC4B51',       # Dark Red
    'Zero-Shot-Deepgram': '#8B8C89',   # Gray
    'Zero-Shot-Whisper': '#5E60CE'     # Indigo
}

# Metrics to visualize (excluding section_completeness)
key_metrics = {
    'BERTScore F1': 'bertscore_f1',
    'Semantic Similarity': 'semantic_similarity',
    'TF-IDF Cosine': 'tfidf_cosine',
    'ROUGE-L (Char)': 'rougeL_f1_char',
    'ROUGE-1 F1': 'rouge1_f1',
    'ROUGE-2 F1': 'rouge2_f1',
    'BLEU (Word)': 'bleu_word',
    'BLEU (Char)': 'bleu_char',
    'Entity Recall': 'entity_recall',
}

print(f"Systems: {len(all_systems)}")
print(f"Metrics: {len(key_metrics)} (excluding section completeness)")

# ============================================================================
# FIGURE 1: Comprehensive Metrics Comparison - All 8 Systems
# ============================================================================

def create_comprehensive_bar_chart():
    """Create grouped bar chart comparing all 8 systems across key metrics"""
    print("\nGenerating Figure 1: Comprehensive Metrics Comparison...")

    # Prepare data
    metrics_data = data["aggregated_metrics_with_std"]
    plot_data = []

    for display_name, metric_key in key_metrics.items():
        for system in all_systems:
            mean_val = metrics_data[system][metric_key]['mean']
            std_val = metrics_data[system][metric_key]['std']
            plot_data.append({
                'Metric': display_name,
                'System': system,
                'Mean': mean_val,
                'Std': std_val
            })

    df_plot = pd.DataFrame(plot_data)

    # Create figure with subplots (3 rows x 3 columns)
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()

    for idx, (display_name, metric_key) in enumerate(key_metrics.items()):
        ax = axes[idx]
        metric_data = df_plot[df_plot['Metric'] == display_name]

        # Create bar positions
        x = np.arange(len(all_systems))
        bars = ax.bar(x, metric_data['Mean'].values,
                     color=[system_colors[s] for s in all_systems],
                     alpha=0.85, edgecolor='black', linewidth=0.8)

        # Add error bars
        ax.errorbar(x, metric_data['Mean'].values,
                   yerr=metric_data['Std'].values,
                   fmt='none', ecolor='black', capsize=3, alpha=0.6, linewidth=1)

        # Formatting
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title(display_name, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('-', '-\n') for s in all_systems],
                          rotation=45, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, max(metric_data['Mean'].values) * 1.2)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, metric_data['Mean'].values)):
            ax.text(bar.get_x() + bar.get_width()/2, val + metric_data['Std'].values[i] + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    # Remove empty subplot
    fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Figure1_Comprehensive_Metrics_Comparison.png",
                dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "Figure1_Comprehensive_Metrics_Comparison.pdf",
                bbox_inches='tight')
    print(f"✓ Saved: Figure1_Comprehensive_Metrics_Comparison.png/pdf")
    plt.close()


# ============================================================================
# FIGURE 2: System Group Comparison (Method Comparison)
# ============================================================================

def create_method_comparison_chart():
    """Compare the 4 main methods (AC-RAG, Original-RAG, FS-GKP, Zero-Shot)"""
    print("\nGenerating Figure 2: Seven-Metric Method Comparison...")

    metrics_data = data["aggregated_metrics_with_std"]

    # Define metric labels with newlines for better readability
    metrics_display = {
        'BERTScore\nF1': 'bertscore_f1',
        'Semantic\nSimilarity': 'semantic_similarity',
        'TF-IDF\nCosine': 'tfidf_cosine',
        'ROUGE-L\n(Char)': 'rougeL_f1_char',
        'BLEU\n(Word)': 'bleu_word',
        'BLEU\n(Char)': 'bleu_char',
        'Entity\nRecall': 'entity_recall',
    }

    # Average across ASR systems (Deepgram + Whisper) for each method
    method_scores = {}
    method_stds = {}
    for method, systems in system_groups.items():
        method_scores[method] = {}
        method_stds[method] = {}
        for display_name, metric_key in metrics_display.items():
            means = [metrics_data[s][metric_key]['mean'] for s in systems]
            stds = [metrics_data[s][metric_key]['std'] for s in systems]
            method_scores[method][display_name] = np.mean(means)
            method_stds[method][display_name] = np.sqrt(np.mean([s**2 for s in stds]))

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Bar positions
    metrics_list = list(metrics_display.keys())
    x = np.arange(len(metrics_list))
    width = 0.2
    methods = list(system_groups.keys())

    # Color scheme matching the grouped bar chart style
    method_colors_grouped = {
        'Zero-Shot':'#E74C3C',
        'FS-GKP': '#F39C12',
        'Original-RAG': '#3498DB',
        'AC-RAG': '#2ECC71',
    }

    # Plot bars for each method
    for i, method in enumerate(methods):
        positions = x + (i - 1.5) * width
        means = [method_scores[method][metric] for metric in metrics_list]
        stds = [method_stds[method][metric] for metric in metrics_list]

        bars = ax.bar(
            positions,
            means,
            width,
            label=method,
            color=method_colors_grouped[method],
            alpha=0.8,
            edgecolor='black',
            linewidth=0.5
        )

        # Add error bars
        ax.errorbar(
            positions,
            means,
            yerr=stds,
            fmt='none',
            ecolor='black',
            capsize=2,
            capthick=0.5,
            linewidth=0.5,
            alpha=0.6
        )

        # Add value labels above error bars
        for pos, mean, std in zip(positions, means, stds):
            ax.text(pos, mean + std + 0.01,
                   f'{mean:.2f}', ha='center', va='bottom',
                   fontsize=7, rotation=0)

    # Customize plot
    ax.set_xlabel('Evaluation Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Method Comparison Across Seven Metrics (Average of Deepgram & Whisper)',
                fontsize=13, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_list, rotation=0, ha='center')
    ax.legend(loc='upper right', ncol=2, frameon=True, fancybox=True, shadow=True)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add footnote
    fig.text(0.1, 0.02,
            'Bars: mean of Deepgram and Whisper means. Error bars: root-mean-square within-ASR SD.',
            ha='left', fontsize=8, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    figure_dir = OUTPUT_DIR
    figure_dir.mkdir(exist_ok=True)
    for extension in ['png', 'pdf']:
        output_path = figure_dir / f'Figure2_Method_Comparison.{extension}'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'Saved: {output_path}')
    plt.close(fig)
    records = [
        {
            'method': method,
            'metric': metric_key,
            'mean': method_scores[method][display_name],
            'rms_within_asr_sd': method_stds[method][display_name],
        }
        for method in methods
        for display_name, metric_key in metrics_display.items()
    ]
    plot_data = pd.DataFrame(records)
    plot_data.to_csv(figure_dir / 'Figure2_Method_Comparison_data.csv', index=False)
    caption = (
        'Figure 4. Comparative performance of four summarization approaches across '
        'seven evaluation metrics. Each method produced 178 summaries from 89 '
        'consultations (89 summaries per ASR system). Bars show the mean of the '
        'Deepgram and Whisper means; error bars show the root-mean-square of the '
        'two within-ASR standard deviations. ROUGE-L was computed at the character '
        'level; word-level ROUGE-1 and ROUGE-2 are omitted because the default '
        'tokenizer discarded Japanese text.\n'
    )
    (figure_dir / 'Figure2_Method_Comparison_caption.txt').write_text(caption, encoding='utf-8')
    return plot_data


# ============================================================================
# FIGURE 3: Heatmap of All Systems vs Metrics
# ============================================================================

def create_heatmap():
    """Create heatmap showing all systems vs all metrics"""
    print("\nGenerating Figure 3: Performance Heatmap...")

    metrics_data = data["aggregated_metrics"]

    # Build matrix
    matrix_data = []
    for system in all_systems:
        row = [metrics_data[system][metric_key] for metric_key in key_metrics.values()]
        matrix_data.append(row)

    df_matrix = pd.DataFrame(matrix_data,
                            index=[s.replace('-', '\n') for s in all_systems],
                            columns=key_metrics.keys())

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Create heatmap
    sns.heatmap(df_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                center=0.5, vmin=0, vmax=1,
                cbar_kws={'label': 'Score', 'shrink': 0.8},
                linewidths=0.5, linecolor='gray',
                ax=ax, annot_kws={'size': 9})

    ax.set_title('Performance Heatmap: All Systems vs All Metrics',
                fontweight='bold', fontsize=14, pad=15)
    ax.set_xlabel('Evaluation Metrics', fontweight='bold', fontsize=12)
    ax.set_ylabel('Systems', fontweight='bold', fontsize=12)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Figure3_Performance_Heatmap.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "Figure3_Performance_Heatmap.pdf", bbox_inches='tight')
    print(f"✓ Saved: Figure3_Performance_Heatmap.png/pdf")
    plt.close()


# ============================================================================
# FIGURE 4: ASR System Comparison (Deepgram vs Whisper)
# ============================================================================

def create_asr_comparison():
    """Compare Deepgram vs Whisper across all methods"""
    print("\nGenerating Figure 4: ASR System Comparison...")

    metrics_data = data["aggregated_metrics"]

    # Calculate average scores for Deepgram and Whisper across all methods
    deepgram_systems = [s for s in all_systems if 'Deepgram' in s]
    whisper_systems = [s for s in all_systems if 'Whisper' in s]

    deepgram_scores = {}
    whisper_scores = {}

    for display_name, metric_key in key_metrics.items():
        deepgram_scores[display_name] = np.mean([metrics_data[s][metric_key]
                                                 for s in deepgram_systems])
        whisper_scores[display_name] = np.mean([metrics_data[s][metric_key]
                                                for s in whisper_systems])

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(key_metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, list(deepgram_scores.values()), width,
                  label='Deepgram', color='#2E86AB', alpha=0.85,
                  edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x + width/2, list(whisper_scores.values()), width,
                  label='Whisper', color='#A23B72', alpha=0.85,
                  edgecolor='black', linewidth=0.8)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Evaluation Metrics', fontweight='bold', fontsize=12)
    ax.set_ylabel('Average Score', fontweight='bold', fontsize=12)
    ax.set_title('ASR System Comparison: Deepgram vs Whisper\n(Averaged Across All Methods)',
                fontweight='bold', fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(key_metrics.keys(), rotation=45, ha='right')
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='black')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(max(deepgram_scores.values()), max(whisper_scores.values())) * 1.15)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Figure4_ASR_Comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "Figure4_ASR_Comparison.pdf", bbox_inches='tight')
    print(f"✓ Saved: Figure4_ASR_Comparison.png/pdf")
    plt.close()


# ============================================================================
# FIGURE 5: Radar Chart - Top 4 Systems
# ============================================================================

def create_radar_chart():
    """Create radar chart comparing top performing systems"""
    print("\nGenerating Figure 5: Radar Chart...")

    metrics_data = data["aggregated_metrics"]

    # Select top 4 systems based on average performance
    system_avg_scores = {}
    for system in all_systems:
        scores = [metrics_data[system][metric_key] for metric_key in key_metrics.values()]
        system_avg_scores[system] = np.mean(scores)

    top_systems = sorted(system_avg_scores.items(), key=lambda x: x[1], reverse=True)[:4]
    top_system_names = [s[0] for s in top_systems]

    # Prepare data
    angles = np.linspace(0, 2 * np.pi, len(key_metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    for system in top_system_names:
        values = [metrics_data[system][metric_key] for metric_key in key_metrics.values()]
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2, label=system,
               color=system_colors[system], alpha=0.8)
        ax.fill(angles, values, alpha=0.15, color=system_colors[system])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(key_metrics.keys(), fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)

    ax.set_title('Performance Radar Chart: Top 4 Systems',
                fontweight='bold', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), framealpha=0.9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Figure5_Radar_Chart_Top4.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "Figure5_Radar_Chart_Top4.pdf", bbox_inches='tight')
    print(f"✓ Saved: Figure5_Radar_Chart_Top4.png/pdf")
    plt.close()


# ============================================================================
# Additional output: Summary Table
# ============================================================================

def create_summary_table():
    """Create publication-ready summary table"""
    print("\nGenerating supplementary metrics summary...")

    metrics_data = data["aggregated_metrics_with_std"]

    # Build table data
    table_data = []
    for system in all_systems:
        row = {'System': system}
        for display_name, metric_key in key_metrics.items():
            mean = metrics_data[system][metric_key]['mean']
            std = metrics_data[system][metric_key]['std']
            row[display_name] = f"{mean:.3f}±{std:.3f}"
        table_data.append(row)

    df_table = pd.DataFrame(table_data)

    # Create figure
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df_table.values, colLabels=df_table.columns,
                    cellLoc='center', loc='center',
                    colWidths=[0.12] * len(df_table.columns))

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)

    # Style header
    for i in range(len(df_table.columns)):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(df_table) + 1):
        for j in range(len(df_table.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E8E8E8')

    ax.set_title('Summary Table: All Systems Performance (Mean ± SD)',
                fontweight='bold', fontsize=16, pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Metrics_Summary_Overview.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "Metrics_Summary_Overview.pdf", bbox_inches='tight')
    print(f"✓ Saved: Metrics_Summary_Overview.png/pdf")
    plt.close()


# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("="*80)
    print("GENERATING JMIR PUBLICATION FIGURES")
    print("="*80)
    print(f"Input: {RESULTS_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Metrics excluded: section_completeness")
    print("="*80)

    create_comprehensive_bar_chart()
    create_method_comparison_chart()
    create_heatmap()
    create_asr_comparison()
    create_radar_chart()
    create_summary_table()

    print("\n" + "="*80)
    print("✅ ALL FIGURES GENERATED SUCCESSFULLY")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    print("\nGenerated files:")
    for file in sorted(OUTPUT_DIR.glob("*")):
        print(f"  • {file.name}")
    print("\nFigures are ready for JMIR publication!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure4-only', action='store_true')
    args = parser.parse_args()
    if args.figure4_only:
        create_method_comparison_chart()
    else:
        main()
