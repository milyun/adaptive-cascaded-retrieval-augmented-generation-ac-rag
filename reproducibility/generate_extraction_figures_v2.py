"""
generate_extraction_figures_v2.py
==================================
Regenerate Figure 2 (grouped Precision/Recall/F1 bar chart) and
Figure 3 (per-parameter precision heatmap with n-count and hatching)
from the corrected matched-only evaluation data.

Source : table2_precision_recall_f1.csv   (produced by compute_extraction_prf1_table2.py)
Output : extractionComparison_v2.png/pdf  (replaces extractionComparison.png)
         extractionHeatmap_v2.png/pdf     (replaces extractionHeatmap.png)
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
N_THRESHOLD  = 10          # cells with n < N_THRESHOLD are hatched / low-confidence
OUTPUT_DIR   = Path(".")   # save alongside the existing extraction figures
DATA_CSV     = Path("table2_precision_recall_f1.csv")

# Publication style (matches existing JMIR figures)
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 1.2,
})

# Display order and labels
SYS_ORDER = [
    "FS-GKP (Deepgram)",
    "FS-GKP (Whisper)",
    "Conservative (Deepgram)",
    "Conservative (Whisper)",
]

SYS_SHORT = {
    "FS-GKP (Deepgram)":         "FS-GKP\n(Deepgram)",
    "FS-GKP (Whisper)":          "FS-GKP\n(Whisper)",
    "Conservative (Deepgram)":   "Conservative\n(Deepgram)",
    "Conservative (Whisper)":    "Conservative\n(Whisper)",
}

PARAM_ORDER = [
    "Body Temperature (°C)",
    "Blood Pressure (mmHg)",
    "Pulse (bpm)",
    "Respiratory Rate (bpm)",
    "SpO2 (%)",
    "Defecation",
    "Appetite",
]

PARAM_SHORT = {
    "Body Temperature (°C)":  "Body\nTemp",
    "Blood Pressure (mmHg)":  "Blood\nPressure",
    "Pulse (bpm)":            "Pulse",
    "Respiratory Rate (bpm)": "Respiratory\nRate",
    "SpO2 (%)":               "SpO2",
    "Defecation":             "Defecation",
    "Appetite":               "Appetite",
}

# Colorblind-safe palette
COLORS = {
    "FS-GKP":       "#2E86AB",   # teal-blue
    "Conservative": "#A23B72",   # purple
}
METRIC_COLORS = {
    "Precision": "#2E86AB",   # blue
    "Recall":    "#F18F01",   # amber
    "F1":        "#6A994E",   # green
}

def load_data(path):
    loaded = pd.read_csv(path)
    if {"strategy", "asr", "parameter"}.issubset(loaded.columns):
        loaded = loaded.loc[loaded["parameter"].ne("Overall (micro)")].rename(columns={
            "parameter": "Parameter",
            "precision_pct": "Precision",
            "recall_pct": "Recall",
            "f1_pct": "F1",
        })
        loaded["System"] = loaded["strategy"].map({"AC-RAG": "Conservative", "FS-GKP": "FS-GKP"})
        loaded["System"] = loaded["System"] + " (" + loaded["asr"] + ")"
    else:
        loaded = loaded.rename(columns={
            "Precision (%)": "Precision",
            "Recall (%)": "Recall",
            "F1 (%)": "F1",
        })
    loaded["n"] = loaded["TP"] + loaded["FP"]
    return loaded


df = None

# ── Figure 2: Grouped Precision / Recall / F1 bar chart ───────────────────────

def make_figure2():
    # Aggregate across parameters (micro-average via TP/FP/FN sums)
    agg = (
        df.groupby("System")[["TP", "FP", "FN"]]
        .sum()
        .reset_index()
    )
    agg["Precision"] = agg["TP"] / (agg["TP"] + agg["FP"]) * 100
    agg["Recall"]    = agg["TP"] / (agg["TP"] + agg["FN"]) * 100
    agg["F1"]        = (2 * agg["Precision"] * agg["Recall"] /
                        (agg["Precision"] + agg["Recall"]).replace(0, np.nan))
    agg = agg.set_index("System").reindex(SYS_ORDER)

    metrics  = ["Precision", "Recall", "F1"]
    x        = np.arange(len(SYS_ORDER))
    n_bars   = len(metrics)
    width    = 0.22
    offsets  = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * width

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, metric in enumerate(metrics):
        vals = agg[metric].values
        bars = ax.bar(
            x + offsets[i], vals, width,
            color=METRIC_COLORS[metric],
            alpha=0.88,
            edgecolor="black", linewidth=0.6,
            label=metric,
            zorder=3,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{v:.1f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
            )

    # Shade Conservative vs FS-GKP background regions
    ax.axvspan(1.5, 3.6, color="#A23B72", alpha=0.05, zorder=0)
    ax.axvspan(-0.6, 1.5, color="#2E86AB", alpha=0.05, zorder=0)
    ax.text(0.5, 96, "FS-GKP", ha="center", va="top",
            fontsize=9, color="#2E86AB", fontstyle="italic")
    ax.text(2.5, 96, "Conservative", ha="center", va="top",
            fontsize=9, color="#A23B72", fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels([SYS_SHORT[s] for s in SYS_ORDER], fontsize=10)
    ax.set_ylabel("Score (%)", fontweight="bold")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        title="Metric", loc="upper right",
        frameon=True, fancybox=False, framealpha=0.9,
    )
    recall_denominator = int(df.loc[df["System"].eq(SYS_ORDER[0]), ["TP", "FN"]].to_numpy().sum())
    fig.text(
        0.5, 0.015,
        "Precision = TP/(TP+FP); recall = TP/T; F1 = harmonic mean.\n"
        f"Recall denominator: {recall_denominator} visit-parameter pairs with scorable reference values.",
        ha="center", fontsize=8, style="italic",
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    for ext in ("png", "pdf"):
        out = OUTPUT_DIR / f"extractionComparison_v2.{ext}"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  Saved → {out.name}")
    plt.close()
    print("Figure 2 done.")


# ── Figure 3: Precision heatmap with n annotation and hatching ────────────────

def make_figure3():
    # Pivot to matrix: rows = systems, columns = parameters
    prec_mat = df.pivot(index="System", columns="Parameter", values="Precision")
    n_mat    = df.pivot(index="System", columns="Parameter", values="n")

    prec_mat = prec_mat.reindex(index=SYS_ORDER, columns=PARAM_ORDER)
    n_mat    = n_mat.reindex(index=SYS_ORDER, columns=PARAM_ORDER)

    # Mask cells with insufficient n for reliable precision estimates
    masked = prec_mat.where(n_mat >= N_THRESHOLD)

    row_labels = [SYS_SHORT[s] for s in SYS_ORDER]
    col_labels = [PARAM_SHORT[p] for p in PARAM_ORDER]

    fig, ax = plt.subplots(figsize=(13, 5.5))

    import matplotlib.colors as mcolors
    cmap = plt.get_cmap("RdYlGn")
    cmap.set_bad(color="#d3d3d3")   # light grey for masked cells

    # Draw the heatmap manually for full control
    norm = mcolors.Normalize(vmin=0, vmax=100)
    img  = ax.imshow(
        masked.values.astype(float),
        cmap=cmap, norm=norm,
        aspect="auto",
    )

    n_rows, n_cols = prec_mat.shape

    for r in range(n_rows):
        for c in range(n_cols):
            prec = prec_mat.iloc[r, c]
            n    = n_mat.iloc[r, c]

            low_n = n < N_THRESHOLD

            # Hatching for low-n cells
            if low_n:
                rect = plt.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    fill=False,
                    hatch="///",
                    edgecolor="#666666",
                    linewidth=0,
                )
                ax.add_patch(rect)
                # Draw cell border
                border = plt.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    fill=False,
                    edgecolor="#333333",
                    linewidth=1.2,
                )
                ax.add_patch(border)

            # Text annotation: precision on top line, n on bottom line
            txt_color = "black"
            if not low_n and prec is not None and not np.isnan(prec):
                brightness = 0.299 * cmap(norm(prec))[0] + \
                             0.587 * cmap(norm(prec))[1] + \
                             0.114 * cmap(norm(prec))[2]
                txt_color = "white" if brightness < 0.45 else "black"

            label = f"{prec:.1f}%\n(n={int(n)})"
            style = "italic" if low_n else "normal"
            ax.text(
                c, r, label,
                ha="center", va="center",
                fontsize=8.5,
                fontweight="bold",
                color=txt_color,
                style=style,
            )

    # Axes formatting
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xlabel("Health Parameter", fontweight="bold", labelpad=8)
    ax.set_ylabel("Strategy (ASR)", fontweight="bold", labelpad=8)

    # Grid lines between cells
    for x in np.arange(-0.5, n_cols, 1):
        ax.axvline(x, color="white", linewidth=1.5)
    for y in np.arange(-0.5, n_rows, 1):
        ax.axhline(y, color="white", linewidth=1.5)

    # Draw separator between FS-GKP (rows 0-1) and Conservative (rows 2-3)
    ax.axhline(1.5, color="black", linewidth=2.5, linestyle="-")

    # Colour bar
    cbar = fig.colorbar(img, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Precision (%)", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.0f}%")
    )

    # Legend for hatching
    hatch_patch = mpatches.Patch(
        facecolor="white", hatch="///",
        edgecolor="#666666", linewidth=0.8,
        label=f"n < {N_THRESHOLD}: insufficient sample for reliable estimate",
    )
    hatch_legend = fig.legend(
        handles=[hatch_patch],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.14),
        bbox_transform=fig.transFigure,
        borderaxespad=0,
        fontsize=8.5,
        frameon=True,
        framealpha=0.9,
    )

    footnote = fig.text(
        0.5, 0.015,
        "Each cell: precision and n = TP + FP (scored predictions, not consultations).\n"
        "Hatched cells (n < 10) have few scored predictions; interpret their precision with caution.",
        ha="center", va="top", fontsize=8, style="italic",
    )

    plt.tight_layout(rect=[0, 0.14, 1, 1])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_bottom = ax.xaxis.label.get_window_extent(renderer).transformed(fig.transFigure.inverted()).y0
    hatch_legend.set_bbox_to_anchor((0.5, label_bottom - 0.035), transform=fig.transFigure)
    fig.canvas.draw()
    legend_bottom = hatch_legend.get_window_extent(renderer).transformed(fig.transFigure.inverted()).y0
    footnote.set_position((0.5, legend_bottom - 0.015))
    fig.canvas.draw()
    label_box = ax.xaxis.label.get_window_extent(renderer)
    legend_box = hatch_legend.get_window_extent(renderer)
    note_box = footnote.get_window_extent(renderer)
    if label_box.overlaps(legend_box) or legend_box.overlaps(note_box):
        raise ValueError("Heatmap label, legend and footnote must not overlap")
    for ext in ("png", "pdf"):
        out = OUTPUT_DIR / f"extractionHeatmap_v2.{ext}"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  Saved → {out.name}")
    plt.close()
    print("Figure 3 done.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-csv", type=Path, default=DATA_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    DATA_CSV = args.data_csv
    OUTPUT_DIR = args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_CSV.exists():
        raise FileNotFoundError(
            f"Source data not found: {DATA_CSV}\n"
            "Run compute_extraction_prf1_table2.py first."
        )

    df = load_data(DATA_CSV)
    print(f"Reading {DATA_CSV} …")

    print("\n[1/2] Figure 2 — Grouped P/R/F1 bar chart …")
    make_figure2()

    print("\n[2/2] Figure 3 — Precision heatmap with n annotation …")
    make_figure3()

    print("\nAll figures saved to:", OUTPUT_DIR.resolve())
