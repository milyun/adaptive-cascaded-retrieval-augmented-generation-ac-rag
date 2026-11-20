"""Figure 6: mean nurse ratings (1-5) by system and dimension, complete cases as in Table 4."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--xlsx", default=str(Path.home() / "Downloads/Nurse Evaluation.xlsx"))
parser.add_argument("--output-dir", default="figures")
args = parser.parse_args()

SYSTEMS = ["Zero-Shot", "FS-GKP", "Original-RAG", "AC-RAG"]
COLORS = ["#b5b0a6", "#85b4e8", "#3a8ae0", "#0d4079"]
DIMS = [("accuracy", "Factual\naccuracy"), ("completeness", "Complete-\nness"),
        ("hallucination", "No\nhallucination"), ("safety", "Clinical\nsafety"),
        ("format", "Format\ncompliance"), ("usefulness", "Overall\nusefulness")]

raw = pd.read_excel(args.xlsx, sheet_name="Responses")
long = pd.DataFrame([
    {"case": r.display_id, "system": r[f"system_{L}"], "dim": dim, "value": r[f"{dim}_{L}"]}
    for _, r in raw.iterrows() for L in "ABCD" for dim, _ in DIMS
])

means, sds, ns = {}, {}, {}
for dim, _ in DIMS:
    wide = long[long.dim == dim].pivot_table(index="case", columns="system", values="value", aggfunc="first")[SYSTEMS].dropna()
    means[dim], sds[dim], ns[dim] = wide.mean(), wide.std(), len(wide)
    print(dim, ns[dim], [f"{wide[s].mean():.2f} ({wide[s].std():.2f})" for s in SYSTEMS])

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.2
x = np.arange(len(DIMS))
for i, system in enumerate(SYSTEMS):
    ax.bar(x + (i - 1.5) * width, [means[d][system] for d, _ in DIMS], width, color=COLORS[i],
           edgecolor="white", label=system,
           yerr=[sds[d][system] for d, _ in DIMS], capsize=2.5, error_kw={"elinewidth": 0.8, "ecolor": "#555555"})
ax.set_ylim(1, 5)
ax.set_ylabel("Mean rating (1-5)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([label for _, label in DIMS], fontsize=11)
ax.yaxis.grid(True, color="#e5e5e5", linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=4, frameon=False, fontsize=11)

out = Path(args.output_dir)
out.mkdir(exist_ok=True)
fig.savefig(out / "Figure6_nurse_ratings.png", dpi=300, bbox_inches="tight")
fig.savefig(out / "Figure6_nurse_ratings.pdf", bbox_inches="tight")
