"""Bar charts for the 2x2 hierarchical ablation."""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

CSV = "eval/ablation/hierarchical_ablation_2x2.csv"
OUT = "eval/ablation/ccl_visualizations/hierarchical_ablation.png"

labels, pfc, beat, fid, div = [], [], [], [], []
with open(CSV) as f:
    for r in csv.DictReader(f):
        labels.append(r["variant"].replace("_", "\n", 1))
        pfc.append(float(r["PFC_down"]))
        beat.append(float(r["beat_align_up"]))
        fid.append(float(r["fid_k_down"]))
        div.append(abs(float(r["div_k"]) - float(r["div_k_gt"])))

x = np.arange(len(labels))
colors = ["#9aa0a6", "#34a853", "#fbbc04", "#ea4335"]
panels = [
    ("PFC  (lower better)", pfc, False),
    ("Beat Align  (higher better)", beat, True),
    ("FID_k  (lower better)", fid, False),
    ("|Div_k - GT|  (lower better)", div, False),
]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, (title, vals, _) in zip(axes, panels):
    ax.bar(x, vals, color=colors)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
fig.suptitle("2x2 Ablation: hierarchical conditioning x hierarchical loss (5-epoch fine-tune)", fontsize=12)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=180, bbox_inches="tight")
print("saved:", OUT)


if __name__ == "__main__":
    main() if False else None
