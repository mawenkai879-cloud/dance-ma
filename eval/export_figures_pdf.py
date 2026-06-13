"""Export the key result visualizations (PNG) to PDF.

- Produces one combined multi-page PDF (docs/figures/all_figures.pdf).
- Also writes one PDF per figure under docs/figures/.
"""
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

VIS = "eval/ablation/ccl_visualizations"
OUT = "docs/figures"

FIGURES = [
    ("hierarchical_ablation_fair_table.png", "Controlled ablation (10-epoch, loss weight 1.5)"),
    ("hierarchical_ablation.png",            "Ablation: per-metric bar charts"),
    ("global_style_tsne.png",                "Global style embedding t-SNE (clusters by genre)"),
    ("style_swap_figure.png",                "Style-swap experiment (global guides local)"),
    ("real_constrained_generation_hierarchical.png", "Constrained / editable generation"),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    combined = os.path.join(OUT, "all_figures.pdf")
    with PdfPages(combined) as pdf:
        for fname, title in FIGURES:
            path = os.path.join(VIS, fname)
            if not os.path.exists(path):
                print("skip (missing):", path)
                continue
            img = mpimg.imread(path)
            h, w = img.shape[:2]
            fig = plt.figure(figsize=(w / 150, h / 150 + 0.4))
            ax = fig.add_axes([0, 0, 1, h / (h + 60)])
            ax.imshow(img)
            ax.axis("off")
            fig.suptitle(title, fontsize=10, y=0.99)
            # per-figure pdf
            single = os.path.join(OUT, fname.replace(".png", ".pdf"))
            fig.savefig(single, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            print("exported:", single)
    print("combined PDF:", combined)


if __name__ == "__main__":
    main()
