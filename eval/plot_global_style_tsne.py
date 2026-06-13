"""t-SNE of the learned hierarchical global style embeddings.

Left: raw jukebox features mean-pooled (no learning, control).
Right: global_encoder output of the hierarchical DanceDecoder.
If the right side clusters by genre much better than the left,
it proves the global branch learned whole-clip music style.
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from EDGE import EDGE

GENRE_NAMES = {
    "BR": "Break", "PO": "Pop", "LO": "Lock", "MH": "Middle Hip-hop",
    "LH": "LA Hip-hop", "HO": "House", "WA": "Waack", "KR": "Krump",
    "JS": "Street Jazz", "JB": "Ballet Jazz",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/train/edge_hierarchical_ft5/weights/train-5.pt")
    parser.add_argument("--feat_dir", default="data/test/jukebox_feats")
    parser.add_argument("--out", default="eval/ablation/ccl_visualizations/global_style_tsne.png")
    return parser.parse_args()


def main():
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.feat_dir, "*.npy")))
    genres = [os.path.basename(f)[1:3] for f in files]
    print(f"{len(files)} slices, genres: {sorted(set(genres))}")

    model = EDGE("jukebox", args.checkpoint, use_hierarchical=True)
    model.eval()
    net = model.model
    device = next(net.parameters()).device

    raw_means, global_embs = [], []
    with torch.no_grad():
        for f in files:
            feat = torch.from_numpy(np.load(f)).float().to(device)  # 150 x 4800
            pooled = feat.mean(dim=0)
            raw_means.append(pooled.cpu().numpy())
            emb = net.global_encoder(pooled.unsqueeze(0)).squeeze(0)
            global_embs.append(emb.cpu().numpy())
    raw_means = np.stack(raw_means)
    global_embs = np.stack(global_embs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    cmap = plt.get_cmap("tab10")
    genre_list = sorted(set(genres))
    for ax, data, title in [
        (axes[0], raw_means, "Raw music features (mean-pooled)"),
        (axes[1], global_embs, "Learned global style embedding"),
    ]:
        z = TSNE(n_components=2, perplexity=12, random_state=0, init="pca").fit_transform(data)
        for gi, g in enumerate(genre_list):
            idx = [i for i, gg in enumerate(genres) if gg == g]
            ax.scatter(z[idx, 0], z[idx, 1], s=22, color=cmap(gi % 10),
                       label=GENRE_NAMES.get(g, g), alpha=0.85)
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=False)
    fig.suptitle("Hierarchical global style token clusters music by genre", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print("saved:", args.out)


if __name__ == "__main__":
    main()
