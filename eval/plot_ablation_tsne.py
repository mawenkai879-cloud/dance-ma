import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_bailando_metrics import extract_features_from_dir, extract_gt_features


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge_dir", required=True)
    parser.add_argument("--no_ccl_dir", required=True)
    parser.add_argument("--gt_dir", default="data/test/motions_sliced")
    parser.add_argument("--out", default="eval/ablation/tsne_motion_features.png")
    parser.add_argument("--max_files", type=int, default=100)
    parser.add_argument("--feature", choices=["kinetic", "manual", "both"], default="both")
    return parser.parse_args()


def select_features(kinetic, manual, feature):
    if feature == "kinetic":
        return kinetic
    if feature == "manual":
        return manual
    return np.concatenate([kinetic, manual], axis=1)


def main():
    args = parse_args()
    edge_k, edge_m, _ = extract_features_from_dir(args.edge_dir, max_files=args.max_files)
    no_ccl_k, no_ccl_m, _ = extract_features_from_dir(args.no_ccl_dir, max_files=args.max_files)
    gt_k, gt_m = extract_gt_features(args.gt_dir, max_files=args.max_files)

    groups = [
        ("EDGE", select_features(edge_k, edge_m, args.feature), "#1f77b4"),
        ("w/o CCL", select_features(no_ccl_k, no_ccl_m, args.feature), "#d62728"),
        ("Ground Truth", select_features(gt_k, gt_m, args.feature), "#2ca02c"),
    ]
    x = np.concatenate([g[1] for g in groups], axis=0)
    labels = np.concatenate([[g[0]] * len(g[1]) for g in groups])

    x = StandardScaler().fit_transform(x)
    perplexity = max(2, min(30, len(x) // 3))
    emb = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=0).fit_transform(x)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6), dpi=180)
    start = 0
    for name, feats, color in groups:
        end = start + len(feats)
        plt.scatter(emb[start:end, 0], emb[start:end, 1], s=34, alpha=0.82, label=f"{name} (n={len(feats)})", c=color, edgecolors="white", linewidths=0.4)
        start = end
    plt.title(f"t-SNE of Motion Features ({args.feature})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
