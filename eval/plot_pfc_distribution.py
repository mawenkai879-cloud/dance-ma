import argparse
import glob
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge_dir", required=True)
    parser.add_argument("--no_ccl_dir", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def calc_pfc_values(motion_dir):
    values = []
    up_dir = 2
    flat_dirs = [i for i in range(3) if i != up_dir]
    dt = 1 / 30
    for path in sorted(glob.glob(os.path.join(motion_dir, "*.pkl"))):
        with open(path, "rb") as f:
            info = pickle.load(f)
        joint3d = info["full_pose"]
        root_v = (joint3d[1:, 0, :] - joint3d[:-1, 0, :]) / dt
        root_a = (root_v[1:] - root_v[:-1]) / dt
        root_a[:, up_dir] = np.maximum(root_a[:, up_dir], 0)
        root_a = np.linalg.norm(root_a, axis=-1)
        scaling = root_a.max()
        if scaling > 0:
            root_a /= scaling
        foot_idx = [7, 10, 8, 11]
        feet = joint3d[:, foot_idx]
        foot_v = np.linalg.norm(feet[2:, :, flat_dirs] - feet[1:-1, :, flat_dirs], axis=-1)
        foot_mins = np.zeros((len(foot_v), 2))
        foot_mins[:, 0] = np.minimum(foot_v[:, 0], foot_v[:, 1])
        foot_mins[:, 1] = np.minimum(foot_v[:, 2], foot_v[:, 3])
        values.append(float((foot_mins[:, 0] * foot_mins[:, 1] * root_a).mean() * 10000))
    return np.asarray(values)


def main():
    args = parse_args()
    edge = calc_pfc_values(args.edge_dir)
    no_ccl = calc_pfc_values(args.no_ccl_dir)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.5, 5.2), dpi=180)
    data = [edge, no_ccl]
    labels = [f"EDGE\nmean={edge.mean():.3f}", f"w/o CCL\nmean={no_ccl.mean():.3f}"]
    box = plt.boxplot(data, labels=labels, patch_artist=True, widths=0.55, showfliers=False)
    colors = ["#1f77b4", "#d62728"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)
        patch.set_edgecolor(color)
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data, start=1):
        x = i + rng.normal(0, 0.035, size=len(vals))
        plt.scatter(x, vals, s=32, color=colors[i - 1], alpha=0.78, edgecolors="white", linewidths=0.45)
    plt.ylabel("PFC ↓")
    plt.title("CCL Ablation: Physical Foot Contact Score Distribution")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(args.out)
    print(args.out)
    print("edge_mean", float(edge.mean()))
    print("no_ccl_mean", float(no_ccl.mean()))
    print("edge_n", len(edge))
    print("no_ccl_n", len(no_ccl))


if __name__ == "__main__":
    main()
