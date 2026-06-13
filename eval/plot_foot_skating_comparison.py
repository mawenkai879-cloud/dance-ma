import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge_pkl", required=True)
    parser.add_argument("--no_ccl_pkl", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="Foot skating comparison")
    return parser.parse_args()


def load_pose(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return np.asarray(data["full_pose"])


def foot_stats(joint3d):
    foot_idx = [7, 10, 8, 11]
    up_dir = 2
    flat_dirs = [0, 1]
    feet = joint3d[:, foot_idx]
    heights = feet[1:-1, :, up_dir]
    horizontal_velocity = np.linalg.norm(feet[2:, :, flat_dirs] - feet[1:-1, :, flat_dirs], axis=-1)
    contact_threshold = np.percentile(heights, 35, axis=0, keepdims=True)
    contact_mask = heights <= contact_threshold
    contact_velocity = np.where(contact_mask, horizontal_velocity, np.nan)
    mean_contact_velocity = np.nanmean(contact_velocity, axis=1)
    mean_all_velocity = np.mean(horizontal_velocity, axis=1)
    return mean_all_velocity, mean_contact_velocity


def main():
    args = parse_args()
    edge_all, edge_contact = foot_stats(load_pose(args.edge_pkl))
    no_all, no_contact = foot_stats(load_pose(args.no_ccl_pkl))
    n = min(len(edge_all), len(no_all))
    frames = np.arange(n)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5.2), dpi=180)
    plt.plot(frames, edge_contact[:n], color="#1f77b4", linewidth=2.0, label=f"EDGE contact foot velocity, mean={np.nanmean(edge_contact[:n]):.4f}")
    plt.plot(frames, no_contact[:n], color="#d62728", linewidth=2.0, label=f"w/o CCL contact foot velocity, mean={np.nanmean(no_contact[:n]):.4f}")
    plt.plot(frames, edge_all[:n], color="#1f77b4", linewidth=1.0, alpha=0.25, linestyle="--", label="EDGE all-frame foot velocity")
    plt.plot(frames, no_all[:n], color="#d62728", linewidth=1.0, alpha=0.25, linestyle="--", label="w/o CCL all-frame foot velocity")
    plt.xlabel("Frame")
    plt.ylabel("Horizontal foot velocity")
    plt.title(args.title)
    plt.grid(alpha=0.25)
    plt.legend(frameon=True, fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out)
    print(args.out)
    print("edge_contact_mean", float(np.nanmean(edge_contact[:n])))
    print("no_ccl_contact_mean", float(np.nanmean(no_contact[:n])))


if __name__ == "__main__":
    main()
