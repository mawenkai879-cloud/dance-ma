import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge_pkl", required=True)
    parser.add_argument("--no_ccl_pkl", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--title", default="Generated Dance Pose Sequence")
    return parser.parse_args()


def load_pose(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if "full_pose" not in data:
        raise KeyError(f"{path} does not contain full_pose")
    pose = np.asarray(data["full_pose"], dtype=np.float32)
    pose = pose - pose[:, :1, :]
    return pose


def equalize_axes(ax, pose, lim):
    center = pose.mean(axis=0)
    ax.set_xlim(center[0] - lim, center[0] + lim)
    ax.set_ylim(center[1] - lim, center[1] + lim)
    ax.set_zlim(max(0, center[2] - lim), center[2] + lim)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=16, azim=-70)
    ax.set_axis_off()


def draw_pose(ax, pose, color, constraint_color=None):
    for joint, parent in enumerate(SMPL_PARENTS):
        if parent < 0:
            continue
        xs = [pose[parent, 0], pose[joint, 0]]
        ys = [pose[parent, 1], pose[joint, 1]]
        zs = [pose[parent, 2], pose[joint, 2]]
        ax.plot(xs, ys, zs, color=color, linewidth=2.2, solid_capstyle="round")
    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], color=color, s=8, depthshade=False)
    if constraint_color is not None:
        highlight = [0, 7, 8, 10, 11]
        ax.scatter(pose[highlight, 0], pose[highlight, 1], pose[highlight, 2], color=constraint_color, s=20, depthshade=False)


def main():
    args = parse_args()
    edge = load_pose(args.edge_pkl)
    no_ccl = load_pose(args.no_ccl_pkl)
    n = min(len(edge), len(no_ccl))
    frames = np.linspace(0, n - 1, args.num_frames, dtype=int)
    all_pose = np.concatenate([edge[frames], no_ccl[frames]], axis=0)
    lim = float(np.max(np.ptp(all_pose.reshape(-1, 3), axis=0)) * 0.28)
    lim = max(lim, 0.8)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(args.num_frames * 1.25, 3.2), dpi=200)
    fig.suptitle(args.title, fontsize=12, y=0.98)
    row_names = ["EDGE", "w/o CCL"]
    colors = ["#48c78e", "#2b7a8b"]
    poses = [edge, no_ccl]
    for row, (name, color, seq) in enumerate(zip(row_names, colors, poses)):
        for col, frame in enumerate(frames):
            ax = fig.add_subplot(2, args.num_frames, row * args.num_frames + col + 1, projection="3d")
            draw_pose(ax, seq[frame], color=color, constraint_color="#f6c35b" if col in (0, args.num_frames // 2) else None)
            equalize_axes(ax, seq[frame], lim)
            if col == 0:
                ax.text2D(-0.15, 0.45, name, transform=ax.transAxes, fontsize=10, fontweight="bold", rotation=90, va="center")
            if row == 1:
                ax.text2D(0.35, -0.08, f"t={frame}", transform=ax.transAxes, fontsize=7)
    plt.tight_layout(rect=[0, 0, 1, 0.94], pad=0.1)
    plt.savefig(args.out, bbox_inches="tight")
    print(args.out)


if __name__ == "__main__":
    main()
