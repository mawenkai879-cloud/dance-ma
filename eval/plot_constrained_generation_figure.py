import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
UPPER_BODY = [9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
LOWER_BODY_ROOT = [0, 1, 2, 4, 5, 7, 8, 10, 11]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_pkl", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames_per_row", type=int, default=7)
    return parser.parse_args()


def load_pose(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    pose = np.asarray(data["full_pose"], dtype=np.float32)
    return pose - pose[:, :1, :]


def setup_axis(ax, pose, lim):
    center = pose.mean(axis=0)
    ax.set_xlim(center[0] - lim, center[0] + lim)
    ax.set_ylim(center[1] - lim, center[1] + lim)
    ax.set_zlim(max(0, center[2] - lim), center[2] + lim)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=13, azim=-68)
    ax.set_axis_off()


def draw_skeleton(ax, pose, base_color="#2f7f88", constrained_joints=None, constrained_frame=False):
    constraint_color = "#62e6ad"
    color = constraint_color if constrained_frame else base_color
    for joint, parent in enumerate(SMPL_PARENTS):
        if parent < 0:
            continue
        xs = [pose[parent, 0], pose[joint, 0]]
        ys = [pose[parent, 1], pose[joint, 1]]
        zs = [pose[parent, 2], pose[joint, 2]]
        edge_color = color
        if constrained_joints is not None and (joint in constrained_joints or parent in constrained_joints):
            edge_color = constraint_color
        ax.plot(xs, ys, zs, color=edge_color, linewidth=2.25, solid_capstyle="round")
    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], color=color, s=7, depthshade=False)
    if constrained_joints is not None:
        cj = np.asarray(constrained_joints, dtype=int)
        ax.scatter(pose[cj, 0], pose[cj, 1], pose[cj, 2], color=constraint_color, s=24, depthshade=False)


def row_config(row_name, frames):
    mid = len(frames) // 2
    if row_name == "seed":
        return {0: "frame", 1: "frame"}
    if row_name == "keyframe":
        return {mid: "frame"}
    if row_name == "upper":
        return {i: "upper" for i in range(len(frames))}
    if row_name == "lower":
        return {i: "lower" for i in range(len(frames))}
    return {}


def main():
    args = parse_args()
    pose = load_pose(args.motion_pkl)
    frames = np.linspace(0, len(pose) - 1, args.frames_per_row, dtype=int)
    selected = pose[frames]
    lim = max(float(np.max(np.ptp(selected.reshape(-1, 3), axis=0)) * 0.28), 0.8)

    row_titles = [
        "completion from seed motion",
        "specified middle keyframe",
        "upper-body joint constraints",
        "lower-body + root constraints",
    ]
    row_names = ["seed", "keyframe", "upper", "lower"]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(args.frames_per_row * 1.22, 5.2), dpi=220)
    for row, (row_name, row_title) in enumerate(zip(row_names, row_titles)):
        constraints = row_config(row_name, frames)
        for col, frame in enumerate(frames):
            ax = fig.add_subplot(4, args.frames_per_row, row * args.frames_per_row + col + 1, projection="3d")
            mode = constraints.get(col)
            constrained_joints = None
            constrained_frame = False
            if mode == "frame":
                constrained_frame = True
            elif mode == "upper":
                constrained_joints = UPPER_BODY
            elif mode == "lower":
                constrained_joints = LOWER_BODY_ROOT
            draw_skeleton(ax, pose[frame], constrained_joints=constrained_joints, constrained_frame=constrained_frame)
            setup_axis(ax, pose[frame], lim)
            if col == 0:
                ax.text2D(-0.25, 0.45, row_title, transform=ax.transAxes, fontsize=7.2, rotation=90, va="center")
    plt.tight_layout(pad=0.05)
    plt.savefig(args.out, bbox_inches="tight")
    print(args.out)


if __name__ == "__main__":
    main()
