"""Real constrained generation using GaussianDiffusion.inpaint_loop.

Takes a ground-truth motion slice from the cached test dataset as the
constraint "value", builds masks for four scenarios, and lets the model
truly inpaint the unconstrained parts.

Channel layout of the 151-dim representation:
  [0:4]   foot contact labels
  [4:7]   root translation
  [7:151] 24 joints x 6d rotation (joint j -> 7 + 6*j : 13 + 6*j)
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from EDGE import EDGE
from vis import SMPLSkeleton
from pytorch3d.transforms import RotateAxisAngle
from model.diffusion import ax_from_6v

UPPER_BODY = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
LOWER_BODY_ROOT = [0, 1, 2, 4, 5, 7, 8, 10, 11]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoint.pt")
    parser.add_argument("--use_hierarchical", action="store_true")
    parser.add_argument("--test_dataset", default="data/dataset_backups/test_tensor_dataset.pkl")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--out_dir", default="eval/ablation/constrained_generation")
    return parser.parse_args()


def joint_channels(joints):
    chans = []
    for j in joints:
        chans.extend(range(7 + 6 * j, 13 + 6 * j))
    return chans


def build_masks(horizon, repr_dim):
    masks = {}

    m = torch.zeros(1, horizon, repr_dim)
    m[:, : horizon // 5, :] = 1.0  # first 20% of frames fully fixed
    masks["seed"] = m

    m = torch.zeros(1, horizon, repr_dim)
    mid = horizon // 2
    m[:, mid - 2 : mid + 3, :] = 1.0  # middle keyframe (small window)
    masks["keyframe"] = m

    m = torch.zeros(1, horizon, repr_dim)
    m[:, :, joint_channels(UPPER_BODY)] = 1.0
    masks["upper"] = m

    m = torch.zeros(1, horizon, repr_dim)
    m[:, :, joint_channels(LOWER_BODY_ROOT)] = 1.0
    m[:, :, 0:7] = 1.0  # contacts + root translation
    masks["lower"] = m

    return masks


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.test_dataset, "rb") as f:
        test_dataset = pickle.load(f)
    pose_gt, cond, filename, *_ = test_dataset[args.sample_idx]
    pose_gt = torch.as_tensor(pose_gt).float().unsqueeze(0)  # 1 x horizon x 151
    cond = torch.as_tensor(cond).float().unsqueeze(0)
    print(f"constraint source: {filename}")

    model = EDGE("jukebox", args.checkpoint, use_hierarchical=args.use_hierarchical)
    model.eval()
    diffusion = model.diffusion
    device = next(diffusion.parameters()).device
    horizon, repr_dim = pose_gt.shape[1], pose_gt.shape[2]

    smpl = SMPLSkeleton(device)

    def decode(samples):
        samples = model.normalizer.unnormalize(samples.cpu())
        _, samples = torch.split(samples, (4, samples.shape[2] - 4), dim=2)
        b, s, _ = samples.shape
        pos = samples[:, :, :3].to(device)
        q = ax_from_6v(samples[:, :, 3:].reshape(b, s, 24, 6).to(device))
        full_pose = smpl.forward(q, pos).detach().cpu().numpy()[0]
        return (
            q.reshape(s, 72).detach().cpu().numpy(),
            pos[0].detach().cpu().numpy(),
            full_pose,
        )

    masks = build_masks(horizon, repr_dim)
    shape = (1, horizon, repr_dim)

    # save the GT constraint source too
    smpl_poses, smpl_trans, full_pose = decode(pose_gt.clone())
    with open(out_dir / "constraint_source_gt.pkl", "wb") as f:
        pickle.dump(
            {"smpl_poses": smpl_poses, "smpl_trans": smpl_trans, "full_pose": full_pose, "filename": str(filename)},
            f,
        )

    for name, mask in masks.items():
        torch.manual_seed(1234)
        constraint = {"mask": mask.to(device), "value": pose_gt.to(device)}
        samples = diffusion.inpaint_loop(shape, cond.to(device), constraint=constraint)
        smpl_poses, smpl_trans, full_pose = decode(samples)
        with open(out_dir / f"constrained_{name}.pkl", "wb") as f:
            pickle.dump(
                {
                    "smpl_poses": smpl_poses,
                    "smpl_trans": smpl_trans,
                    "full_pose": full_pose,
                    "mask_type": name,
                    "mask_frames": mask[0].any(dim=-1).numpy(),
                    "filename": str(filename),
                },
                f,
            )
        print(f"saved {name}")


if __name__ == "__main__":
    main()
