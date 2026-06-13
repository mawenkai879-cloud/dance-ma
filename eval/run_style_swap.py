"""Global style swap experiment for the hierarchical model.

Fix the local per-frame music condition and the sampling noise; only swap the
global style source (mean-pooled features from a different genre's music).
If the generated motion changes, the global token causally guides local motion.
"""
import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from EDGE import EDGE
from dataset.quaternion import ax_from_6v
from render_smpl_mesh_grid import GENERATED_BG, GENERATED_COLOR, build_meshes, render_mesh

GENRE_NAMES = {
    "BR": "Break", "PO": "Pop", "LO": "Lock", "MH": "Middle Hip-hop",
    "LH": "LA Hip-hop", "HO": "House", "WA": "Waack", "KR": "Krump",
    "JS": "Street Jazz", "JB": "Ballet Jazz",
}
ROW_BGS = [
    (0.92, 0.92, 0.92),  # original
    (0.98, 0.92, 0.80),
    (0.85, 0.93, 0.98),
    (0.88, 0.97, 0.88),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/train/edge_hierarchical_ft5/weights/train-5.pt")
    parser.add_argument("--feat_dir", default="data/test/jukebox_feats")
    parser.add_argument("--local_song", default="gBR_sBM_cAll_d04_mBR0_ch02_slice0")
    parser.add_argument("--swap_genres", default="JB,HO,KR")
    parser.add_argument("--out_dir", default="eval/ablation/style_swap")
    parser.add_argument("--smpl_model", default="smpl_model/SMPL_NEUTRAL.pkl")
    parser.add_argument("--num_frames", type=int, default=7)
    return parser.parse_args()


def pooled_feat(path, device):
    return torch.from_numpy(np.load(path)).float().to(device).mean(dim=0, keepdim=True)


def decode(model, samples):
    samples = model.normalizer.unnormalize(samples.cpu())
    _, samples = torch.split(samples, (4, samples.shape[2] - 4), dim=2)
    pos, q_6d = torch.split(samples, (3, samples.shape[2] - 3), dim=2)
    q = ax_from_6v(q_6d.reshape(1, -1, 24, 6)).squeeze(0).numpy()  # T x 24 x 3
    return pos.squeeze(0).numpy(), q.reshape(-1, 72)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = EDGE("jukebox", args.checkpoint, use_hierarchical=True)
    model.eval()
    diffusion = model.diffusion
    net = model.model

    local_path = os.path.join(args.feat_dir, args.local_song + ".npy")
    cond = torch.from_numpy(np.load(local_path)).float().unsqueeze(0).to(device)
    local_genre = args.local_song[1:3]

    # pick one swap song per requested genre
    swap_rows = [("original", local_genre, None)]
    for g in args.swap_genres.split(","):
        cands = sorted(glob.glob(os.path.join(args.feat_dir, f"g{g}_*slice0.npy")))
        assert cands, f"no song for genre {g}"
        swap_rows.append((os.path.basename(cands[0])[:-4], g, cands[0]))

    torch.manual_seed(42)
    noise = torch.randn(1, cond.shape[1], 151)

    results = []
    for name, genre, path in swap_rows:
        net.global_pool_override = None if path is None else pooled_feat(path, device)
        with torch.no_grad():
            sample = diffusion.ddim_sample((1, cond.shape[1], 151), cond, noise=noise.clone())
        pos, q = decode(model, sample)
        results.append((name, genre, pos, q))
        with open(out_dir / f"styleswap_{genre}.pkl", "wb") as f:
            pickle.dump({"smpl_trans": pos, "smpl_poses": q, "global_from": name}, f)
        print("generated:", name)
    net.global_pool_override = None

    # quantitative: joint-angle difference vs original
    print("\nMean |pose diff| vs original global:")
    base_q = results[0][3]
    for name, genre, _, q in results[1:]:
        print(f"  global from {GENRE_NAMES[genre]:14s}: {np.abs(q - base_q).mean():.4f} rad")

    # figure
    frames = np.linspace(0, cond.shape[1] - 1, args.num_frames, dtype=int)
    fig, axes = plt.subplots(len(results), args.num_frames,
                             figsize=(args.num_frames * 1.6, len(results) * 1.75), dpi=180)
    for row, (name, genre, pos, q) in enumerate(results):
        vertices, faces, _ = build_meshes(args.smpl_model, q, pos, frames)
        label = ("original global\n(%s)" % GENRE_NAMES[local_genre]) if row == 0 \
            else "global from\n%s" % GENRE_NAMES[genre]
        for col in range(args.num_frames):
            rgb = render_mesh(vertices[col], faces, GENERATED_COLOR, 400, ROW_BGS[row % len(ROW_BGS)])
            ax = axes[row, col]
            ax.imshow(rgb)
            ax.set_axis_off()
            if col == 0:
                ax.text(-0.14, 0.5, label, transform=ax.transAxes, fontsize=7,
                        rotation=90, va="center", ha="center")
    fig.suptitle(f"Style swap: same local music ({GENRE_NAMES[local_genre]}), different global style token",
                 fontsize=11, y=0.995)
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    out_png = "eval/ablation/ccl_visualizations/style_swap_figure.png"
    plt.savefig(out_png, bbox_inches="tight")
    print("saved:", out_png)


if __name__ == "__main__":
    main()
