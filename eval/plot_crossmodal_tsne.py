import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_bailando_metrics import extract_kinetic_features, extract_manual_features, load_edge_motion


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_dir", required=True)
    parser.add_argument("--audio_feat_dir", default="data/test/jukebox_feats")
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="Cross-modal t-SNE")
    parser.add_argument("--max_files", type=int, default=50)
    return parser.parse_args()


def motion_stem_to_audio_base(path):
    stem = Path(path).stem
    for prefix in ("test_", "train_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    return stem


def load_music_feature(audio_feat_dir, base):
    paths = sorted(Path(audio_feat_dir).glob(base + "_slice*.npy"))
    if not paths:
        paths = sorted(Path(audio_feat_dir).glob(base + "*.npy"))
    if not paths:
        raise FileNotFoundError(base)
    feats = []
    for path in paths:
        arr = np.load(path)
        feats.append(arr.reshape(-1, arr.shape[-1]).mean(axis=0))
    return np.stack(feats, axis=0).mean(axis=0)


def load_motion_feature(path):
    positions = load_edge_motion(path)
    kinetic = extract_kinetic_features(positions)
    manual = extract_manual_features(positions)
    return np.concatenate([kinetic, manual], axis=0)


def main():
    args = parse_args()
    motion_paths = sorted(glob.glob(os.path.join(args.motion_dir, "*.pkl")))[: args.max_files]
    music_feats = []
    motion_feats = []
    names = []
    for path in motion_paths:
        base = motion_stem_to_audio_base(path)
        try:
            music_feats.append(load_music_feature(args.audio_feat_dir, base))
            motion_feats.append(load_motion_feature(path))
            names.append(base)
        except FileNotFoundError:
            continue
    if len(names) < 3:
        raise ValueError(f"Need at least 3 matched music-motion pairs, got {len(names)}")

    music = StandardScaler().fit_transform(np.asarray(music_feats))
    motion = StandardScaler().fit_transform(np.asarray(motion_feats))
    n_components = max(1, min(len(names) - 1, music.shape[1], motion.shape[1], 10))
    music_common, motion_common = CCA(n_components=n_components, max_iter=10000, tol=1e-3).fit_transform(music, motion)

    combined = np.concatenate([music_common, motion_common], axis=0)
    perplexity = max(2, min(10, (len(combined) - 1) // 3))
    emb = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=0).fit_transform(combined)
    music_2d = emb[: len(names)]
    motion_2d = emb[len(names) :]
    pair_dist = np.linalg.norm(music_2d - motion_2d, axis=1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6), dpi=180)
    for i, name in enumerate(names):
        plt.plot([music_2d[i, 0], motion_2d[i, 0]], [music_2d[i, 1], motion_2d[i, 1]], color="0.68", linewidth=1.0, alpha=0.75)
    plt.scatter(music_2d[:, 0], music_2d[:, 1], s=55, marker="o", c="#1f77b4", label="Music features", edgecolors="white", linewidths=0.5)
    plt.scatter(motion_2d[:, 0], motion_2d[:, 1], s=65, marker="^", c="#d62728", label="Generated motion features", edgecolors="white", linewidths=0.5)
    for i, name in enumerate(names):
        short = name.split("_cAll_")[0].replace("g", "")
        plt.text((music_2d[i, 0] + motion_2d[i, 0]) / 2, (music_2d[i, 1] + motion_2d[i, 1]) / 2, str(i + 1), fontsize=7, color="0.25")
    plt.title(f"{args.title}\nmean paired distance={pair_dist.mean():.3f}, n={len(names)}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(args.out)
    print(args.out)
    print("matched_pairs", len(names))
    print("mean_pair_distance", float(pair_dist.mean()))


if __name__ == "__main__":
    main()
