import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_bailando_metrics import infer_wav_path, load_edge_motion, motion_beats, music_beats_from_wav


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_pkl", required=True)
    parser.add_argument("--wav_dir", default="data/test/wavs_sliced")
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="Beat Alignment between Music and Generated Dance")
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def kinetic_velocity(positions):
    positions = np.asarray(positions).reshape(-1, 24, 3)
    velocity = np.mean(np.sqrt(np.sum((positions[1:] - positions[:-1]) ** 2, axis=2)), axis=1)
    return gaussian_filter(velocity, 5)


def main():
    args = parse_args()
    positions = load_edge_motion(args.motion_pkl)
    wav_path = infer_wav_path(args.motion_pkl, args.wav_dir)
    if wav_path is None:
        raise FileNotFoundError(f"No wav found for {args.motion_pkl} in {args.wav_dir}")

    velocity = kinetic_velocity(positions)
    k_beats = motion_beats(positions)
    m_beats = music_beats_from_wav(wav_path, len(positions), fps=args.fps)
    k_beats = k_beats[k_beats < len(velocity)]
    m_beats = m_beats[m_beats < len(velocity)]
    t = np.arange(len(velocity)) / args.fps

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 3.8), dpi=180)
    plt.plot(t, velocity, color="#3776d6", linewidth=2.0, label="kinetic velocity")
    for i, beat in enumerate(m_beats):
        plt.axvline(beat / args.fps, color="#ff9d28", linestyle="--", linewidth=1.1, alpha=0.9, label="music beats" if i == 0 else None)
    for i, beat in enumerate(k_beats):
        plt.axvline(beat / args.fps, color="#4c9f38", linestyle="--", linewidth=1.1, alpha=0.9, label="kinematic beats" if i == 0 else None)
    plt.xlabel("time (s)")
    plt.ylabel("kinetic velocity")
    plt.title(args.title)
    plt.legend(loc="upper right", frameon=True, ncol=3, fontsize=8)
    plt.grid(axis="y", alpha=0.22)
    plt.tight_layout()
    plt.savefig(args.out)
    print(args.out)
    print("wav", wav_path)
    print("music_beats", len(m_beats))
    print("kinematic_beats", len(k_beats))


if __name__ == "__main__":
    main()
