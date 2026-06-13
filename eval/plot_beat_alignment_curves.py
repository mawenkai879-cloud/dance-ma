import argparse
import pickle
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_bailando_metrics import load_edge_motion


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_pkl", required=True)
    parser.add_argument("--edge_pkl", required=True)
    parser.add_argument("--no_ccl_pkl", required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="Beat Alignment")
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def kinetic_velocity(path):
    positions = load_edge_motion(path).reshape(-1, 24, 3)
    velocity = np.mean(np.sqrt(np.sum((positions[1:] - positions[:-1]) ** 2, axis=2)), axis=1)
    velocity = gaussian_filter(velocity, 5)
    return velocity


def normalize_curve(curve):
    curve = np.asarray(curve, dtype=np.float32)
    curve = curve - curve.mean()
    scale = np.max(np.abs(curve))
    if scale > 0:
        curve = curve / scale
    return curve


def music_beats(wav_path, length, fps):
    y, sr = librosa.load(wav_path, sr=None)
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    beat_axis = np.rint(beat_times * fps).astype(int)
    return beat_axis[(beat_axis >= 0) & (beat_axis < length)]


def main():
    args = parse_args()
    curves = {
        "GT": kinetic_velocity(args.gt_pkl),
        "EDGE": kinetic_velocity(args.edge_pkl),
        "w/o CCL": kinetic_velocity(args.no_ccl_pkl),
    }
    n = min(len(v) for v in curves.values())
    curves = {k: normalize_curve(v[:n]) for k, v in curves.items()}
    beats = music_beats(args.wav, n, args.fps)
    t = np.arange(n) / args.fps

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 3.2), dpi=180)
    colors = {"GT": "#2ca25f", "EDGE": "#e78ac3", "w/o CCL": "#e41a1c"}
    for name, curve in curves.items():
        plt.plot(t, curve, linewidth=1.8, color=colors[name], label=name, alpha=0.95)
    y_min = min(float(v.min()) for v in curves.values())
    y_max = max(float(v.max()) for v in curves.values())
    beat_bottom = y_min - 0.18 * (y_max - y_min)
    beat_top = y_min - 0.03 * (y_max - y_min)
    for beat in beats:
        plt.vlines(beat / args.fps, beat_bottom, beat_top, color="black", linewidth=1.0, alpha=0.85)
    plt.text(t[min(n - 1, max(0, int(n * 0.46)))], beat_bottom, "audio beats", fontsize=8, ha="center", va="top")
    plt.xlabel("time")
    plt.ylabel("normalized kinetic velocity")
    plt.title(args.title)
    plt.legend(loc="upper left", ncol=3, frameon=False, fontsize=8)
    plt.ylim(beat_bottom - 0.08 * (y_max - y_min), y_max + 0.08 * (y_max - y_min))
    plt.grid(axis="y", alpha=0.15)
    plt.tight_layout()
    plt.savefig(args.out)
    print(args.out)
    print("audio_beats", len(beats))


if __name__ == "__main__":
    main()
