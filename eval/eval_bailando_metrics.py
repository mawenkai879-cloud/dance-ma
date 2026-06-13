import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from scipy import linalg
import scipy.signal
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vis import SMPLSkeleton

if not hasattr(scipy.signal, "hann"):
    scipy.signal.hann = scipy.signal.windows.hann

SMPL_JOINT_NAMES = [
    "root",
    "lhip", "rhip", "belly",
    "lknee", "rknee", "spine",
    "lankle", "rankle", "chest",
    "ltoes", "rtoes", "neck",
    "linshoulder", "rinshoulder",
    "head", "lshoulder", "rshoulder",
    "lelbow", "relbow",
    "lwrist", "rwrist",
    "lhand", "rhand",
]


def distance_between_points(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def distance_from_plane(a, b, c, p, threshold):
    ba = np.array(b) - np.array(a)
    ca = np.array(c) - np.array(a)
    cross = np.cross(ca, ba)
    norm = np.linalg.norm(cross)
    if norm < 1e-8:
        return False
    pa = np.array(p) - np.array(a)
    return np.dot(cross, pa) / norm > threshold


def distance_from_plane_normal(n1, n2, a, p, threshold):
    normal = np.array(n2) - np.array(n1)
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        return False
    pa = np.array(p) - np.array(a)
    return np.dot(normal, pa) / norm > threshold


def angle_within_range(j1, j2, k1, k2, angle_range):
    j = np.array(j2) - np.array(j1)
    k = np.array(k2) - np.array(k1)
    denom = np.linalg.norm(j) * np.linalg.norm(k)
    if denom < 1e-8:
        return False
    angle = np.arccos(np.clip(np.dot(j, k) / denom, -1.0, 1.0))
    angle = np.degrees(angle)
    return angle_range[0] < angle < angle_range[1]


def velocity_direction_above_threshold(j1, j1_prev, j2, j2_prev, p, p_prev, threshold, time_per_frame=1 / 120.0):
    velocity = np.array(p) - np.array(j1) - (np.array(p_prev) - np.array(j1_prev))
    direction = np.array(j2) - np.array(j1)
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return False
    velocity_along_direction = np.dot(velocity, direction) / norm
    velocity_along_direction = velocity_along_direction / time_per_frame
    return velocity_along_direction > threshold


def velocity_direction_above_threshold_normal(j1, j1_prev, j2, j3, p, p_prev, threshold, time_per_frame=1 / 120.0):
    velocity = np.array(p) - np.array(j1) - (np.array(p_prev) - np.array(j1_prev))
    j31 = np.array(j3) - np.array(j1)
    j21 = np.array(j2) - np.array(j1)
    direction = np.cross(j31, j21)
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return False
    velocity_along_direction = np.dot(velocity, direction) / norm
    velocity_along_direction = velocity_along_direction / time_per_frame
    return velocity_along_direction > threshold


def velocity_above_threshold(p, p_prev, threshold, time_per_frame=1 / 120.0):
    velocity = np.linalg.norm(np.array(p) - np.array(p_prev)) / time_per_frame
    return velocity > threshold


def calc_average_velocity(positions, i, joint_idx, sliding_window, frame_time):
    current_window = 0
    average_velocity = np.zeros(len(positions[0][joint_idx]))
    for j in range(-sliding_window, sliding_window + 1):
        if i + j - 1 < 0 or i + j >= len(positions):
            continue
        average_velocity += positions[i + j][joint_idx] - positions[i + j - 1][joint_idx]
        current_window += 1
    if current_window == 0:
        return 0.0
    return np.linalg.norm(average_velocity / (current_window * frame_time))


def calc_average_acceleration(positions, i, joint_idx, sliding_window, frame_time):
    current_window = 0
    average_acceleration = np.zeros(len(positions[0][joint_idx]))
    for j in range(-sliding_window, sliding_window + 1):
        if i + j - 1 < 0 or i + j + 1 >= len(positions):
            continue
        v2 = (positions[i + j + 1][joint_idx] - positions[i + j][joint_idx]) / frame_time
        v1 = (positions[i + j][joint_idx] - positions[i + j - 1][joint_idx]) / frame_time
        average_acceleration += (v2 - v1) / frame_time
        current_window += 1
    if current_window == 0:
        return 0.0
    return np.linalg.norm(average_acceleration / current_window)


def calc_average_velocity_horizontal(positions, i, joint_idx, sliding_window, frame_time, up_vec="z"):
    current_window = 0
    average_velocity = np.zeros(len(positions[0][joint_idx]))
    for j in range(-sliding_window, sliding_window + 1):
        if i + j - 1 < 0 or i + j >= len(positions):
            continue
        average_velocity += positions[i + j][joint_idx] - positions[i + j - 1][joint_idx]
        current_window += 1
    if current_window == 0:
        return 0.0
    if up_vec == "y":
        average_velocity = np.array([average_velocity[0], average_velocity[2]]) / (current_window * frame_time)
    elif up_vec == "z":
        average_velocity = np.array([average_velocity[0], average_velocity[1]]) / (current_window * frame_time)
    else:
        raise NotImplementedError
    return np.linalg.norm(average_velocity)


def calc_average_velocity_vertical(positions, i, joint_idx, sliding_window, frame_time, up_vec):
    current_window = 0
    average_velocity = np.zeros(len(positions[0][joint_idx]))
    for j in range(-sliding_window, sliding_window + 1):
        if i + j - 1 < 0 or i + j >= len(positions):
            continue
        average_velocity += positions[i + j][joint_idx] - positions[i + j - 1][joint_idx]
        current_window += 1
    if current_window == 0:
        return 0.0
    if up_vec == "y":
        average_velocity = np.array([average_velocity[1]]) / (current_window * frame_time)
    elif up_vec == "z":
        average_velocity = np.array([average_velocity[2]]) / (current_window * frame_time)
    else:
        raise NotImplementedError
    return np.linalg.norm(average_velocity)


class KineticFeatures:
    def __init__(self, positions, frame_time=1.0 / 30, up_vec="z", sliding_window=2):
        self.positions = positions
        self.frame_time = frame_time
        self.up_vec = up_vec
        self.sliding_window = sliding_window

    def average_kinetic_energy_horizontal(self, joint):
        val = 0.0
        for i in range(1, len(self.positions)):
            average_velocity = calc_average_velocity_horizontal(self.positions, i, joint, self.sliding_window, self.frame_time, self.up_vec)
            val += average_velocity ** 2
        return val / max(len(self.positions) - 1.0, 1.0)

    def average_kinetic_energy_vertical(self, joint):
        val = 0.0
        for i in range(1, len(self.positions)):
            average_velocity = calc_average_velocity_vertical(self.positions, i, joint, self.sliding_window, self.frame_time, self.up_vec)
            val += average_velocity ** 2
        return val / max(len(self.positions) - 1.0, 1.0)

    def average_energy_expenditure(self, joint):
        val = 0.0
        for i in range(1, len(self.positions)):
            val += calc_average_acceleration(self.positions, i, joint, self.sliding_window, self.frame_time)
        return val / max(len(self.positions) - 1.0, 1.0)


def extract_kinetic_features(positions):
    positions = np.asarray(positions)
    features = KineticFeatures(positions)
    kinetic_feature_vector = []
    for i in range(positions.shape[1]):
        feature_vector = np.hstack([
            features.average_kinetic_energy_horizontal(i),
            features.average_kinetic_energy_vertical(i),
            features.average_energy_expenditure(i),
        ])
        kinetic_feature_vector.extend(feature_vector)
    return np.array(kinetic_feature_vector, dtype=np.float32)


class ManualFeatures:
    def __init__(self, positions, joint_names=SMPL_JOINT_NAMES):
        self.positions = positions
        self.joint_names = joint_names
        self.frame_num = 1
        self.hl = distance_between_points([1.99113488e-01, 2.36807942e-01, -1.80702247e-02], [4.54445392e-01, 2.21158922e-01, -4.10167128e-02])
        self.sw = distance_between_points([1.99113488e-01, 2.36807942e-01, -1.80702247e-02], [-1.91692337e-01, 2.36928746e-01, -1.23055102e-02])
        self.hw = distance_between_points([5.64076714e-02, -3.23069185e-01, 1.09197125e-02], [-6.24834076e-02, -3.31302464e-01, 1.50412619e-02])

    def next_frame(self):
        self.frame_num += 1

    def transform_and_fetch_position(self, j):
        if j == "y_unit":
            return [0, 1, 0]
        if j == "minus_y_unit":
            return [0, -1, 0]
        if j == "zero":
            return [0, 0, 0]
        if j == "y_min":
            return [0, min([y for (_, y, _) in self.positions[self.frame_num]]), 0]
        return self.positions[self.frame_num][self.joint_names.index(j)]

    def transform_and_fetch_prev_position(self, j):
        return self.positions[self.frame_num - 1][self.joint_names.index(j)]

    def f_move(self, j1, j2, j3, j4, range_value):
        j1_prev, j2_prev, j3_prev, j4_prev = [self.transform_and_fetch_prev_position(j) for j in [j1, j2, j3, j4]]
        j1, j2, j3, j4 = [self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]]
        return velocity_direction_above_threshold(j1, j1_prev, j2, j2_prev, j3, j3_prev, range_value)

    def f_nmove(self, j1, j2, j3, j4, range_value):
        j1_prev, j2_prev, j3_prev, j4_prev = [self.transform_and_fetch_prev_position(j) for j in [j1, j2, j3, j4]]
        j1, j2, j3, j4 = [self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]]
        return velocity_direction_above_threshold_normal(j1, j1_prev, j2, j3, j4, j4_prev, range_value)

    def f_plane(self, j1, j2, j3, j4, threshold):
        j1, j2, j3, j4 = [self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]]
        return distance_from_plane(j1, j2, j3, j4, threshold)

    def f_nplane(self, j1, j2, j3, j4, threshold):
        j1, j2, j3, j4 = [self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]]
        return distance_from_plane_normal(j1, j2, j3, j4, threshold)

    def f_angle(self, j1, j2, j3, j4, range_value):
        j1, j2, j3, j4 = [self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]]
        return angle_within_range(j1, j2, j3, j4, range_value)

    def f_fast(self, j1, threshold):
        j1_prev = self.transform_and_fetch_prev_position(j1)
        j1 = self.transform_and_fetch_position(j1)
        return velocity_above_threshold(j1, j1_prev, threshold)


def extract_manual_features(positions):
    positions = np.asarray(positions)
    features = []
    f = ManualFeatures(positions)
    for _ in range(1, positions.shape[0]):
        pose_features = []
        pose_features.append(f.f_nmove("neck", "rhip", "lhip", "rwrist", 1.8 * f.hl))
        pose_features.append(f.f_nmove("neck", "lhip", "rhip", "lwrist", 1.8 * f.hl))
        pose_features.append(f.f_nplane("chest", "neck", "neck", "rwrist", 0.2 * f.hl))
        pose_features.append(f.f_nplane("chest", "neck", "neck", "lwrist", 0.2 * f.hl))
        pose_features.append(f.f_move("belly", "chest", "chest", "rwrist", 1.8 * f.hl))
        pose_features.append(f.f_move("belly", "chest", "chest", "lwrist", 1.8 * f.hl))
        pose_features.append(f.f_angle("relbow", "rshoulder", "relbow", "rwrist", [0, 110]))
        pose_features.append(f.f_angle("lelbow", "lshoulder", "lelbow", "lwrist", [0, 110]))
        pose_features.append(f.f_nplane("lshoulder", "rshoulder", "lwrist", "rwrist", 2.5 * f.sw))
        pose_features.append(f.f_move("lwrist", "rwrist", "rwrist", "lwrist", 1.4 * f.hl))
        pose_features.append(f.f_move("rwrist", "root", "lwrist", "root", 1.4 * f.hl))
        pose_features.append(f.f_move("lwrist", "root", "rwrist", "root", 1.4 * f.hl))
        pose_features.append(f.f_fast("rwrist", 2.5 * f.hl))
        pose_features.append(f.f_fast("lwrist", 2.5 * f.hl))
        pose_features.append(f.f_plane("root", "lhip", "ltoes", "rankle", 0.38 * f.hl))
        pose_features.append(f.f_plane("root", "rhip", "rtoes", "lankle", 0.38 * f.hl))
        pose_features.append(f.f_nplane("zero", "y_unit", "y_min", "rankle", 1.2 * f.hl))
        pose_features.append(f.f_nplane("zero", "y_unit", "y_min", "lankle", 1.2 * f.hl))
        pose_features.append(f.f_nplane("lhip", "rhip", "lankle", "rankle", 2.1 * f.hw))
        pose_features.append(f.f_angle("rknee", "rhip", "rknee", "rankle", [0, 110]))
        pose_features.append(f.f_angle("lknee", "lhip", "lknee", "lankle", [0, 110]))
        pose_features.append(f.f_fast("rankle", 2.5 * f.hl))
        pose_features.append(f.f_fast("lankle", 2.5 * f.hl))
        pose_features.append(f.f_angle("neck", "root", "rshoulder", "relbow", [25, 180]))
        pose_features.append(f.f_angle("neck", "root", "lshoulder", "lelbow", [25, 180]))
        pose_features.append(f.f_angle("neck", "root", "rhip", "rknee", [50, 180]))
        pose_features.append(f.f_angle("neck", "root", "lhip", "lknee", [50, 180]))
        pose_features.append(f.f_plane("rankle", "neck", "lankle", "root", 0.5 * f.hl))
        pose_features.append(f.f_angle("neck", "root", "zero", "y_unit", [70, 110]))
        pose_features.append(f.f_nplane("zero", "minus_y_unit", "y_min", "rwrist", -1.2 * f.hl))
        pose_features.append(f.f_nplane("zero", "minus_y_unit", "y_min", "lwrist", -1.2 * f.hl))
        pose_features.append(f.f_fast("root", 2.3 * f.hl))
        features.append(pose_features)
        f.next_frame()
    return np.array(features, dtype=np.float32).mean(axis=0)


def calc_fid(feats_gen, feats_gt):
    if len(feats_gen) < 2 or len(feats_gt) < 2:
        return float("nan")
    mu_gen = np.mean(feats_gen, axis=0)
    sigma_gen = np.cov(feats_gen, rowvar=False)
    mu_gt = np.mean(feats_gt, axis=0)
    sigma_gt = np.cov(feats_gt, rowvar=False)
    diff = mu_gen - mu_gt
    eps = 1e-5
    covmean, _ = linalg.sqrtm(sigma_gen.dot(sigma_gt), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_gen.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_gen + offset).dot(sigma_gt + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma_gen) + np.trace(sigma_gt) - 2 * np.trace(covmean))


def calculate_avg_distance(features):
    features = np.asarray(features)
    n = features.shape[0]
    if n < 2:
        return float("nan")
    dist = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dist += np.linalg.norm(features[i] - features[j])
    return float(dist / ((n * n - n) / 2))


def normalize_by_gt(gt_features, pred_features):
    mean = gt_features.mean(axis=0)
    std = gt_features.std(axis=0)
    return (gt_features - mean) / (std + 1e-10), (pred_features - mean) / (std + 1e-10)


def load_edge_motion(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if "full_pose" in data:
        return np.asarray(data["full_pose"], dtype=np.float32)
    if "pos" in data and "q" in data:
        smpl = SMPLSkeleton()
        pos = torch.as_tensor(data["pos"], dtype=torch.float32).unsqueeze(0)
        q = torch.as_tensor(data["q"], dtype=torch.float32).reshape(1, len(data["q"]), 24, 3)
        return smpl.forward(q, pos).squeeze(0).detach().cpu().numpy().astype(np.float32)
    raise KeyError(f"{path} does not contain full_pose")


def relative_root(positions):
    positions = np.asarray(positions, dtype=np.float32)
    return positions - positions[:1, :1, :]


def extract_features_from_dir(motion_dir, cache_dir=None, max_files=None):
    paths = sorted(glob.glob(os.path.join(motion_dir, "*.pkl")))
    if max_files is not None:
        paths = paths[:max_files]
    kinetic = []
    manual = []
    names = []
    if cache_dir:
        Path(cache_dir, "kinetic_features").mkdir(parents=True, exist_ok=True)
        Path(cache_dir, "manual_features").mkdir(parents=True, exist_ok=True)
    for path in tqdm(paths, desc=f"features:{motion_dir}"):
        stem = Path(path).stem
        k_cache = Path(cache_dir, "kinetic_features", stem + ".npy") if cache_dir else None
        m_cache = Path(cache_dir, "manual_features", stem + ".npy") if cache_dir else None
        if cache_dir and k_cache.exists() and m_cache.exists():
            k_feat = np.load(k_cache)
            m_feat = np.load(m_cache)
        else:
            positions = relative_root(load_edge_motion(path))
            k_feat = extract_kinetic_features(positions)
            m_feat = extract_manual_features(positions)
            if cache_dir:
                np.save(k_cache, k_feat)
                np.save(m_cache, m_feat)
        kinetic.append(k_feat)
        manual.append(m_feat)
        names.append(path)
    if not kinetic:
        raise ValueError(f"No pkl files found in {motion_dir}")
    return np.stack(kinetic), np.stack(manual), names


def extract_gt_features(gt_motion_dir, cache_dir=None, max_files=None):
    paths = sorted(glob.glob(os.path.join(gt_motion_dir, "*.pkl")))
    if max_files is not None:
        paths = paths[:max_files]
    kinetic = []
    manual = []
    if cache_dir:
        Path(cache_dir, "gt_kinetic_features").mkdir(parents=True, exist_ok=True)
        Path(cache_dir, "gt_manual_features").mkdir(parents=True, exist_ok=True)
    for path in tqdm(paths, desc=f"gt_features:{gt_motion_dir}"):
        stem = Path(path).stem
        k_cache = Path(cache_dir, "gt_kinetic_features", stem + ".npy") if cache_dir else None
        m_cache = Path(cache_dir, "gt_manual_features", stem + ".npy") if cache_dir else None
        if cache_dir and k_cache.exists() and m_cache.exists():
            k_feat = np.load(k_cache)
            m_feat = np.load(m_cache)
        else:
            positions = load_edge_motion(path)
            positions = relative_root(positions)
            k_feat = extract_kinetic_features(positions)
            m_feat = extract_manual_features(positions)
            if cache_dir:
                np.save(k_cache, k_feat)
                np.save(m_cache, m_feat)
        kinetic.append(k_feat)
        manual.append(m_feat)
    if not kinetic:
        raise ValueError(f"No full_pose pkl files found in {gt_motion_dir}; generate a GT full_pose cache or use a generated-motion directory as reference")
    return np.stack(kinetic), np.stack(manual)


def motion_beats(positions):
    positions = np.asarray(positions).reshape(-1, 24, 3)
    kinetic_vel = np.mean(np.sqrt(np.sum((positions[1:] - positions[:-1]) ** 2, axis=2)), axis=1)
    kinetic_vel = gaussian_filter(kinetic_vel, 5)
    return argrelextrema(kinetic_vel, np.less)[0]


def music_beats_from_wav(wav_path, length, fps=30):
    y, sr = librosa.load(wav_path, sr=None)
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    beat_axis = np.rint(beat_times * fps).astype(int)
    return beat_axis[(beat_axis >= 0) & (beat_axis < length)]


def beat_align_score(motion_beats_value, music_beats_value):
    if len(music_beats_value) == 0 or len(motion_beats_value) == 0:
        return float("nan")
    score = 0.0
    for beat in music_beats_value:
        score += np.exp(-np.min((motion_beats_value - beat) ** 2) / 2 / 9)
    return float(score / len(music_beats_value))


def infer_wav_path(motion_pkl, wav_dir):
    stem = Path(motion_pkl).stem
    for prefix in ("test_", "train_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    if "_slice" in stem:
        stem = stem.split("_slice")[0]
    candidates = [
        Path(wav_dir, stem + ".wav"),
        Path(wav_dir, Path(motion_pkl).stem + ".wav"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    matches = sorted(Path(wav_dir).glob(stem + "*.wav"))
    if matches:
        return str(matches[0])
    return None


def calc_beat_align(motion_dir, wav_dir, max_files=None):
    paths = sorted(glob.glob(os.path.join(motion_dir, "*.pkl")))
    if max_files is not None:
        paths = paths[:max_files]
    scores = []
    missing = []
    for path in tqdm(paths, desc="beat_align"):
        wav_path = infer_wav_path(path, wav_dir)
        if wav_path is None:
            missing.append(path)
            continue
        positions = load_edge_motion(path)
        m_beats = motion_beats(positions)
        a_beats = music_beats_from_wav(wav_path, len(positions))
        score = beat_align_score(m_beats, a_beats)
        if not np.isnan(score):
            scores.append(score)
    return float(np.mean(scores)) if scores else float("nan"), len(scores), len(missing)


def calc_metrics(pred_motion_dir, gt_motion_dir, wav_dir, cache_dir=None, max_files=None):
    pred_k, pred_g, _ = extract_features_from_dir(pred_motion_dir, cache_dir, max_files)
    gt_k, gt_g, _ = extract_features_from_dir(gt_motion_dir, cache_dir and os.path.join(cache_dir, "gt"), max_files)
    gt_k_norm, pred_k_norm = normalize_by_gt(gt_k, pred_k)
    gt_g_norm, pred_g_norm = normalize_by_gt(gt_g, pred_g)
    ba, ba_count, ba_missing = calc_beat_align(pred_motion_dir, wav_dir, max_files)
    return {
        "fid_k": calc_fid(pred_k_norm, gt_k_norm),
        "fid_g_manual": calc_fid(pred_g_norm, gt_g_norm),
        "div_k": calculate_avg_distance(pred_k_norm),
        "div_g_manual": calculate_avg_distance(pred_g_norm),
        "div_k_gt": calculate_avg_distance(gt_k_norm),
        "div_g_manual_gt": calculate_avg_distance(gt_g_norm),
        "beat_align": ba,
        "beat_align_count": ba_count,
        "beat_align_missing_wav": ba_missing,
        "pred_count": int(pred_k.shape[0]),
        "gt_count": int(gt_k.shape[0]),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_motion_dir", default="eval/motions")
    parser.add_argument("--gt_motion_dir", default="data/test/motions_sliced")
    parser.add_argument("--wav_dir", default="data/test/wavs")
    parser.add_argument("--cache_dir", default="eval/metric_cache")
    parser.add_argument("--max_files", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = calc_metrics(args.pred_motion_dir, args.gt_motion_dir, args.wav_dir, args.cache_dir, args.max_files)
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
