import argparse
import os
import pickle
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import matplotlib.pyplot as plt
import numpy as np
import pyrender
import torch
import trimesh

import smplx


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_pkl", required=True)
    parser.add_argument("--smpl_model", default="smpl_model/SMPL_NEUTRAL.pkl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--num_frames", type=int, default=7)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--constraint_rows", action="store_true", help="render 4 paper-style constraint rows")
    parser.add_argument("--img_size", type=int, default=400)
    return parser.parse_args()


GENERATED_COLOR = (0.23, 0.45, 0.52, 1.0)
CONSTRAINT_COLOR = (0.42, 0.85, 0.65, 1.0)
UPPER_BODY = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
LOWER_BODY_ROOT = [0, 1, 2, 4, 5, 7, 8, 10, 11]
GENERATED_BG = (0.88, 0.88, 0.88)
CONSTRAINT_BG = (0.96, 0.87, 0.73)


def load_motion(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    poses = np.asarray(data["smpl_poses"], dtype=np.float32)  # (S, 72)
    trans = np.asarray(data["smpl_trans"], dtype=np.float32)  # (S, 3)
    return poses, trans


def build_meshes(smpl_model_path, poses, trans, frames):
    model = smplx.SMPL(model_path=str(Path(smpl_model_path).parent), gender="neutral")
    body_pose = torch.from_numpy(poses[frames, 3:]).float()
    global_orient = torch.from_numpy(poses[frames, :3]).float()
    transl = torch.from_numpy(trans[frames]).float()
    transl = transl - transl.mean(dim=0, keepdim=True)
    with torch.no_grad():
        output = model(
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl * 0,
        )
    vertices = output.vertices.cpu().numpy()
    skin_weights = model.lbs_weights.cpu().numpy()
    return vertices, model.faces, skin_weights


def render_mesh(vertices, faces, color, img_size, bg_color, vertex_colors=None):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    # rotate from SMPL (y-up after AIST convention z-up) for nicer view
    rot = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
    mesh.apply_transform(rot)
    mesh.vertices -= mesh.vertices.mean(axis=0)

    if vertex_colors is not None:
        mesh.visual.vertex_colors = (vertex_colors * 255).astype(np.uint8)
        render_mesh_obj = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    else:
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=color, metallicFactor=0.1, roughnessFactor=0.7
        )
        render_mesh_obj = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)

    scene = pyrender.Scene(bg_color=list(bg_color) + [1.0], ambient_light=(0.35, 0.35, 0.35))
    scene.add(render_mesh_obj)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.5)
    cam_pose = np.eye(4)
    cam_pose[:3, 3] = [0, 0.1, 2.6]
    scene.add(camera, pose=cam_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.2)
    light_pose = np.eye(4)
    light_pose[:3, 3] = [0.5, 1.0, 2.0]
    scene.add(light, pose=light_pose)

    renderer = pyrender.OffscreenRenderer(img_size, img_size)
    rgb, _ = renderer.render(scene)
    renderer.delete()
    return rgb


def main():
    args = parse_args()
    if not Path(args.smpl_model).exists():
        raise FileNotFoundError(
            f"SMPL model not found at {args.smpl_model}.\n"
            "Download SMPL_NEUTRAL.pkl from https://smpl.is.tue.mpg.de (registration required)\n"
            "and place it at smpl_model/SMPL_NEUTRAL.pkl"
        )
    poses, trans = load_motion(args.motion_pkl)
    frames = np.linspace(0, len(poses) - 1, args.num_frames, dtype=int)
    vertices, faces, skin_weights = build_meshes(args.smpl_model, poses, trans, frames)

    def part_colors(joints):
        w = skin_weights[:, joints].sum(axis=1)
        mask = (w > 0.5)[:, None]
        base = np.tile(np.array(GENERATED_COLOR), (skin_weights.shape[0], 1))
        green = np.tile(np.array(CONSTRAINT_COLOR), (skin_weights.shape[0], 1))
        return np.where(mask, green, base)

    if args.constraint_rows:
        row_specs = [
            ("completion from seed motion", {0: "full", 1: "full"}),
            ("specified middle keyframe", {args.num_frames // 2: "full"}),
            ("upper-body constraints", {i: "upper" for i in range(args.num_frames)}),
            ("lower-body + root constraints", {i: "lower" for i in range(args.num_frames)}),
        ]
        n_rows = 4
    else:
        row_specs = [("generated dance", {})] * args.rows
        n_rows = args.rows

    fig, axes = plt.subplots(n_rows, args.num_frames, figsize=(args.num_frames * 1.6, n_rows * 1.75), dpi=180)
    axes = np.atleast_2d(axes)
    for row, (row_title, constrained_cols) in enumerate(row_specs):
        for col in range(args.num_frames):
            mode = constrained_cols.get(col)
            vertex_colors = None
            color = GENERATED_COLOR
            bg = GENERATED_BG
            if mode == "full":
                color = CONSTRAINT_COLOR
                bg = CONSTRAINT_BG
            elif mode == "upper":
                vertex_colors = part_colors(UPPER_BODY)
                bg = CONSTRAINT_BG
            elif mode == "lower":
                vertex_colors = part_colors(LOWER_BODY_ROOT)
                bg = CONSTRAINT_BG
            rgb = render_mesh(vertices[col], faces, color, args.img_size, bg, vertex_colors=vertex_colors)
            ax = axes[row, col]
            ax.imshow(rgb)
            ax.set_axis_off()
            if col == 0:
                ax.text(-0.12, 0.5, row_title, transform=ax.transAxes, fontsize=7, rotation=90, va="center")
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight")
    print(args.out)


if __name__ == "__main__":
    main()
