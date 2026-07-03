#!/usr/bin/env python3
"""Generate MotionBricks G1 motions and export them as GEAR-SONIC motion_lib PKLs.

This is the first, low-risk integration path:

    MotionBricks pretrained model -> G1 MuJoCo qpos -> SONIC robot motion_lib

The exported robot motion directory can be passed to GEAR-SONIC as
``manager_env.commands.motion.motion_lib_cfg.motion_file``. Use
``smpl_motion_file=dummy`` for the first tests so SONIC evaluates the G1 encoder
path without requiring paired SMPL data.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

# Headless SSH sessions usually do not have DISPLAY. Set this before importing
# MotionBricks controllers, because they import pynput at module import time.
os.environ.setdefault("PYNPUT_BACKEND", "dummy")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/motion_sonic_matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
MOTIONBRICKS_ROOT = REPO_ROOT / "motionbricks"

for path in (REPO_ROOT, MOTIONBRICKS_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


@contextlib.contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _parse_speed_scale(value: str) -> list[float]:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--speed_scale must have exactly two comma-separated floats")
    return parts


def _summarize_qpos(qpos: np.ndarray) -> dict:
    dof = qpos[:, 7:]
    root = qpos[:, :3]
    return {
        "num_frames": int(qpos.shape[0]),
        "root_xyz_first": root[0].round(6).tolist(),
        "root_xyz_last": root[-1].round(6).tolist(),
        "root_xyz_min": root.min(axis=0).round(6).tolist(),
        "root_xyz_max": root.max(axis=0).round(6).tolist(),
        "dof_mean": float(dof.mean()),
        "dof_min": float(dof.min()),
        "dof_max": float(dof.max()),
    }


def qpos_to_sonic_motion_entry(qpos, fps: int) -> dict:
    """Convert MotionBricks/MuJoCo G1 qpos to one SONIC motion_lib entry."""
    import numpy as np

    from gear_sonic.data_process.convert_soma_csv_to_motion_lib import convert_sequence

    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"Expected qpos shape (T, 36), got {qpos.shape}")

    root_quat_wxyz = qpos[:, 3:7].copy()
    quat_norm = np.linalg.norm(root_quat_wxyz, axis=1, keepdims=True)
    root_quat_wxyz = root_quat_wxyz / np.maximum(quat_norm, 1e-8)

    body_pos_w = np.zeros((qpos.shape[0], 14, 3), dtype=np.float32)
    body_pos_w[:, 0, :] = qpos[:, :3]

    body_quat_w = np.zeros((qpos.shape[0], 14, 4), dtype=np.float32)
    body_quat_w[:, :, 0] = 1.0
    body_quat_w[:, 0, :] = root_quat_wxyz

    seq_data = {
        "joint_pos": qpos[:, 7:36],
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "joint_order": "mj",
    }
    return convert_sequence(seq_data, fps=fps)


def _make_motionbricks_args(args: argparse.Namespace) -> argparse.Namespace:
    mb_args = argparse.Namespace()

    humanoid_scene_xml = str(MOTIONBRICKS_ROOT / "assets" / "skeletons" / "g1" / "scene_29dof.xml")
    mb_args.humanoid_xml = humanoid_scene_xml
    mb_args.humanoid_scene_xml = humanoid_scene_xml
    mb_args.result_dir = str(MOTIONBRICKS_ROOT / "out")
    mb_args.data_root = str(MOTIONBRICKS_ROOT / "datasets")
    mb_args.explicit_dataset_folder = args.explicit_dataset_folder
    mb_args.reprocess_clips = args.reprocess_clips

    mb_args.controller = args.controller
    mb_args.lookat_movement_direction = args.lookat_movement_direction
    mb_args.pre_filter_qpos = args.pre_filter_qpos
    mb_args.source_root_realignment = args.source_root_realignment
    mb_args.target_root_realignment = args.target_root_realignment
    mb_args.force_canonicalization = args.force_canonicalization
    mb_args.skip_ending_target_cond = args.skip_ending_target_cond
    mb_args.random_speed_scale = args.random_speed_scale
    mb_args.speed_scale = args.speed_scale
    mb_args.generate_dt = args.generate_dt

    mb_args.use_qpos = args.use_qpos
    mb_args.planner = args.planner
    mb_args.allowed_mode = args.allowed_mode
    mb_args.clips = args.clips

    mb_args.has_viewer = 0
    mb_args.return_model_configs = True
    mb_args.return_dataloader = True
    mb_args.recording_dir = None
    mb_args.EXP = args.planner
    return mb_args


def generate_qpos_clip(demo_agent, args: argparse.Namespace, seed: int) -> np.ndarray:
    import mujoco
    import numpy as np
    import torch as t

    np.random.seed(seed)
    t.manual_seed(seed)
    demo_agent.full_agent.reset()

    frames = []
    for step in range(1, args.max_steps + 1):
        force_idle = step + args.force_idle_tail > args.max_steps

        qpos = demo_agent.full_agent.get_next_frame()
        frames.append(np.asarray(qpos, dtype=np.float32).copy())

        context_motion_features = demo_agent.full_agent.get_context_motion_features()
        context_mujoco_qpos = demo_agent.full_agent.get_context_mujoco_qpos()
        demo_agent.mj_data.qpos[:] = qpos

        control_signals = demo_agent.controller.generate_control_signals(
            None,
            demo_agent.mj_model,
            demo_agent.mj_data,
            visualize=False,
            control_info={"force_idle": force_idle, "allowed_mode": args.allowed_mode},
        )
        if args.use_qpos:
            control_signals["context_mujoco_qpos"] = context_mujoco_qpos
        else:
            control_signals["context_motion_features"] = context_motion_features

        with t.no_grad():
            demo_agent.full_agent.generate_new_frames(
                control_signals,
                demo_agent.controller.get_controller_dt() * args.generate_dt,
            )

        mujoco.mj_forward(demo_agent.mj_model, demo_agent.mj_data)

    return np.stack(frames, axis=0)


def main(args: argparse.Namespace) -> None:
    import joblib
    import numpy as np
    import torch as t

    from motionbricks.motion_backbone.demo.utils import navigation_demo

    if not t.cuda.is_available():
        raise RuntimeError("MotionBricks inference expects CUDA; torch.cuda.is_available() is False.")

    output_dir = Path(args.output_dir).resolve()
    robot_dir = output_dir / "robot_filtered" / args.session
    qpos_dir = output_dir / "qpos"
    robot_dir.mkdir(parents=True, exist_ok=True)
    qpos_dir.mkdir(parents=True, exist_ok=True)

    mb_args = _make_motionbricks_args(args)
    started = time.time()
    with _pushd(MOTIONBRICKS_ROOT):
        demo_agent = navigation_demo(mb_args)

    manifest = {
        "created_unix": started,
        "fps": args.fps,
        "max_steps": args.max_steps,
        "num_motions": args.num_motions,
        "controller": args.controller,
        "allowed_mode": args.allowed_mode,
        "use_qpos": args.use_qpos,
        "motions": [],
    }

    for motion_idx in range(args.num_motions):
        seed = args.random_seed + motion_idx
        motion_name = f"{args.motion_prefix}_seed_{seed:08d}"
        with _pushd(MOTIONBRICKS_ROOT):
            qpos = generate_qpos_clip(demo_agent, args, seed=seed)
        entry = qpos_to_sonic_motion_entry(qpos, fps=args.fps)

        pkl_path = robot_dir / f"{motion_name}.pkl"
        qpos_path = qpos_dir / f"{motion_name}.npy"
        joblib.dump({motion_name: entry}, pkl_path, compress=True)
        np.save(qpos_path, qpos)

        summary = _summarize_qpos(qpos)
        manifest["motions"].append(
            {
                "name": motion_name,
                "seed": seed,
                "robot_pkl": str(pkl_path.relative_to(REPO_ROOT)),
                "qpos_npy": str(qpos_path.relative_to(REPO_ROOT)),
                "summary": summary,
            }
        )
        if args.print_summary:
            print(
                f"wrote {pkl_path} frames={summary['num_frames']} "
                f"root_last={summary['root_xyz_last']} dof_range=({summary['dof_min']:.4f}, {summary['dof_max']:.4f})",
                flush=True,
            )

    manifest_path = output_dir / "manifest.json"
    manifest["elapsed_sec"] = time.time() - started
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("\nMotionBricks -> SONIC export complete.")
    print(f"motion_file={robot_dir.relative_to(REPO_ROOT)}")
    print("smpl_motion_file=dummy")
    print(f"manifest={manifest_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate MotionBricks G1 motions and export GEAR-SONIC motion_lib PKLs."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(REPO_ROOT / "motion_sonic" / "motion" / "motionbricks_headless"),
    )
    parser.add_argument("--session", type=str, default="motionbricks")
    parser.add_argument("--motion_prefix", type=str, default="motionbricks_g1")
    parser.add_argument("--num_motions", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--force_idle_tail", type=int, default=100)
    parser.add_argument("--print_summary", type=int, default=1)

    parser.add_argument("--explicit_dataset_folder", type=str, default=None)
    parser.add_argument("--reprocess_clips", type=int, default=0)
    parser.add_argument("--controller", type=str, default="random", choices=["wasd", "random"])
    parser.add_argument("--lookat_movement_direction", type=int, default=0)
    parser.add_argument("--pre_filter_qpos", type=int, default=1)
    parser.add_argument("--source_root_realignment", type=int, default=1)
    parser.add_argument("--target_root_realignment", type=int, default=1)
    parser.add_argument("--force_canonicalization", type=int, default=1)
    parser.add_argument("--skip_ending_target_cond", type=int, default=0)
    parser.add_argument("--random_speed_scale", type=int, default=0)
    parser.add_argument("--speed_scale", type=_parse_speed_scale, default=[0.8, 1.2])
    parser.add_argument("--generate_dt", type=float, default=2.0)

    parser.add_argument("--use_qpos", type=int, default=1)
    parser.add_argument("--planner", type=str, default="default")
    parser.add_argument("--allowed_mode", type=str, default=None)
    parser.add_argument("--clips", type=str, default="G1")

    main(parser.parse_args())
