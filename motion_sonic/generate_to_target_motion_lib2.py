#!/usr/bin/env python3
"""Generate a 5 m MotionBricks walk that ends in a hand-raise pose.

This script follows the public MotionBricks demo path documented in
``motionbricks/README.md``: build ``navigation_demo`` for the G1 MuJoCo model,
feed it target root position/heading controls, then export the generated qpos
sequence as a GEAR-Sonic ``motion_lib`` PKL.

MotionBricks' public G1 navigation API strongly conditions root targets through
``specific_target_positions`` and ``specific_target_headings``. It does not
expose a direct per-joint final-pose target in that control dictionary, so the
last segment deliberately blends the generated qpos into a configurable
hand-raise pose while holding the root exactly at the 5 m target.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYNPUT_BACKEND", "dummy")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/motion_sonic_matplotlib")

import joblib
import numpy as np
from scipy.spatial.transform import Rotation

try:
    import torch
except ImportError:
    torch = None


REPO_ROOT = Path(__file__).resolve().parents[1]
MOTIONBRICKS_ROOT = REPO_ROOT / "motionbricks"

for path in (REPO_ROOT, MOTIONBRICKS_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from motion_sonic.generate_motionbricks_motion_lib import qpos_to_sonic_motion_entry


DEFAULT_HUMANOID_XML = MOTIONBRICKS_ROOT / "assets" / "skeletons" / "g1" / "scene_29dof.xml"
DEFAULT_SKELETON_XML = MOTIONBRICKS_ROOT / "assets" / "skeletons" / "g1" / "g1.xml"

MUJOCO_DOF_AXES = np.asarray(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

ARM_DOF = {
    "left_shoulder_pitch": 15,
    "left_shoulder_roll": 16,
    "left_shoulder_yaw": 17,
    "left_elbow": 18,
    "left_wrist_roll": 19,
    "left_wrist_pitch": 20,
    "left_wrist_yaw": 21,
    "right_shoulder_pitch": 22,
    "right_shoulder_roll": 23,
    "right_shoulder_yaw": 24,
    "right_elbow": 25,
    "right_wrist_roll": 26,
    "right_wrist_pitch": 27,
    "right_wrist_yaw": 28,
}

DOF_LIMITS = {
    "left_shoulder_pitch": (-3.0892, 2.6704),
    "left_shoulder_roll": (-1.5882, 2.2515),
    "left_shoulder_yaw": (-2.618, 2.618),
    "left_elbow": (-1.0472, 2.0944),
    "left_wrist_roll": (-1.97222, 1.97222),
    "left_wrist_pitch": (-1.61443, 1.61443),
    "left_wrist_yaw": (-1.61443, 1.61443),
    "right_shoulder_pitch": (-3.0892, 2.6704),
    "right_shoulder_roll": (-2.2515, 1.5882),
    "right_shoulder_yaw": (-2.618, 2.618),
    "right_elbow": (-1.0472, 2.0944),
    "right_wrist_roll": (-1.97222, 1.97222),
    "right_wrist_pitch": (-1.61443, 1.61443),
    "right_wrist_yaw": (-1.61443, 1.61443),
}


@contextlib.contextmanager
def pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    quat_xyzw = Rotation.from_euler("z", yaw).as_quat()
    return quat_xyzw[[3, 0, 1, 2]].astype(np.float32)


def quat_wxyz_to_yaw(quat_wxyz: np.ndarray) -> float:
    quat_xyzw = np.asarray(quat_wxyz, dtype=np.float64)[[1, 2, 3, 0]]
    return float(Rotation.from_quat(quat_xyzw).as_euler("xyz")[2])


def smoothstep(num_frames: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    return t * t * (3.0 - 2.0 * t)


def qpos_to_pose_aa(qpos: np.ndarray) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float32)
    pose_aa = np.zeros((qpos.shape[0], 30, 3), dtype=np.float32)
    quat_xyzw = qpos[:, 3:7][:, [1, 2, 3, 0]]
    pose_aa[:, 0, :] = Rotation.from_quat(quat_xyzw).as_rotvec().astype(np.float32)
    pose_aa[:, 1:, :] = qpos[:, 7:36, None] * MUJOCO_DOF_AXES[None, :, :]
    return pose_aa


def make_forward_target(args: argparse.Namespace) -> np.ndarray:
    target = np.zeros(36, dtype=np.float32)
    target[:3] = [args.forward_meters, 0.0, args.target_height]
    target[3:7] = yaw_to_quat_wxyz(args.target_yaw)
    return target


def make_motionbricks_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        has_viewer=0,
        return_model_configs=True,
        return_dataloader=True,
        recording_dir=None,
        EXP=args.planner,
        controller="random",
        use_qpos=1,
        humanoid_xml=str(repo_path(args.humanoid_xml)),
        humanoid_scene_xml=str(repo_path(args.humanoid_xml)),
        skeleton_xml=str(repo_path(args.skeleton_xml)),
        result_dir=str(repo_path(args.result_dir)),
        data_root=str(repo_path(args.data_root)),
        explicit_dataset_folder=None
        if args.explicit_dataset_folder is None
        else str(repo_path(args.explicit_dataset_folder)),
        reprocess_clips=args.reprocess_clips,
        lookat_movement_direction=args.lookat_movement_direction,
        pre_filter_qpos=args.pre_filter_qpos,
        source_root_realignment=args.source_root_realignment,
        target_root_realignment=args.target_root_realignment,
        force_canonicalization=args.force_canonicalization,
        skip_ending_target_cond=args.skip_ending_target_cond,
        random_speed_scale=0,
        speed_scale=[1.0, 1.0],
        generate_dt=args.generate_dt,
        planner=args.planner,
        allowed_mode=None,
        clips=args.clips,
    )


def build_motionbricks_demo(args: argparse.Namespace) -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for MotionBricks inference.")
    if not torch.cuda.is_available():
        raise RuntimeError("MotionBricks inference expects CUDA; torch.cuda.is_available() is False.")

    from motionbricks.motion_backbone.demo.utils import navigation_demo

    with pushd(MOTIONBRICKS_ROOT):
        return navigation_demo(make_motionbricks_args(args))


def make_control(demo: Any, args: argparse.Namespace, qpos: np.ndarray, target_qpos: np.ndarray, mode_idx: int) -> dict:
    delta_xy = target_qpos[:2] - qpos[:2]
    distance = float(np.linalg.norm(delta_xy))
    if distance > 1e-6:
        direction_xy = delta_xy / distance
    else:
        direction_xy = np.asarray([math.cos(args.target_yaw), math.sin(args.target_yaw)], dtype=np.float32)

    movement = np.asarray([direction_xy[0], direction_xy[1], 0.0], dtype=np.float32)
    facing = np.asarray([math.cos(args.target_yaw), math.sin(args.target_yaw), 0.0], dtype=np.float32)
    target_positions = np.repeat(
        target_qpos[None, :3].astype(np.float32),
        demo.full_agent.NUM_FRAMES_PER_TOKEN,
        axis=0,
    )
    target_headings = np.repeat(
        np.asarray(args.target_yaw, dtype=np.float32),
        demo.full_agent.NUM_FRAMES_PER_TOKEN,
    )

    control = {
        "movement_direction": torch.from_numpy(movement).view(1, 3),
        "facing_direction": torch.from_numpy(facing).view(1, 3),
        "mode": torch.tensor([[mode_idx]], dtype=torch.long),
        "context_mujoco_qpos": demo.full_agent.get_context_mujoco_qpos(),
        "target_vel": torch.tensor([[float(args.target_vel)]], dtype=torch.float32),
        "specific_target_positions": torch.from_numpy(target_positions).view(1, demo.full_agent.NUM_FRAMES_PER_TOKEN, 3),
        "specific_target_headings": torch.from_numpy(target_headings).view(1, demo.full_agent.NUM_FRAMES_PER_TOKEN),
        "has_specific_target": torch.ones((1, 1), dtype=torch.bool),
        "random_seed": torch.tensor([int(args.random_seed)], dtype=torch.long),
    }
    control["allowed_pred_num_tokens"] = demo.controller.get_default_allowed_pred_num_tokens(mode_idx)
    return control


def resolve_mode(demo: Any, requested_mode: str) -> tuple[str, int, list[str]]:
    modes = list(demo.full_agent._clip_holder.CLIPS.keys())
    if requested_mode in modes:
        return requested_mode, modes.index(requested_mode), modes
    if requested_mode in ("jog", "run") and "walk" in modes:
        return "walk", modes.index("walk"), modes
    raise ValueError(f"Unknown mode {requested_mode!r}. Available modes: {modes}")


def generate_walk(demo: Any, args: argparse.Namespace, target_qpos: np.ndarray) -> tuple[np.ndarray, str, list[str]]:
    import mujoco

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    demo.full_agent.reset()
    demo.full_agent.BYPASS_SPRING_MODEL = bool(args.bypass_spring_model)

    mode_name, mode_idx, modes = resolve_mode(demo, args.mode)
    frames = []
    reached_at = None

    with pushd(MOTIONBRICKS_ROOT):
        for step in range(1, args.max_steps + 1):
            qpos = np.asarray(demo.full_agent.get_next_frame(), dtype=np.float32).copy()
            frames.append(qpos)

            root_err = float(np.linalg.norm(qpos[:2] - target_qpos[:2]))
            if root_err <= args.arrival_radius_meters:
                reached_at = step
                break

            demo.mj_data.qpos[:] = qpos
            control = make_control(demo, args, qpos, target_qpos, mode_idx)
            with torch.no_grad():
                demo.full_agent.generate_new_frames(
                    control,
                    demo.controller.get_controller_dt() * args.generate_dt,
                    force_generation=(step == 1),
                )
            mujoco.mj_forward(demo.mj_model, demo.mj_data)

            if step == 1 or step % args.print_every == 0:
                print(
                    f"step={step:04d} root_xy=({qpos[0]:.3f}, {qpos[1]:.3f}) "
                    f"target_xy=({target_qpos[0]:.3f}, {target_qpos[1]:.3f}) err={root_err:.3f}",
                    flush=True,
                )

    if not frames:
        raise RuntimeError("MotionBricks did not produce any frames.")

    qpos = np.asarray(frames, dtype=np.float32)
    print(
        f"MotionBricks walk frames={qpos.shape[0]} reached_at={reached_at} "
        f"last_xy_error={np.linalg.norm(qpos[-1, :2] - target_qpos[:2]):.4f}",
        flush=True,
    )
    return qpos, mode_name, modes


def apply_hand_raise_pose(qpos: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.asarray(qpos, dtype=np.float32)
    final_pose = qpos[-1].copy()
    final_pose[:3] = [args.forward_meters, 0.0, args.target_height]
    final_pose[3:7] = yaw_to_quat_wxyz(args.target_yaw)

    dof = final_pose[7:36]
    if args.hand in ("left", "both"):
        dof[ARM_DOF["left_shoulder_pitch"]] = args.shoulder_pitch
        dof[ARM_DOF["left_shoulder_roll"]] = args.left_shoulder_roll
        dof[ARM_DOF["left_shoulder_yaw"]] = args.left_shoulder_yaw
        dof[ARM_DOF["left_elbow"]] = args.elbow
        dof[ARM_DOF["left_wrist_roll"]] = args.left_wrist_roll
        dof[ARM_DOF["left_wrist_pitch"]] = args.wrist_pitch
        dof[ARM_DOF["left_wrist_yaw"]] = args.left_wrist_yaw
    if args.hand in ("right", "both"):
        dof[ARM_DOF["right_shoulder_pitch"]] = args.shoulder_pitch
        dof[ARM_DOF["right_shoulder_roll"]] = args.right_shoulder_roll
        dof[ARM_DOF["right_shoulder_yaw"]] = args.right_shoulder_yaw
        dof[ARM_DOF["right_elbow"]] = args.elbow
        dof[ARM_DOF["right_wrist_roll"]] = args.right_wrist_roll
        dof[ARM_DOF["right_wrist_pitch"]] = args.wrist_pitch
        dof[ARM_DOF["right_wrist_yaw"]] = args.right_wrist_yaw

    for name, idx in ARM_DOF.items():
        lo, hi = DOF_LIMITS[name]
        dof[idx] = np.clip(dof[idx], lo, hi)

    pieces = [qpos]
    if args.root_settle_frames > 0:
        start = qpos[-1].copy()
        settle_alphas = smoothstep(args.root_settle_frames)
        settle = []
        for alpha in settle_alphas:
            frame = start.copy()
            frame[:3] = (1.0 - alpha) * start[:3] + alpha * final_pose[:3]
            frame[3:7] = final_pose[3:7]
            settle.append(frame)
        pieces.append(np.asarray(settle, dtype=np.float32))

    if args.raise_frames > 0:
        start = pieces[-1][-1].copy() if len(pieces) > 1 else qpos[-1].copy()
        raise_alphas = smoothstep(args.raise_frames)
        raise_frames = []
        for alpha in raise_alphas:
            frame = start.copy()
            frame[:3] = final_pose[:3]
            frame[3:7] = final_pose[3:7]
            frame[7:36] = (1.0 - alpha) * start[7:36] + alpha * final_pose[7:36]
            raise_frames.append(frame)
        pieces.append(np.asarray(raise_frames, dtype=np.float32))

    if args.hold_frames > 0:
        pieces.append(np.repeat(final_pose[None, :], args.hold_frames, axis=0).astype(np.float32))

    return np.concatenate(pieces, axis=0).astype(np.float32), final_pose


def write_markers(output_dir: Path, motion_name: str, qpos: np.ndarray, target_qpos: np.ndarray, args: argparse.Namespace) -> dict:
    viz_dir = output_dir / "visualization"
    viz_dir.mkdir(parents=True, exist_ok=True)
    stride = max(1, args.marker_stride)
    sampled = qpos[::stride, :3]

    marker_json = viz_dir / f"{motion_name}_trajectory_markers.json"
    marker_xml = viz_dir / f"{motion_name}_trajectory_markers.xml"
    payload = {
        "intermediate_xyz": sampled.round(6).tolist(),
        "target_xyz": target_qpos[:3].round(6).tolist(),
        "marker_stride": stride,
    }
    marker_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    geoms = []
    for idx, xyz in enumerate(sampled):
        geoms.append(
            f'    <geom name="mb_path_{idx:04d}" type="sphere" size="0.035" '
            f'pos="{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}" rgba="1 0.72 0.05 0.85"/>'
        )
    geoms.append(
        f'    <geom name="mb_target" type="sphere" size="0.08" '
        f'pos="{target_qpos[0]:.6f} {target_qpos[1]:.6f} {target_qpos[2]:.6f}" rgba="0.1 0.35 1 1"/>'
    )
    marker_xml.write_text(
        "<mujoco model=\"motionbricks hand raise trajectory markers\">\n"
        "  <worldbody>\n"
        + "\n".join(geoms)
        + "\n  </worldbody>\n"
        "</mujoco>\n",
        encoding="utf-8",
    )
    return {
        "trajectory_markers_json": display_path(marker_json),
        "trajectory_markers_xml": display_path(marker_xml),
        "marker_count": int(sampled.shape[0]),
        "marker_stride": stride,
    }


def export_outputs(
    args: argparse.Namespace,
    qpos: np.ndarray,
    target_qpos: np.ndarray,
    mode_name: str,
    available_modes: list[str],
) -> None:
    output_dir = repo_path(args.output_dir)
    qpos_dir = output_dir / "qpos"
    robot_dir = output_dir / "robot_filtered" / args.session
    target_dir = output_dir / "target_reference"
    for directory in (qpos_dir, robot_dir, target_dir):
        directory.mkdir(parents=True, exist_ok=True)

    motion_name = args.motion_name
    qpos_npy = qpos_dir / f"{motion_name}.npy"
    qpos_npz = qpos_dir / f"{motion_name}.npz"
    robot_pkl = robot_dir / f"{motion_name}.pkl"
    target_npz = target_dir / f"{args.forward_target_name}.npz"
    manifest_path = output_dir / "manifest.json"

    np.save(qpos_npy, qpos)
    np.savez_compressed(
        qpos_npz,
        qpos=qpos,
        target_qpos=target_qpos,
        root_xyz=qpos[:, :3],
        dof=qpos[:, 7:36],
        fps=np.asarray(args.fps, dtype=np.int32),
    )
    np.savez_compressed(
        target_npz,
        target_qpos=target_qpos,
        qpos=target_qpos[None, :],
        fps=np.asarray(args.fps, dtype=np.int32),
    )

    entry = qpos_to_sonic_motion_entry(qpos, fps=args.fps)
    entry.setdefault("pose_aa", qpos_to_pose_aa(qpos))
    joblib.dump({motion_name: entry}, robot_pkl, compress=True)

    visualization = write_markers(output_dir, motion_name, qpos, target_qpos, args)
    manifest = {
        "created_unix": time.time(),
        "source": "MotionBricks navigation_demo root trajectory plus terminal qpos hand-raise blend",
        "motion_name": motion_name,
        "mode": mode_name,
        "requested_mode": args.mode,
        "available_modes": available_modes,
        "fps": args.fps,
        "forward_meters": args.forward_meters,
        "target_vel": args.target_vel,
        "arrival_radius_meters": args.arrival_radius_meters,
        "root_settle_frames": args.root_settle_frames,
        "raise_frames": args.raise_frames,
        "hold_frames": args.hold_frames,
        "hand": args.hand,
        "target_root_xyz": target_qpos[:3].round(6).tolist(),
        "final_root_xyz": qpos[-1, :3].round(6).tolist(),
        "final_root_xy_error_m": float(np.linalg.norm(qpos[-1, :2] - target_qpos[:2])),
        "final_hand_raise_dof": {
            name: float(qpos[-1, 7 + idx])
            for name, idx in ARM_DOF.items()
            if args.hand == "both" or name.startswith(args.hand)
        },
        "qpos_npy": display_path(qpos_npy),
        "qpos_npz": display_path(qpos_npz),
        "target_npz": display_path(target_npz),
        "robot_pkl": display_path(robot_pkl),
        "visualization": visualization,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("\nMotionBricks 5 m hand-raise export complete.")
    print(f"motion_file={display_path(robot_pkl)}")
    print("smpl_motion_file=dummy")
    print(f"qpos_npy={display_path(qpos_npy)}")
    print(f"manifest={display_path(manifest_path)}")
    print(f"final_root_xy_error_m={manifest['final_root_xy_error_m']:.6f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a 5 m MotionBricks trajectory ending in hand raise.")
    parser.add_argument("--output_dir", type=str, default="motion_sonic/motion_5m_hand_raise")
    parser.add_argument("--session", type=str, default="motionbricks_target")
    parser.add_argument("--motion_name", type=str, default="motionbricks_to_target_forward_5m_hand_raise")
    parser.add_argument("--forward_meters", type=float, default=5.0)
    parser.add_argument("--forward_target_name", type=str, default="forward_5m_hand_raise_target")
    parser.add_argument("--target_height", type=float, default=0.78)
    parser.add_argument("--target_yaw", type=float, default=0.0)
    parser.add_argument("--mode", type=str, default="walk")
    parser.add_argument("--target_vel", type=float, default=0.20)
    parser.add_argument("--max_steps", type=int, default=750)
    parser.add_argument("--arrival_radius_meters", type=float, default=0.08)
    parser.add_argument("--root_settle_frames", type=int, default=30)
    parser.add_argument("--raise_frames", type=int, default=45)
    parser.add_argument("--hold_frames", type=int, default=90)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--print_every", type=int, default=30)

    parser.add_argument("--hand", choices=["left", "right", "both"], default="both")
    parser.add_argument("--shoulder_pitch", type=float, default=-1.55)
    parser.add_argument("--left_shoulder_roll", type=float, default=0.35)
    parser.add_argument("--right_shoulder_roll", type=float, default=-0.35)
    parser.add_argument("--left_shoulder_yaw", type=float, default=0.0)
    parser.add_argument("--right_shoulder_yaw", type=float, default=0.0)
    parser.add_argument("--elbow", type=float, default=0.45)
    parser.add_argument("--left_wrist_roll", type=float, default=0.0)
    parser.add_argument("--right_wrist_roll", type=float, default=0.0)
    parser.add_argument("--wrist_pitch", type=float, default=0.0)
    parser.add_argument("--left_wrist_yaw", type=float, default=0.0)
    parser.add_argument("--right_wrist_yaw", type=float, default=0.0)

    parser.add_argument("--bypass_spring_model", type=int, default=1)
    parser.add_argument("--humanoid_xml", type=str, default=str(DEFAULT_HUMANOID_XML))
    parser.add_argument("--skeleton_xml", type=str, default=str(DEFAULT_SKELETON_XML))
    parser.add_argument("--result_dir", type=str, default=str(MOTIONBRICKS_ROOT / "out"))
    parser.add_argument("--data_root", type=str, default=str(MOTIONBRICKS_ROOT / "datasets"))
    parser.add_argument("--explicit_dataset_folder", type=str, default=None)
    parser.add_argument("--reprocess_clips", type=int, default=0)
    parser.add_argument("--lookat_movement_direction", type=int, default=1)
    parser.add_argument("--pre_filter_qpos", type=int, default=1)
    parser.add_argument("--source_root_realignment", type=int, default=1)
    parser.add_argument("--target_root_realignment", type=int, default=1)
    parser.add_argument("--force_canonicalization", type=int, default=1)
    parser.add_argument("--skip_ending_target_cond", type=int, default=0)
    parser.add_argument("--generate_dt", type=float, default=2.0)
    parser.add_argument("--planner", type=str, default="default")
    parser.add_argument("--clips", type=str, default="G1")
    parser.add_argument("--marker_stride", type=int, default=10)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    target_qpos = make_forward_target(args)
    demo = build_motionbricks_demo(args)
    qpos_walk, mode_name, modes = generate_walk(demo, args, target_qpos)
    qpos_final, hand_target_qpos = apply_hand_raise_pose(qpos_walk, args)
    export_outputs(args, qpos_final, hand_target_qpos, mode_name, modes)


if __name__ == "__main__":
    main()
