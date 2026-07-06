#!/usr/bin/env python3
"""Generate a MotionBricks target trajectory and export it for GEAR-Sonic.

Example:

    python3 motion_sonic/motionbricks_trajectory.py \
      --reference forward \
      --output_dir motion_sonic/motion/ \
      --motion_name motionbricks_to_target_forward_5m_target \
      --forward_meters 5.0 \
      --forward_target_name forward_5m_target \
      --mode walk \
      --target_vel 0.20 \
      --target_lookahead_meters 0.35 \
      --max_steps 750 \
      --append_target_hold 100
"""

from __future__ import annotations

import argparse
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
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

DEFAULT_HUMANOID_XML = MOTIONBRICKS_ROOT / "assets" / "skeletons" / "g1" / "scene_29dof.xml"
DEFAULT_SKELETON_XML = MOTIONBRICKS_ROOT / "assets" / "skeletons" / "g1" / "g1.xml"

MUJOCO_DOF_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

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


def repo_path(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    quat_xyzw = Rotation.from_euler("z", yaw).as_quat()
    return quat_xyzw[[3, 0, 1, 2]].astype(np.float32)


def quat_wxyz_to_yaw(quat_wxyz: np.ndarray) -> float:
    quat_xyzw = np.asarray(quat_wxyz, dtype=np.float64)[[1, 2, 3, 0]]
    return float(Rotation.from_quat(quat_xyzw).as_euler("xyz")[2])


def qpos_to_pose_aa(qpos: np.ndarray) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float32)
    pose_aa = np.zeros((qpos.shape[0], 30, 3), dtype=np.float32)
    quat_xyzw = qpos[:, 3:7][:, [1, 2, 3, 0]]
    pose_aa[:, 0, :] = Rotation.from_quat(quat_xyzw).as_rotvec().astype(np.float32)
    pose_aa[:, 1:, :] = qpos[:, 7:36, None] * MUJOCO_DOF_AXES[None, :, :]
    return pose_aa


def make_forward_target(forward_meters: float, target_height: float, yaw: float = 0.0) -> np.ndarray:
    target = np.zeros(36, dtype=np.float32)
    target[:3] = np.asarray([forward_meters, 0.0, target_height], dtype=np.float32)
    target[3:7] = yaw_to_quat_wxyz(yaw)
    target[7:] = 0.0
    return target


def maybe_load_target(path: str | None, frame: int, qpos_key: str) -> np.ndarray:
    if path is None:
        raise ValueError("--target is required unless --reference forward is used.")
    target_path = repo_path(path)
    if not target_path.exists():
        raise FileNotFoundError(target_path)

    if target_path.suffix == ".npy":
        arr = np.load(target_path, allow_pickle=True)
        qpos = arr[frame] if arr.ndim > 1 else arr
    elif target_path.suffix == ".npz":
        data = np.load(target_path, allow_pickle=True)
        for key in (qpos_key, "target_qpos", "qpos", "mujoco_qpos", "robot_qpos"):
            if key in data:
                arr = data[key]
                qpos = arr[frame] if arr.ndim > 1 else arr
                break
        else:
            raise KeyError(f"No qpos-like key found in {target_path}; keys={data.files}")
    elif target_path.suffix == ".pkl":
        data = joblib.load(target_path)
        if isinstance(data, dict) and len(data) == 1 and qpos_key not in data:
            data = next(iter(data.values()))
        if isinstance(data, dict) and qpos_key in data:
            arr = np.asarray(data[qpos_key])
        elif isinstance(data, dict) and {"root_trans_offset", "root_rot", "dof"} <= set(data):
            root = np.asarray(data["root_trans_offset"])[frame]
            quat_xyzw = np.asarray(data["root_rot"])[frame]
            dof = np.asarray(data["dof"])[frame]
            qpos = np.zeros(36, dtype=np.float32)
            qpos[:3] = root[:3]
            qpos[3:7] = quat_xyzw[[3, 0, 1, 2]]
            qpos[7:] = dof[:29]
            return qpos
        else:
            arr = np.asarray(data)
        qpos = arr[frame] if arr.ndim > 1 else arr
    else:
        raise ValueError(f"Unsupported target extension: {target_path.suffix}")

    qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
    if qpos.shape[0] < 36:
        padded = np.zeros(36, dtype=np.float32)
        padded[: qpos.shape[0]] = qpos
        if np.linalg.norm(padded[3:7]) < 1e-6:
            padded[3] = 1.0
        qpos = padded
    return qpos[:36]


def build_motionbricks_demo(args: argparse.Namespace) -> Any:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for MotionBricks inference. Run this in the MotionBricks/Isaac "
            "environment where `import torch` works."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "MotionBricks inference expects CUDA. Run in an environment where "
            "`python3 -c \"import torch; print(torch.cuda.is_available())\"` prints True."
        )
    from motionbricks.motion_backbone.demo.utils import navigation_demo

    demo_args = argparse.Namespace(**vars(args))
    demo_args.has_viewer = 0
    demo_args.return_model_configs = True
    demo_args.return_dataloader = True
    demo_args.recording_dir = None
    demo_args.EXP = args.planner
    demo_args.controller = "random"
    demo_args.use_qpos = 1
    demo_args.humanoid_scene_xml = str(repo_path(args.humanoid_xml))
    demo_args.skeleton_xml = str(repo_path(args.skeleton_xml))
    demo_args.result_dir = str(repo_path(args.result_dir))
    demo_args.data_root = str(repo_path(args.data_root))
    demo_args.explicit_dataset_folder = (
        None if args.explicit_dataset_folder is None else str(repo_path(args.explicit_dataset_folder))
    )
    demo_args.speed_scale = [float(item) for item in str(args.speed_scale).split(",")]
    prev_cwd = os.getcwd()
    try:
        # MotionBricks checkpoints store several asset paths relative to the
        # MotionBricks project root (for example out/.../skeleton). GEAR-Sonic
        # eval runs from the parent repo, so initialize MotionBricks from here.
        os.chdir(MOTIONBRICKS_ROOT)
        return navigation_demo(demo_args)
    finally:
        os.chdir(prev_cwd)


def make_control_signals(
    demo: Any,
    mode_name: str,
    current_qpos: np.ndarray,
    target_qpos: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    clip_names = list(demo.full_agent._clip_holder.CLIPS.keys())
    if mode_name not in clip_names:
        raise ValueError(f"Unknown mode {mode_name!r}. Available modes: {clip_names}")
    mode_id = clip_names.index(mode_name)

    delta_xy = target_qpos[:2] - current_qpos[:2]
    distance_xy = float(np.linalg.norm(delta_xy))
    if distance_xy > 1e-6:
        direction = delta_xy / distance_xy
    else:
        direction = np.asarray([math.cos(quat_wxyz_to_yaw(target_qpos[3:7])), math.sin(quat_wxyz_to_yaw(target_qpos[3:7]))])

    movement_direction = np.asarray([direction[0], direction[1], 0.0], dtype=np.float32)
    facing_direction = np.asarray(
        [math.cos(quat_wxyz_to_yaw(target_qpos[3:7])), math.sin(quat_wxyz_to_yaw(target_qpos[3:7])), 0.0],
        dtype=np.float32,
    )
    if np.linalg.norm(facing_direction[:2]) < 1e-6:
        facing_direction = movement_direction.copy()

    mode = torch.tensor([[mode_id]], dtype=torch.long)
    control = {
        "movement_direction": torch.from_numpy(movement_direction).view(1, 3),
        "facing_direction": torch.from_numpy(facing_direction).view(1, 3),
        "mode": mode,
        "context_mujoco_qpos": demo.full_agent.get_context_mujoco_qpos(),
        "target_vel": torch.tensor([float(args.target_vel)], dtype=torch.float32),
        "specific_target_positions": torch.from_numpy(target_qpos[:3].astype(np.float32)).view(1, 1, 3),
        "specific_target_headings": torch.tensor([[quat_wxyz_to_yaw(target_qpos[3:7])]], dtype=torch.float32),
        "has_specific_target": torch.ones((1, 1), dtype=torch.int32),
        "random_seed": torch.tensor([int(args.random_seed)], dtype=torch.long),
    }
    if args.use_default_allowed_tokens:
        control["allowed_pred_num_tokens"] = demo.controller.get_default_allowed_pred_num_tokens(mode_id)
    return control


def generate_qpos(args: argparse.Namespace, target_qpos: np.ndarray) -> np.ndarray:
    demo = build_motionbricks_demo(args)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    demo.full_agent.reset()

    qpos_frames: list[np.ndarray] = []
    started = time.time()
    reached_at = None
    for step in range(args.max_steps):
        qpos = np.asarray(demo.full_agent.get_next_frame(), dtype=np.float32).copy()
        qpos_frames.append(qpos)

        dist_xy = float(np.linalg.norm(qpos[:2] - target_qpos[:2]))
        if dist_xy <= args.target_lookahead_meters:
            reached_at = step
            break

        demo.mj_data.qpos[:] = qpos
        control = make_control_signals(demo, args.mode, qpos, target_qpos, args)
        with torch.no_grad():
            demo.full_agent.generate_new_frames(
                control,
                demo.controller.get_controller_dt() * args.generate_dt,
                force_generation=args.force_generation,
            )
        if step == 0 or (step + 1) % args.print_every == 0:
            print(
                f"step={step + 1:04d} root_xy=({qpos[0]:.3f}, {qpos[1]:.3f}) "
                f"target_xy=({target_qpos[0]:.3f}, {target_qpos[1]:.3f}) dist_xy={dist_xy:.3f}",
                flush=True,
            )

    if not qpos_frames:
        raise RuntimeError("MotionBricks did not produce any frames.")

    qpos_arr = np.asarray(qpos_frames, dtype=np.float32)
    qpos_arr = append_target_transition(qpos_arr, target_qpos, args)
    elapsed = time.time() - started
    print(
        f"generated frames={qpos_arr.shape[0]} reached_at={reached_at} "
        f"final_xy_error={np.linalg.norm(qpos_arr[-1, :2] - target_qpos[:2]):.4f} elapsed_sec={elapsed:.2f}",
        flush=True,
    )
    return qpos_arr


def append_target_transition(qpos: np.ndarray, target_qpos: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    pieces = [qpos]
    if args.snap_to_target_frames > 0:
        start = qpos[-1].copy()
        transition = []
        for idx in range(1, args.snap_to_target_frames + 1):
            alpha = idx / float(args.snap_to_target_frames)
            frame = (1.0 - alpha) * start + alpha * target_qpos
            frame[3:7] = target_qpos[3:7]
            transition.append(frame.astype(np.float32))
        pieces.append(np.asarray(transition, dtype=np.float32))
    if args.append_target_hold > 0:
        pieces.append(np.repeat(target_qpos[None, :], args.append_target_hold, axis=0).astype(np.float32))
    return np.concatenate(pieces, axis=0).astype(np.float32)


def export_outputs(args: argparse.Namespace, qpos: np.ndarray, target_qpos: np.ndarray) -> dict[str, Any]:
    output_dir = repo_path(args.output_dir)
    qpos_dir = output_dir / "qpos"
    target_dir = output_dir / "target_reference"
    robot_dir = output_dir / "robot_filtered" / "motionbricks_target"
    viz_dir = output_dir / "visualization"
    for directory in (qpos_dir, target_dir, robot_dir, viz_dir):
        directory.mkdir(parents=True, exist_ok=True)

    motion_name = args.motion_name
    target_name = args.forward_target_name if args.reference == "forward" else f"{motion_name}_target"
    qpos_npy = qpos_dir / f"{motion_name}.npy"
    qpos_npz = qpos_dir / f"{motion_name}.npz"
    target_npz = target_dir / f"{target_name}.npz"
    target_pkl = target_dir / f"{target_name}.pkl"
    robot_pkl = robot_dir / f"{motion_name}.pkl"
    markers_json = viz_dir / f"{motion_name}_trajectory_markers.json"
    markers_xml = viz_dir / f"{motion_name}_trajectory_markers.xml"

    np.save(qpos_npy, qpos)
    np.savez(
        qpos_npz,
        qpos=qpos,
        target_qpos=target_qpos,
        root_xyz=qpos[:, :3],
        dof=qpos[:, 7:36],
        fps=np.asarray(args.fps, dtype=np.int32),
    )
    np.savez(target_npz, target_qpos=target_qpos, qpos=target_qpos[None, :], fps=np.asarray(args.fps, dtype=np.int32))
    joblib.dump({target_name: {"target_qpos": target_qpos, "qpos": target_qpos[None, :], "fps": args.fps}}, target_pkl)

    root_rot_xyzw = qpos[:, 3:7][:, [1, 2, 3, 0]].astype(np.float32)
    entry = {
        "root_trans_offset": qpos[:, :3].astype(np.float32),
        "pose_aa": qpos_to_pose_aa(qpos),
        "dof": qpos[:, 7:36].astype(np.float32),
        "root_rot": root_rot_xyzw,
        "smpl_joints": np.zeros((qpos.shape[0], 24, 3), dtype=np.float32),
        "fps": int(args.fps),
    }
    joblib.dump({motion_name: entry}, robot_pkl)

    write_markers(markers_json, markers_xml, qpos[:, :3], target_qpos[:3], args.marker_stride)
    manifest = write_manifest(
        output_dir,
        args,
        target_qpos,
        qpos,
        qpos_npy,
        qpos_npz,
        target_npz,
        target_pkl,
        robot_pkl,
        markers_json,
        markers_xml,
    )
    return manifest


def write_markers(
    json_path: Path,
    xml_path: Path,
    root_xyz: np.ndarray,
    target_xyz: np.ndarray,
    marker_stride: int,
) -> None:
    stride = max(1, int(marker_stride))
    sampled = root_xyz[::stride]
    payload = {
        "intermediate_xyz": sampled.astype(float).round(6).tolist(),
        "target_xyz": np.asarray(target_xyz, dtype=float).round(6).tolist(),
        "marker_stride": stride,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    geoms = []
    for idx, xyz in enumerate(sampled):
        geoms.append(
            f'    <geom name="mb_path_{idx:04d}" type="sphere" size="0.035" '
            f'pos="{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}" rgba="1 0.72 0.05 0.85"/>'
        )
    geoms.append(
        f'    <geom name="mb_target" type="sphere" size="0.07" '
        f'pos="{target_xyz[0]:.6f} {target_xyz[1]:.6f} {target_xyz[2]:.6f}" rgba="0.1 0.35 1 1"/>'
    )
    xml_path.write_text(
        "<mujoco model=\"motionbricks trajectory markers\">\n"
        "  <worldbody>\n"
        + "\n".join(geoms)
        + "\n  </worldbody>\n"
        "</mujoco>\n",
        encoding="utf-8",
    )


def write_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    target_qpos: np.ndarray,
    qpos: np.ndarray,
    qpos_npy: Path,
    qpos_npz: Path,
    target_npz: Path,
    target_pkl: Path,
    robot_pkl: Path,
    markers_json: Path,
    markers_xml: Path,
) -> dict[str, Any]:
    def display_path(path: Path) -> str:
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)

    manifest = {
        "created_unix": time.time(),
        "target": {
            "source_kind": "generated_forward_target" if args.reference == "forward" else "file_target",
            "motion_key": args.forward_target_name if args.reference == "forward" else Path(args.target).stem,
            "frame": args.target_frame,
            "forward_meters": args.forward_meters if args.reference == "forward" else None,
            "target_height": args.target_height,
            "npz_path": str(target_npz),
            "pkl_path": str(target_pkl),
            "target_dof_order": "mujoco",
        },
        "mode": args.mode,
        "target_vel": args.target_vel,
        "target_lookahead_meters": args.target_lookahead_meters,
        "fps": args.fps,
        "max_steps": args.max_steps,
        "append_target_hold": args.append_target_hold,
        "snap_to_target_frames": args.snap_to_target_frames,
        "target_root_xyz": target_qpos[:3].round(6).tolist(),
        "target_yaw": quat_wxyz_to_yaw(target_qpos[3:7]),
        "final_root_xyz": qpos[-1, :3].round(6).tolist(),
        "final_root_xy_error": float(np.linalg.norm(qpos[-1, :2] - target_qpos[:2])),
        "final_dof_rmse": float(np.sqrt(np.mean((qpos[-1, 7:36] - target_qpos[7:36]) ** 2))),
        "qpos_summary": {
            "num_frames": int(qpos.shape[0]),
            "root_xyz_first": qpos[0, :3].round(6).tolist(),
            "root_xyz_last": qpos[-1, :3].round(6).tolist(),
            "root_xyz_min": qpos[:, :3].min(axis=0).round(6).tolist(),
            "root_xyz_max": qpos[:, :3].max(axis=0).round(6).tolist(),
            "dof_mean": float(qpos[:, 7:36].mean()),
            "dof_min": float(qpos[:, 7:36].min()),
            "dof_max": float(qpos[:, 7:36].max()),
        },
        "robot_pkl": display_path(robot_pkl),
        "qpos_npy": display_path(qpos_npy),
        "qpos_npz": display_path(qpos_npz),
        "visualization": {
            "trajectory_markers_json": display_path(markers_json),
            "trajectory_markers_xml": display_path(markers_xml),
            "marker_count": int(math.ceil(qpos.shape[0] / max(1, args.marker_stride))),
            "marker_stride": args.marker_stride,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote manifest: {manifest_path}")
    print(f"wrote GEAR-Sonic motion: {robot_pkl}")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MotionBricks target trajectory exporter for GEAR-Sonic.")
    parser.add_argument("--reference", choices=["forward", "target"], default="forward")
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--target_frame", type=int, default=-1)
    parser.add_argument("--target_qpos_key", type=str, default="target_qpos")
    parser.add_argument("--output_dir", type=str, default="motion_sonic/motion")
    parser.add_argument("--motion_name", type=str, default="motionbricks_to_target_forward_5m_target")
    parser.add_argument("--forward_meters", type=float, default=5.0)
    parser.add_argument("--forward_target_name", type=str, default="forward_5m_target")
    parser.add_argument("--target_height", type=float, default=0.78)
    parser.add_argument("--mode", type=str, default="walk")
    parser.add_argument("--target_vel", type=float, default=0.20)
    parser.add_argument("--target_lookahead_meters", type=float, default=0.35)
    parser.add_argument("--max_steps", type=int, default=750)
    parser.add_argument("--append_target_hold", type=int, default=100)
    parser.add_argument("--snap_to_target_frames", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--marker_stride", type=int, default=10)
    parser.add_argument("--print_every", type=int, default=30)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--force_generation", action="store_true")
    parser.add_argument("--use_default_allowed_tokens", type=int, default=1)

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
    parser.add_argument("--random_speed_scale", type=int, default=0)
    parser.add_argument("--speed_scale", type=str, default="1.0,1.0")
    parser.add_argument("--generate_dt", type=float, default=2.0)
    parser.add_argument("--planner", type=str, default="default")
    parser.add_argument("--allowed_mode", type=str, default=None)
    parser.add_argument("--clips", type=str, default="G1")
    return parser


def parse_args_for_live_defaults() -> argparse.Namespace:
    """Return parser defaults without consuming Hydra/Isaac command-line args."""
    return build_arg_parser().parse_args([])


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def main() -> None:
    args = parse_args()
    if args.reference == "forward":
        target_qpos = make_forward_target(args.forward_meters, args.target_height)
    else:
        target_qpos = maybe_load_target(args.target, args.target_frame, args.target_qpos_key)
        target_qpos[7:] = 0.0
    qpos = generate_qpos(args, target_qpos)
    export_outputs(args, qpos, target_qpos)


if __name__ == "__main__":
    main()
