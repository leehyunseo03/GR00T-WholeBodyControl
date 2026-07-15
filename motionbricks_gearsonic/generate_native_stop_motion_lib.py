#!/usr/bin/env python3
"""Generate a MotionBricks-native path, then append a GEAR-Sonic final pose.

The key difference from ``motion_sonic/generate_to_target_motion_lib.py`` is the
default split of responsibility:

* MotionBricks receives only root position/heading target conditioning.
* Generation stops when the root reaches the destination radius.
* The terminal 29-DOF qpos is appended afterward as a SONIC reference tail.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYNPUT_BACKEND", "dummy")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/motionbricks_gearsonic_matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
MOTIONBRICKS_ROOT = REPO_ROOT / "motionbricks"
for path in (REPO_ROOT, MOTIONBRICKS_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from motion_sonic import generate_to_target_motion_lib as base  # noqa: E402
from motion_sonic.generate_motionbricks_motion_lib import _parse_speed_scale  # noqa: E402


def _append_gearsonic_final_pose_tail(
    qpos_seq: np.ndarray,
    target_qpos: np.ndarray,
    blend_frames: int,
    hold_frames: int,
) -> np.ndarray:
    pieces = [np.asarray(qpos_seq, dtype=np.float32)]
    target = np.asarray(target_qpos, dtype=np.float32)

    if blend_frames > 0:
        start = pieces[0][-1].copy()
        blend = []
        for frame_idx in range(1, blend_frames + 1):
            alpha = frame_idx / float(blend_frames)
            frame = (1.0 - alpha) * start + alpha * target
            frame[3:7] = target[3:7]
            blend.append(frame.astype(np.float32))
        pieces.append(np.asarray(blend, dtype=np.float32))

    if hold_frames > 0:
        pieces.append(np.repeat(target[None, :], hold_frames, axis=0).astype(np.float32))

    return np.concatenate(pieces, axis=0)


def _set_base_compat_defaults(args: argparse.Namespace) -> argparse.Namespace:
    args.target = None
    args.target_frame = -1
    args.qpos_key = None
    args.motion_key = None
    args.root_quat_order = "auto"
    args.reference = "forward"
    args.direct_reference_export = 0
    args.target_only = 0
    args.start_from_target_first = 1
    args.bypass_spring_model = 0
    args.target_yaw = None
    args.arrival_target_vel = 0.0
    args.arrival_force_final_target = 1
    args.arm_swing_seconds = 6.0
    args.arm_swing_frequency = 0.5
    args.arm_pitch_bias = 0.0
    args.arm_pitch_amplitude = 0.75
    args.arm_roll_bias = 0.28
    args.arm_roll_amplitude = 0.12
    args.arm_yaw_amplitude = 0.18
    args.elbow_bias = 0.35
    args.elbow_amplitude = 0.55
    args.wrist_roll_amplitude = 0.20
    args.wrist_yaw_amplitude = 0.18
    return args


def generate(args: argparse.Namespace) -> None:
    import torch as t
    from motionbricks.motion_backbone.demo.utils import navigation_demo

    if not t.cuda.is_available():
        raise RuntimeError("MotionBricks inference expects CUDA; torch.cuda.is_available() is False.")

    args = _set_base_compat_defaults(args)
    started = time.time()
    target_qpos, start_qpos, target_info = base._make_forward_target_reference(args)
    args.target_yaw = base.yaw_from_wxyz(target_qpos[3:7])

    mb_args = base._make_motionbricks_args(args)
    with base._pushd(MOTIONBRICKS_ROOT):
        demo_agent = navigation_demo(mb_args)

    available_modes = list(demo_agent.full_agent._clip_holder.CLIPS.keys())
    base._resolve_requested_mode(args, available_modes=available_modes)
    if args.requested_mode != args.effective_mode:
        print(
            f"Resolved --mode {args.requested_mode!r} to MotionBricks mode {args.effective_mode!r} "
            f"with preset {args.mode_preset!r}.",
            flush=True,
        )

    hold_frames = int(args.append_target_hold)
    args.append_target_hold = 0
    with base._pushd(MOTIONBRICKS_ROOT):
        qpos_seq, modes = base.generate_to_target(demo_agent, args, target_qpos, start_qpos=start_qpos)
    args.append_target_hold = hold_frames
    motionbricks_stop_qpos = qpos_seq[-1].copy()

    qpos_seq = _append_gearsonic_final_pose_tail(
        qpos_seq,
        target_qpos,
        blend_frames=max(0, int(args.final_pose_blend_frames)),
        hold_frames=max(0, int(args.append_target_hold)),
    )

    output_dir = Path(args.output_dir).resolve()
    base.export_qpos_sequence(args, output_dir, qpos_seq, target_qpos, target_info, args.mode, modes)
    _patch_manifest(output_dir, args, qpos_seq, target_qpos, motionbricks_stop_qpos, started)


def _patch_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    qpos_seq: np.ndarray,
    target_qpos: np.ndarray,
    motionbricks_stop_qpos: np.ndarray,
    started: float,
) -> None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    native_stop = {
        "motionbricks_goal_semantics": "root_position_heading_only",
        "motionbricks_target_pose_condition": bool(args.target_pose_condition),
        "stop_radius_meters": float(args.arrival_radius_meters),
        "final_pose_blend_frames": int(args.final_pose_blend_frames),
        "final_pose_hold_frames": int(args.append_target_hold),
        "terminal_pose_owner": "gear_sonic_reference_tail",
        "motionbricks_stop_root_xyz": motionbricks_stop_qpos[:3].round(6).tolist(),
        "motionbricks_stop_root_xy_error": float(
            np.linalg.norm(motionbricks_stop_qpos[:2] - target_qpos[:2])
        ),
        "final_root_xy_error": float(np.linalg.norm(qpos_seq[-1, :2] - target_qpos[:2])),
        "final_dof_rmse": float(np.linalg.norm(qpos_seq[-1, 7:] - target_qpos[7:]) / np.sqrt(29)),
    }
    manifest["native_stop"] = native_stop
    manifest["elapsed_sec"] = time.time() - started
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a native MotionBricks forward path and append a GEAR-Sonic final zero-pose tail."
    )
    parser.add_argument("--output_dir", type=str, default=str(REPO_ROOT / "motionbricks_gearsonic" / "motion"))
    parser.add_argument("--session", type=str, default="native_stop")
    parser.add_argument("--motion_name", type=str, default="motionbricks_native_forward_5m_zero_pose")
    parser.add_argument("--forward_meters", type=float, default=5.0)
    parser.add_argument("--forward_target_name", type=str, default="forward_5m_zero_pose_target")
    parser.add_argument("--target_output_dir", type=str, default=None)
    parser.add_argument("--target_height", type=float, default=0.78)
    parser.add_argument("--target_dof", nargs=29, type=float, default=None, metavar="RAD")
    parser.add_argument("--target_dof_order", choices=["mujoco", "isaaclab"], default="mujoco")

    parser.add_argument("--mode", type=str, default="walk")
    parser.add_argument("--target_vel", type=float, default=0.40)
    parser.add_argument("--target_lookahead_meters", type=float, default=0.50)
    parser.add_argument("--arrival_radius_meters", type=float, default=0.15)
    parser.add_argument("--arrival_mode", type=str, default="idle")
    parser.add_argument("--arrival_settle_frames", type=int, default=24)
    parser.add_argument("--target_pose_condition", type=int, default=0)
    parser.add_argument("--final_pose_blend_frames", type=int, default=60)
    parser.add_argument("--append_target_hold", type=int, default=100)

    parser.add_argument("--max_steps", type=int, default=750)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--print_every", type=int, default=30)
    parser.add_argument("--zero_waist", type=int, default=0)
    parser.add_argument("--zero_waist_pitch", type=int, default=0)

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

    parser.add_argument("--write_visualization", type=int, default=1)
    parser.add_argument("--marker_stride", type=int, default=10)
    parser.add_argument("--marker_size", type=float, default=0.045)
    parser.add_argument("--target_marker_size", type=float, default=0.10)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
