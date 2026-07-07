#!/usr/bin/env python3
"""Generate a MotionBricks/GEAR-SONIC trajectory toward a target teleop frame.

Input can be a teleop/robot ``.npz``, ``.npy``, or ``.pkl`` containing either:

- full MuJoCo qpos, shape ``(36,)`` or ``(T, 36)``; or
- SONIC-style fields ``root_trans_offset``, ``root_rot``, and ``dof``.

The script extracts one target frame, asks MotionBricks to generate a trajectory
toward that target root position/heading, optionally appends the exact target
qpos as a short hold segment, and exports the result as a GEAR-SONIC
``motion_lib`` PKL directory. For in-place target motions such as standing arm
swinging, it can also export the generated qpos reference directly, because a
single final target frame cannot describe a full upper-body time sequence.
"""

from __future__ import annotations

import argparse
import contextlib
import html
import json
import os
import pickle
import sys
import time
from pathlib import Path

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

from motion_sonic.generate_motionbricks_motion_lib import (
    _make_motionbricks_args,
    _parse_speed_scale,
    _summarize_qpos,
    qpos_to_sonic_motion_entry,
)


QP0S_KEYS = (
    "qpos",
    "mujoco_qpos",
    "robot_qpos",
    "target_qpos",
    "body_qpos",
    "joint_qpos",
)
ROOT_POS_KEYS = (
    "root_trans_offset",
    "root_pos",
    "root_xyz",
    "base_trans_target",
    "base_trans_measured",
)
ROOT_QUAT_KEYS = (
    "root_rot",
    "root_quat",
    "root_quat_wxyz",
    "base_quat_target",
    "base_quat_measured",
)
DOF_KEYS = (
    "dof",
    "joint_pos",
    "body_q_target",
    "body_q_measured",
)

DEFAULT_TARGET_VEL = 0.35
DEFAULT_TARGET_LOOKAHEAD_METERS = 0.65
JOG_PRESET_TARGET_VEL = 1.4
JOG_PRESET_LOOKAHEAD_METERS = 1.0
RUN_PRESET_TARGET_VEL = 2.2
RUN_PRESET_LOOKAHEAD_METERS = 1.5

MJ_TO_IL = (
    0,
    3,
    6,
    9,
    13,
    17,
    1,
    4,
    7,
    10,
    14,
    18,
    2,
    5,
    8,
    11,
    15,
    19,
    21,
    23,
    25,
    27,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
)

ARM_DOF_INDICES = {
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

WAIST_DOF_INDICES = (12, 13, 14)
WAIST_QPOS_INDICES = tuple(7 + idx for idx in WAIST_DOF_INDICES)
WAIST_PITCH_DOF_INDEX = 14
WAIST_PITCH_QPOS_INDEX = 7 + WAIST_PITCH_DOF_INDEX

ARM_DOF_LIMITS = {
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


def _zero_waist_dof_inplace(dof):
    dof[..., list(WAIST_DOF_INDICES)] = 0.0
    return dof


def _zero_waist_qpos_inplace(qpos):
    qpos[..., list(WAIST_QPOS_INDICES)] = 0.0
    return qpos


def _zero_waist_pitch_dof_inplace(dof):
    dof[..., WAIST_PITCH_DOF_INDEX] = 0.0
    return dof


def _zero_waist_pitch_qpos_inplace(qpos):
    qpos[..., WAIST_PITCH_QPOS_INDEX] = 0.0
    return qpos


def _apply_waist_lock_dof_inplace(dof, args):
    if args.zero_waist:
        return _zero_waist_dof_inplace(dof)
    if args.zero_waist_pitch:
        return _zero_waist_pitch_dof_inplace(dof)
    return dof


def _apply_waist_lock_qpos_inplace(qpos, args):
    if args.zero_waist:
        return _zero_waist_qpos_inplace(qpos)
    if args.zero_waist_pitch:
        return _zero_waist_pitch_qpos_inplace(qpos)
    return qpos


def _load_pickle(path: Path):
    try:
        import joblib

        return joblib.load(path)
    except Exception:
        with path.open("rb") as f:
            return pickle.load(f)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _as_array(value):
    import numpy as np

    value = np.asarray(value)
    if value.dtype == object and value.shape == ():
        value = np.asarray(value.item())
    return value


def _select_frame(arr, frame: int):
    import numpy as np

    arr = _as_array(arr)
    if arr.ndim == 1:
        return arr.astype(np.float32)
    if arr.ndim >= 2:
        return arr[frame].astype(np.float32)
    raise ValueError(f"Cannot select a frame from shape {arr.shape}")


def _pick_key(data: dict, keys: tuple[str, ...], explicit_key: str | None = None):
    if explicit_key:
        if explicit_key not in data:
            raise KeyError(f"Requested key '{explicit_key}' not found. Available keys: {list(data.keys())}")
        return explicit_key
    for key in keys:
        if key in data:
            return key
    return None


def _unwrap_motion_dict(data, motion_key: str | None):
    if not isinstance(data, dict):
        return data, None

    if motion_key:
        if motion_key not in data:
            raise KeyError(f"Requested motion_key '{motion_key}' not found. Available keys: {list(data.keys())}")
        return data[motion_key], motion_key

    if any(key in data for key in QP0S_KEYS + ROOT_POS_KEYS + ROOT_QUAT_KEYS + DOF_KEYS):
        return data, None

    dict_values = [(key, value) for key, value in data.items() if isinstance(value, dict)]
    if len(dict_values) == 1:
        return dict_values[0][1], dict_values[0][0]

    for key, value in dict_values:
        if any(field in value for field in QP0S_KEYS + ROOT_POS_KEYS + ROOT_QUAT_KEYS + DOF_KEYS):
            return value, key

    return data, None


def _quat_to_wxyz(quat, order: str, source_key: str = ""):
    import numpy as np

    quat = np.asarray(quat, dtype=np.float32).reshape(4)
    if order == "auto":
        if "xyzw" in source_key or source_key == "root_rot":
            order = "xyzw"
        else:
            order = "wxyz"
    if order == "xyzw":
        quat = quat[[3, 0, 1, 2]]
    elif order != "wxyz":
        raise ValueError(f"Unsupported quaternion order: {order}")

    norm = np.linalg.norm(quat)
    return quat / max(norm, 1e-8)


def _qpos_from_parts(data: dict, frame: int, quat_order: str):
    import numpy as np

    root_pos_key = _pick_key(data, ROOT_POS_KEYS)
    root_quat_key = _pick_key(data, ROOT_QUAT_KEYS)
    dof_key = _pick_key(data, DOF_KEYS)
    if not root_pos_key or not root_quat_key or not dof_key:
        raise KeyError(
            "Could not infer qpos fields. Need either a qpos-like key, or "
            f"root/dof fields. Available keys: {list(data.keys())}"
        )

    root_pos = _select_frame(data[root_pos_key], frame)[:3]
    root_quat = _quat_to_wxyz(_select_frame(data[root_quat_key], frame), quat_order, root_quat_key)
    dof = _select_frame(data[dof_key], frame)[:29]
    return np.concatenate([root_pos, root_quat, dof]).astype(np.float32)


def _qpos_from_any(data, frame: int, qpos_key: str | None, motion_key: str | None, quat_order: str):
    import numpy as np

    data, used_motion_key = _unwrap_motion_dict(data, motion_key)

    if isinstance(data, dict):
        key = _pick_key(data, QP0S_KEYS, qpos_key)
        if key:
            qpos = _select_frame(data[key], frame)
            if qpos.shape[0] < 36:
                raise ValueError(f"Key '{key}' has frame shape {qpos.shape}; expected at least 36 values")
            return qpos[:36].astype(np.float32), {"source_kind": "qpos", "key": key, "motion_key": used_motion_key}

        qpos = _qpos_from_parts(data, frame, quat_order)
        return qpos, {"source_kind": "root_dof", "key": None, "motion_key": used_motion_key}

    arr = np.asarray(data)
    if arr.ndim == 1 and arr.shape[0] >= 36:
        return arr[:36].astype(np.float32), {"source_kind": "array", "key": None, "motion_key": used_motion_key}
    if arr.ndim >= 2 and arr.shape[-1] >= 36:
        return arr[frame, :36].astype(np.float32), {"source_kind": "array", "key": None, "motion_key": used_motion_key}

    raise ValueError(f"Cannot infer target qpos from object of type {type(data).__name__}")


def load_target_qpos(path: str, frame: int, qpos_key: str | None, motion_key: str | None, quat_order: str):
    import numpy as np

    target_path = Path(path).expanduser().resolve()
    if not target_path.exists():
        raise FileNotFoundError(target_path)

    if target_path.suffix == ".npz":
        npz = np.load(target_path, allow_pickle=True)
        data = {key: npz[key] for key in npz.files}
    elif target_path.suffix == ".npy":
        data = np.load(target_path, allow_pickle=True)
    elif target_path.suffix == ".pkl":
        data = _load_pickle(target_path)
    else:
        raise ValueError(f"Unsupported target extension '{target_path.suffix}'. Use .npz, .npy, or .pkl")

    qpos, info = _qpos_from_any(data, frame, qpos_key, motion_key, quat_order)
    info["path"] = str(target_path)
    info["frame"] = frame
    return qpos, info


def yaw_from_wxyz(quat):
    import numpy as np

    w, x, y, z = _quat_to_wxyz(quat, "wxyz")
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _target_dof_to_mujoco(values: list[float] | None, order: str):
    import numpy as np

    if values is None:
        return np.zeros(29, dtype=np.float32)

    dof = np.asarray(values, dtype=np.float32)
    if dof.shape != (29,):
        raise ValueError(f"--target_dof must contain exactly 29 values, got {dof.shape}")
    if order == "mujoco":
        return dof
    if order == "isaaclab":
        return dof[np.asarray(MJ_TO_IL, dtype=np.int32)]
    raise ValueError(f"Unsupported --target_dof_order: {order}")


def _make_forward_target_reference(args):
    import joblib
    import numpy as np

    target_dir = Path(args.target_output_dir or (Path(args.output_dir) / "target_reference")).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    dof = _target_dof_to_mujoco(args.target_dof, args.target_dof_order)
    _apply_waist_lock_dof_inplace(dof, args)
    start_qpos = np.zeros(36, dtype=np.float32)
    start_qpos[:3] = [0.0, 0.0, args.target_height]
    start_qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    start_qpos[7:] = dof

    target_qpos = start_qpos.copy()
    target_qpos[0] += args.forward_meters

    qpos = np.stack([start_qpos, target_qpos], axis=0)
    root_trans = qpos[:, :3].copy()
    root_rot_xyzw = qpos[:, [4, 5, 6, 3]].copy()
    dof_seq = qpos[:, 7:].copy()

    motion_name = args.forward_target_name
    npz_path = target_dir / f"{motion_name}.npz"
    pkl_path = target_dir / f"{motion_name}.pkl"

    np.savez_compressed(
        npz_path,
        qpos=qpos,
        target_qpos=target_qpos,
        start_qpos=start_qpos,
        dof_mujoco=dof,
        fps=np.asarray(args.fps, dtype=np.int32),
    )
    joblib.dump(
        {
            motion_name: {
                "root_trans_offset": root_trans,
                "root_rot": root_rot_xyzw,
                "dof": dof_seq,
                "fps": args.fps,
            }
        },
        pkl_path,
        compress=True,
    )

    info = {
        "source_kind": "generated_forward_target",
        "motion_key": motion_name,
        "frame": -1,
        "forward_meters": float(args.forward_meters),
        "target_height": float(args.target_height),
        "npz_path": str(npz_path),
        "pkl_path": str(pkl_path),
        "target_dof_order": args.target_dof_order,
    }
    return target_qpos, start_qpos, info


def _base_standing_qpos(args):
    import numpy as np

    dof = _target_dof_to_mujoco(args.target_dof, args.target_dof_order)
    _apply_waist_lock_dof_inplace(dof, args)
    qpos = np.zeros(36, dtype=np.float32)
    qpos[:3] = [0.0, 0.0, args.target_height]
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    qpos[7:] = dof
    return qpos


def _clip_arm_dof(dof, name: str):
    import numpy as np

    lo, hi = ARM_DOF_LIMITS[name]
    dof[..., ARM_DOF_INDICES[name]] = np.clip(dof[..., ARM_DOF_INDICES[name]], lo, hi)


def _make_standing_arm_swing_reference(args):
    import joblib
    import numpy as np

    target_dir = Path(args.target_output_dir or (Path(args.output_dir) / "target_reference")).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    num_frames = max(2, int(round(args.arm_swing_seconds * args.fps)))
    time_s = np.arange(num_frames, dtype=np.float32) / float(args.fps)
    phase = 2.0 * np.pi * float(args.arm_swing_frequency) * time_s

    base_qpos = _base_standing_qpos(args)
    qpos = np.repeat(base_qpos[None, :], num_frames, axis=0)
    dof = qpos[:, 7:]

    swing = np.sin(phase)
    bend = 0.5 + 0.5 * np.sin(phase - np.pi / 2.0)
    twist = np.sin(phase + np.pi / 2.0)

    # MuJoCo qpos order: left leg, right leg, waist, left arm, right arm.
    # Opposite shoulder phases create a clear standing arm-swing reference.
    dof[:, ARM_DOF_INDICES["left_shoulder_pitch"]] = args.arm_pitch_bias + args.arm_pitch_amplitude * swing
    dof[:, ARM_DOF_INDICES["right_shoulder_pitch"]] = args.arm_pitch_bias - args.arm_pitch_amplitude * swing
    dof[:, ARM_DOF_INDICES["left_shoulder_roll"]] = args.arm_roll_bias + args.arm_roll_amplitude * twist
    dof[:, ARM_DOF_INDICES["right_shoulder_roll"]] = -args.arm_roll_bias + args.arm_roll_amplitude * twist
    dof[:, ARM_DOF_INDICES["left_shoulder_yaw"]] = args.arm_yaw_amplitude * np.sin(phase + np.pi)
    dof[:, ARM_DOF_INDICES["right_shoulder_yaw"]] = args.arm_yaw_amplitude * np.sin(phase)
    dof[:, ARM_DOF_INDICES["left_elbow"]] = args.elbow_bias + args.elbow_amplitude * bend
    dof[:, ARM_DOF_INDICES["right_elbow"]] = args.elbow_bias + args.elbow_amplitude * (1.0 - bend)
    dof[:, ARM_DOF_INDICES["left_wrist_roll"]] = args.wrist_roll_amplitude * np.sin(phase * 2.0)
    dof[:, ARM_DOF_INDICES["right_wrist_roll"]] = -args.wrist_roll_amplitude * np.sin(phase * 2.0)
    dof[:, ARM_DOF_INDICES["left_wrist_yaw"]] = args.wrist_yaw_amplitude * np.sin(phase + np.pi / 3.0)
    dof[:, ARM_DOF_INDICES["right_wrist_yaw"]] = -args.wrist_yaw_amplitude * np.sin(phase + np.pi / 3.0)

    for name in ARM_DOF_INDICES:
        _clip_arm_dof(dof, name)

    root_trans = qpos[:, :3].copy()
    root_rot_xyzw = qpos[:, [4, 5, 6, 3]].copy()
    dof_seq = qpos[:, 7:].copy()

    motion_name = args.standing_target_name
    npz_path = target_dir / f"{motion_name}.npz"
    pkl_path = target_dir / f"{motion_name}.pkl"

    np.savez_compressed(
        npz_path,
        qpos=qpos,
        target_qpos=qpos[-1],
        start_qpos=qpos[0],
        root_xyz=root_trans,
        dof=dof_seq,
        fps=np.asarray(args.fps, dtype=np.int32),
    )
    joblib.dump(
        {
            motion_name: {
                "root_trans_offset": root_trans,
                "root_rot": root_rot_xyzw,
                "dof": dof_seq,
                "fps": args.fps,
            }
        },
        pkl_path,
        compress=True,
    )

    info = {
        "source_kind": "generated_standing_arm_swing",
        "motion_key": motion_name,
        "frame": -1,
        "duration_sec": float(args.arm_swing_seconds),
        "frequency_hz": float(args.arm_swing_frequency),
        "target_height": float(args.target_height),
        "npz_path": str(npz_path),
        "pkl_path": str(pkl_path),
        "target_dof_order": args.target_dof_order,
    }
    return qpos[-1].copy(), qpos[0].copy(), info, qpos


def _write_trajectory_visualization(output_dir: Path, motion_name: str, qpos_seq, target_qpos, args):
    import numpy as np

    if not args.write_visualization:
        return {}

    viz_dir = output_dir / "visualization"
    viz_dir.mkdir(parents=True, exist_ok=True)

    stride = max(1, int(args.marker_stride))
    generated_frames = qpos_seq[:-args.append_target_hold] if args.append_target_hold > 0 else qpos_seq
    marker_xyz = generated_frames[::stride, :3].astype(np.float32)
    target_xyz = np.asarray(target_qpos[:3], dtype=np.float32)

    marker_json = viz_dir / f"{motion_name}_trajectory_markers.json"
    marker_xml = viz_dir / f"{motion_name}_trajectory_markers.xml"

    marker_payload = {
        "motion_name": motion_name,
        "coordinate_frame": "MuJoCo qpos root translation: x forward, y left, z up",
        "intermediate_marker_color": "yellow",
        "target_marker_color": "blue",
        "marker_stride": stride,
        "intermediate_xyz": marker_xyz.round(6).tolist(),
        "target_xyz": target_xyz.round(6).tolist(),
    }
    marker_json.write_text(json.dumps(marker_payload, indent=2) + "\n", encoding="utf-8")

    g1_xml = MOTIONBRICKS_ROOT / "assets" / "skeletons" / "g1" / "g1_29dof.xml"
    sphere_size = float(args.marker_size)
    target_size = float(args.target_marker_size)
    marker_lines = []
    for idx, xyz in enumerate(marker_xyz):
        marker_lines.append(
            "    "
            f'<geom name="motionbricks_traj_{idx:04d}" type="sphere" '
            f'pos="{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}" '
            f'size="{sphere_size:.4f}" rgba="1.0 0.86 0.0 0.85" contype="0" conaffinity="0"/>'
        )
    marker_lines.append(
        "    "
        f'<geom name="final_target_blue" type="sphere" '
        f'pos="{target_xyz[0]:.6f} {target_xyz[1]:.6f} {target_xyz[2]:.6f}" '
        f'size="{target_size:.4f}" rgba="0.0 0.25 1.0 0.95" contype="0" conaffinity="0"/>'
    )

    all_xyz = np.concatenate([marker_xyz, target_xyz[None, :]], axis=0)
    center = all_xyz.mean(axis=0)
    extent = max(1.5, float(np.ptp(all_xyz[:, :2], axis=0).max()) + 1.0)

    marker_xml.write_text(
        "\n".join(
            [
                '<mujoco model="motionbricks target trajectory markers">',
                f'  <include file="{html.escape(str(g1_xml))}"/>',
                "",
                f'  <statistic center="{center[0]:.6f} {center[1]:.6f} {center[2]:.6f}" extent="{extent:.6f}"/>',
                "",
                "  <visual>",
                "    <headlight diffuse=\"0.6 0.6 0.6\" ambient=\"0.3 0.3 0.3\" specular=\"0 0 0\"/>",
                "    <global azimuth=\"-130\" elevation=\"-20\"/>",
                "  </visual>",
                "",
                "  <asset>",
                "    <texture type=\"2d\" name=\"groundplane\" builtin=\"checker\" mark=\"edge\" "
                "rgb1=\"0.2 0.3 0.4\" rgb2=\"0.1 0.2 0.3\" markrgb=\"0.8 0.8 0.8\" width=\"300\" height=\"300\"/>",
                "    <material name=\"groundplane\" texture=\"groundplane\" texuniform=\"true\" texrepeat=\"8 8\" reflectance=\"0.2\"/>",
                "  </asset>",
                "",
                "  <worldbody>",
                "    <light pos=\"0 0 1.5\" dir=\"0 0 -1\" directional=\"true\"/>",
                "    <geom name=\"floor\" size=\"0 0 0.05\" type=\"plane\" material=\"groundplane\"/>",
                *marker_lines,
                "  </worldbody>",
                "</mujoco>",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "trajectory_markers_json": _display_path(marker_json),
        "trajectory_markers_xml": _display_path(marker_xml),
        "marker_count": int(marker_xyz.shape[0]),
        "marker_stride": stride,
    }


def _mode_index(demo_agent, mode_name: str):
    modes = list(demo_agent.full_agent._clip_holder.CLIPS.keys())
    if mode_name not in modes:
        raise ValueError(f"Unknown mode '{mode_name}'. Available modes: {modes}")
    return modes.index(mode_name), modes


def _resolve_requested_mode(args, available_modes=None):
    available_modes = list(available_modes or [])
    requested_mode = args.mode
    effective_mode = requested_mode
    mode_preset = "native"

    if requested_mode in ("jog", "run") and requested_mode not in available_modes:
        if "walk" not in available_modes:
            raise ValueError(
                f"Requested --mode {requested_mode!r}, but this MotionBricks clip set has no "
                f"{requested_mode!r} or fallback 'walk' mode. Available modes: {available_modes}"
            )
        effective_mode = "walk"
        mode_preset = f"{requested_mode}_via_walk_velocity"
        if requested_mode == "run":
            args.target_vel = RUN_PRESET_TARGET_VEL if args.target_vel is None else args.target_vel
            args.target_lookahead_meters = (
                RUN_PRESET_LOOKAHEAD_METERS
                if args.target_lookahead_meters is None
                else args.target_lookahead_meters
            )
        else:
            args.target_vel = JOG_PRESET_TARGET_VEL if args.target_vel is None else args.target_vel
            args.target_lookahead_meters = (
                JOG_PRESET_LOOKAHEAD_METERS
                if args.target_lookahead_meters is None
                else args.target_lookahead_meters
            )
    else:
        args.target_vel = DEFAULT_TARGET_VEL if args.target_vel is None else args.target_vel
        args.target_lookahead_meters = (
            DEFAULT_TARGET_LOOKAHEAD_METERS if args.target_lookahead_meters is None else args.target_lookahead_meters
        )

    args.requested_mode = requested_mode
    args.effective_mode = effective_mode
    args.mode_preset = mode_preset
    args.mode = effective_mode


def _lookahead_target_qpos(qpos, target_qpos, args):
    import numpy as np

    lookahead = float(args.target_lookahead_meters)
    if lookahead <= 0:
        return target_qpos

    delta_xy = target_qpos[:2] - qpos[:2]
    dist = float(np.linalg.norm(delta_xy))
    if dist <= lookahead:
        return target_qpos

    clipped_target = target_qpos.copy()
    clipped_target[:2] = qpos[:2] + delta_xy / max(dist, 1e-6) * lookahead
    return clipped_target


def _target_control(
    demo_agent,
    args,
    qpos,
    target_qpos,
    mode_idx: int,
    *,
    target_vel: float | None = None,
    force_final_target: bool = False,
):
    import numpy as np
    import torch as t

    control_target_qpos = target_qpos if force_final_target else _lookahead_target_qpos(qpos, target_qpos, args)
    root_xy = qpos[:2]
    target_xy = control_target_qpos[:2]
    delta = target_xy - root_xy
    norm = np.linalg.norm(delta)
    if norm < 1e-5:
        movement = np.array([np.cos(args.target_yaw), np.sin(args.target_yaw), 0.0], dtype=np.float32)
    else:
        movement = np.array([delta[0] / norm, delta[1] / norm, 0.0], dtype=np.float32)

    facing = np.array([np.cos(args.target_yaw), np.sin(args.target_yaw), 0.0], dtype=np.float32)
    target_positions = np.repeat(
        control_target_qpos[None, :3],
        demo_agent.full_agent.NUM_FRAMES_PER_TOKEN,
        axis=0,
    )
    target_headings = np.repeat(args.target_yaw, demo_agent.full_agent.NUM_FRAMES_PER_TOKEN).astype(np.float32)

    control = {
        "movement_direction": t.from_numpy(movement).view(1, -1),
        "facing_direction": t.from_numpy(facing).view(1, -1),
        "mode": t.tensor([[mode_idx]], dtype=t.long),
        "specific_target_positions": t.from_numpy(target_positions).view(1, demo_agent.full_agent.NUM_FRAMES_PER_TOKEN, 3),
        "specific_target_headings": t.from_numpy(target_headings).view(1, demo_agent.full_agent.NUM_FRAMES_PER_TOKEN),
        "has_specific_target": t.tensor([[True]], dtype=t.bool),
    }
    control["allowed_pred_num_tokens"] = demo_agent.controller.get_default_allowed_pred_num_tokens(mode_idx)
    effective_target_vel = args.target_vel if target_vel is None else target_vel
    if effective_target_vel > 0:
        control["target_vel"] = t.tensor([[effective_target_vel]], dtype=t.float32)
    return control


def _set_start_context(demo_agent, start_qpos):
    import torch as t

    qpos = t.tensor(start_qpos, dtype=t.float32, device="cuda").view(1, 1, 36)
    qpos = qpos.repeat(1, 64, 1)
    demo_agent.full_agent.frames["mujoco_qpos"] = qpos
    demo_agent.full_agent._current_frame_idx = 0


def generate_to_target(demo_agent, args, target_qpos, start_qpos=None):
    import mujoco
    import numpy as np
    import torch as t

    np.random.seed(args.random_seed)
    t.manual_seed(args.random_seed)
    demo_agent.full_agent.reset()
    demo_agent.full_agent.BYPASS_SPRING_MODEL = bool(args.bypass_spring_model)

    if start_qpos is not None:
        _set_start_context(demo_agent, start_qpos)

    mode_idx, modes = _mode_index(demo_agent, args.mode)
    arrival_mode_idx = None
    if args.arrival_radius_meters > 0:
        arrival_mode_idx, _ = _mode_index(demo_agent, args.arrival_mode)

    frames = []
    arrival_active = False
    arrival_frames = 0

    for step in range(1, args.max_steps + 1):
        qpos = demo_agent.full_agent.get_next_frame()
        frames.append(np.asarray(qpos, dtype=np.float32).copy())
        root_err = float(np.linalg.norm(qpos[:2] - target_qpos[:2]))

        if (
            arrival_mode_idx is not None
            and not arrival_active
            and root_err <= args.arrival_radius_meters
        ):
            arrival_active = True
            print(
                f"arrival phase: step={step:05d} root_xy_err={root_err:.4f} m; "
                f"switching to mode={args.arrival_mode!r}, target_vel={args.arrival_target_vel}.",
                flush=True,
            )

        context_mujoco_qpos = demo_agent.full_agent.get_context_mujoco_qpos()
        demo_agent.mj_data.qpos[:] = qpos

        if arrival_active:
            arrival_frames += 1
            control_signals = _target_control(
                demo_agent,
                args,
                qpos,
                target_qpos,
                arrival_mode_idx,
                target_vel=args.arrival_target_vel,
                force_final_target=bool(args.arrival_force_final_target),
            )
        else:
            control_signals = _target_control(demo_agent, args, qpos, target_qpos, mode_idx)
        control_signals["context_mujoco_qpos"] = context_mujoco_qpos

        with t.no_grad():
            demo_agent.full_agent.generate_new_frames(
                control_signals,
                demo_agent.controller.get_controller_dt() * args.generate_dt,
                force_generation=(step == 1),
            )

        mujoco.mj_forward(demo_agent.mj_model, demo_agent.mj_data)

        if step % args.print_every == 0 or step == args.max_steps:
            dof_err = float(np.linalg.norm(qpos[7:] - target_qpos[7:]) / np.sqrt(29))
            print(f"step={step:05d} root_xy_err={root_err:.4f} dof_rmse={dof_err:.4f}", flush=True)

        if arrival_active and arrival_frames >= args.arrival_settle_frames:
            print(
                f"arrival phase complete: settle_frames={arrival_frames}, "
                f"root_xy_err={root_err:.4f} m.",
                flush=True,
            )
            break

    qpos_seq = np.stack(frames, axis=0)
    if args.append_target_hold > 0:
        hold = np.repeat(target_qpos[None, :], args.append_target_hold, axis=0).astype(np.float32)
        qpos_seq = np.concatenate([qpos_seq, hold], axis=0)

    return qpos_seq, modes


def export_qpos_sequence(args, output_dir: Path, qpos_seq, target_qpos, target_info, mode_name: str, modes):
    import joblib
    import numpy as np

    started = time.time()
    if args.zero_waist or args.zero_waist_pitch:
        qpos_seq = np.asarray(qpos_seq, dtype=np.float32).copy()
        target_qpos = np.asarray(target_qpos, dtype=np.float32).copy()
        _apply_waist_lock_qpos_inplace(qpos_seq, args)
        _apply_waist_lock_qpos_inplace(target_qpos, args)

    robot_dir = output_dir / "robot_filtered" / args.session
    qpos_dir = output_dir / "qpos"
    robot_dir.mkdir(parents=True, exist_ok=True)
    qpos_dir.mkdir(parents=True, exist_ok=True)

    target_stem = Path(args.target).stem if args.target else target_info["motion_key"]
    motion_name = args.motion_name or f"motionbricks_to_target_{target_stem}"
    entry = qpos_to_sonic_motion_entry(qpos_seq, fps=args.fps)

    pkl_path = robot_dir / f"{motion_name}.pkl"
    qpos_path = qpos_dir / f"{motion_name}.npy"
    npz_path = qpos_dir / f"{motion_name}.npz"
    joblib.dump({motion_name: entry}, pkl_path, compress=True)
    np.save(qpos_path, qpos_seq)
    np.savez_compressed(
        npz_path,
        qpos=qpos_seq,
        target_qpos=target_qpos,
        root_xyz=qpos_seq[:, :3],
        dof=qpos_seq[:, 7:],
        fps=np.asarray(args.fps, dtype=np.int32),
    )

    visualization = _write_trajectory_visualization(output_dir, motion_name, qpos_seq, target_qpos, args)

    final_generated = qpos_seq[-1]
    manifest = {
        "created_unix": started,
        "elapsed_sec": 0.0,
        "target": target_info,
        "mode": mode_name,
        "requested_mode": getattr(args, "requested_mode", mode_name),
        "effective_mode": getattr(args, "effective_mode", mode_name),
        "mode_preset": getattr(args, "mode_preset", "native"),
        "available_modes": modes,
        "target_vel": float(args.target_vel),
        "target_lookahead_meters": float(args.target_lookahead_meters),
        "arrival_radius_meters": float(args.arrival_radius_meters),
        "arrival_mode": args.arrival_mode,
        "arrival_target_vel": float(args.arrival_target_vel),
        "arrival_settle_frames": int(args.arrival_settle_frames),
        "arrival_force_final_target": bool(args.arrival_force_final_target),
        "fps": args.fps,
        "max_steps": args.max_steps,
        "append_target_hold": args.append_target_hold,
        "zero_waist": bool(args.zero_waist),
        "zero_waist_pitch": bool(args.zero_waist_pitch),
        "start_from_target_first": bool(args.start_from_target_first),
        "target_root_xyz": target_qpos[:3].round(6).tolist(),
        "target_yaw": float(args.target_yaw),
        "final_root_xyz": final_generated[:3].round(6).tolist(),
        "final_root_xy_error": float(np.linalg.norm(final_generated[:2] - target_qpos[:2])),
        "final_dof_rmse": float(np.linalg.norm(final_generated[7:] - target_qpos[7:]) / np.sqrt(29)),
        "qpos_summary": _summarize_qpos(qpos_seq),
        "robot_pkl": _display_path(pkl_path),
        "qpos_npy": _display_path(qpos_path),
        "qpos_npz": _display_path(npz_path),
        "visualization": visualization,
    }
    manifest_path = output_dir / "manifest.json"
    manifest["elapsed_sec"] = time.time() - started
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    export_label = "Target reference" if mode_name == "direct_reference" else "Target MotionBricks"
    print(f"\n{export_label} -> SONIC export complete.")
    print(f"target_root_xyz={manifest['target_root_xyz']} target_yaw={manifest['target_yaw']:.4f}")
    print(f"motion_file={_display_path(robot_dir)}")
    print("smpl_motion_file=dummy")
    print(f"robot_pkl={_display_path(pkl_path)}")
    print(f"qpos_npy={_display_path(qpos_path)}")
    print(f"qpos_npz={_display_path(npz_path)}")
    if visualization:
        print(f"trajectory_markers_xml={visualization['trajectory_markers_xml']}")
        print(f"trajectory_markers_json={visualization['trajectory_markers_json']}")
    print(f"manifest={_display_path(manifest_path)}")


def main(args: argparse.Namespace) -> None:
    import numpy as np

    generated_reference_qpos = None
    if args.target:
        target_qpos, target_info = load_target_qpos(
            args.target,
            args.target_frame,
            args.qpos_key,
            args.motion_key,
            args.root_quat_order,
        )
        generated_start_qpos = None
    else:
        if args.reference == "forward":
            target_qpos, generated_start_qpos, target_info = _make_forward_target_reference(args)
        elif args.reference == "standing_arm_swing":
            target_qpos, generated_start_qpos, target_info, generated_reference_qpos = _make_standing_arm_swing_reference(args)
        else:
            raise ValueError(f"Unsupported --reference: {args.reference}")

    if args.zero_waist or args.zero_waist_pitch:
        target_qpos = np.asarray(target_qpos, dtype=np.float32).copy()
        _apply_waist_lock_qpos_inplace(target_qpos, args)
        if generated_start_qpos is not None:
            generated_start_qpos = np.asarray(generated_start_qpos, dtype=np.float32).copy()
            _apply_waist_lock_qpos_inplace(generated_start_qpos, args)
        if generated_reference_qpos is not None:
            generated_reference_qpos = np.asarray(generated_reference_qpos, dtype=np.float32).copy()
            _apply_waist_lock_qpos_inplace(generated_reference_qpos, args)

    if args.target_only:
        print("Target reference export complete.")
        print(f"target_root_xyz={target_qpos[:3].round(6).tolist()}")
        if "npz_path" in target_info:
            print(f"target_npz={_display_path(Path(target_info['npz_path']))}")
        if "pkl_path" in target_info:
            print(f"target_pkl={_display_path(Path(target_info['pkl_path']))}")
        return

    args.target_yaw = yaw_from_wxyz(target_qpos[3:7]) if args.target_yaw is None else args.target_yaw
    output_dir = Path(args.output_dir).resolve()

    direct_reference_export = args.direct_reference_export == 1 or (
        args.direct_reference_export < 0 and generated_reference_qpos is not None
    )
    if direct_reference_export:
        _resolve_requested_mode(args, available_modes=[args.mode])
        if generated_reference_qpos is None:
            raise ValueError("--direct_reference_export=1 requires a generated multi-frame reference such as standing_arm_swing.")
        export_qpos_sequence(
            args,
            output_dir,
            generated_reference_qpos,
            target_qpos,
            target_info,
            mode_name="direct_reference",
            modes=[],
        )
        return

    import torch as t

    from motionbricks.motion_backbone.demo.utils import navigation_demo

    if not t.cuda.is_available():
        raise RuntimeError("MotionBricks inference expects CUDA; torch.cuda.is_available() is False.")

    start_qpos = None
    if args.start_from_target_first and args.target:
        start_qpos, _ = load_target_qpos(args.target, 0, args.qpos_key, args.motion_key, args.root_quat_order)
        if args.zero_waist or args.zero_waist_pitch:
            start_qpos = np.asarray(start_qpos, dtype=np.float32).copy()
            _apply_waist_lock_qpos_inplace(start_qpos, args)
    elif args.start_from_target_first and generated_start_qpos is not None:
        start_qpos = generated_start_qpos

    mb_args = _make_motionbricks_args(args)
    with _pushd(MOTIONBRICKS_ROOT):
        demo_agent = navigation_demo(mb_args)

    available_modes = list(demo_agent.full_agent._clip_holder.CLIPS.keys())
    _resolve_requested_mode(args, available_modes=available_modes)
    if args.requested_mode != args.effective_mode:
        print(
            f"Resolved --mode {args.requested_mode!r} to MotionBricks mode {args.effective_mode!r} "
            f"with preset {args.mode_preset!r} "
            f"(target_vel={args.target_vel}, lookahead={args.target_lookahead_meters}).",
            flush=True,
        )

    with _pushd(MOTIONBRICKS_ROOT):
        qpos_seq, modes = generate_to_target(demo_agent, args, target_qpos, start_qpos=start_qpos)
    export_qpos_sequence(args, output_dir, qpos_seq, target_qpos, target_info, args.mode, modes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a MotionBricks trajectory toward a target teleop frame and export SONIC motion_lib."
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Target .npz, .npy, or .pkl file. If omitted, --reference is generated.",
    )
    parser.add_argument("--target_frame", type=int, default=-1, help="Frame index to use as target.")
    parser.add_argument("--qpos_key", type=str, default=None, help="Explicit qpos key for npz/pkl files.")
    parser.add_argument("--motion_key", type=str, default=None, help="Explicit top-level motion key for PKL files.")
    parser.add_argument(
        "--root_quat_order",
        choices=["auto", "wxyz", "xyzw"],
        default="auto",
        help="Quaternion order for root_rot fields when reconstructing qpos.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(REPO_ROOT / "motion_sonic" / "motion" / "standing_arm_swing"),
    )
    parser.add_argument("--session", type=str, default="motionbricks_target")
    parser.add_argument("--motion_name", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--print_every", type=int, default=30)
    parser.add_argument("--append_target_hold", type=int, default=30)
    parser.add_argument("--start_from_target_first", type=int, default=0)

    parser.add_argument(
        "--reference",
        choices=["standing_arm_swing", "forward"],
        default="standing_arm_swing",
        help="Generated reference to use when --target is omitted.",
    )
    parser.add_argument(
        "--direct_reference_export",
        type=int,
        default=-1,
        help="1 exports a generated qpos sequence directly, 0 forces MotionBricks, -1 auto-enables direct export for standing_arm_swing.",
    )
    parser.add_argument("--forward_meters", type=float, default=5.0)
    parser.add_argument("--target_height", type=float, default=0.78)
    parser.add_argument("--forward_target_name", type=str, default="forward_5m_target")
    parser.add_argument("--standing_target_name", type=str, default="standing_arm_swing")
    parser.add_argument("--target_output_dir", type=str, default=None)
    parser.add_argument(
        "--target_dof",
        nargs=29,
        type=float,
        default=None,
        metavar="RAD",
        help="Optional 29-DOF final target pose. Defaults to all-zero neutral joints.",
    )
    parser.add_argument("--target_dof_order", choices=["mujoco", "isaaclab"], default="mujoco")
    parser.add_argument(
        "--zero_waist",
        type=int,
        default=0,
        help="Set waist yaw/roll/pitch to 0 in generated targets and exported qpos frames.",
    )
    parser.add_argument(
        "--zero_waist_pitch",
        type=int,
        default=0,
        help="Set only waist pitch to 0 while preserving waist yaw and roll.",
    )
    parser.add_argument(
        "--target_only",
        type=int,
        default=0,
        help="Write the generated target reference .npz/.pkl and exit before MotionBricks CUDA inference.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="slow_walk",
        help=(
            "MotionBricks mode. The public G1 clip set has no native run/jog mode; "
            "'run' and 'jog' are accepted as velocity presets that fall back to walk when needed."
        ),
    )
    parser.add_argument("--target_yaw", type=float, default=None)
    parser.add_argument(
        "--target_vel",
        type=float,
        default=None,
        help=f"Target velocity command. Defaults to {DEFAULT_TARGET_VEL}; --mode run/jog use faster presets.",
    )
    parser.add_argument(
        "--target_lookahead_meters",
        type=float,
        default=None,
        help=(
            "Give MotionBricks a moving local waypoint this far ahead instead of "
            f"the final target every step. Defaults to {DEFAULT_TARGET_LOOKAHEAD_METERS}; "
            "set 0 to condition directly on the final target."
        ),
    )
    parser.add_argument("--bypass_spring_model", type=int, default=0)
    parser.add_argument(
        "--arrival_radius_meters",
        type=float,
        default=0.0,
        help=(
            "When >0, switch from the requested locomotion mode to --arrival_mode once root XY is "
            "within this radius of the final target. Use this to stop at the destination instead "
            "of continuing to walk."
        ),
    )
    parser.add_argument(
        "--arrival_mode",
        type=str,
        default="idle",
        help="MotionBricks mode used after entering --arrival_radius_meters.",
    )
    parser.add_argument(
        "--arrival_target_vel",
        type=float,
        default=0.0,
        help="Target velocity command during arrival settle. 0 disables positive walking velocity.",
    )
    parser.add_argument(
        "--arrival_settle_frames",
        type=int,
        default=30,
        help="Number of generated frames to keep after entering arrival mode before stopping generation.",
    )
    parser.add_argument(
        "--arrival_force_final_target",
        type=int,
        default=1,
        help="1 feeds the final target directly during arrival instead of the lookahead waypoint.",
    )

    parser.add_argument("--arm_swing_seconds", type=float, default=6.0)
    parser.add_argument("--arm_swing_frequency", type=float, default=0.5)
    parser.add_argument("--arm_pitch_bias", type=float, default=0.0)
    parser.add_argument("--arm_pitch_amplitude", type=float, default=0.75)
    parser.add_argument("--arm_roll_bias", type=float, default=0.28)
    parser.add_argument("--arm_roll_amplitude", type=float, default=0.12)
    parser.add_argument("--arm_yaw_amplitude", type=float, default=0.18)
    parser.add_argument("--elbow_bias", type=float, default=0.35)
    parser.add_argument("--elbow_amplitude", type=float, default=0.55)
    parser.add_argument("--wrist_roll_amplitude", type=float, default=0.20)
    parser.add_argument("--wrist_yaw_amplitude", type=float, default=0.18)

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

    main(parser.parse_args())
