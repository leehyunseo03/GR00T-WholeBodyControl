#!/usr/bin/env python3
"""Plot MotionSonic tracking metrics for a 5 m GEAR-Sonic rollout.

This script compares:

* MotionBricks/KIMODO-style reference qpos and target reference frame.
* GEAR-Sonic rollout recording saved as first_episode_body_tracking.npz.
* Per-joint reference-vs-robot error when the recording contains joint traces.

The matching recorder/callback stores joint traces after this script was added.
Older recordings still work, but only scalar joint metrics can be plotted.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/motion_sonic_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOTION_NAME = "forward_5m_target"
DEFAULT_RECORDING = REPO_ROOT / "metrics" / "motionsonic" / DEFAULT_MOTION_NAME / "recording"
DEFAULT_MOTIONBRICKS_QPOS = (
    REPO_ROOT
    / "motion_sonic"
    / "motion"
    / DEFAULT_MOTION_NAME
    / "qpos"
    / "motionbricks_to_target_forward_5m_target.npz"
)
DEFAULT_TARGET = (
    REPO_ROOT
    / "motion_sonic"
    / "motion"
    / DEFAULT_MOTION_NAME
    / "target_reference"
    / "forward_5m_target.npz"
)
DEFAULT_OUT_DIR = REPO_ROOT / "metrics" / "motionsonic" / DEFAULT_MOTION_NAME / "plots"

UNIT_SCALES = {
    "m": (1.0, "m"),
    "cm": (100.0, "cm"),
    "mm": (1000.0, "mm"),
}

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


def repo_path(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if path.exists():
        return path
    text = str(path)
    prefix = "/workspace/GR00T-WholeBodyControl"
    if text.startswith(prefix):
        candidate = REPO_ROOT / text[len(prefix) + 1 :]
        if candidate.exists():
            return candidate
    return path


def resolve_recording(path_text: str | Path) -> Path:
    path = repo_path(path_text)
    if path.is_dir():
        path = path / "first_episode_body_tracking.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Recording not found: {path}. Run GEAR-Sonic with BodyTrackingCallback first."
        )
    return path


def load_npz(path: Path) -> dict[str, np.ndarray]:
    # Compatibility for npz object arrays written by newer NumPy.
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def load_recording(path: Path) -> dict[str, np.ndarray]:
    return load_npz(path)


def load_qpos(path_text: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    path = repo_path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"qpos file not found: {path}")
    if path.suffix == ".npy":
        qpos = np.load(path, allow_pickle=True)
        return np.asarray(qpos, dtype=float), {}
    if path.suffix == ".npz":
        data = load_npz(path)
        for key in ("qpos", "mujoco_qpos", "robot_qpos", "target_qpos"):
            if key in data and np.asarray(data[key]).ndim >= 2:
                return np.asarray(data[key], dtype=float), data
        raise KeyError(f"No qpos-like array found in {path}. Keys: {list(data)}")
    raise ValueError(f"Unsupported qpos extension: {path.suffix}")


def load_target_qpos(qpos_meta: dict[str, np.ndarray], target_path_text: str | Path | None) -> np.ndarray:
    if "target_qpos" in qpos_meta:
        return np.asarray(qpos_meta["target_qpos"], dtype=float).reshape(-1)[:36]
    if target_path_text is None:
        raise ValueError("No target_qpos in qpos file and --target was not provided.")

    target_path = repo_path(target_path_text)
    if not target_path.exists():
        raise FileNotFoundError(f"target file not found: {target_path}")
    if target_path.suffix == ".npz":
        data = load_npz(target_path)
        for key in ("target_qpos", "qpos", "mujoco_qpos", "robot_qpos"):
            if key in data:
                arr = np.asarray(data[key], dtype=float)
                return arr[-1].reshape(-1)[:36] if arr.ndim >= 2 else arr.reshape(-1)[:36]
    raise ValueError(f"Could not read target qpos from {target_path}")


def object_array_to_list(value: np.ndarray | None) -> list[str]:
    if value is None:
        return []
    return [str(item) for item in np.asarray(value, dtype=object).reshape(-1).tolist()]


def find_root_body_index(body_names: list[str], preferred: str | None) -> int:
    if preferred:
        if preferred not in body_names:
            raise ValueError(f"Root body '{preferred}' not found. Available: {body_names}")
        return body_names.index(preferred)
    for name in ("pelvis", "pelvis_link", "torso_link", "base_link", "root"):
        if name in body_names:
            return body_names.index(name)
    return 0


def maybe_metric(data: dict[str, np.ndarray], key: str) -> np.ndarray | None:
    value = data.get(key)
    if value is None or value.size == 0:
        return None
    return np.asarray(value, dtype=float).reshape(-1)


def frame_axis(data: dict[str, np.ndarray], length: int) -> np.ndarray:
    if "time_step" in data and len(data["time_step"]) >= length:
        return np.asarray(data["time_step"][:length], dtype=float)
    return np.arange(length, dtype=float)


def same_length(*arrays: np.ndarray) -> list[np.ndarray]:
    count = min(arr.shape[0] for arr in arrays if arr is not None)
    return [arr[:count] for arr in arrays]


def matched_motionbricks_roots(
    data: dict[str, np.ndarray], qpos: np.ndarray, root_idx: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map local MotionBricks qpos roots into the GEAR-Sonic recording world frame."""
    robot_body = np.asarray(data["robot_body_pos_w"], dtype=float)
    robot_root_all = robot_body[:, root_idx, :]
    qpos_root = np.asarray(qpos[:, :3], dtype=float)

    if "time_step" in data and len(data["time_step"]) >= len(robot_root_all):
        steps_int = np.asarray(data["time_step"][: len(robot_root_all)], dtype=int)
        valid = (steps_int >= 0) & (steps_int < len(qpos_root))
        if np.any(valid):
            steps = steps_int[valid].astype(float)
            mb_local = qpos_root[steps_int[valid]]
            robot_root = robot_root_all[valid]
            ref_body = np.asarray(data.get("ref_body_pos_w"), dtype=float) if "ref_body_pos_w" in data else None
            if ref_body is not None and ref_body.ndim == 3 and ref_body.shape[0] >= len(robot_root_all):
                anchor_root = ref_body[valid, root_idx, :]
            else:
                anchor_root = robot_root
            offset = anchor_root[0] - mb_local[0]
            return steps, mb_local + offset, robot_root, offset, mb_local

    count = min(len(qpos_root), len(robot_root_all))
    steps = np.arange(count, dtype=float)
    mb_local = qpos_root[:count]
    robot_root = robot_root_all[:count]
    ref_body = np.asarray(data.get("ref_body_pos_w"), dtype=float) if "ref_body_pos_w" in data else None
    if ref_body is not None and ref_body.ndim == 3 and ref_body.shape[0] >= count:
        anchor_root = ref_body[:count, root_idx, :]
    else:
        anchor_root = robot_root
    offset = anchor_root[0] - mb_local[0]
    return steps, mb_local + offset, robot_root, offset, mb_local


def motionbricks_world_offset(data: dict[str, np.ndarray], qpos: np.ndarray, root_idx: int) -> np.ndarray:
    _, _, _, offset, _ = matched_motionbricks_roots(data, qpos, root_idx)
    return offset


def plot_target_path(
    data: dict[str, np.ndarray],
    qpos: np.ndarray,
    target_qpos: np.ndarray,
    out_path: Path,
    root_idx: int,
    root_name: str,
    scale: float,
    unit: str,
) -> dict[str, float]:
    robot_body = np.asarray(data["robot_body_pos_w"], dtype=float)
    robot_root = robot_body[:, root_idx, :]
    offset = motionbricks_world_offset(data, qpos, root_idx)
    ref_root = qpos[:, :3] + offset[None, :]
    target = target_qpos[:3] + offset

    robot_xy_err = np.linalg.norm(robot_root[:, :2] - target[None, :2], axis=1)
    robot_xyz_err = np.linalg.norm(robot_root - target[None, :], axis=1)
    ref_xy_err = np.linalg.norm(ref_root[:, :2] - target[None, :2], axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    axes[0].plot(ref_root[:, 0], ref_root[:, 1], label="MotionBricks reference", linewidth=2.0)
    axes[0].plot(robot_root[:, 0], robot_root[:, 1], label="GEAR-Sonic robot", linewidth=1.7)
    axes[0].scatter([target[0]], [target[1]], marker="*", s=180, label="target reference frame")
    axes[0].scatter([robot_root[0, 0]], [robot_root[0, 1]], marker="o", s=55, label="robot start")
    axes[0].scatter([robot_root[-1, 0]], [robot_root[-1, 1]], marker="x", s=80, label="robot end")
    axes[0].set_title("XY Path: reference vs robot")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].axis("equal")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    t_robot = np.arange(len(robot_xy_err))
    t_ref = np.arange(len(ref_xy_err))
    axes[1].plot(t_robot, robot_xy_err * scale, label=f"robot {root_name} XY -> target")
    axes[1].plot(t_robot, robot_xyz_err * scale, label=f"robot {root_name} XYZ -> target", alpha=0.75)
    axes[1].plot(t_ref, ref_xy_err * scale, label="reference root XY -> target", alpha=0.75)
    axes[1].set_title("Target reference frame error")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel(unit)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    closest_idx = int(np.nanargmin(robot_xy_err)) if robot_xy_err.size else 0
    return {
        "robot_final_x": float(robot_root[-1, 0]),
        "robot_final_y": float(robot_root[-1, 1]),
        "robot_final_z": float(robot_root[-1, 2]),
        "robot_final_xy_error_m": float(robot_xy_err[-1]),
        "robot_final_xyz_error_m": float(robot_xyz_err[-1]),
        "robot_closest_xy_error_m": float(robot_xy_err[closest_idx]),
        "robot_closest_frame": float(closest_idx),
        "reference_final_xy_error_m": float(ref_xy_err[-1]),
    }


def plot_absolute_xy_by_step(
    data: dict[str, np.ndarray],
    qpos: np.ndarray,
    target_qpos: np.ndarray,
    out_path: Path,
    root_idx: int,
    root_name: str,
    scale: float,
    unit: str,
) -> None:
    """Plot absolute X and Y coordinates separately over rollout/reference step."""
    robot_body = np.asarray(data["robot_body_pos_w"], dtype=float)
    robot_root = robot_body[:, root_idx, :]
    offset = motionbricks_world_offset(data, qpos, root_idx)
    ref_root = np.asarray(qpos[:, :3], dtype=float) + offset[None, :]
    target = np.asarray(target_qpos[:3], dtype=float) + offset

    t_robot = frame_axis(data, len(robot_root))
    t_ref = np.arange(len(ref_root), dtype=float)
    t_max = max(float(t_robot[-1]) if len(t_robot) else 0.0, float(t_ref[-1]) if len(t_ref) else 0.0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
    coord_specs = [(0, "x"), (1, "y")]
    for ax, (axis_idx, axis_name) in zip(axes, coord_specs, strict=False):
        ax.plot(
            t_robot,
            robot_root[:, axis_idx] * scale,
            label=f"GEAR-Sonic actual {root_name}",
            linewidth=1.8,
        )
        ax.plot(
            t_ref,
            ref_root[:, axis_idx] * scale,
            label="MotionBricks reference root",
            linewidth=1.5,
            alpha=0.8,
        )
        ax.hlines(
            target[axis_idx] * scale,
            xmin=0.0,
            xmax=t_max,
            colors="tab:red",
            linestyles="--",
            linewidth=1.8,
            label="target reference frame",
        )
        ax.set_title(f"Absolute {axis_name.upper()} coordinate vs step")
        ax.set_xlabel("step")
        ax.set_ylabel(f"{axis_name} ({unit})")
        ax.grid(alpha=0.25)
        ax.legend()

    fig.suptitle("Actual body position vs target absolute coordinates")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_absolute_x_zoom_4000_5000_mm(
    data: dict[str, np.ndarray],
    qpos: np.ndarray,
    target_qpos: np.ndarray,
    out_path: Path,
    root_idx: int,
    root_name: str,
    scale: float,
    unit: str,
) -> None:
    """Plot absolute X coordinate with the y-axis zoomed to 4000-5000 mm."""
    robot_body = np.asarray(data["robot_body_pos_w"], dtype=float)
    robot_root = robot_body[:, root_idx, :]
    offset = motionbricks_world_offset(data, qpos, root_idx)
    ref_root = np.asarray(qpos[:, :3], dtype=float) + offset[None, :]
    target = np.asarray(target_qpos[:3], dtype=float) + offset

    t_robot = frame_axis(data, len(robot_root))
    t_ref = np.arange(len(ref_root), dtype=float)
    t_max = max(float(t_robot[-1]) if len(t_robot) else 0.0, float(t_ref[-1]) if len(t_ref) else 0.0)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(
        t_robot,
        robot_root[:, 0] * scale,
        label=f"GEAR-Sonic actual {root_name}",
        linewidth=1.8,
    )
    ax.plot(
        t_ref,
        ref_root[:, 0] * scale,
        label="MotionBricks reference root",
        linewidth=1.5,
        alpha=0.8,
    )
    ax.hlines(
        target[0] * scale,
        xmin=0.0,
        xmax=t_max,
        colors="tab:red",
        linestyles="--",
        linewidth=1.8,
        label="target reference frame",
    )
    ax.set_title("Absolute X coordinate vs step (4000-5000 mm zoom)")
    ax.set_xlabel("step")
    ax.set_ylabel(f"x ({unit})")
    ax.set_ylim(4000.0, 5000.0)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_motionbricks_vs_gearsonic_absolute_xy(
    data: dict[str, np.ndarray],
    qpos: np.ndarray,
    out_path: Path,
    root_idx: int,
    root_name: str,
    scale: float,
    unit: str,
) -> dict[str, float]:
    """Compare MotionBricks dummy root and GEAR-Sonic actual body on matched steps."""
    steps, mb_root, robot_root, offset, mb_local = matched_motionbricks_roots(data, qpos, root_idx)
    delta = robot_root - mb_root
    xy_error = np.linalg.norm(delta[:, :2], axis=1)
    xyz_error = np.linalg.norm(delta, axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for axis_idx, axis_name in ((0, "x"), (1, "y")):
        ax = axes[axis_idx]
        ax.plot(
            steps,
            mb_root[:, axis_idx] * scale,
            label="MotionBricks G1 dummy root (world-aligned)",
            linewidth=2.0,
        )
        ax.plot(
            steps,
            robot_root[:, axis_idx] * scale,
            label=f"GEAR-Sonic actual {root_name}",
            linewidth=1.7,
        )
        ax.fill_between(
            steps,
            mb_root[:, axis_idx] * scale,
            robot_root[:, axis_idx] * scale,
            alpha=0.18,
            label=f"{axis_name} error area",
        )
        ax.set_title(f"Matched absolute {axis_name.upper()} position")
        ax.set_ylabel(f"{axis_name} ({unit})")
        ax.grid(alpha=0.25)
        ax.legend()

    axes[2].plot(steps, xy_error * scale, label="XY error", linewidth=2.0)
    axes[2].plot(steps, xyz_error * scale, label="XYZ error", linewidth=1.4, alpha=0.75)
    axes[2].set_title("Position error: GEAR-Sonic actual - MotionBricks dummy")
    axes[2].set_xlabel("motion time step")
    axes[2].set_ylabel(unit)
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    fig.suptitle("MotionBricks dummy vs GEAR-Sonic actual absolute position")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return {
        "matched_position_frames": int(len(steps)),
        "motionbricks_world_offset_x": float(offset[0]),
        "motionbricks_world_offset_y": float(offset[1]),
        "motionbricks_world_offset_z": float(offset[2]),
        "raw_initial_motionbricks_x": float(mb_local[0, 0]),
        "raw_initial_motionbricks_y": float(mb_local[0, 1]),
        "raw_initial_gearsonic_x": float(robot_root[0, 0]),
        "raw_initial_gearsonic_y": float(robot_root[0, 1]),
        "motionbricks_vs_gearsonic_final_x_error_m": float(delta[-1, 0]),
        "motionbricks_vs_gearsonic_final_y_error_m": float(delta[-1, 1]),
        "motionbricks_vs_gearsonic_final_xy_error_m": float(xy_error[-1]),
        "motionbricks_vs_gearsonic_mean_xy_error_m": float(np.nanmean(xy_error)),
        "motionbricks_vs_gearsonic_max_xy_error_m": float(np.nanmax(xy_error)),
    }


def plot_body_tracking_timeseries(
    data: dict[str, np.ndarray], out_path: Path, scale: float, unit: str
) -> None:
    body_mean = np.asarray(data["body_error_mean"], dtype=float)
    body_max = np.asarray(data["body_error_max"], dtype=float)
    anchor = np.asarray(data["anchor_pos_error"], dtype=float)
    t = frame_axis(data, len(body_mean))

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(t, body_mean * scale, label="mean body position error")
    ax.plot(t, body_max * scale, label="max body position error", alpha=0.8)
    ax.plot(t, anchor * scale, label="anchor position error", alpha=0.8)
    ax.set_title("GEAR-Sonic tracking error")
    ax.set_xlabel("motion time step")
    ax.set_ylabel(unit)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_joint_scalar_fallback(data: dict[str, np.ndarray], out_path: Path) -> bool:
    values = maybe_metric(data, "command_metric__error_joint_pos")
    if values is None:
        return False
    t = frame_axis(data, len(values))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(t, values, label="mean abs joint position error")
    ax.set_title("Scalar joint error from command metrics")
    ax.set_xlabel("motion time step")
    ax.set_ylabel("rad")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def plot_joint_heatmap(joint_error: np.ndarray, joint_names: list[str], out_path: Path) -> None:
    error_deg = np.abs(joint_error) * 180.0 / np.pi
    vmax = float(np.nanpercentile(error_deg, 95)) if error_deg.size else 1.0

    fig, ax = plt.subplots(figsize=(13, max(7, len(joint_names) * 0.28)))
    image = ax.imshow(
        error_deg.T,
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=0.0,
        vmax=max(vmax, 1e-6),
    )
    ax.set_title("Per-joint absolute tracking error")
    ax.set_xlabel("frame")
    ax.set_ylabel("joint")
    ax.set_yticks(np.arange(len(joint_names)))
    ax.set_yticklabels(joint_names)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("deg")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_joint_bar(joint_error: np.ndarray, joint_names: list[str], out_path: Path) -> None:
    error_deg = np.abs(joint_error) * 180.0 / np.pi
    mean = np.nanmean(error_deg, axis=0)
    p95 = np.nanpercentile(error_deg, 95, axis=0)
    order = np.argsort(mean)

    fig, ax = plt.subplots(figsize=(10, max(7, len(joint_names) * 0.32)))
    y = np.arange(len(order))
    ax.barh(y, mean[order], label="mean abs")
    ax.scatter(p95[order], y, color="tab:red", s=22, label="p95 abs")
    ax.set_yticks(y)
    ax.set_yticklabels([joint_names[idx] for idx in order])
    ax.set_xlabel("deg")
    ax.set_title("Per-joint tracking error summary")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_joint_timeseries(joint_error: np.ndarray, joint_names: list[str], out_path: Path) -> None:
    error_deg = joint_error * 180.0 / np.pi
    joint_count = len(joint_names)
    cols = 3
    rows = int(np.ceil(joint_count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, max(6, rows * 2.15)), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    t = np.arange(error_deg.shape[0])

    for idx, ax in enumerate(axes[:joint_count]):
        values = error_deg[:, idx]
        ax.plot(t, values, linewidth=1.0)
        ax.axhline(0.0, color="black", linewidth=0.7)
        ax.set_title(joint_names[idx], fontsize=9)
        ax.grid(alpha=0.25)
    for ax in axes[joint_count:]:
        ax.axis("off")
    fig.suptitle("Signed joint position error: robot - reference (deg)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_joint_stats(joint_error: np.ndarray, joint_names: list[str], out_path: Path) -> dict[str, float]:
    abs_rad = np.abs(joint_error)
    abs_deg = abs_rad * 180.0 / np.pi
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "joint",
                "mean_abs_rad",
                "p50_abs_rad",
                "p95_abs_rad",
                "max_abs_rad",
                "mean_abs_deg",
                "p50_abs_deg",
                "p95_abs_deg",
                "max_abs_deg",
            ],
        )
        writer.writeheader()
        for idx, name in enumerate(joint_names):
            writer.writerow(
                {
                    "joint": name,
                    "mean_abs_rad": f"{np.nanmean(abs_rad[:, idx]):.8f}",
                    "p50_abs_rad": f"{np.nanpercentile(abs_rad[:, idx], 50):.8f}",
                    "p95_abs_rad": f"{np.nanpercentile(abs_rad[:, idx], 95):.8f}",
                    "max_abs_rad": f"{np.nanmax(abs_rad[:, idx]):.8f}",
                    "mean_abs_deg": f"{np.nanmean(abs_deg[:, idx]):.8f}",
                    "p50_abs_deg": f"{np.nanpercentile(abs_deg[:, idx], 50):.8f}",
                    "p95_abs_deg": f"{np.nanpercentile(abs_deg[:, idx], 95):.8f}",
                    "max_abs_deg": f"{np.nanmax(abs_deg[:, idx]):.8f}",
                }
            )

    return {
        "joint_mean_abs_error_rad": float(np.nanmean(abs_rad)),
        "joint_p95_abs_error_rad": float(np.nanpercentile(abs_rad, 95)),
        "joint_max_abs_error_rad": float(np.nanmax(abs_rad)),
        "joint_mean_abs_error_deg": float(np.nanmean(abs_deg)),
        "joint_p95_abs_error_deg": float(np.nanpercentile(abs_deg, 95)),
        "joint_max_abs_error_deg": float(np.nanmax(abs_deg)),
    }


def plot_motionbricks_qpos_reference(qpos: np.ndarray, out_path: Path) -> None:
    dof = qpos[:, 7 : 7 + len(MUJOCO_DOF_NAMES)]
    if dof.shape[1] == 0:
        return
    cols = 3
    rows = int(np.ceil(dof.shape[1] / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, max(6, rows * 2.15)), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    t = np.arange(dof.shape[0])
    for idx, ax in enumerate(axes[: dof.shape[1]]):
        ax.plot(t, dof[:, idx], linewidth=1.0)
        ax.set_title(MUJOCO_DOF_NAMES[idx], fontsize=9)
        ax.grid(alpha=0.25)
    for ax in axes[dof.shape[1] :]:
        ax.axis("off")
    fig.suptitle("MotionBricks reference 29DOF qpos (MuJoCo order)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_29dof_qpos_reference_vs_robot(
    data: dict[str, np.ndarray], qpos: np.ndarray, out_path: Path
) -> bool:
    ref_joint = data.get("ref_joint_pos")
    robot_joint = data.get("robot_joint_pos")
    if ref_joint is None or robot_joint is None:
        plot_motionbricks_qpos_reference(qpos, out_path)
        return False

    ref_joint = np.asarray(ref_joint, dtype=float)
    robot_joint = np.asarray(robot_joint, dtype=float)
    if ref_joint.ndim != 2 or robot_joint.ndim != 2 or ref_joint.size == 0 or robot_joint.size == 0:
        plot_motionbricks_qpos_reference(qpos, out_path)
        return False

    frame_count = min(ref_joint.shape[0], robot_joint.shape[0])
    joint_count = min(ref_joint.shape[1], robot_joint.shape[1], len(MUJOCO_DOF_NAMES))
    ref_joint = ref_joint[:frame_count, :joint_count]
    robot_joint = robot_joint[:frame_count, :joint_count]

    joint_names = object_array_to_list(data.get("joint_names"))
    if not joint_names:
        joint_names = MUJOCO_DOF_NAMES
    joint_names = joint_names[:joint_count]

    cols = 3
    rows = int(np.ceil(joint_count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, max(6, rows * 2.25)), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    t = frame_axis(data, frame_count)

    for idx, ax in enumerate(axes[:joint_count]):
        ax.plot(t, ref_joint[:, idx], label="reference", linewidth=1.15)
        ax.plot(t, robot_joint[:, idx], label="actual robot", linewidth=1.0, alpha=0.82)
        ax.set_title(joint_names[idx], fontsize=9)
        ax.grid(alpha=0.25)
        if idx % cols == 0:
            ax.set_ylabel("qpos (rad)")
        if idx == 0:
            ax.legend(fontsize=8)
    for ax in axes[joint_count:]:
        ax.axis("off")
    for ax in axes[-cols:]:
        if ax.has_data():
            ax.set_xlabel("motion time step")

    fig.suptitle("29DOF qpos: reference vs actual robot")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def save_summary(summary: dict[str, Any], out_path: Path) -> None:
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recording",
        default=str(DEFAULT_RECORDING),
        help="first_episode_body_tracking.npz or its directory.",
    )
    parser.add_argument(
        "--motionbricks-qpos",
        default=str(DEFAULT_MOTIONBRICKS_QPOS),
        help="MotionBricks/KIMODO reference qpos .npz/.npy.",
    )
    parser.add_argument(
        "--target",
        default=str(DEFAULT_TARGET),
        help="Target reference .npz. Used only if qpos file has no target_qpos.",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for plots.")
    parser.add_argument("--unit", choices=sorted(UNIT_SCALES), default="mm")
    parser.add_argument(
        "--root-body",
        default=None,
        help="Body name used as GEAR-Sonic robot root for target-frame error. Defaults to pelvis/torso.",
    )
    parser.add_argument(
        "--plot-reference-dof",
        action="store_true",
        help="Also plot 29DOF qpos traces. Overlays recording reference and actual robot when available.",
    )
    args = parser.parse_args()

    recording_path = resolve_recording(args.recording)
    qpos, qpos_meta = load_qpos(args.motionbricks_qpos)
    target_qpos = load_target_qpos(qpos_meta, args.target)
    data = load_recording(recording_path)
    out_dir = repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scale, unit = UNIT_SCALES[args.unit]

    body_names = object_array_to_list(data.get("body_names"))
    if not body_names:
        raise KeyError("Recording does not contain body_names.")
    root_idx = find_root_body_index(body_names, args.root_body)
    root_name = body_names[root_idx]

    summary: dict[str, Any] = {
        "recording": str(recording_path),
        "motionbricks_qpos": str(repo_path(args.motionbricks_qpos)),
        "target": str(repo_path(args.target)),
        "num_recorded_frames": int(len(data["body_error_mean"])),
        "num_reference_frames": int(len(qpos)),
        "root_body": root_name,
        "target_x": float(target_qpos[0]),
        "target_y": float(target_qpos[1]),
        "target_z": float(target_qpos[2]),
        "body_mean_error_m": float(np.nanmean(np.asarray(data["body_error_mean"], dtype=float))),
        "body_max_error_m": float(np.nanmax(np.asarray(data["body_error_max"], dtype=float))),
        "anchor_mean_error_m": float(np.nanmean(np.asarray(data["anchor_pos_error"], dtype=float))),
    }

    target_summary = plot_target_path(
        data,
        qpos,
        target_qpos,
        out_dir / f"target_frame_vs_robot_path_{unit}.png",
        root_idx,
        root_name,
        scale,
        unit,
    )
    summary.update(target_summary)
    plot_absolute_xy_by_step(
        data,
        qpos,
        target_qpos,
        out_dir / f"absolute_xy_by_step_{unit}.png",
        root_idx,
        root_name,
        scale,
        unit,
    )
    plot_absolute_x_zoom_4000_5000_mm(
        data,
        qpos,
        target_qpos,
        out_dir / f"absolute_x_zoom_4000_5000_{unit}.png",
        root_idx,
        root_name,
        scale,
        unit,
    )
    summary.update(
        plot_motionbricks_vs_gearsonic_absolute_xy(
            data,
            qpos,
            out_dir / f"motionbricks_dummy_vs_gearsonic_absolute_xy_{unit}.png",
            root_idx,
            root_name,
            scale,
            unit,
        )
    )
    plot_body_tracking_timeseries(data, out_dir / f"gearsonic_tracking_timeseries_{unit}.png", scale, unit)

    joint_error = data.get("joint_error")
    joint_names = object_array_to_list(data.get("joint_names"))
    has_joint_traces = (
        joint_error is not None
        and np.asarray(joint_error).ndim == 2
        and np.asarray(joint_error).shape[0] > 0
        and np.asarray(joint_error).shape[1] > 0
    )
    summary["has_per_joint_traces"] = bool(has_joint_traces)
    if has_joint_traces:
        joint_error = np.asarray(joint_error, dtype=float)
        if not joint_names:
            joint_names = [f"joint_{idx:02d}" for idx in range(joint_error.shape[1])]
        joint_names = joint_names[: joint_error.shape[1]]
        plot_joint_heatmap(joint_error, joint_names, out_dir / "per_joint_error_heatmap_deg.png")
        plot_joint_bar(joint_error, joint_names, out_dir / "per_joint_error_bar_deg.png")
        plot_joint_timeseries(joint_error, joint_names, out_dir / "per_joint_error_timeseries_deg.png")
        summary.update(write_joint_stats(joint_error, joint_names, out_dir / "per_joint_error_stats.csv"))
    else:
        plotted_scalar = plot_joint_scalar_fallback(data, out_dir / "joint_error_scalar_timeseries_rad.png")
        summary["joint_trace_note"] = (
            "This recording has no ref_joint_pos/robot_joint_pos arrays. "
            "Re-run eval with the updated BodyTrackingCallback to get per-joint plots."
        )
        summary["has_scalar_joint_metric"] = bool(plotted_scalar)

    if args.plot_reference_dof:
        summary["qpos_reference_vs_robot_plotted"] = plot_29dof_qpos_reference_vs_robot(
            data,
            qpos,
            out_dir / "motionbricks_reference_29dof_qpos.png",
        )

    save_summary(summary, out_dir / "summary.json")

    print(f"Wrote MotionSonic plots to: {out_dir}")
    print(f"Frames: recorded={summary['num_recorded_frames']} reference={summary['num_reference_frames']}")
    print(
        "Target error: "
        f"final_xy={summary['robot_final_xy_error_m'] * scale:.3f}{unit}, "
        f"closest_xy={summary['robot_closest_xy_error_m'] * scale:.3f}{unit}"
    )
    if summary["has_per_joint_traces"]:
        print(
            "Joint error: "
            f"mean={summary['joint_mean_abs_error_deg']:.3f}deg, "
            f"p95={summary['joint_p95_abs_error_deg']:.3f}deg"
        )
    else:
        print(summary["joint_trace_note"])


if __name__ == "__main__":
    main()
