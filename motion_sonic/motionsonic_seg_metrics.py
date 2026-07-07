#!/usr/bin/env python3
"""Plot MotionSonic metrics for live segmented MotionBricks evaluation.

This has the same CLI shape as ``motionsonic_metrics.py`` but treats
``ref_body_pos_w`` and ``ref_joint_pos`` in the BodyTrackingCallback recording as
the primary reference. That matters for receding-horizon MotionBricks eval,
because the active SONIC motion-lib tensors are replaced during rollout.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/motion_sonic_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from motion_sonic import motionsonic_metrics as base  # noqa: E402


def segment_boundaries(data: dict[str, np.ndarray]) -> np.ndarray:
    """Return frame indexes where the live reference appears to restart."""
    time_step = np.asarray(data.get("time_step", []), dtype=float).reshape(-1)
    if time_step.size <= 1:
        return np.asarray([], dtype=int)
    jumps = np.where(np.diff(time_step) <= 0)[0] + 1
    return jumps.astype(int)


def frame_numbers(length: int) -> np.ndarray:
    return np.arange(length, dtype=float)


def live_roots(
    data: dict[str, np.ndarray], root_idx: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref_body = np.asarray(data["ref_body_pos_w"], dtype=float)
    robot_body = np.asarray(data["robot_body_pos_w"], dtype=float)
    count = min(ref_body.shape[0], robot_body.shape[0])
    return frame_numbers(count), ref_body[:count, root_idx, :], robot_body[:count, root_idx, :]


def draw_segment_lines(ax, boundaries: np.ndarray) -> None:
    for idx in boundaries:
        ax.axvline(float(idx), color="0.25", linewidth=0.7, alpha=0.18)


def plot_live_reference_path(
    data: dict[str, np.ndarray],
    target_qpos: np.ndarray,
    out_path: Path,
    root_idx: int,
    root_name: str,
    scale: float,
    unit: str,
) -> dict[str, float]:
    _, ref_root, robot_root = live_roots(data, root_idx)
    target_xy = np.asarray(target_qpos[:2], dtype=float)
    boundaries = segment_boundaries(data)

    delta = robot_root - ref_root
    xy_error = np.linalg.norm(delta[:, :2], axis=1)
    xyz_error = np.linalg.norm(delta, axis=1)
    robot_target_xy = np.linalg.norm(robot_root[:, :2] - target_xy[None, :], axis=1)
    ref_target_xy = np.linalg.norm(ref_root[:, :2] - target_xy[None, :], axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].plot(ref_root[:, 0], ref_root[:, 1], label="live segmented reference", linewidth=2.0)
    axes[0].plot(robot_root[:, 0], robot_root[:, 1], label=f"GEAR-Sonic {root_name}", linewidth=1.7)
    axes[0].scatter([target_xy[0]], [target_xy[1]], marker="*", s=180, label="final target")
    axes[0].scatter([robot_root[0, 0]], [robot_root[0, 1]], marker="o", s=55, label="robot start")
    axes[0].scatter([robot_root[-1, 0]], [robot_root[-1, 1]], marker="x", s=80, label="robot end")
    axes[0].set_title("XY path: live reference vs robot")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].axis("equal")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    t = frame_numbers(len(xy_error))
    axes[1].plot(t, xy_error * scale, label="robot - live reference XY")
    axes[1].plot(t, xyz_error * scale, label="robot - live reference XYZ", alpha=0.75)
    axes[1].plot(t, robot_target_xy * scale, label="robot XY -> final target", alpha=0.8)
    axes[1].plot(t, ref_target_xy * scale, label="live reference XY -> final target", alpha=0.65)
    draw_segment_lines(axes[1], boundaries)
    axes[1].set_title("Segmented tracking and target error")
    axes[1].set_xlabel("recorded frame")
    axes[1].set_ylabel(unit)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    closest_idx = int(np.nanargmin(robot_target_xy)) if robot_target_xy.size else 0
    return {
        "segmented_reference_frames": int(len(ref_root)),
        "segmented_replan_count": int(len(boundaries)),
        "segmented_live_ref_mean_xy_error_m": float(np.nanmean(xy_error)),
        "segmented_live_ref_p95_xy_error_m": float(np.nanpercentile(xy_error, 95)),
        "segmented_live_ref_max_xy_error_m": float(np.nanmax(xy_error)),
        "segmented_live_ref_final_xy_error_m": float(xy_error[-1]),
        "robot_final_target_xy_error_m": float(robot_target_xy[-1]),
        "robot_closest_target_xy_error_m": float(robot_target_xy[closest_idx]),
        "robot_closest_target_frame": int(closest_idx),
        "live_ref_final_target_xy_error_m": float(ref_target_xy[-1]),
        "robot_final_x": float(robot_root[-1, 0]),
        "robot_final_y": float(robot_root[-1, 1]),
        "robot_final_z": float(robot_root[-1, 2]),
        "live_ref_final_x": float(ref_root[-1, 0]),
        "live_ref_final_y": float(ref_root[-1, 1]),
        "live_ref_final_z": float(ref_root[-1, 2]),
    }


def plot_live_absolute_xy(
    data: dict[str, np.ndarray],
    out_path: Path,
    root_idx: int,
    root_name: str,
    scale: float,
    unit: str,
) -> None:
    t, ref_root, robot_root = live_roots(data, root_idx)
    boundaries = segment_boundaries(data)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, axis_idx, label in ((axes[0], 0, "x"), (axes[1], 1, "y")):
        ax.plot(t, ref_root[:, axis_idx] * scale, label="live segmented reference", linewidth=1.8)
        ax.plot(t, robot_root[:, axis_idx] * scale, label=f"GEAR-Sonic {root_name}", linewidth=1.5)
        draw_segment_lines(ax, boundaries)
        ax.set_title(f"Absolute {label.upper()} coordinate")
        ax.set_ylabel(f"{label} ({unit})")
        ax.grid(alpha=0.25)
        ax.legend()

    xy_error = np.linalg.norm(robot_root[:, :2] - ref_root[:, :2], axis=1)
    axes[2].plot(t, xy_error * scale, label="XY tracking error", linewidth=1.8)
    draw_segment_lines(axes[2], boundaries)
    axes[2].set_title("Live reference tracking error")
    axes[2].set_xlabel("recorded frame")
    axes[2].set_ylabel(unit)
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_live_absolute_x_zoom_4000_5000_mm(
    data: dict[str, np.ndarray],
    out_path: Path,
    root_idx: int,
    root_name: str,
    scale: float,
    unit: str,
) -> None:
    """Plot the live absolute X coordinate with y-axis fixed to 4000-5000 mm."""
    t, ref_root, robot_root = live_roots(data, root_idx)
    boundaries = segment_boundaries(data)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(t, ref_root[:, 0] * scale, label="live segmented reference", linewidth=1.8)
    ax.plot(t, robot_root[:, 0] * scale, label=f"GEAR-Sonic {root_name}", linewidth=1.5)
    draw_segment_lines(ax, boundaries)
    ax.set_title("Absolute X coordinate (4000-5000 mm zoom)")
    ax.set_xlabel("recorded frame")
    ax.set_ylabel(f"x ({unit})")
    ax.set_ylim(4000.0, 5000.0)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_live_x_tail_zoom(
    data: dict[str, np.ndarray],
    target_qpos: np.ndarray,
    out_path: Path,
    root_idx: int,
    root_name: str,
    scale: float,
    unit: str,
    tail_frames: int,
) -> dict[str, float]:
    t, ref_root, robot_root = live_roots(data, root_idx)
    target_x = float(target_qpos[0])
    start = max(0, len(t) - max(1, int(tail_frames)))
    t_tail = t[start:]
    ref_x = ref_root[start:, 0]
    robot_x = robot_root[start:, 0]
    boundaries = segment_boundaries(data)
    boundaries = boundaries[boundaries >= start] - start

    final_signed_x_error = target_x - float(robot_root[-1, 0])
    final_abs_x_error = abs(final_signed_x_error)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(t_tail, ref_x * scale, label="live segmented reference x", linewidth=1.9)
    ax.plot(t_tail, robot_x * scale, label=f"GEAR-Sonic {root_name} x", linewidth=1.6)
    ax.hlines(
        target_x * scale,
        xmin=float(t_tail[0]),
        xmax=float(t_tail[-1]),
        colors="tab:red",
        linestyles="--",
        linewidth=1.5,
        label=f"target x = {target_x * scale:.1f}{unit}",
    )
    for idx in boundaries:
        ax.axvline(float(t_tail[0] + idx), color="0.25", linewidth=0.7, alpha=0.18)
    ax.scatter([t_tail[-1]], [robot_x[-1] * scale], marker="x", s=75, label="robot final x")
    ax.set_title(f"Final X coordinate zoom: last {len(t_tail)} frames")
    ax.set_xlabel("recorded frame")
    ax.set_ylabel(f"x ({unit})")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return {
        "tail_zoom_frames": int(len(t_tail)),
        "robot_final_target_x_error_m": float(final_signed_x_error),
        "robot_final_target_abs_x_error_m": float(final_abs_x_error),
        "robot_final_target_x_error_mm": float(final_signed_x_error * 1000.0),
        "robot_final_target_abs_x_error_mm": float(final_abs_x_error * 1000.0),
    }


def plot_segment_time_steps(data: dict[str, np.ndarray], out_path: Path) -> None:
    time_step = np.asarray(data.get("time_step", []), dtype=float).reshape(-1)
    if time_step.size == 0:
        return
    boundaries = segment_boundaries(data)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(frame_numbers(len(time_step)), time_step, linewidth=1.5)
    draw_segment_lines(ax, boundaries)
    ax.set_title("Recorded motion time step resets")
    ax.set_xlabel("recorded frame")
    ax.set_ylabel("motion time_step")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_segment_table(data: dict[str, np.ndarray], out_path: Path) -> None:
    boundaries = segment_boundaries(data)
    starts = np.concatenate(([0], boundaries))
    frame_count = len(np.asarray(data.get("time_step", [])))
    stops = np.concatenate((boundaries, [frame_count]))
    body_mean = np.asarray(data["body_error_mean"], dtype=float)
    anchor = np.asarray(data["anchor_pos_error"], dtype=float)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "segment_index",
                "start_frame",
                "end_frame_exclusive",
                "num_frames",
                "body_mean_error_m",
                "anchor_mean_error_m",
            ],
        )
        writer.writeheader()
        for seg_idx, (start, stop) in enumerate(zip(starts, stops, strict=False)):
            if stop <= start:
                continue
            writer.writerow(
                {
                    "segment_index": seg_idx,
                    "start_frame": int(start),
                    "end_frame_exclusive": int(stop),
                    "num_frames": int(stop - start),
                    "body_mean_error_m": f"{np.nanmean(body_mean[start:stop]):.8f}",
                    "anchor_mean_error_m": f"{np.nanmean(anchor[start:stop]):.8f}",
                }
            )


def save_summary(summary: dict[str, Any], out_path: Path) -> None:
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recording",
        default=str(base.DEFAULT_RECORDING),
        help="first_episode_body_tracking.npz or its directory.",
    )
    parser.add_argument(
        "--motionbricks-qpos",
        default=str(base.DEFAULT_MOTIONBRICKS_QPOS),
        help="Original/offline MotionBricks qpos .npz/.npy, used for target metadata and comparison plots.",
    )
    parser.add_argument(
        "--target",
        default=str(base.DEFAULT_TARGET),
        help="Target reference .npz. Used only if qpos file has no target_qpos.",
    )
    parser.add_argument("--out-dir", default=str(base.DEFAULT_OUT_DIR), help="Output directory for plots.")
    parser.add_argument("--unit", choices=sorted(base.UNIT_SCALES), default="mm")
    parser.add_argument(
        "--root-body",
        default=None,
        help="Body name used as GEAR-Sonic robot root. Defaults to pelvis/torso.",
    )
    parser.add_argument(
        "--plot-reference-dof",
        action="store_true",
        help="Also plot 29DOF qpos traces from the recording reference and actual robot.",
    )
    parser.add_argument(
        "--tail-zoom-frames",
        type=int,
        default=300,
        help="Number of final recorded frames to show in the X-coordinate zoom plot.",
    )
    args = parser.parse_args()

    recording_path = base.resolve_recording(args.recording)
    qpos, qpos_meta = base.load_qpos(args.motionbricks_qpos)
    target_qpos = base.load_target_qpos(qpos_meta, args.target)
    data = base.load_recording(recording_path)
    out_dir = base.repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scale, unit = base.UNIT_SCALES[args.unit]

    body_names = base.object_array_to_list(data.get("body_names"))
    if not body_names:
        raise KeyError("Recording does not contain body_names.")
    root_idx = base.find_root_body_index(body_names, args.root_body)
    root_name = body_names[root_idx]

    summary: dict[str, Any] = {
        "mode": "segmented_live_motionbricks",
        "recording": str(recording_path),
        "motionbricks_qpos": str(base.repo_path(args.motionbricks_qpos)),
        "target": str(base.repo_path(args.target)),
        "num_recorded_frames": int(len(data["body_error_mean"])),
        "num_offline_reference_frames": int(len(qpos)),
        "root_body": root_name,
        "target_x": float(target_qpos[0]),
        "target_y": float(target_qpos[1]),
        "target_z": float(target_qpos[2]),
        "body_mean_error_m": float(np.nanmean(np.asarray(data["body_error_mean"], dtype=float))),
        "body_max_error_m": float(np.nanmax(np.asarray(data["body_error_max"], dtype=float))),
        "anchor_mean_error_m": float(np.nanmean(np.asarray(data["anchor_pos_error"], dtype=float))),
    }

    summary.update(
        plot_live_reference_path(
            data,
            target_qpos,
            out_dir / f"segmented_live_reference_vs_robot_path_{unit}.png",
            root_idx,
            root_name,
            scale,
            unit,
        )
    )
    plot_live_absolute_xy(
        data,
        out_dir / f"segmented_live_reference_absolute_xy_{unit}.png",
        root_idx,
        root_name,
        scale,
        unit,
    )
    plot_live_absolute_x_zoom_4000_5000_mm(
        data,
        out_dir / f"segmented_live_reference_absolute_x_zoom_4000_5000_{unit}.png",
        root_idx,
        root_name,
        scale,
        unit,
    )
    summary.update(
        plot_live_x_tail_zoom(
            data,
            target_qpos,
            out_dir / f"segmented_final_x_zoom_{unit}.png",
            root_idx,
            root_name,
            scale,
            unit,
            args.tail_zoom_frames,
        )
    )
    plot_segment_time_steps(data, out_dir / "segmented_motion_time_steps.png")
    write_segment_table(data, out_dir / "segmented_replan_stats.csv")
    base.plot_body_tracking_timeseries(
        data, out_dir / f"gearsonic_tracking_timeseries_{unit}.png", scale, unit
    )

    joint_error = data.get("joint_error")
    joint_names = base.object_array_to_list(data.get("joint_names"))
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
        base.plot_joint_heatmap(joint_error, joint_names, out_dir / "per_joint_error_heatmap_deg.png")
        base.plot_joint_bar(joint_error, joint_names, out_dir / "per_joint_error_bar_deg.png")
        base.plot_joint_timeseries(joint_error, joint_names, out_dir / "per_joint_error_timeseries_deg.png")
        summary.update(base.write_joint_stats(joint_error, joint_names, out_dir / "per_joint_error_stats.csv"))
    else:
        plotted_scalar = base.plot_joint_scalar_fallback(data, out_dir / "joint_error_scalar_timeseries_rad.png")
        summary["joint_trace_note"] = (
            "This recording has no ref_joint_pos/robot_joint_pos arrays. "
            "Re-run eval with BodyTrackingCallback to get per-joint plots."
        )
        summary["has_scalar_joint_metric"] = bool(plotted_scalar)

    if args.plot_reference_dof:
        summary["qpos_reference_vs_robot_plotted"] = base.plot_29dof_qpos_reference_vs_robot(
            data,
            qpos,
            out_dir / "motionbricks_reference_29dof_qpos.png",
        )

    save_summary(summary, out_dir / "summary.json")

    print(f"Wrote segmented MotionSonic plots to: {out_dir}")
    print(
        "Live reference error: "
        f"mean_xy={summary['segmented_live_ref_mean_xy_error_m'] * scale:.3f}{unit}, "
        f"p95_xy={summary['segmented_live_ref_p95_xy_error_m'] * scale:.3f}{unit}, "
        f"final_xy={summary['segmented_live_ref_final_xy_error_m'] * scale:.3f}{unit}"
    )
    print(
        "Final target error: "
        f"robot_xy={summary['robot_final_target_xy_error_m'] * scale:.3f}{unit}, "
        f"closest_xy={summary['robot_closest_target_xy_error_m'] * scale:.3f}{unit}, "
        f"x_signed={summary['robot_final_target_x_error_m'] * scale:.3f}{unit}"
    )
    print(f"Detected replans/time-step resets: {summary['segmented_replan_count']}")
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
