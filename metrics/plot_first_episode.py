#!/usr/bin/env python3
"""Visualize first-episode body tracking recordings."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


UNIT_SCALES = {
    "m": (1.0, "m"),
    "cm": (100.0, "cm"),
    "mm": (1000.0, "mm"),
}


def resolve_recording(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_dir():
        path = path / "first_episode_body_tracking.npz"
    if not path.exists():
        raise FileNotFoundError(f"Recording not found: {path}")
    return path


def load_recording(path: Path) -> dict[str, np.ndarray]:
    # Recordings may be written by a newer NumPy that pickles object arrays
    # under numpy._core. Ubuntu's older NumPy exposes the same modules as
    # numpy.core, so provide aliases before loading.
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def maybe_metric(data: dict[str, np.ndarray], key: str) -> np.ndarray | None:
    value = data.get(key)
    if value is None or value.size == 0:
        return None
    return np.asarray(value, dtype=float).reshape(-1)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="valid")


def selected_link_indices(body_names: list[str], selected_links: list[str] | None) -> list[int]:
    if not selected_links:
        return list(range(len(body_names)))
    missing = [name for name in selected_links if name not in body_names]
    if missing:
        print(f"Warning: requested links not found and will be skipped: {missing}")
    return [body_names.index(name) for name in selected_links if name in body_names]


def plot_timeseries(data: dict[str, np.ndarray], out_path: Path, scale: float, unit: str) -> None:
    t = np.arange(len(data["body_error_mean"]))
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(t, data["body_error_mean"] * scale, label="mean body pos error")
    axes[0].plot(t, data["body_error_max"] * scale, label="max body pos error", alpha=0.8)
    axes[0].plot(t, data["anchor_pos_error"] * scale, label="anchor pos error", alpha=0.8)
    axes[0].set_ylabel(unit)
    axes[0].set_title("Reference Yellow Markers vs Robot")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(t, data["vr3_error_mean"] * scale, label="VR3 mean: torso + wrists")
    axes[1].plot(t, data["foot_error_mean"] * scale, label="feet mean")
    axes[1].set_ylabel(unit)
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    extra_keys = [
        "command_metric__error_body_pos",
        "command_metric__error_anchor_pos",
        "command_metric__error_joint_pos",
        "command_metric__error_body_rot",
    ]
    plotted = False
    for key in extra_keys:
        values = maybe_metric(data, key)
        if values is not None:
            label = key.replace("command_metric__", "")
            if key.endswith("_pos"):
                values = values * scale
                label = f"{label} ({unit})"
            axes[2].plot(t[: len(values)], values, label=label)
            plotted = True
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("metric")
    axes[2].grid(alpha=0.25)
    if plotted:
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, "No command metrics recorded", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_body_heatmap(
    data: dict[str, np.ndarray], out_path: Path, scale: float, unit: str, vmax: float | None
) -> None:
    body_error = np.asarray(data["body_error"], dtype=float) * scale
    body_names = [str(x) for x in data["body_names"].tolist()]
    if vmax is None and body_error.size:
        vmax = float(np.nanpercentile(body_error, 95))

    fig, ax = plt.subplots(figsize=(12, max(5, len(body_names) * 0.35)))
    image = ax.imshow(
        body_error.T,
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
    )
    ax.set_title("Body Position Error Heatmap")
    ax.set_xlabel("step")
    ax.set_ylabel("body")
    ax.set_yticks(np.arange(len(body_names)))
    ax.set_yticklabels(body_names)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(unit)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_xy_trace(data: dict[str, np.ndarray], out_path: Path, scale: float, unit: str) -> None:
    ref = np.asarray(data["ref_body_pos_w"], dtype=float)
    robot = np.asarray(data["robot_body_pos_w"], dtype=float)
    body_names = [str(x) for x in data["body_names"].tolist()]

    interesting = [
        "pelvis",
        "torso_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ]
    indices = [body_names.index(name) for name in interesting if name in body_names]
    if not indices:
        return

    cols = 2
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, max(4, rows * 3.2)))
    axes = np.asarray(axes).reshape(-1)
    for ax, idx in zip(axes, indices, strict=False):
        ref_xy = (ref[:, idx, :2] - ref[0:1, idx, :2]) * scale
        robot_xy = (robot[:, idx, :2] - ref[0:1, idx, :2]) * scale
        ax.plot(ref_xy[:, 0], ref_xy[:, 1], label="ref", linewidth=2)
        ax.plot(robot_xy[:, 0], robot_xy[:, 1], label="robot", linewidth=1.5)
        ax.set_title(body_names[idx])
        ax.set_xlabel(f"x delta from first ref ({unit})")
        ax.set_ylabel(f"y delta from first ref ({unit})")
        ax.axis("equal")
        ax.grid(alpha=0.25)
        ax.legend()
    for ax in axes[len(indices) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_mpjpe(data: dict[str, np.ndarray], out_path: Path, scale: float, unit: str, smooth: int) -> None:
    """Plot per-step MPJPE. Here MPJPE is mean body position error per frame."""
    mpjpe = np.asarray(data["body_error_mean"], dtype=float) * scale
    t = np.arange(len(mpjpe))
    cumulative = np.cumsum(mpjpe) / np.arange(1, len(mpjpe) + 1)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(t, mpjpe, label="frame MPJPE", linewidth=1.5)
    ax.plot(t, cumulative, label="cumulative mean MPJPE", linewidth=2.0)
    if smooth > 1 and len(mpjpe) >= smooth:
        smoothed = rolling_mean(mpjpe, smooth)
        ax.plot(np.arange(smooth - 1, len(mpjpe)), smoothed, label=f"rolling mean ({smooth})")
    ax.set_title("MPJPE Over Time")
    ax.set_xlabel("step")
    ax.set_ylabel(unit)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_link_error_timeseries(
    data: dict[str, np.ndarray],
    out_path: Path,
    scale: float,
    unit: str,
    selected_links: list[str] | None,
) -> None:
    body_error = np.asarray(data["body_error"], dtype=float) * scale
    body_names = [str(x) for x in data["body_names"].tolist()]
    indices = selected_link_indices(body_names, selected_links)
    if not indices:
        return

    cols = 2
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, max(4, rows * 2.4)), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    t = np.arange(body_error.shape[0])
    for ax, idx in zip(axes, indices, strict=False):
        values = body_error[:, idx]
        ax.plot(t, values, linewidth=1.3)
        ax.axhline(np.mean(values), color="tab:orange", linestyle="--", linewidth=1.0)
        ax.set_title(f"{body_names[idx]} mean={np.mean(values):.3f}{unit}")
        ax.set_ylabel(unit)
        ax.grid(alpha=0.25)
    for ax in axes[len(indices) :]:
        ax.axis("off")
    axes[min(len(indices) - 1, len(axes) - 1)].set_xlabel("step")
    fig.suptitle("Per-Link Position Error Magnitude")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_link_xyz_delta(
    data: dict[str, np.ndarray],
    out_path: Path,
    scale: float,
    unit: str,
    selected_links: list[str] | None,
) -> None:
    ref = np.asarray(data["ref_body_pos_w"], dtype=float)
    robot = np.asarray(data["robot_body_pos_w"], dtype=float)
    delta = (robot - ref) * scale
    body_names = [str(x) for x in data["body_names"].tolist()]
    indices = selected_link_indices(body_names, selected_links)
    if not indices:
        return

    cols = 2
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, max(4, rows * 2.7)), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    t = np.arange(delta.shape[0])
    axis_names = ["dx", "dy", "dz"]
    for ax, idx in zip(axes, indices, strict=False):
        for axis, name in enumerate(axis_names):
            ax.plot(t, delta[:, idx, axis], label=name, linewidth=1.2)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(body_names[idx])
        ax.set_ylabel(unit)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
    for ax in axes[len(indices) :]:
        ax.axis("off")
    axes[min(len(indices) - 1, len(axes) - 1)].set_xlabel("step")
    fig.suptitle("Per-Link Signed Position Delta (robot - reference)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_per_link_mpjpe_bar(
    data: dict[str, np.ndarray], out_path: Path, scale: float, unit: str
) -> None:
    body_error = np.asarray(data["body_error"], dtype=float) * scale
    body_names = [str(x) for x in data["body_names"].tolist()]
    means = body_error.mean(axis=0)
    p95 = np.percentile(body_error, 95, axis=0)
    order = np.argsort(means)

    fig, ax = plt.subplots(figsize=(10, max(5, len(body_names) * 0.35)))
    y = np.arange(len(order))
    ax.barh(y, means[order], label="mean")
    ax.scatter(p95[order], y, color="tab:red", s=22, label="p95")
    ax.set_yticks(y)
    ax.set_yticklabels([body_names[i] for i in order])
    ax.set_xlabel(unit)
    ax.set_title("Per-Link MPJPE")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_link_stats_csv(data: dict[str, np.ndarray], out_path: Path, scale: float) -> None:
    body_error = np.asarray(data["body_error"], dtype=float) * scale
    body_names = [str(x) for x in data["body_names"].tolist()]
    lines = ["link,mean,p50,p95,max\n"]
    for idx, name in enumerate(body_names):
        values = body_error[:, idx]
        lines.append(
            f"{name},{np.mean(values):.8f},{np.percentile(values, 50):.8f},"
            f"{np.percentile(values, 95):.8f},{np.max(values):.8f}\n"
        )
    out_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "recording",
        nargs="?",
        default="/home/hslee/IsaacLab_ws/GR00T-WholeBodyControl/metrics/first_episode",
        help="Path to first_episode_body_tracking.npz or its containing directory.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to <recording_dir>/plots.",
    )
    parser.add_argument(
        "--unit",
        choices=sorted(UNIT_SCALES),
        default="mm",
        help="Plot distance unit. Default is mm for fine-grained error inspection.",
    )
    parser.add_argument(
        "--links",
        nargs="+",
        default=None,
        help="Optional subset of links to show in per-link plots. Default: all links.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=25,
        help="Rolling window for MPJPE plot. Use 1 to disable.",
    )
    parser.add_argument(
        "--heatmap-vmax",
        type=float,
        default=None,
        help="Optional heatmap max in selected unit. Default: 95th percentile.",
    )
    args = parser.parse_args()

    recording_path = resolve_recording(args.recording)
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else recording_path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    scale, unit = UNIT_SCALES[args.unit]

    data = load_recording(recording_path)
    plot_timeseries(data, out_dir / f"tracking_timeseries_{unit}.png", scale, unit)
    plot_body_heatmap(data, out_dir / f"body_error_heatmap_{unit}.png", scale, unit, args.heatmap_vmax)
    plot_xy_trace(data, out_dir / f"xy_traces_{unit}.png", scale, unit)
    plot_mpjpe(data, out_dir / f"mpjpe_timeseries_{unit}.png", scale, unit, args.smooth)
    plot_link_error_timeseries(
        data, out_dir / f"per_link_error_timeseries_{unit}.png", scale, unit, args.links
    )
    plot_link_xyz_delta(data, out_dir / f"per_link_xyz_delta_{unit}.png", scale, unit, args.links)
    plot_per_link_mpjpe_bar(data, out_dir / f"per_link_mpjpe_bar_{unit}.png", scale, unit)
    write_link_stats_csv(data, out_dir / f"per_link_stats_{unit}.csv", scale)

    print(f"Wrote plots to: {out_dir}")
    print(f"Frames: {len(data['body_error_mean'])}")
    print(f"Mean body error: {np.mean(data['body_error_mean']) * scale:.6f} {unit}")
    print(f"Max body error: {np.max(data['body_error_max']) * scale:.6f} {unit}")


if __name__ == "__main__":
    main()
