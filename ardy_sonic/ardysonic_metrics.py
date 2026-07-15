#!/usr/bin/env python3
"""Plot ArdySonic tracking metrics for Ardy replanning + GEAR-Sonic rollouts.

The output layout intentionally mirrors the MotionSonic metrics scripts, but the
primary reference is the live reference captured by BodyTrackingCallback. That is
the meaningful comparison for receding-horizon Ardy runs because the active
GEAR-Sonic reference is replaced every time Ardy replans.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_sonic import motionsonic_metrics as base  # noqa: E402
from motion_sonic import motionsonic_seg_metrics as segmented  # noqa: E402


DEFAULT_RECORDING = REPO_ROOT / "ardy_sonic" / "metrics" / "recording"
DEFAULT_OUT_DIR = REPO_ROOT / "ardy_sonic" / "metrics" / "plot"


def repo_or_workspace_path(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    text = str(path)
    repo_prefix = "/workspace/GR00T-WholeBodyControl"
    if text.startswith(repo_prefix):
        return REPO_ROOT / text[len(repo_prefix) + 1 :]
    shared_prefix = "/workspace/shared_io"
    if text.startswith(shared_prefix):
        return WORKSPACE_ROOT / "shared_io" / text[len(shared_prefix) + 1 :]
    return base.repo_path(path)


def runtime_candidates(runtime_text: str | None) -> list[Path]:
    candidates: list[Path] = []
    for value in (
        runtime_text,
        os.environ.get("ARDY_SONIC_RUNTIME"),
        WORKSPACE_ROOT / "shared_io" / "runtime",
        Path("/workspace/shared_io/runtime"),
        REPO_ROOT / "ardy_sonic" / "runtime",
    ):
        if value is None:
            continue
        path = repo_or_workspace_path(value)
        if path not in candidates:
            candidates.append(path)
    return candidates


def plan_sort_key(path: Path) -> tuple[int, float, str]:
    match = re.search(r"(\d+)$", path.stem)
    index = int(match.group(1)) if match else -1
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return index, mtime, path.name


def response_has_qpos(path: Path) -> bool:
    try:
        data = base.load_npz(path)
    except Exception:  # noqa: BLE001
        return False
    qpos = np.asarray(data.get("qpos", []))
    ok = bool(np.asarray(data.get("ok", True)).reshape(-1)[0])
    return ok and qpos.ndim == 2 and qpos.shape[0] > 0 and qpos.shape[1] >= 36


def latest_ardy_qpos(runtime_text: str | None) -> Path:
    checked: list[str] = []
    for runtime in runtime_candidates(runtime_text):
        checked.append(str(runtime))
        responses = runtime / "responses"
        if responses.exists():
            plans = sorted(responses.glob("plan_*.npz"), key=plan_sort_key, reverse=True)
            plans.extend(sorted(responses.glob("*.npz"), key=plan_sort_key, reverse=True))
            for path in plans:
                if response_has_qpos(path):
                    return path

        plan_dir = runtime / "plans"
        if plan_dir.exists():
            for path in sorted(plan_dir.glob("plan_*.npy"), key=plan_sort_key, reverse=True):
                return path
            for path in sorted(plan_dir.glob("*.npy"), key=plan_sort_key, reverse=True):
                return path

    raise FileNotFoundError(
        "No Ardy qpos response found. Checked runtime dirs: " + ", ".join(checked)
    )


def load_ardy_qpos(path_text: str | Path | None, runtime_text: str | None) -> tuple[np.ndarray, dict[str, np.ndarray], Path]:
    path = latest_ardy_qpos(runtime_text) if path_text is None else repo_or_workspace_path(path_text)
    qpos, meta = base.load_qpos(path)
    return qpos, meta, path


def target_qpos_from_args(
    qpos: np.ndarray, qpos_meta: dict[str, np.ndarray], target_path_text: str | None
) -> np.ndarray:
    if target_path_text:
        return base.load_target_qpos(qpos_meta, repo_or_workspace_path(target_path_text))
    if "target_qpos" in qpos_meta:
        return np.asarray(qpos_meta["target_qpos"], dtype=float).reshape(-1)[:36]
    return np.asarray(qpos[-1], dtype=float).reshape(-1)[:36]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recording",
        default=str(DEFAULT_RECORDING),
        help="first_episode_body_tracking.npz or its directory.",
    )
    parser.add_argument(
        "--ardy-qpos",
        "--motionbricks-qpos",
        dest="ardy_qpos",
        default=None,
        help="Ardy qpos .npz/.npy. Defaults to the newest runtime response plan.",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Optional target reference .npz. Defaults to the final frame of the selected Ardy qpos.",
    )
    parser.add_argument(
        "--runtime",
        default=None,
        help="Ardy runtime dir used to discover the newest response when --ardy-qpos is omitted.",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for plots.")
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

    recording_path = base.resolve_recording(repo_or_workspace_path(args.recording))
    qpos, qpos_meta, qpos_path = load_ardy_qpos(args.ardy_qpos, args.runtime)
    target_qpos = target_qpos_from_args(qpos, qpos_meta, args.target)
    data = base.load_recording(recording_path)
    out_dir = repo_or_workspace_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scale, unit = base.UNIT_SCALES[args.unit]

    body_names = base.object_array_to_list(data.get("body_names"))
    if not body_names:
        raise KeyError("Recording does not contain body_names.")
    root_idx = base.find_root_body_index(body_names, args.root_body)
    root_name = body_names[root_idx]

    summary: dict[str, Any] = {
        "mode": "segmented_live_ardy",
        "recording": str(recording_path),
        "ardy_qpos": str(qpos_path),
        "target": str(repo_or_workspace_path(args.target)) if args.target else "selected_ardy_qpos_final_frame",
        "num_recorded_frames": int(len(data["body_error_mean"])),
        "num_selected_ardy_reference_frames": int(len(qpos)),
        "root_body": root_name,
        "target_x": float(target_qpos[0]),
        "target_y": float(target_qpos[1]),
        "target_z": float(target_qpos[2]),
        "body_mean_error_m": float(np.nanmean(np.asarray(data["body_error_mean"], dtype=float))),
        "body_max_error_m": float(np.nanmax(np.asarray(data["body_error_max"], dtype=float))),
        "anchor_mean_error_m": float(np.nanmean(np.asarray(data["anchor_pos_error"], dtype=float))),
    }

    summary.update(
        segmented.plot_live_reference_path(
            data,
            target_qpos,
            out_dir / f"segmented_live_reference_vs_robot_path_{unit}.png",
            root_idx,
            root_name,
            scale,
            unit,
        )
    )
    segmented.plot_live_absolute_xy(
        data,
        out_dir / f"segmented_live_reference_absolute_xy_{unit}.png",
        root_idx,
        root_name,
        scale,
        unit,
    )
    segmented.plot_live_absolute_x_zoom_4000_5000_mm(
        data,
        out_dir / f"segmented_live_reference_absolute_x_zoom_4000_5000_{unit}.png",
        root_idx,
        root_name,
        scale,
        unit,
    )
    summary.update(
        segmented.plot_live_x_tail_zoom(
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
    segmented.plot_segment_time_steps(data, out_dir / "segmented_motion_time_steps.png")
    segmented.write_segment_table(data, out_dir / "segmented_replan_stats.csv")
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
            out_dir / "ardy_reference_29dof_qpos.png",
        )

    segmented.save_summary(summary, out_dir / "summary.json")

    print(f"Wrote ArdySonic plots to: {out_dir}")
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
