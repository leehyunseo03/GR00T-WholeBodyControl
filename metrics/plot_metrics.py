#!/usr/bin/env python3
"""Plot SONIC imitation metrics saved by im_eval.

Example:
    python metrics/plot_metrics.py \
        --runs baseline=metrics/baseline/metrics_eval.json finetuned=metrics/finetuned/metrics_eval.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SUMMARY_KEYS = [
    ("eval/all/mpjpe", "all_mpjpe"),
    ("eval/all/mpjpe_vr_3points", "vr3_mpjpe"),
    ("eval/all/mpjpe_foot", "foot_mpjpe"),
    ("eval/all/mpjpe_legs", "legs_mpjpe"),
    ("eval/all/mpjpe_other_upper_bodies", "upper_mpjpe"),
    ("eval/success/mpjpe", "success_mpjpe"),
    ("eval/success/success_rate", "success_rate"),
    ("eval/success/progress_rate", "progress_rate"),
]

DEFAULT_PER_MOTION_KEYS = [
    "mpjpe",
    "mpjpe_vr_3points",
    "mpjpe_foot",
    "mpjpe_legs",
    "mpjpe_other_upper_bodies",
    "progress",
    "terminated",
]


def resolve_metrics_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_dir():
        path = path / "metrics_eval.json"
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    return path


def load_run(spec: str) -> tuple[str, dict[str, Any], Path]:
    if "=" in spec:
        label, path_text = spec.split("=", 1)
    else:
        path_text = spec
        label = Path(path_text).expanduser().parent.name or Path(path_text).stem
    path = resolve_metrics_path(path_text)
    with path.open("r", encoding="utf-8") as f:
        return label, json.load(f), path


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and len(value) == 1:
        return as_float(value[0])
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    return None


def numeric_array(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(value)
        if arr.dtype.kind not in "biufc":
            arr = arr.astype(float)
        return arr.astype(float).reshape(-1)
    except (TypeError, ValueError):
        return None


def get_all_metrics_dict(run: dict[str, Any]) -> dict[str, Any]:
    value = run.get("eval/all_metrics_dict", {})
    return value if isinstance(value, dict) else {}


def build_summary_rows(runs: list[tuple[str, dict[str, Any], Path]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, data, path in runs:
        row: dict[str, Any] = {"run": label, "path": str(path)}
        for json_key, short_key in DEFAULT_SUMMARY_KEYS:
            row[short_key] = as_float(data.get(json_key))
        all_metrics = get_all_metrics_dict(data)
        motion_keys = all_metrics.get("motion_keys", [])
        row["num_motions"] = len(motion_keys) if isinstance(motion_keys, list) else None
        rows.append(row)
    return rows


def write_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    labels = [row["run"] for row in rows]
    metric_keys = [short for _, short in DEFAULT_SUMMARY_KEYS]
    available = [
        key for key in metric_keys if any(row.get(key) is not None and np.isfinite(row[key]) for row in rows)
    ]
    if not available:
        return

    x = np.arange(len(available))
    width = min(0.8 / max(len(rows), 1), 0.35)

    fig, ax = plt.subplots(figsize=(max(10, len(available) * 1.25), 5.5))
    for idx, row in enumerate(rows):
        values = [row.get(key, np.nan) for key in available]
        offset = (idx - (len(rows) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=labels[idx])

    ax.set_xticks(x)
    ax.set_xticklabels(available, rotation=35, ha="right")
    ax.set_title("Evaluation Summary")
    ax.set_ylabel("Metric value")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_per_motion_hist(
    runs: list[tuple[str, dict[str, Any], Path]], metric: str, output_path: Path
) -> None:
    series: list[tuple[str, np.ndarray]] = []
    for label, data, _ in runs:
        arr = numeric_array(get_all_metrics_dict(data).get(metric))
        if arr is not None and arr.size > 0:
            series.append((label, arr))
    if not series:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, arr in series:
        ax.hist(arr, bins=40, alpha=0.45, label=f"{label} (mean={np.nanmean(arr):.4g})")
    ax.set_title(f"Per-Motion Distribution: {metric}")
    ax.set_xlabel(metric)
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def motion_metric_by_key(data: dict[str, Any], metric: str) -> dict[str, float]:
    all_metrics = get_all_metrics_dict(data)
    motion_keys = all_metrics.get("motion_keys")
    values = numeric_array(all_metrics.get(metric))
    if not isinstance(motion_keys, list) or values is None:
        return {}
    count = min(len(motion_keys), len(values))
    return {str(motion_keys[i]): float(values[i]) for i in range(count)}


def plot_pair_delta(
    baseline: tuple[str, dict[str, Any], Path],
    candidate: tuple[str, dict[str, Any], Path],
    metric: str,
    output_path: Path,
) -> None:
    base_label, base_data, _ = baseline
    cand_label, cand_data, _ = candidate
    base = motion_metric_by_key(base_data, metric)
    cand = motion_metric_by_key(cand_data, metric)
    common_keys = sorted(set(base) & set(cand))
    if not common_keys:
        return

    base_values = np.array([base[key] for key in common_keys], dtype=float)
    cand_values = np.array([cand[key] for key in common_keys], dtype=float)
    delta = cand_values - base_values
    order = np.argsort(delta)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(base_values, cand_values, s=14, alpha=0.65)
    lo = float(np.nanmin([base_values.min(), cand_values.min()]))
    hi = float(np.nanmax([base_values.max(), cand_values.max()]))
    axes[0].plot([lo, hi], [lo, hi], "k--", linewidth=1)
    axes[0].set_xlabel(f"{base_label} {metric}")
    axes[0].set_ylabel(f"{cand_label} {metric}")
    axes[0].set_title("Per-Motion Before/After")
    axes[0].grid(alpha=0.25)

    axes[1].plot(delta[order], linewidth=1.5)
    axes[1].axhline(0.0, color="k", linestyle="--", linewidth=1)
    axes[1].set_title(f"Delta: {cand_label} - {base_label}")
    axes[1].set_xlabel("motions sorted by delta")
    axes[1].set_ylabel(f"delta {metric}")
    axes[1].grid(alpha=0.25)

    improved = int(np.sum(delta < 0.0))
    fig.suptitle(f"{metric}: {improved}/{len(delta)} motions improved")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_per_motion_csv(
    runs: list[tuple[str, dict[str, Any], Path]], output_path: Path, metrics: list[str]
) -> None:
    rows: dict[str, dict[str, Any]] = {}
    for label, data, _ in runs:
        all_metrics = get_all_metrics_dict(data)
        motion_keys = all_metrics.get("motion_keys")
        if not isinstance(motion_keys, list):
            continue
        for idx, motion_key in enumerate(motion_keys):
            row = rows.setdefault(str(motion_key), {"motion_key": str(motion_key)})
            for metric in metrics:
                arr = numeric_array(all_metrics.get(metric))
                if arr is not None and idx < arr.size:
                    row[f"{label}.{metric}"] = float(arr[idx])

    if not rows:
        return

    fieldnames = ["motion_key"]
    for row in rows.values():
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for motion_key in sorted(rows):
            writer.writerow(rows[motion_key])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run specs: label=/path/to/metrics_eval.json or label=/path/to/dir",
    )
    parser.add_argument(
        "--out-dir",
        default="/workspace/GR00T-WholeBodyControl/metrics/plots",
        help="Directory for PNG/CSV outputs.",
    )
    parser.add_argument(
        "--per-motion-metric",
        default="mpjpe",
        help="Per-motion metric used for histogram and before/after delta plots.",
    )
    args = parser.parse_args()

    runs = [load_run(spec) for spec in args.runs]
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_summary_rows(runs)
    write_summary_csv(rows, out_dir / "summary.csv")
    write_per_motion_csv(runs, out_dir / "per_motion.csv", DEFAULT_PER_MOTION_KEYS)
    plot_summary(rows, out_dir / "summary.png")
    plot_per_motion_hist(runs, args.per_motion_metric, out_dir / f"{args.per_motion_metric}_hist.png")

    if len(runs) >= 2:
        plot_pair_delta(runs[0], runs[1], args.per_motion_metric, out_dir / f"{args.per_motion_metric}_delta.png")

    print(f"Wrote plots and CSV files to: {out_dir}")
    print("Summary:")
    for row in rows:
        compact = {k: v for k, v in row.items() if k not in {"path"}}
        print(compact)


if __name__ == "__main__":
    main()
