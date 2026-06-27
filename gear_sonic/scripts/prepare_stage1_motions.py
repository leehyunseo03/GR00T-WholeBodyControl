#!/usr/bin/env python3
"""Prepare weighted stage-1 motion directories without modifying source PKLs.

The SONIC motion loader samples one motion key per PKL.  To make a simple
weighted mixture from small sample/custom sets, this script creates uniquely
named relative symlinks to the original robot and SMPL PKLs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


def parse_group(value: str) -> dict:
    parts = value.split(":")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--group must be NAME:ROBOT_DIR:SMPL_DIR:FRACTION, "
            f"got {value!r}"
        )
    name, robot_dir, smpl_dir, fraction = parts
    try:
        fraction_value = float(fraction)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid fraction in {value!r}") from exc
    if fraction_value < 0:
        raise argparse.ArgumentTypeError(f"Fraction must be non-negative in {value!r}")
    return {
        "name": name,
        "robot_dir": Path(robot_dir),
        "smpl_dir": Path(smpl_dir),
        "fraction": fraction_value,
    }


def collect_pairs(robot_dir: Path, smpl_dir: Path) -> list[tuple[Path, Path]]:
    robot_files = sorted(p for p in robot_dir.rglob("*.pkl") if p.name != "metadata.pkl")
    smpl_files = {
        p.stem: p for p in smpl_dir.rglob("*.pkl") if p.name != "metadata.pkl"
    }
    pairs: list[tuple[Path, Path]] = []
    missing: list[str] = []
    for robot_path in robot_files:
        smpl_path = smpl_files.get(robot_path.stem)
        if smpl_path is None:
            missing.append(robot_path.name)
        else:
            pairs.append((robot_path, smpl_path))
    if missing:
        raise FileNotFoundError(
            f"Missing SMPL PKLs in {smpl_dir} for robot files: {', '.join(missing[:8])}"
        )
    if not pairs:
        raise FileNotFoundError(f"No robot/SMPL PKL pairs found in {robot_dir} and {smpl_dir}")
    return pairs


def allocate_counts(fractions: list[float], total_slots: int) -> list[int]:
    total_fraction = sum(fractions)
    if total_fraction <= 0:
        raise ValueError("At least one group fraction must be positive")

    raw = [fraction / total_fraction * total_slots for fraction in fractions]
    counts = [math.floor(value) for value in raw]
    remainder = total_slots - sum(counts)
    order = sorted(range(len(raw)), key=lambda idx: raw[idx] - counts[idx], reverse=True)
    for idx in order[:remainder]:
        counts[idx] += 1

    for idx, fraction in enumerate(fractions):
        if fraction > 0 and counts[idx] == 0:
            counts[idx] = 1

    while sum(counts) > total_slots:
        idx = max(range(len(counts)), key=lambda i: counts[i])
        counts[idx] -= 1
    while sum(counts) < total_slots:
        idx = max(range(len(raw)), key=lambda i: raw[i] - counts[i])
        counts[idx] += 1
    return counts


def relative_symlink(source: Path, destination: Path, copy_instead: bool) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing path: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_instead:
        import shutil

        shutil.copy2(source, destination)
        return
    rel_source = os.path.relpath(source.resolve(), start=destination.parent.resolve())
    destination.symlink_to(rel_source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory containing robot_filtered/ and smpl_filtered/.",
    )
    parser.add_argument(
        "--group",
        action="append",
        required=True,
        type=parse_group,
        help="Weighted source group: NAME:ROBOT_DIR:SMPL_DIR:FRACTION",
    )
    parser.add_argument(
        "--total-slots",
        type=int,
        default=20,
        help="Total symlinked motion keys to create across all groups.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy PKLs instead of creating relative symlinks.",
    )
    args = parser.parse_args()

    if args.total_slots <= 0:
        raise ValueError("--total-slots must be positive")

    output = args.output
    robot_output = output / "robot_filtered"
    smpl_output = output / "smpl_filtered"
    if any(path.exists() and any(path.iterdir()) for path in (robot_output, smpl_output)):
        raise FileExistsError(
            f"{output} already contains files. Use a new output path to preserve prior runs."
        )

    groups = []
    for group in args.group:
        pairs = collect_pairs(group["robot_dir"], group["smpl_dir"])
        groups.append({**group, "pairs": pairs})

    counts = allocate_counts([group["fraction"] for group in groups], args.total_slots)
    manifest = {
        "output": str(output),
        "total_slots": args.total_slots,
        "copy": args.copy,
        "groups": [],
        "links": [],
    }

    for group, count in zip(groups, counts, strict=True):
        group_manifest = {
            "name": group["name"],
            "requested_fraction": group["fraction"],
            "slots": count,
            "source_robot_dir": str(group["robot_dir"]),
            "source_smpl_dir": str(group["smpl_dir"]),
            "source_pairs": len(group["pairs"]),
        }
        manifest["groups"].append(group_manifest)

        for slot_idx in range(count):
            robot_source, smpl_source = group["pairs"][slot_idx % len(group["pairs"])]
            stem = f"{group['name']}_{slot_idx:04d}_{robot_source.stem}"
            robot_dest = robot_output / f"{stem}.pkl"
            smpl_dest = smpl_output / f"{stem}.pkl"
            relative_symlink(robot_source, robot_dest, args.copy)
            relative_symlink(smpl_source, smpl_dest, args.copy)
            manifest["links"].append(
                {
                    "group": group["name"],
                    "robot": str(robot_dest),
                    "smpl": str(smpl_dest),
                    "source_robot": str(robot_source),
                    "source_smpl": str(smpl_source),
                }
            )

    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(manifest['links'])} motion slots in {output}")
    for group in manifest["groups"]:
        print(
            f"  {group['name']}: {group['slots']} slots "
            f"from {group['source_pairs']} source pairs"
        )


if __name__ == "__main__":
    main()
