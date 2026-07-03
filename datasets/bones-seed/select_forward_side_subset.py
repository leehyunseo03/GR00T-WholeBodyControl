#!/usr/bin/env python3
"""Select simple forward/side walking motions from BONES-SEED motion_lib files.

Run this after:
  1. converting BONES-SEED G1 CSVs to motion_lib PKLs
  2. applying the official broad robot_filtered pass

The script uses BONES-SEED metadata to choose simple standing locomotion:
forward walking plus left/right or sideways walking. It then copies matching
PKLs from robot_filtered into a smaller fine-tuning dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import pandas as pd


DEFAULT_BAD_PATTERN = (
    "jump|danc|jog|run|climb|crawl|sit|chair|kneel|crouch|"
    "kick|punch|throw|cartwheel|handstand|stair|ladder|bike|scooter|"
    "dog|crutch|smoking|phone|weapon|ball|box"
)

DEFAULT_FORWARD_PATTERN = (
    "walk forward|walking forward|move forward|moving forward|"
    "steps forward|step forward|go forward"
)

DEFAULT_SIDE_PATTERN = (
    "walk right|walk left|walks right|walks left|walking sideways|"
    "walk sideways|steps sideways|step sideways|side step|sidestep|"
    "lateral|shift weight|weight shift"
)


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]

    parser = argparse.ArgumentParser(
        description="Copy forward/side walking BONES-SEED PKLs for SONIC fine-tuning."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=repo_root / "datasets/bones-seed/metadata/seed_metadata_v004.csv",
        help="Path to seed_metadata_v004.csv.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=repo_root / "data/motion_lib_bones_seed/robot_filtered",
        help="Source motion_lib directory after official filtering.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=repo_root / "data/motion_lib_bones_seed/robot_forward_side",
        help="Destination directory for selected subset.",
    )
    parser.add_argument(
        "--selection-csv",
        type=Path,
        default=repo_root / "datasets/filters/forward_side_selected_metadata.csv",
        help="Where to write selected metadata rows.",
    )
    parser.add_argument(
        "--exclude-pattern",
        default=DEFAULT_BAD_PATTERN,
        help="Regex pattern for motions to exclude.",
    )
    parser.add_argument(
        "--include-turning",
        action="store_true",
        help="Keep walking, turning motions. By default they are excluded.",
    )
    parser.add_argument(
        "--include-fast",
        action="store_true",
        help="Keep fast and very_fast motions. By default they are excluded.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print counts without copying.")
    return parser.parse_args()


def build_selection(df: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    text = df.fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    category = df["category"].fillna("")
    movement = df["content_type_of_movement"].fillna("").str.lower()
    body = df["content_body_position"].fillna("").str.lower()

    basic_locomotion = category.str.contains("Basic Locomotion", regex=False)
    standing = body.str.contains("standing")
    horizontal = df["content_horizontal_move"].fillna(0).astype(int).eq(1)
    no_props = df["content_props"].fillna(0).astype(str).eq("0")
    simple = df["content_complex_action"].fillna(0).astype(int).eq(0)

    walking = movement.eq("walking") | movement.eq("walking, turning")
    if not args.include_turning:
        walking &= ~movement.str.contains("turning", regex=False)

    bad = text.str.contains(args.exclude_pattern, regex=True)
    if not args.include_fast:
        bad |= text.str.contains("fast|very_fast|very fast", regex=True)

    forward = text.str.contains(DEFAULT_FORWARD_PATTERN, regex=True)
    side = text.str.contains(DEFAULT_SIDE_PATTERN, regex=True)

    return basic_locomotion & standing & horizontal & walking & no_props & simple & ~bad & (
        forward | side
    )


def copy_selected(df: pd.DataFrame, args: argparse.Namespace) -> tuple[int, int]:
    copied = 0
    missing = 0

    for _, row in df.iterrows():
        g1_path = Path(str(row["move_g1_path"]))
        session = g1_path.parent.name
        name = Path(str(row["filename"])).stem

        src = args.source / session / f"{name}.pkl"
        dst = args.dest / session / f"{name}.pkl"

        if not src.exists():
            missing += 1
            continue

        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        copied += 1

    return copied, missing


def main() -> int:
    args = parse_args()

    df = pd.read_csv(args.metadata)
    selected_mask = build_selection(df, args)
    selected = df[selected_mask].copy()

    forward_count = selected.fillna("").astype(str).agg(" ".join, axis=1).str.lower().str.contains(
        DEFAULT_FORWARD_PATTERN, regex=True
    ).sum()
    side_count = selected.fillna("").astype(str).agg(" ".join, axis=1).str.lower().str.contains(
        DEFAULT_SIDE_PATTERN, regex=True
    ).sum()

    copied, missing = copy_selected(selected, args)

    if not args.dry_run:
        args.selection_csv.parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(args.selection_csv, index=False)

    print("Selection summary")
    print(f"  metadata rows selected: {len(selected)}")
    print(f"  forward-like rows:      {forward_count}")
    print(f"  side/shift-like rows:   {side_count}")
    print(f"  copied PKLs:            {copied}")
    print(f"  missing PKLs:           {missing}")
    print(f"  source:                 {args.source}")
    print(f"  destination:            {args.dest}")
    print(f"  selection csv:          {args.selection_csv}")
    if args.dry_run:
        print("  dry run:                no files copied")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
