#!/usr/bin/env python3
"""Generate a naive forward MotionBricks walk and export it for GEAR-Sonic."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYNPUT_BACKEND", "dummy")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/motionbricks_gearsonic_matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
MOTIONBRICKS_ROOT = REPO_ROOT / "motionbricks"
for path in (REPO_ROOT, MOTIONBRICKS_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from motion_sonic.generate_motionbricks_motion_lib import (  # noqa: E402
    _make_motionbricks_args,
    _parse_speed_scale,
    _summarize_qpos,
    qpos_to_sonic_motion_entry,
)


@contextlib.contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _mode_index(demo_agent, mode_name: str) -> tuple[int, list[str]]:
    modes = list(demo_agent.full_agent._clip_holder.CLIPS.keys())
    if mode_name not in modes:
        raise ValueError(f"Unknown MotionBricks mode {mode_name!r}. Available modes: {modes}")
    return modes.index(mode_name), modes


def generate_forward_qpos(demo_agent, args: argparse.Namespace):
    import mujoco
    import numpy as np
    import torch

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    demo_agent.full_agent.reset()

    mode_idx, modes = _mode_index(demo_agent, args.mode)
    movement = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    facing = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    mode = torch.tensor([[mode_idx]], dtype=torch.long)

    frames = []
    for step in range(1, args.max_steps + 1):
        qpos = demo_agent.full_agent.get_next_frame()
        frames.append(np.asarray(qpos, dtype=np.float32).copy())

        context_mujoco_qpos = demo_agent.full_agent.get_context_mujoco_qpos()
        demo_agent.mj_data.qpos[:] = qpos

        control_signals = {
            "movement_direction": movement,
            "facing_direction": facing,
            "mode": mode,
            "context_mujoco_qpos": context_mujoco_qpos,
            "allowed_pred_num_tokens": demo_agent.controller.get_default_allowed_pred_num_tokens(mode_idx),
            "random_seed": torch.tensor([args.random_seed + step], dtype=torch.long),
        }
        if args.target_vel > 0.0:
            control_signals["target_vel"] = torch.tensor([[args.target_vel]], dtype=torch.float32)

        with torch.no_grad():
            demo_agent.full_agent.generate_new_frames(
                control_signals,
                demo_agent.controller.get_controller_dt() * args.generate_dt,
                force_generation=(step == 1),
            )

        mujoco.mj_forward(demo_agent.mj_model, demo_agent.mj_data)

        if args.print_every > 0 and (step % args.print_every == 0 or step == args.max_steps):
            print(
                f"step={step:05d} root_xyz=({qpos[0]:.3f}, {qpos[1]:.3f}, {qpos[2]:.3f})",
                flush=True,
            )

    return np.stack(frames, axis=0), modes


def export_motion(args: argparse.Namespace, qpos, modes: list[str]) -> None:
    import joblib
    import numpy as np

    output_dir = Path(args.output_dir).resolve()
    robot_dir = output_dir / "robot_filtered" / args.session
    qpos_dir = output_dir / "qpos"
    robot_dir.mkdir(parents=True, exist_ok=True)
    qpos_dir.mkdir(parents=True, exist_ok=True)

    entry = qpos_to_sonic_motion_entry(qpos, fps=args.fps)
    pkl_path = robot_dir / f"{args.motion_name}.pkl"
    npy_path = qpos_dir / f"{args.motion_name}.npy"
    npz_path = qpos_dir / f"{args.motion_name}.npz"

    joblib.dump({args.motion_name: entry}, pkl_path, compress=True)
    np.save(npy_path, qpos)
    np.savez_compressed(
        npz_path,
        qpos=qpos,
        root_xyz=qpos[:, :3],
        dof=qpos[:, 7:36],
        fps=np.asarray(args.fps, dtype=np.int32),
    )

    summary = _summarize_qpos(qpos)
    manifest = {
        "created_unix": time.time(),
        "source": "naive MotionBricks forward walk without target position or terminal pose constraint",
        "motion_name": args.motion_name,
        "mode": args.mode,
        "available_modes": modes,
        "target_vel": float(args.target_vel),
        "fps": int(args.fps),
        "max_steps": int(args.max_steps),
        "generate_dt": float(args.generate_dt),
        "random_seed": int(args.random_seed),
        "robot_pkl": _display_path(pkl_path),
        "qpos_npy": _display_path(npy_path),
        "qpos_npz": _display_path(npz_path),
        "qpos_summary": summary,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("\nNaive MotionBricks forward export complete.")
    print(f"root_first={summary['root_xyz_first']} root_last={summary['root_xyz_last']}")
    print(f"motion_file={_display_path(pkl_path)}")
    print("smpl_motion_file=dummy")
    print(f"qpos_npz={_display_path(npz_path)}")
    print(f"manifest={_display_path(manifest_path)}")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main(args: argparse.Namespace) -> None:
    import torch
    from motionbricks.motion_backbone.demo.utils import navigation_demo

    if not torch.cuda.is_available():
        raise RuntimeError("MotionBricks inference expects CUDA; torch.cuda.is_available() is False.")

    mb_args = _make_motionbricks_args(args)
    with _pushd(MOTIONBRICKS_ROOT):
        demo_agent = navigation_demo(mb_args)
        qpos, modes = generate_forward_qpos(demo_agent, args)
    export_motion(args, qpos, modes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a naive MotionBricks forward walk and export a GEAR-Sonic motion_lib PKL."
    )
    parser.add_argument("--output_dir", type=str, default=str(REPO_ROOT / "motionbricks_gearsonic" / "naive_forward"))
    parser.add_argument("--session", type=str, default="naive_forward")
    parser.add_argument("--motion_name", type=str, default="motionbricks_naive_forward_walk")
    parser.add_argument("--mode", type=str, default="walk")
    parser.add_argument("--target_vel", type=float, default=0.40)
    parser.add_argument("--max_steps", type=int, default=900)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--print_every", type=int, default=60)

    parser.add_argument("--explicit_dataset_folder", type=str, default=None)
    parser.add_argument("--reprocess_clips", type=int, default=0)
    parser.add_argument("--controller", type=str, default="random", choices=["wasd", "random"])
    parser.add_argument("--lookat_movement_direction", type=int, default=1)
    parser.add_argument("--pre_filter_qpos", type=int, default=1)
    parser.add_argument("--source_root_realignment", type=int, default=1)
    parser.add_argument("--target_root_realignment", type=int, default=1)
    parser.add_argument("--force_canonicalization", type=int, default=1)
    parser.add_argument("--skip_ending_target_cond", type=int, default=0)
    parser.add_argument("--random_speed_scale", type=int, default=0)
    parser.add_argument("--speed_scale", type=_parse_speed_scale, default=[1.0, 1.0])
    parser.add_argument("--generate_dt", type=float, default=2.0)

    parser.add_argument("--use_qpos", type=int, default=1)
    parser.add_argument("--planner", type=str, default="default")
    parser.add_argument("--allowed_mode", type=str, default=None)
    parser.add_argument("--clips", type=str, default="G1")
    parser.add_argument("--force_idle_tail", type=int, default=0)
    parser.add_argument("--num_motions", type=int, default=1)
    parser.add_argument("--motion_prefix", type=str, default="motionbricks_naive_forward")
    main(parser.parse_args())
