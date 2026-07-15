#!/usr/bin/env python3
"""Ardy walk-to-goal planner server (runs on the HOST in ``conda activate ardy``).

Loads the pretrained Ardy G1 diffusion model once, then serves plan requests that
arrive on the shared ``runtime/requests/`` directory. For each request it:

  1. builds Ardy constraints (root-2D forward path + optional final full-body pose),
  2. runs Ardy generation (mirrors ``ardy/scripts/generate.py``),
  3. optionally lands the last frames exactly on the target pose,
  4. exports a MuJoCo qpos CSV (root xyz + quat wxyz + 29 joints),
  5. applies an SE(2) transform to place the canonical +x walk at the robot's
     current world pose and heading,
  6. writes the world-frame ``(T, 36)`` qpos back as ``runtime/responses/<id>.npz``.

The GEAR-SONIC tracker (in the container) reads those responses and feeds them to
``TrackingCommand.install_live_qpos_segment``.

Two generation engines:
  --engine inprocess (default) : keeps the model resident -> fast replanning.
  --engine subprocess          : shells out to ``generate_g1_walk_5m.py`` per plan
                                 (slow, reloads the model each time, but uses the
                                 proven end-to-end script verbatim as a fallback).

Usage (host):
    conda activate ardy
    python ardy_sonic/ardy_planner_server.py --serve
    # or validate one plan without the tracker:
    python ardy_sonic/ardy_planner_server.py --once --distance 5 --duration 10 \
        --start-xy 0 0 --heading 0
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

# --- locate the Ardy workspace (qpos_ardy helpers) and this package ----------- #
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import protocol as P  # noqa: E402


def _resolve_ardy_repo(explicit: Optional[str]) -> Path:
    for cand in (
        explicit,
        os.environ.get("ARDY_REPO"),
        "/home/hslee/IsaacLab_ws/ardy",
        str(THIS_DIR.parents[1] / "ardy"),  # <ws>/GR00T-WholeBodyControl/ardy_sonic -> <ws>/ardy
    ):
        if cand and (Path(cand) / "workspace" / "qpos_ardy.py").exists():
            return Path(cand)
    raise FileNotFoundError(
        "Could not locate the Ardy repo (looked for <repo>/workspace/qpos_ardy.py). "
        "Pass --ardy-repo or set ARDY_REPO."
    )


# --------------------------------------------------------------------------- #
# Helpers copied from ardy/scripts/generate.py (small, avoids importing the CLI) #
# --------------------------------------------------------------------------- #
def _default_history_frames(fps: float, gen_horizon_len: int, num_frames_per_token: int) -> int:
    max_window_len = (int(10 * fps) // num_frames_per_token) * num_frames_per_token
    return ((max_window_len - gen_horizon_len) // num_frames_per_token) * num_frames_per_token


# --------------------------------------------------------------------------- #
# In-process Ardy planner                                                      #
# --------------------------------------------------------------------------- #
class ArdyWalkPlanner:
    """Resident Ardy G1 model that turns a PlanRequest into a world-frame qpos."""

    def __init__(
        self,
        ardy_repo: Path,
        model_name: str = "g1",
        device: str = "auto",
        checkpoints_dir: Optional[str] = None,
        constraint_stride: int = 5,
        velocity_profile: str = "smoothstep",
        target_hold_frames: int = 8,
        terminal_blend_frames: int = 12,
        diffusion_steps: Optional[int] = None,
        work_dir: Optional[Path] = None,
    ):
        self.ardy_repo = Path(ardy_repo)
        workspace = self.ardy_repo / "workspace"
        if str(workspace) not in sys.path:
            sys.path.insert(0, str(workspace))

        import torch  # noqa: F401 (import here so --serve fails fast with a clear msg)

        self.torch = torch
        self.constraint_stride = int(constraint_stride)
        self.velocity_profile = velocity_profile
        self.target_hold_frames = int(target_hold_frames)
        self.terminal_blend_frames = int(terminal_blend_frames)
        self.diffusion_steps_override = diffusion_steps
        self.work_dir = Path(work_dir) if work_dir else (THIS_DIR / "runtime" / "_ardy_work")
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Ardy imports (available in the `ardy` conda env).
        from ardy.constraints import load_constraints_lst
        from ardy.exports.mujoco import MujocoQposConverter
        from ardy.model import load_model
        from ardy.model.loading import get_env_var
        from ardy.model.registry import resolve_model_name
        from ardy.motion_rep.tools import length_to_mask
        from ardy.tools import seed_everything, to_numpy

        # Workspace helpers (constraint builders + smooth terminal landing).
        from qpos_ardy import (
            QposArdyConverter,
            build_root2d_constraint,
            resolve_target_qpos,
            apply_terminal_landing,
        )

        self._load_constraints_lst = load_constraints_lst
        self._length_to_mask = length_to_mask
        self._to_numpy = to_numpy
        self._seed_everything = seed_everything
        self._build_root2d = build_root2d_constraint
        self._resolve_target_qpos = resolve_target_qpos
        self._apply_terminal_landing = apply_terminal_landing

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device

        checkpoints_dir = checkpoints_dir or get_env_var("CHECKPOINTS_DIR")
        resolved = resolve_model_name(model_name, checkpoints_dir=checkpoints_dir)
        print(f"[planner] loading Ardy model '{resolved}' on {device} ...", flush=True)
        self.model = load_model(resolved, device=device, checkpoints_dir=checkpoints_dir)
        self.resolved_model = resolved
        self.fps = float(self.model.motion_rep.fps)
        self.qpos_converter = QposArdyConverter()          # 29-dof qpos -> ARDY constraint
        self.mj_converter = MujocoQposConverter(self.model.skeleton)  # ARDY out -> qpos CSV
        print(f"[planner] ready. model fps={self.fps:g}", flush=True)

    # ---- core generation (mirrors ardy/scripts/generate.py) ----------------- #
    def _generate_canonical_csv(self, req: P.PlanRequest, csv_path: Path) -> None:
        torch = self.torch
        model = self.model
        num_frames = int(round(req.duration * self.fps))
        if num_frames < 2:
            raise ValueError(f"duration {req.duration}s too short at {self.fps} fps")

        # Target 36-DOF qpos (root at +x=distance, target-height, +x facing; joints=req or zeros).
        target_qpos = self._resolve_target_qpos(
            joint_qpos=list(req.target_joint_qpos) if req.target_joint_qpos is not None else None,
            default_root_x=req.distance,
            default_root_z=req.target_height,
        )

        # Constraints: forward root-2D path (+ optional final full-body target pose).
        constraints = [
            self._build_root2d(
                distance=req.distance,
                num_frames=num_frames,
                stride=self.constraint_stride,
                profile=self.velocity_profile,
            )
        ]
        if req.reach_target_pose:
            hold = max(1, self.target_hold_frames)
            frame_indices = list(range(num_frames - hold, num_frames))
            constraints.append(self.qpos_converter.build_fullbody_constraint(target_qpos, frame_indices))

        constraints_json = self.work_dir / f"{req.id}_constraints.json"
        constraints_json.write_text(json.dumps(constraints, indent=2) + "\n")
        constraint_lst = self._load_constraints_lst(str(constraints_json), model.skeleton)

        # CFG weights.
        cw = list(req.cfg_weight)
        cfg_weight = float(cw[0]) if len(cw) == 1 else (float(cw[0]), float(cw[1]))

        num_base_steps = int(model.diffusion.num_base_steps)
        steps = self.diffusion_steps_override or num_base_steps
        steps = max(1, min(steps, num_base_steps))
        patch = model.num_frames_per_token
        history_frames = _default_history_frames(self.fps, model.gen_horizon_len, patch)

        self._seed_everything(int(req.seed))
        lengths = torch.tensor([num_frames], device=self.device)
        pad_mask = self._length_to_mask(lengths)
        first_heading_angle = torch.zeros(1, device=self.device)  # canonical: face +Z(ardy)=+X(isaac)

        observed_motion, motion_mask = model.motion_rep.create_conditions_from_constraints_batched(
            constraint_lst, lengths, to_normalize=True, device=self.device
        )
        with torch.no_grad():
            motion = model(
                [req.prompt.strip()],
                num_frames,
                num_denoising_steps=steps,
                pad_mask=pad_mask,
                first_heading_angle=first_heading_angle,
                motion_mask=motion_mask,
                observed_motion=observed_motion,
                cfg_weight=cfg_weight,
                crop_history_length=history_frames,
            )
            output = model.motion_rep.inverse(motion, is_normalized=True)
        # G1 postprocessing is disabled (as in generate.py); export MuJoCo qpos CSV.
        output = self._to_numpy(output)
        qpos = self.mj_converter.dict_to_qpos(output, self.device)
        self.mj_converter.save_csv(qpos, str(csv_path))

    def plan(self, req: P.PlanRequest) -> tuple[np.ndarray, float]:
        """Return (world_frame_qpos (T,36) float32, fps)."""
        csv_path = self.work_dir / f"{req.id}.csv"
        self._generate_canonical_csv(req, csv_path)

        # Target for the smooth exact landing (same target as the constraint).
        if req.reach_target_pose:
            target_qpos = self._resolve_target_qpos(
                joint_qpos=list(req.target_joint_qpos) if req.target_joint_qpos is not None else None,
                default_root_x=req.distance,
                default_root_z=req.target_height,
            )
            self._apply_terminal_landing(
                str(csv_path),
                target_qpos,
                blend_frames=self.terminal_blend_frames,
                hold_frames=max(1, self.target_hold_frames),
                preserve_root_z=True,
            )

        canonical = np.loadtxt(csv_path, delimiter=",").astype(np.float32)
        if canonical.ndim == 1:
            canonical = canonical[None, :]
        world = P.transform_qpos_traj_se2(canonical, req.start_xy, req.heading, anchor_start=True)
        return world, self.fps


# --------------------------------------------------------------------------- #
# Subprocess engine (fallback): run the proven generate_g1_walk_5m.py verbatim  #
# --------------------------------------------------------------------------- #
def plan_via_subprocess(req: P.PlanRequest, ardy_repo: Path, work_dir: Path,
                        python_exe: str, fps: float = 25.0) -> tuple[np.ndarray, float]:
    work_dir.mkdir(parents=True, exist_ok=True)
    stem = work_dir / f"{req.id}"
    cmd = [
        python_exe,
        str(ardy_repo / "workspace" / "generate_g1_walk_5m.py"),
        "--distance", str(req.distance),
        "--duration", str(req.duration),
        "--output", str(stem),
        "--seed", str(req.seed),
        "--prompt", req.prompt,
        "--cfg-weight", *[str(x) for x in req.cfg_weight],
        "--target-height", str(req.target_height),
    ]
    if req.target_joint_qpos is not None:
        cmd += ["--target-joint-qpos", *[str(x) for x in req.target_joint_qpos]]
    if not req.reach_target_pose:
        cmd += ["--constraint-mode", "root2d", "--no-terminal-landing"]
    print("[planner:subprocess] +", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ardy_repo), check=True)
    canonical = np.loadtxt(str(stem) + ".csv", delimiter=",").astype(np.float32)
    if canonical.ndim == 1:
        canonical = canonical[None, :]
    world = P.transform_qpos_traj_se2(canonical, req.start_xy, req.heading, anchor_start=True)
    return world, fps


# --------------------------------------------------------------------------- #
# Server loop                                                                  #
# --------------------------------------------------------------------------- #
def serve(args: argparse.Namespace) -> None:
    ardy_repo = _resolve_ardy_repo(args.ardy_repo)
    paths = P.RuntimePaths(args.runtime).ensure()
    print(f"[planner] runtime={paths.root}  engine={args.engine}  ardy_repo={ardy_repo}", flush=True)

    planner: Optional[ArdyWalkPlanner] = None
    if args.engine == "inprocess":
        planner = ArdyWalkPlanner(
            ardy_repo=ardy_repo,
            model_name=args.model,
            device=args.device,
            checkpoints_dir=args.checkpoints_dir,
            constraint_stride=args.constraint_stride,
            velocity_profile=args.velocity_profile,
            target_hold_frames=args.target_hold_frames,
            terminal_blend_frames=args.terminal_blend_frames,
            diffusion_steps=args.diffusion_steps,
        )

    def handle(req: P.PlanRequest) -> None:
        t0 = time.time()
        try:
            if planner is not None:
                world, fps = planner.plan(req)
            else:
                world, fps = plan_via_subprocess(
                    req, ardy_repo, THIS_DIR / "runtime" / "_ardy_work", sys.executable
                )
            # Debug copy of the world-frame plan.
            np.save(paths.plans / f"{req.id}.npy", world)
            P.write_response(paths, req.id, world, fps=fps, ok=True,
                             final_root_xy=world[-1, :2])
            print(f"[planner] plan '{req.id}' ok: frames={world.shape[0]} "
                  f"start={np.round(world[0,:2],3).tolist()} end={np.round(world[-1,:2],3).tolist()} "
                  f"({time.time()-t0:.1f}s)", flush=True)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            P.write_response(paths, req.id, None, fps=0.0, ok=False, error=repr(exc))
            print(f"[planner] plan '{req.id}' FAILED: {exc}", flush=True)

    print("[planner] serving. waiting for requests ... (Ctrl-C to stop)", flush=True)
    seen: set[str] = set()
    while True:
        reqs = sorted(paths.requests.glob("*.json"))
        for rp in reqs:
            req_id = rp.stem
            if req_id in seen:
                continue
            try:
                req = P.PlanRequest.from_json(rp.read_text())
            except Exception:  # partially written; retry next tick
                continue
            seen.add(req_id)
            print(f"[planner] got request '{req_id}'", flush=True)
            handle(req)
            try:
                rp.unlink()
            except OSError:
                pass
        time.sleep(args.poll_interval)


def run_once(args: argparse.Namespace) -> None:
    """Generate a single plan from CLI args (validate the planner standalone)."""
    ardy_repo = _resolve_ardy_repo(args.ardy_repo)
    paths = P.RuntimePaths(args.runtime).ensure()
    req = P.PlanRequest(
        id=args.req_id,
        start_xy=[args.start_xy[0], args.start_xy[1]],
        heading=args.heading,
        distance=args.distance,
        duration=args.duration,
        target_joint_qpos=None,
        reach_target_pose=not args.no_target_pose,
        seed=args.seed,
        cfg_weight=args.cfg_weight,
        target_height=args.target_height,
    )
    if args.engine == "inprocess":
        planner = ArdyWalkPlanner(
            ardy_repo=ardy_repo, model_name=args.model, device=args.device,
            checkpoints_dir=args.checkpoints_dir, constraint_stride=args.constraint_stride,
            velocity_profile=args.velocity_profile, target_hold_frames=args.target_hold_frames,
            terminal_blend_frames=args.terminal_blend_frames, diffusion_steps=args.diffusion_steps,
        )
        world, fps = planner.plan(req)
    else:
        world, fps = plan_via_subprocess(req, ardy_repo, THIS_DIR / "runtime" / "_ardy_work", sys.executable)
    out = paths.plans / f"{req.id}.npy"
    np.save(out, world)
    P.write_response(paths, req.id, world, fps=fps, ok=True, final_root_xy=world[-1, :2])
    print(f"[planner] one-shot plan '{req.id}': frames={world.shape[0]} fps={fps:g} "
          f"start={np.round(world[0,:2],3).tolist()} end={np.round(world[-1,:2],3).tolist()}")
    print(f"[planner] saved {out} and {paths.response_path(req.id)}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--serve", action="store_true", help="Run the request/response server loop (default).")
    mode.add_argument("--once", action="store_true", help="Generate one plan from CLI args and exit.")
    ap.add_argument("--engine", choices=("inprocess", "subprocess"), default="inprocess")
    ap.add_argument("--runtime", default=None, help="Shared runtime dir (default: ardy_sonic/runtime).")
    ap.add_argument("--ardy-repo", default=None, help="Ardy repo root (default: $ARDY_REPO or /home/hslee/IsaacLab_ws/ardy).")
    ap.add_argument("--model", default="g1")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--checkpoints-dir", default=None)
    ap.add_argument("--constraint-stride", type=int, default=5)
    ap.add_argument("--velocity-profile", choices=("smoothstep", "linear", "cruise-decel"), default="smoothstep")
    ap.add_argument("--target-hold-frames", type=int, default=8)
    ap.add_argument("--terminal-blend-frames", type=int, default=12)
    ap.add_argument("--diffusion-steps", type=int, default=None)
    ap.add_argument("--poll-interval", type=float, default=0.2)
    # --once params
    ap.add_argument("--req-id", default="once")
    ap.add_argument("--start-xy", type=float, nargs=2, default=[0.0, 0.0])
    ap.add_argument("--heading", type=float, default=0.0)
    ap.add_argument("--distance", type=float, default=5.0)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cfg-weight", type=float, nargs=2, default=[2.0, 3.0])
    ap.add_argument("--target-height", type=float, default=0.72)
    ap.add_argument("--no-target-pose", action="store_true", help="Skip the final full-body pose landing.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.once:
        run_once(args)
    else:
        serve(args)


if __name__ == "__main__":
    main()
