import argparse
import os
import time

# SSH/headless sessions usually do not have DISPLAY. Set this before importing
# MotionBricks controllers, because controllers import pynput at module import.
os.environ.setdefault("PYNPUT_BACKEND", "dummy")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/motionbricks_matplotlib")

import mujoco
import numpy as np
import torch as t

from motionbricks.motion_backbone.demo.utils import navigation_demo


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _summarize_qpos(step: int, qpos: np.ndarray) -> str:
    root_pos = qpos[:3]
    root_quat_wxyz = qpos[3:7]
    dof = qpos[7:]
    return (
        f"step={step:05d} "
        f"root_xyz=[{root_pos[0]: .4f}, {root_pos[1]: .4f}, {root_pos[2]: .4f}] "
        f"root_quat_wxyz=[{root_quat_wxyz[0]: .4f}, {root_quat_wxyz[1]: .4f}, "
        f"{root_quat_wxyz[2]: .4f}, {root_quat_wxyz[3]: .4f}] "
        f"dof_mean={dof.mean(): .4f} dof_min={dof.min(): .4f} dof_max={dof.max(): .4f}"
    )


def main(args) -> None:
    if not t.cuda.is_available():
        raise RuntimeError(
            "MotionBricks demo code currently expects CUDA. "
            "Run this on a shell where `python -c \"import torch; print(torch.cuda.is_available())\"` prints True."
        )

    args.has_viewer = 0
    args.return_model_configs = True
    args.return_dataloader = True
    args.recording_dir = None
    args.EXP = args.planner
    args.speed_scale = [float(i) for i in args.speed_scale.split(",")]

    demo_agent = navigation_demo(args)

    np.random.seed(args.random_seed)
    t.manual_seed(args.random_seed)
    demo_agent.full_agent.reset()

    started = time.time()
    for step in range(1, args.max_steps + 1):
        force_idle = step + 100 > args.max_steps

        qpos = demo_agent.full_agent.get_next_frame()
        context_motion_features = demo_agent.full_agent.get_context_motion_features()
        context_mujoco_qpos = demo_agent.full_agent.get_context_mujoco_qpos()
        demo_agent.mj_data.qpos[:] = qpos

        control_signals = demo_agent.controller.generate_control_signals(
            None,
            demo_agent.mj_model,
            demo_agent.mj_data,
            visualize=False,
            control_info={"force_idle": force_idle, "allowed_mode": args.allowed_mode},
        )
        if args.use_qpos:
            control_signals["context_mujoco_qpos"] = context_mujoco_qpos
        else:
            control_signals["context_motion_features"] = context_motion_features

        with t.no_grad():
            demo_agent.full_agent.generate_new_frames(
                control_signals,
                demo_agent.controller.get_controller_dt() * args.generate_dt,
            )

        mujoco.mj_forward(demo_agent.mj_model, demo_agent.mj_data)

        if step == 1 or step % args.print_every == 0 or step == args.max_steps:
            print(_summarize_qpos(step, qpos), flush=True)

    elapsed = time.time() - started
    print(f"finished steps={args.max_steps} elapsed_sec={elapsed:.2f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless MotionBricks G1 demo that prints qpos stats.")

    parser.add_argument(
        "--humanoid_xml",
        type=str,
        default=os.path.join(PROJECT_ROOT, "assets", "skeletons", "g1", "scene_29dof.xml"),
    )
    parser.add_argument("--result_dir", type=str, default=os.path.join(PROJECT_ROOT, "out"))
    parser.add_argument("--data_root", type=str, default=os.path.join(PROJECT_ROOT, "datasets"))
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
    parser.add_argument("--speed_scale", type=str, default="0.8,1.2")
    parser.add_argument("--generate_dt", type=float, default=2.0)

    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--print_every", type=int, default=30)
    parser.add_argument("--random_seed", type=int, default=1234)

    parser.add_argument("--use_qpos", type=int, default=1)
    parser.add_argument("--planner", type=str, default="default")
    parser.add_argument("--allowed_mode", type=str, default=None)
    parser.add_argument("--clips", type=str, default="G1")

    main(parser.parse_args())
