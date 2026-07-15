#!/usr/bin/env python3
"""Create a long static G1 placeholder motion clip for the SONIC tracker.

The Ardy<->SONIC tracker installs freshly planned reference segments at run time via
``TrackingCommand.install_live_qpos_segment``. That method *overwrites* an already
loaded motion clip in place and truncates the live segment to the placeholder's
frame count (``replace_len = min(old_len, num_frames)``). So the env still needs a
valid ``motion_file`` at launch, and it must be **at least as long** (in 50 fps
frames) as the longest plan we will ever inject. A 5 m / 10 s walk upsamples to
~500 frames at 50 fps, so the default 1500 frames (30 s) leaves comfortable margin.

The clip content is irrelevant (it is overwritten on the first control step); we
just emit a valid neutral standing pose. This script is self-contained: numpy +
joblib only, no gear_sonic / motionbricks imports, so it runs anywhere the
container python can reach joblib.

Run inside the container:
    /workspace/isaaclab/isaaclab.sh -p \
        GR00T-WholeBodyControl/ardy_sonic/make_placeholder_motion.py
(or any python with numpy + joblib).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

# Per-joint rotation axes in MuJoCo/Ardy G1 joint order (29), matching
# TrackingCommand._qpos_to_pose_aa_for_live_segment (commands.py) exactly.
DOF_AXIS_MUJOCO = np.array(
    [
        [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0],  # left leg
        [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0],  # right leg
        [0, 0, 1], [1, 0, 0], [0, 1, 0],                                    # waist yaw/roll/pitch
        [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],  # left arm
        [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],  # right arm
    ],
    dtype=np.float32,
)


def build_static_entry(frames: int, fps: float, root_z: float, joints29: np.ndarray) -> dict:
    q = np.zeros((frames, 36), dtype=np.float32)
    q[:, 2] = root_z
    q[:, 3] = 1.0  # root quat wxyz = identity
    q[:, 7:36] = joints29[None, :]

    pose_aa = np.zeros((frames, 30, 3), dtype=np.float32)
    # root orientation identity -> zero rotvec at index 0; joints at 1..29.
    pose_aa[:, 1:, :] = DOF_AXIS_MUJOCO[None, :, :] * q[:, 7:36, None]

    root_rot = np.zeros((frames, 4), dtype=np.float32)
    root_rot[:, 3] = 1.0  # xyzw identity (auxiliary; robot loader ignores it)

    return {
        "root_trans_offset": q[:, :3].copy(),
        "pose_aa": pose_aa,
        "dof": q[:, 7:36].copy(),
        "root_rot": root_rot,
        "smpl_joints": np.zeros((frames, 24, 3), dtype=np.float32),
        "fps": float(fps),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path,
                    default=Path(__file__).resolve().parent / "runtime" / "placeholder_motion.pkl")
    ap.add_argument("--name", default="ardy_sonic_placeholder")
    ap.add_argument("--frames", type=int, default=1500, help="Clip length in frames at --fps (>= longest plan).")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--root-z", type=float, default=0.78)
    args = ap.parse_args()

    joints = np.zeros(29, dtype=np.float32)  # neutral standing pose
    entry = build_static_entry(args.frames, args.fps, args.root_z, joints)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({args.name: entry}, args.output, compress=True)
    print(f"Wrote placeholder motion: {args.output}")
    print(f"  key={args.name} frames={args.frames} fps={args.fps:g} "
          f"pose_aa={entry['pose_aa'].shape} root_trans_offset={entry['root_trans_offset'].shape}")
    print("Use as:  ++manager_env.commands.motion.motion_lib_cfg.motion_file=" + str(args.output))
    print("         ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy")


if __name__ == "__main__":
    main()
