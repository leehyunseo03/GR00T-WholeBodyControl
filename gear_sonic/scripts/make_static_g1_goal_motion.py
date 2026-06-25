#!/usr/bin/env python3
"""Create a static G1 29-DOF goal motion for SONIC evaluation."""

import argparse
from pathlib import Path

import joblib
import numpy as np


ISAACLAB_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

MUJOCO_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

DOF_AXIS_MUJOCO = np.array(
    [
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="custom_motions/robot_static_goal/static_goal.pkl")
    parser.add_argument("--smpl-output", default="custom_motions/smpl_static_goal/static_goal.pkl")
    parser.add_argument("--name", default="static_goal")
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--root-height", type=float, default=0.76)
    parser.add_argument(
        "--order",
        choices=["isaaclab", "mujoco"],
        default="isaaclab",
        help="Order of --dof values. Default matches IsaacLab action log order.",
    )
    parser.add_argument(
        "--dof",
        nargs=29,
        type=float,
        required=True,
        metavar="RAD",
        help="29 joint targets in radians.",
    )
    return parser.parse_args()


def to_mujoco_order(dof: np.ndarray, order: str) -> np.ndarray:
    if order == "mujoco":
        return dof

    by_name = dict(zip(ISAACLAB_JOINT_NAMES, dof, strict=True))
    return np.array([by_name[name] for name in MUJOCO_JOINT_NAMES], dtype=np.float32)


def main() -> None:
    args = parse_args()
    dof_input = np.array(args.dof, dtype=np.float32)
    dof_mujoco = to_mujoco_order(dof_input, args.order)

    num_frames = max(2, int(round(args.seconds * args.fps)))
    root_trans = np.zeros((num_frames, 3), dtype=np.float32)
    root_trans[:, 2] = args.root_height

    root_rot_xyzw = np.zeros((num_frames, 4), dtype=np.float32)
    root_rot_xyzw[:, 3] = 1.0

    dof = np.repeat(dof_mujoco[None, :], num_frames, axis=0)
    pose_aa = np.zeros((num_frames, 30, 3), dtype=np.float32)
    pose_aa[:, 1:, :] = DOF_AXIS_MUJOCO[None, :, :] * dof[:, :, None]

    robot_entry = {
        args.name: {
            "root_trans_offset": root_trans,
            "pose_aa": pose_aa,
            "dof": dof,
            "root_rot": root_rot_xyzw,
            "smpl_joints": np.zeros((num_frames, 24, 3), dtype=np.float32),
            "fps": args.fps,
        }
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(robot_entry, output, compress=True)

    smpl_output = Path(args.smpl_output)
    smpl_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pose_aa": np.zeros((num_frames, 72), dtype=np.float32),
            "transl": root_trans.copy(),
            "smpl_joints": np.zeros((num_frames, 24, 3), dtype=np.float32),
            "fps": float(args.fps),
            "original_pose_aa": np.zeros((num_frames, 72), dtype=np.float32),
            "original_fps": float(args.fps),
        },
        smpl_output,
        compress=True,
    )

    print(f"Wrote robot goal: {output}")
    print(f"Wrote dummy SMPL: {smpl_output}")
    print(f"Input order: {args.order}")
    print("IsaacLab order values:")
    for name, value in zip(ISAACLAB_JOINT_NAMES, dof_input, strict=True):
        print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    main()
