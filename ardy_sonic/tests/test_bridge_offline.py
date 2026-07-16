"""Offline end-to-end test of the Ardy<->SONIC bridge (no Isaac Sim, no Ardy model).

Exercises the real ``ArdyReplanCallback`` control loop and the real ``protocol``
transform/IO against:
  * a fake planner thread that returns a straight canonical walk transformed with
    the same ``protocol`` placement the real Ardy planner uses (including the
    end-anchored settle plans), and
  * a fake SONIC command/env that reports a robot pose (root xy + yaw + 29 joints)
    and "tracks" the installed segment with drift.

It verifies the request/response handshake, live-segment install, the continuous
walk -> exact-landing flow (the final plan is tracked to completion instead of
being cut for a replan), and that arrival is only declared when root xy AND yaw
AND all 29 joints converge to the goal pose.

Run inside the container (needs torch):
    /workspace/isaaclab/isaaclab.sh -p \
        /workspace/GR00T-WholeBodyControl/ardy_sonic/tests/test_bridge_offline.py
"""

import math
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch

# The fake env below stores root quaternions as wxyz directly. The callback's
# _robot_qpos_mujoco passes IsaacLab-read quats through gear_sonic's
# quaternion_adapter, which by default treats them as xyzw (IsaacLab 3.x) and
# would scramble the fake's wxyz values -- so pin the older wxyz boundary here.
os.environ["GEAR_SONIC_ISAACLAB_XYZW"] = "0"

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ardy_sonic import protocol as P
from ardy_sonic.ardy_replan_callback import ArdyReplanCallback

TARGET_JOINTS = np.linspace(-0.3, 0.3, 29).astype(np.float32)  # non-trivial 29-DOF target


def _fake_planner_loop(paths: P.RuntimePaths, stop: threading.Event, response_delay_s: float = 0.0) -> None:
    seen: set[str] = set()
    while not stop.is_set():
        for rp in sorted(paths.requests.glob("*.json")):
            if rp.stem in seen:
                continue
            try:
                req = P.PlanRequest.from_json(rp.read_text())
            except Exception:
                continue
            seen.add(rp.stem)
            if response_delay_s > 0.0:
                time.sleep(response_delay_s)
            fps = 25.0
            T = max(2, int(round(req.duration * fps)))
            canon = np.zeros((T, 36), np.float32)
            canon[:, 0] = np.linspace(0, req.distance, T)
            canon[:, 2] = 0.75
            canon[:, 3] = 1.0
            if req.reach_target_pose:
                # mimic the terminal landing: last frames hold the exact target pose
                tj = np.asarray(req.target_joint_qpos if req.target_joint_qpos is not None
                                else np.zeros(29), np.float32)
                hold = min(8, T)
                canon[-hold:, 7:36] = tj
            world = P.transform_qpos_traj_se2(canon, req.start_xy, req.heading,
                                              anchor_start=True)
            P.write_response(paths, req.id, world, fps=fps, ok=True, final_root_xy=world[-1, :2])
            try:
                rp.unlink()
            except OSError:
                pass
        time.sleep(0.02)


class _FakeData:
    def __init__(self):
        self.root_pos_w = torch.zeros(1, 3)
        self.root_pos_w[0, 2] = 0.78
        self.root_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # wxyz identity
        self.joint_pos = torch.zeros(1, 29)


class _FakeCommand:
    """Tracks the installed segment's END pose (xy + yaw + joints) with drift."""

    def __init__(self):
        self.robot = type("R", (), {"data": _FakeData()})()
        self._env = type("E", (), {"scene": type("S", (), {"env_origins": torch.zeros(1, 3)})()})()
        self.isaaclab_to_mujoco_dof = torch.arange(29)
        self.device = "cpu"
        self._seg_end = None  # (xy, yaw, joints) of the installed segment's last frame
        self.install_count = 0

    def install_live_qpos_segment(self, qpos, fps, env_ids, reset_time=True, install_at_current_time=False):
        q = torch.as_tensor(qpos).cpu().numpy()
        assert q.ndim == 2 and q.shape[1] >= 36, q.shape
        assert q.shape[0] >= 2
        last = q[-1]
        self._seg_end = (
            last[:2].copy(),
            P.yaw_from_quat_wxyz(last[3:7]),
            last[7:36].copy(),
        )
        self.install_count += 1

    def advance(self, rng):
        if self._seg_end is None:
            return
        end_xy, end_yaw, end_joints = self._seg_end
        data = self.robot.data
        # root xy: bounded step toward the segment end + noise
        cur = data.root_pos_w[0, :2].numpy()
        delta = end_xy - cur
        dist = float(np.linalg.norm(delta))
        step = min(0.03, dist)
        if dist > 1e-6:
            cur = cur + delta / dist * step
        cur = cur + rng.normal(0, 0.004, size=2)
        data.root_pos_w[0, 0] = float(cur[0])
        data.root_pos_w[0, 1] = float(cur[1])
        # yaw: rotate toward the segment-end yaw + noise
        cur_yaw = P.yaw_from_quat_wxyz(data.root_quat_w[0].numpy())
        dyaw = (end_yaw - cur_yaw + np.pi) % (2 * np.pi) - np.pi
        new_yaw = cur_yaw + np.clip(dyaw, -0.05, 0.05) + rng.normal(0, 0.002)
        data.root_quat_w[0] = torch.tensor(
            [math.cos(new_yaw / 2), 0.0, 0.0, math.sin(new_yaw / 2)], dtype=torch.float32
        )
        # joints: first-order tracking toward the segment-end joints + noise
        j = data.joint_pos[0].numpy()
        j = j + 0.15 * (end_joints - j) + rng.normal(0, 0.002, size=29)
        data.joint_pos[0] = torch.tensor(j, dtype=torch.float32)


class _FakeEnv:
    def __init__(self):
        self.motion_command = _FakeCommand()
        self.num_envs = 1


def run_scenario(
    name: str,
    planner_delay_s: float = 0.0,
    sleep_per_step_s: float = 0.0,
    callback_overrides: dict | None = None,
    **goal_kwargs,
) -> None:
    print(f"\n######## scenario: {name} {goal_kwargs} ########")
    runtime = Path(tempfile.mkdtemp(prefix="ardy_sonic_test_"))
    paths = P.RuntimePaths(runtime).ensure()
    stop = threading.Event()
    th = threading.Thread(target=_fake_planner_loop, args=(paths, stop, planner_delay_s), daemon=True)
    th.start()

    env = _FakeEnv()
    overrides = dict(callback_overrides or {})
    cb = ArdyReplanCallback(
        target_joint_qpos=TARGET_JOINTS.tolist(),
        **goal_kwargs,
        arrival_radius=0.10, yaw_tol_deg=8.0, joint_tol_rad=0.35,
        final_leg_distance=1.0, landing_hold_seconds=1.0,
        stall_patience=3, min_improve=0.05, hold_seconds=1.0,
        show_target_markers=False, show_plan_markers=False,
        max_plan_distance=overrides.pop("max_plan_distance", 6.0),
        seconds_per_meter=overrides.pop("seconds_per_meter", 2.0),
        track_fraction=overrides.pop("track_fraction", 0.9),
        replan_request_fraction=overrides.pop("replan_request_fraction", 0.8),
        max_async_start_xy_error=overrides.pop("max_async_start_xy_error", 0.75),
        max_replans=40, max_steps=12000, control_hz=50.0, placeholder_frames=1500,
        plan_timeout_s=30.0, runtime_dir=str(runtime), verbose=True,
        **overrides,
    )
    cb.on_step_end(env=env)

    rng = np.random.default_rng(0)
    done = False
    for _ in range(12000):
        env.motion_command.advance(rng)
        done = cb.eval_step(env, None)
        if done:
            break
        if sleep_per_step_s > 0.0:
            time.sleep(sleep_per_step_s)
    stop.set()
    th.join(timeout=2)

    data = env.motion_command.robot.data
    final_xy = data.root_pos_w[0, :2].numpy()
    final_yaw = P.yaw_from_quat_wxyz(data.root_quat_w[0].numpy())
    final_joints = data.joint_pos[0].numpy()
    pos_err = float(np.linalg.norm(cb._goal_xy - final_xy))
    yaw_err = abs((final_yaw - cb._goal_heading + np.pi) % (2 * np.pi) - np.pi)
    joint_err = float(np.abs(final_joints - TARGET_JOINTS).max())
    print(f"\n==== RESULT ({name}) ====")
    print("goal_xy    :", cb._goal_xy.round(3).tolist())
    print("final_xy   :", final_xy.round(3).tolist())
    print("pos_err    :", round(pos_err, 3), " yaw_err(deg):", round(math.degrees(yaw_err), 2),
          " joint_err :", round(joint_err, 3))
    print("installs   :", env.motion_command.install_count)
    print("replans    :", cb._replans, " landings:", cb._landings, " steps:", cb._step)

    assert done, "callback never signalled done"
    assert cb._arrived, "callback stopped without declaring arrival"
    assert cb._landings >= 1, "arrival must come from a completed exact landing"
    assert pos_err <= cb.arrival_radius + 0.02, f"pos_err too large: {pos_err}"
    assert yaw_err <= cb.yaw_tol_rad + 0.02, f"yaw_err too large: {yaw_err}"
    assert joint_err <= cb.joint_tol_rad + 0.02, f"joint_err too large: {joint_err}"
    assert env.motion_command.install_count >= 1
    assert cb._pending_plan is None, "no async request should be left pending after completion"
    print(f"scenario '{name}' OK")


def main() -> None:
    # straight ahead (goal heading == start heading)
    run_scenario("forward-5m", goal_mode="forward", forward_meters=5.0)
    # off-axis absolute goal: exercises the SE(2) rotation and the yaw gate
    # (goal heading = atan2(2, 3) = 33.7 deg while the robot starts facing 0 deg)
    run_scenario("absolute-3-2", goal_mode="absolute", goal_xy=[3.0, 2.0])
    # chunked plan: exercises non-blocking async replanning. The fake planner
    # intentionally responds after the request step while eval keeps advancing.
    run_scenario(
        "forward-5m-async-prefetch",
        goal_mode="forward",
        forward_meters=5.0,
        planner_delay_s=0.05,
        sleep_per_step_s=0.001,
        callback_overrides={"max_plan_distance": 2.0, "replan_request_fraction": 0.8},
    )
    print("\nBRIDGE CONTROL-LOOP TEST OK")


if __name__ == "__main__":
    main()
