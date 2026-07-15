"""Offline end-to-end test of the Ardy<->SONIC bridge (no Isaac Sim, no Ardy model).

Exercises the real ``ArdyReplanCallback`` control loop and the real ``protocol``
transform/IO against:
  * a fake planner thread that returns a straight canonical walk transformed with
    the same ``protocol.transform_qpos_traj_se2`` the real Ardy planner uses, and
  * a fake SONIC command/env that reports a robot pose and "tracks" the installed
    segment with drift.

It verifies the request/response handshake, live-segment install, receding-horizon
replanning cadence, and arrival detection converge the robot to the goal.

Run inside the container (needs torch):
    /isaac-sim/kit/python/bin/python3 \
        /workspace/GR00T-WholeBodyControl/ardy_sonic/tests/test_bridge_offline.py
"""

import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ardy_sonic import protocol as P
from ardy_sonic.ardy_replan_callback import ArdyReplanCallback


def _fake_planner_loop(paths: P.RuntimePaths, stop: threading.Event) -> None:
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
            fps = 25.0
            T = max(2, int(round(req.duration * fps)))
            canon = np.zeros((T, 36), np.float32)
            canon[:, 0] = np.linspace(0, req.distance, T)
            canon[:, 2] = 0.75
            canon[:, 3] = 1.0
            world = P.transform_qpos_traj_se2(canon, req.start_xy, req.heading, anchor_start=True)
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
    def __init__(self):
        self.robot = type("R", (), {"data": _FakeData()})()
        self._env = type("E", (), {"scene": type("S", (), {"env_origins": torch.zeros(1, 3)})()})()
        self.isaaclab_to_mujoco_dof = torch.arange(29)
        self.device = "cpu"
        self._seg_end_xy = None
        self.install_count = 0

    def install_live_qpos_segment(self, qpos, fps, env_ids, reset_time=True):
        q = torch.as_tensor(qpos).cpu().numpy()
        assert q.ndim == 2 and q.shape[1] >= 36, q.shape
        assert q.shape[0] >= 2
        self._seg_end_xy = q[-1, :2].copy()
        self.install_count += 1

    def advance(self, rng):
        if self._seg_end_xy is None:
            return
        cur = self.robot.data.root_pos_w[0, :2].numpy()
        delta = self._seg_end_xy - cur
        dist = float(np.linalg.norm(delta))
        step = min(0.03, dist)
        if dist > 1e-6:
            cur = cur + delta / dist * step
        cur = cur + rng.normal(0, 0.004, size=2)
        self.robot.data.root_pos_w[0, 0] = float(cur[0])
        self.robot.data.root_pos_w[0, 1] = float(cur[1])


class _FakeEnv:
    def __init__(self):
        self.motion_command = _FakeCommand()
        self.num_envs = 1


def main() -> None:
    runtime = Path(tempfile.mkdtemp(prefix="ardy_sonic_test_"))
    paths = P.RuntimePaths(runtime).ensure()
    stop = threading.Event()
    th = threading.Thread(target=_fake_planner_loop, args=(paths, stop), daemon=True)
    th.start()

    env = _FakeEnv()
    cb = ArdyReplanCallback(
        goal_mode="forward", forward_meters=5.0, arrival_radius=0.30,
        max_plan_distance=6.0, seconds_per_meter=2.0, track_fraction=0.9,
        max_replans=20, max_steps=4000, control_hz=50.0, placeholder_frames=1500,
        plan_timeout_s=30.0, runtime_dir=str(runtime), verbose=True,
    )
    cb.on_step_end(env=env)

    rng = np.random.default_rng(0)
    done = False
    for _ in range(4000):
        env.motion_command.advance(rng)
        done = cb.eval_step(env, None)
        if done:
            break
    stop.set()
    th.join(timeout=2)

    final_xy = env.motion_command.robot.data.root_pos_w[0, :2].numpy()
    remaining = float(np.linalg.norm(cb._goal_xy - final_xy))
    print("\n==== RESULT ====")
    print("goal_xy   :", cb._goal_xy.round(3).tolist())
    print("final_xy  :", final_xy.round(3).tolist())
    print("remaining :", round(remaining, 3))
    print("installs  :", env.motion_command.install_count)
    print("replans   :", cb._replans, " steps:", cb._step)

    assert done, "callback never signalled done"
    assert remaining <= cb.arrival_radius + 1e-6, f"did not arrive: {remaining}"
    assert env.motion_command.install_count >= 1
    print("BRIDGE CONTROL-LOOP TEST OK")


if __name__ == "__main__":
    main()
