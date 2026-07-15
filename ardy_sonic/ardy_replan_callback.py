"""Receding-horizon Ardy planner callback for GEAR-SONIC eval.

Runs inside ``gear_sonic/eval_agent_trl.py`` as an ``eval_step`` callback. It closes
the loop between the Ardy motion planner (a separate process on the host, in the
``ardy`` conda env) and the SONIC whole-body tracking policy (this container):

  * The Ardy plan is generated in a canonical frame and returned to us as a
    world-frame MuJoCo qpos ``(T, 36)`` over the shared ``runtime/`` directory.
  * We inject it into the active tracking reference with
    ``TrackingCommand.install_live_qpos_segment`` and let SONIC physically track it.
  * After tracking most of a segment, we read the robot's *actual* pose and ask
    Ardy to re-plan the residual path to the goal -- correcting the drift between
    the kinematic plan and the physically tracked motion -- until the robot arrives.

Start the host planner first:  ``python ardy_sonic/ardy_planner_server.py --serve``
Then launch the tracker with ``ardy_sonic/run_ardy_sonic.sh`` (registers this
callback via Hydra ``++eval_callbacks=[ardy_replan]``).
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from ardy_sonic import protocol as P


class ArdyReplanCallback:
    def __init__(
        self,
        goal_mode: str = "forward",          # "forward" (start + forward_meters along start heading) or "absolute"
        forward_meters: float = 5.0,
        goal_xy: Optional[Sequence[float]] = None,   # used when goal_mode == "absolute"
        target_joint_qpos: Optional[Sequence[float]] = None,  # terminal 29-DOF pose (None -> zeros)
        target_height: float = 0.72,
        arrival_radius: float = 0.30,        # stop when within this planar distance of the goal (m)
        max_plan_distance: float = 6.0,      # cap per-plan forward distance (m)
        seconds_per_meter: float = 2.0,      # plan duration = max(min_duration, distance * this)
        min_duration: float = 2.5,
        track_fraction: float = 0.9,         # replan after tracking this fraction of the installed segment
        max_replans: int = 12,
        max_steps: int = 6000,               # hard stop on total control steps
        control_hz: float = 50.0,
        placeholder_frames: int = 1500,      # length of the placeholder clip (caps installable segment)
        env_index: int = 0,
        plan_timeout_s: float = 900.0,
        prompt: str = "A person walks forward at a steady natural pace and comes to a stop.",
        cfg_weight: Sequence[float] = (2.0, 3.0),
        seed: int = 0,
        runtime_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        self.goal_mode = goal_mode
        self.forward_meters = float(forward_meters)
        self.goal_xy_arg = None if goal_xy is None else [float(v) for v in goal_xy]
        self.target_joint_qpos = None if target_joint_qpos is None else [float(v) for v in target_joint_qpos]
        self.target_height = float(target_height)
        self.arrival_radius = float(arrival_radius)
        self.max_plan_distance = float(max_plan_distance)
        self.seconds_per_meter = float(seconds_per_meter)
        self.min_duration = float(min_duration)
        self.track_fraction = float(np.clip(track_fraction, 0.1, 1.0))
        self.max_replans = int(max_replans)
        self.max_steps = int(max_steps)
        self.control_hz = float(control_hz)
        self.placeholder_frames = int(placeholder_frames)
        self.env_index = int(env_index)
        self.plan_timeout_s = float(plan_timeout_s)
        self.prompt = prompt
        self.cfg_weight = [float(v) for v in cfg_weight]
        self.seed = int(seed)
        self.verbose = bool(verbose)

        self._paths = P.RuntimePaths(runtime_dir).ensure()
        # eval_agent_trl sets callback.model if the attribute exists.
        self.model = None

        self._step = 0
        self._req_counter = 0
        self._replans = 0
        self._initialized = False
        self._done = False
        self._need_replan = True
        self._seg_end_step = 0
        self._goal_xy = np.zeros(2, dtype=np.float32)
        self._start_xy = np.zeros(2, dtype=np.float32)

    # -- lifecycle hooks -------------------------------------------------------
    def on_step_end(self, *args, **kwargs) -> None:
        """Called once before the eval loop. Clear any stale responses."""
        for f in self._paths.responses.glob("plan_*.npz"):
            try:
                f.unlink()
            except OSError:
                pass
        self._log(f"[ardy_replan] runtime={self._paths.root} goal_mode={self.goal_mode} "
                  f"forward_meters={self.forward_meters} arrival_radius={self.arrival_radius}")

    # -- main step -------------------------------------------------------------
    def eval_step(self, env, _results) -> bool:
        if self._done:
            return True
        command = env.motion_command
        env_idx = min(self.env_index, env.num_envs - 1)
        cur = self._robot_qpos_mujoco(command, env_idx)
        cur_xy = cur[:2]
        cur_yaw = P.yaw_from_quat_wxyz(cur[3:7])

        if not self._initialized:
            self._start_xy = cur_xy.copy()
            self._goal_xy = self._compute_goal(cur_xy, cur_yaw)
            self._initialized = True
            self._need_replan = True
            self._log(f"[ardy_replan] start_xy={cur_xy.round(3).tolist()} "
                      f"start_yaw={math.degrees(cur_yaw):.1f}deg goal_xy={self._goal_xy.round(3).tolist()}")

        self._step += 1

        replan_now = self._need_replan or (self._step >= self._seg_end_step)
        if replan_now:
            remaining = float(np.linalg.norm(self._goal_xy - cur_xy))
            self._log(f"[ardy_replan] step={self._step} cur_xy={cur_xy.round(3).tolist()} "
                      f"remaining={remaining:.3f} replans={self._replans}")
            if remaining <= self.arrival_radius:
                self._log(f"[ardy_replan] ARRIVED: remaining={remaining:.3f} <= {self.arrival_radius}. Stopping.")
                self._done = True
                return True
            if self._replans >= self.max_replans:
                self._log(f"[ardy_replan] max_replans={self.max_replans} reached; remaining={remaining:.3f}. Stopping.")
                self._done = True
                return True

            dist = min(remaining, self.max_plan_distance)
            reach_target = dist >= remaining - 1e-6
            delta = self._goal_xy - cur_xy
            heading = math.atan2(float(delta[1]), float(delta[0]))
            duration = max(self.min_duration, dist * self.seconds_per_meter)

            try:
                qpos, fps = self._request_plan(cur_xy, heading, dist, duration, reach_target)
            except Exception as exc:  # noqa: BLE001
                self._log(f"[ardy_replan] planning failed: {exc}. Stopping.")
                self._done = True
                return True

            command.install_live_qpos_segment(
                torch.as_tensor(qpos, dtype=torch.float32),
                fps=int(round(fps)),
                env_ids=torch.tensor([env_idx], device=command.device),
                reset_time=True,
            )
            self._replans += 1
            self._need_replan = False

            seg_ctrl_steps = min(int(round(duration * self.control_hz)), self.placeholder_frames)
            self._seg_end_step = self._step + max(1, int(self.track_fraction * seg_ctrl_steps))
            self._log(f"[ardy_replan] installed plan '{qpos.shape[0]}x{qpos.shape[1]}' @ {fps:g}fps "
                      f"dist={dist:.3f} dur={duration:.2f}s reach_target={reach_target} "
                      f"track_until_step={self._seg_end_step}")

        if self._step >= self.max_steps:
            self._log(f"[ardy_replan] max_steps={self.max_steps} reached. Stopping.")
            self._done = True
            return True
        return False

    # -- helpers ---------------------------------------------------------------
    def _compute_goal(self, cur_xy: np.ndarray, cur_yaw: float) -> np.ndarray:
        if self.goal_mode == "absolute":
            if self.goal_xy_arg is None:
                raise ValueError("goal_mode='absolute' requires goal_xy")
            return np.asarray(self.goal_xy_arg, dtype=np.float32)
        # forward: goal is forward_meters ahead along the robot's current heading.
        return (cur_xy + self.forward_meters * np.array([math.cos(cur_yaw), math.sin(cur_yaw)],
                                                        dtype=np.float32)).astype(np.float32)

    def _request_plan(self, cur_xy, heading, dist, duration, reach_target):
        req_id = f"plan_{self._req_counter:04d}"
        self._req_counter += 1
        req = P.PlanRequest(
            id=req_id,
            start_xy=[float(cur_xy[0]), float(cur_xy[1])],
            heading=float(heading),
            distance=float(dist),
            duration=float(duration),
            target_joint_qpos=self.target_joint_qpos,
            reach_target_pose=bool(reach_target),
            seed=self.seed + self._req_counter,
            prompt=self.prompt,
            cfg_weight=self.cfg_weight,
            target_height=self.target_height,
        )
        resp_path = self._paths.response_path(req_id)
        if resp_path.exists():
            resp_path.unlink()
        P.atomic_write_text(self._paths.request_path(req_id), req.to_json())
        self._log(f"[ardy_replan] requested plan '{req_id}' dist={dist:.3f} dur={duration:.2f}s "
                  f"heading={math.degrees(heading):.1f}deg; waiting for planner ...")
        t0 = time.time()
        while not resp_path.exists():
            if time.time() - t0 > self.plan_timeout_s:
                raise TimeoutError(f"planner did not respond within {self.plan_timeout_s}s for '{req_id}'")
            time.sleep(0.1)
        resp = P.read_response(resp_path)
        if not resp["ok"]:
            raise RuntimeError(f"planner error for '{req_id}': {resp['error']}")
        qpos = resp["qpos"]
        if qpos.ndim != 2 or qpos.shape[1] < P.QPOS_DIM or qpos.shape[0] < 2:
            raise ValueError(f"planner returned bad qpos shape {qpos.shape}")
        self._log(f"[ardy_replan] received plan '{req_id}' in {time.time()-t0:.1f}s "
                  f"frames={qpos.shape[0]} final_xy={resp['final_root_xy'].round(3).tolist()}")
        return qpos, resp["fps"]

    def _robot_qpos_mujoco(self, command, env_idx: int) -> np.ndarray:
        """Current robot state as MuJoCo qpos [root xyz(env-local), quat wxyz, 29 joints]."""
        root_pos = command.robot.data.root_pos_w[env_idx].detach().clone()
        env_origins = getattr(command._env.scene, "env_origins", None)  # noqa: SLF001
        if env_origins is not None:
            root_pos = root_pos - env_origins[env_idx].to(root_pos.device)
        root_quat_wxyz = command.robot.data.root_quat_w[env_idx].detach().clone()
        try:
            from gear_sonic.isaac_utils import quaternion_adapter as quat_adapter

            root_quat_wxyz = quat_adapter.isaaclab_to_wxyz(root_quat_wxyz)
        except Exception:  # noqa: BLE001
            pass
        joint_pos_isaac = command.robot.data.joint_pos[env_idx].detach()
        joint_pos_mujoco = joint_pos_isaac[command.isaaclab_to_mujoco_dof]
        qpos = torch.cat([root_pos, root_quat_wxyz, joint_pos_mujoco[:29]], dim=0)
        return qpos.detach().cpu().numpy().astype(np.float32)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)
