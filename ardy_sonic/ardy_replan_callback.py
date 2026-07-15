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
        hold_seconds: float = 3.0,           # after arriving, hold at the destination this long before stopping
        show_target_markers: bool = True,    # draw the destination 29-DOF pose as blue spheres
        target_marker_radius: float = 0.05,  # radius of each destination sphere (m)
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
        self.hold_seconds = max(0.0, float(hold_seconds))
        self.show_target_markers = bool(show_target_markers)
        self.target_marker_radius = float(target_marker_radius)
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
        self._goal_heading = 0.0
        # arrival hold
        self._arrived = False
        self._hold_until_step = 0
        # destination markers
        self._markers = None
        self._markers_failed = False
        self._target_body_pos_w = None

    # -- lifecycle hooks -------------------------------------------------------
    def on_step_end(self, *args, **kwargs) -> None:
        """Called once before the eval loop. Clear stale responses + create markers."""
        for f in self._paths.responses.glob("plan_*.npz"):
            try:
                f.unlink()
            except OSError:
                pass
        self._log(f"[ardy_replan] runtime={self._paths.root} goal_mode={self.goal_mode} "
                  f"forward_meters={self.forward_meters} arrival_radius={self.arrival_radius}")
        # Create the destination-pose marker prim BEFORE the eval loop so it attaches to
        # the renderer (creating markers mid-loop can leave them invisible). Positions
        # are filled in on the first eval_step once the goal is known.
        if self.show_target_markers and self._markers is None:
            try:
                from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
                import isaaclab.sim as sim_utils

                cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/ArdyDestinationPose",
                    markers={
                        "pt": sim_utils.SphereCfg(
                            radius=self.target_marker_radius,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.0, 0.25, 1.0), opacity=0.9
                            ),
                        )
                    },
                )
                self._markers = VisualizationMarkers(cfg)
                self._markers.set_visibility(True)
                self._log("[ardy_replan] destination marker prim created at /Visuals/ArdyDestinationPose")
            except Exception as exc:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                self._markers_failed = True
                self._log(f"[ardy_replan] marker prim creation failed: {exc}; continuing without markers.")

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
            d = self._goal_xy - self._start_xy
            self._goal_heading = math.atan2(float(d[1]), float(d[0])) if float(np.linalg.norm(d)) > 1e-6 else cur_yaw
            self._initialized = True
            self._need_replan = True
            self._log(f"[ardy_replan] start_xy={cur_xy.round(3).tolist()} "
                      f"start_yaw={math.degrees(cur_yaw):.1f}deg goal_xy={self._goal_xy.round(3).tolist()} "
                      f"goal_heading={math.degrees(self._goal_heading):.1f}deg")

        # Draw / refresh the destination pose as blue spheres (prim created in on_step_end).
        self._update_target_markers(command, env_idx)

        self._step += 1

        # Holding at the destination after arrival: keep tracking (the reference tail
        # holds the target pose) until the hold window elapses, then stop.
        if self._arrived:
            if self._step >= self._hold_until_step:
                self._log("[ardy_replan] hold complete; stopping at destination.")
                self._done = True
                return True
            return False

        replan_now = self._need_replan or (self._step >= self._seg_end_step)
        if replan_now:
            remaining = float(np.linalg.norm(self._goal_xy - cur_xy))
            self._log(f"[ardy_replan] step={self._step} cur_xy={cur_xy.round(3).tolist()} "
                      f"remaining={remaining:.3f} replans={self._replans}")
            if remaining <= self.arrival_radius:
                self._arrived = True
                self._hold_until_step = self._step + max(1, int(self.hold_seconds * self.control_hz))
                self._log(f"[ardy_replan] ARRIVED: remaining={remaining:.3f} <= {self.arrival_radius}. "
                          f"Holding {self.hold_seconds:.1f}s at destination.")
                return False
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

    # -- destination markers ---------------------------------------------------
    def _update_target_markers(self, command, env_idx: int) -> None:
        """Fill the destination-pose spheres with FK body positions, then re-visualize."""
        if self._markers is None or self._markers_failed or not self._initialized:
            return
        try:
            if self._target_body_pos_w is None:
                self._target_body_pos_w = self._compute_target_body_pos(command, env_idx)
                if self._target_body_pos_w is not None:
                    self._log(f"[ardy_replan] destination pose = {int(self._target_body_pos_w.shape[0])} "
                              f"blue spheres at goal_xy={self._goal_xy.round(3).tolist()} "
                              f"z={self.target_height:.2f} heading={math.degrees(self._goal_heading):.1f}deg")
            if self._target_body_pos_w is not None:
                self._markers.visualize(translations=self._target_body_pos_w)
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            self._markers_failed = True
            self._log(f"[ardy_replan] marker update failed: {exc}; continuing without markers.")

    def _compute_target_body_pos(self, command, env_idx: int):
        """World body positions of the destination 29-DOF pose via the motion-lib FK."""
        h = self._goal_heading
        quat = [math.cos(h * 0.5), 0.0, 0.0, math.sin(h * 0.5)]  # wxyz, yaw about +z
        joints = self.target_joint_qpos if self.target_joint_qpos is not None else [0.0] * 29
        row = [float(self._goal_xy[0]), float(self._goal_xy[1]), float(self.target_height), *quat, *joints]
        parser = command.motion_lib.mesh_parsers
        try:
            qpos = torch.tensor(row, dtype=torch.float32, device=command.device)
            body_pos, _ = parser.qpos_to_global_transforms(qpos, root_quat_wxyz=True)  # (num_bodies, 3)
        except Exception:  # noqa: BLE001  # fall back to the exact FK path install uses
            qpos2 = torch.tensor([row, row], dtype=torch.float32, device="cpu")  # 2 frames for velocities
            pose_aa = command._qpos_to_pose_aa_for_live_segment(qpos2)  # noqa: SLF001  (2, 30, 3)
            curr = parser.fk_batch(
                pose_aa.unsqueeze(0), qpos2[:, :3].unsqueeze(0),
                return_full=True, fps=int(command.motion_lib.target_fps),
                target_fps=int(command.motion_lib.target_fps), interpolate_data=False,
                use_parallel_fk=getattr(command.motion_lib, "use_parallel_fk", False),
            )
            body_pos = curr.global_translation[0, 0].to(command.device)
        body_pos = body_pos.reshape(-1, 3)
        env_origins = getattr(command._env.scene, "env_origins", None)  # noqa: SLF001
        if env_origins is not None:
            body_pos = body_pos + env_origins[env_idx].to(body_pos.device)
        return body_pos

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
