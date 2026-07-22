"""Receding-horizon Ardy planner callback for GEAR-SONIC eval.

Runs inside ``gear_sonic/eval_agent_trl.py`` as an ``eval_step`` callback. It closes
the loop between the Ardy motion planner (a separate process on the host, in the
``ardy`` conda env) and the SONIC whole-body tracking policy (this container):

  * The Ardy plan is generated in a canonical frame and returned to us as a
    world-frame MuJoCo qpos ``(T, 36)`` over the shared ``runtime/`` directory.
  * We inject it into the active tracking reference with
    ``TrackingCommand.install_live_qpos_segment`` and let SONIC physically track it.
  * EVERY plan is START-anchored at the robot's actual current pose (the path
    re-attaches to the G1 on each replan) and -- because the planner pins the
    canonical start to (0,0) and the terminal landing pins the last frames to the
    target -- ends EXACTLY on the goal pose (x, y, heading, 29-DOF joints).

There is no separate settle/correction stage. The robot walks to the goal in one
continuous motion:

  * While far from the goal, the next plan is requested once the current plan is
    about ``replan_request_fraction`` consumed. The sim keeps tracking the
    remaining current reference while the planner works; when a valid response is
    available, it is installed without blocking the eval loop.
  * At a replan boundary where the active plan already reaches the goal and the
    robot is within ``final_leg_distance``, the plan is NOT cut: it is tracked to
    COMPLETION (including the exact-landing frames) plus ``landing_hold_seconds``
    while the installed reference tail freezes on the exact target pose.
  * Arrival is declared when a plan that reaches the goal has been tracked through
    its exact-landing frames. The callback prints the achieved robot errors for
    diagnostics, then holds the destination reference for ``hold_seconds`` before
    stopping.

Every installed plan's root path is drawn as an orange ground-track (marker prim
``/Visuals/ArdyPlanPath``), so each replan is visible re-attaching to the robot;
the destination pose is drawn as blue spheres (``/Visuals/ArdyDestinationPose``).

Start the host planner first:  ``python ardy_sonic/ardy_planner_server.py --serve``
Then launch the tracker with ``ardy_sonic/run_ardy_sonic.sh`` (registers this
callback via Hydra ``++eval_callbacks=[ardy_replan]``).
"""

from __future__ import annotations

import math
import os
import time
from typing import Optional, Sequence

import numpy as np
import torch

from ardy_sonic.base_state_estimator import FootOdometryBaseEstimator
from ardy_sonic import protocol as P


def _wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return float(default)
    return float(value)


def _env_xy(name: str):
    value = os.environ.get(name)
    return _parse_optional_xy(value, name)


def _parse_optional_xy(value, name: str = "xy"):
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip() == "":
            return None
        parts = value.replace(",", " ").split()
        if len(parts) != 2:
            raise ValueError(f"{name} must contain exactly two numbers, got {value!r}")
        return [float(parts[0]), float(parts[1])]
    return value


class ArdyReplanCallback:
    def __init__(
        self,
        goal_mode: str = "forward",          # "forward" (start + forward_meters along start heading) or "absolute"
        forward_meters: float = 5.0,
        goal_xy: Optional[Sequence[float]] = None,   # used when goal_mode == "absolute"
        target_joint_qpos: Optional[Sequence[float]] = None,  # terminal 29-DOF pose (None -> zeros)
        target_height: float = 0.72,
        # ---- diagnostic tolerances printed when the Ardy goal-reaching plan ends ---
        arrival_radius: float = 0.10,        # root xy diagnostic tolerance (m)
        yaw_tol_deg: float = 8.0,            # root heading diagnostic tolerance (deg)
        joint_tol_rad: float = 0.35,         # max |joint - target| diagnostic tolerance (rad)
        # ---- landing behaviour -------------------------------------------------------
        final_leg_distance: float = 1.0,     # within this of the goal, stop cutting: track the plan to its exact landing (m)
        landing_hold_seconds: float = 2.0,   # extra tracking of the frozen exact-pose tail after a landing
        stall_patience: int = 3,             # legacy compatibility; planner-goal stop no longer uses it
        min_improve: float = 0.05,           # legacy compatibility; planner-goal stop no longer uses it
        hold_seconds: float = 3.0,           # after arriving, hold at the destination this long before stopping
        show_target_markers: bool = True,    # draw the destination 29-DOF pose as blue spheres
        target_marker_radius: float = 0.05,  # radius of each destination sphere (m)
        show_plan_markers: bool = True,      # draw each installed Ardy plan's root path as an orange ground track
        plan_marker_radius: float = 0.03,    # radius of each plan-path sphere (m)
        marker_z_offset: float = 0.06,       # visual-only marker lift; does not change qpos/reference
        max_plan_distance: float = 6.0,      # cap per-plan forward distance (m)
        seconds_per_meter: float = 2.0,      # plan duration = max(min_duration, distance * this)
        min_duration: float = 2.5,
        track_fraction: float = 0.9,         # legacy compatibility; async replans use replan_request_fraction
        replan_request_fraction: float = 0.8,  # asynchronously request the next non-final plan here
        max_async_start_xy_error: float = 0.75,  # discard async plans whose start became too stale (m)
        handoff_blend_frames: int = 12,       # non-first replans start from current robot qpos, then blend into Ardy
        max_replans: int = 40,
        max_steps: int = 12000,              # hard stop on total control steps
        control_hz: float = 50.0,
        placeholder_frames: int = 1500,      # length of the placeholder clip (caps installable segment)
        env_index: int = 0,
        plan_timeout_s: float = 900.0,
        prompt: str = "A person walks forward at a steady natural pace and comes to a stop.",
        cfg_weight: Sequence[float] = (2.0, 3.0),
        seed: int = 0,
        runtime_dir: Optional[str] = None,
        use_base_state_estimator: bool | str = False,
        strict_no_privileged_state: Optional[bool | str] = None,
        estimator_initial_xy: Optional[Sequence[float]] = None,
        estimator_base_z: Optional[float] = None,
        estimator_contact_source: str = "kinematic",  # kinematic | sim_sensor
        estimator_kinematic_contact_z_margin: float = 0.035,
        estimator_contact_force_threshold: float = 10.0,
        estimator_contact_enter_steps: int = 2,
        estimator_contact_exit_steps: int = 2,
        estimator_xy_correction_alpha: float = 0.75,
        estimator_max_xy_correction_per_step: float = 0.08,
        estimator_max_anchor_residual: float = 0.18,
        estimator_max_yaw_rate: float = 3.5,
        estimator_log_interval: int = 100,
        estimator_left_foot_body: str = "left_ankle_roll_link",
        estimator_right_foot_body: str = "right_ankle_roll_link",
        estimator_init_from_sim: bool | str = False,
        verbose: bool = True,
    ):
        self.goal_mode = goal_mode
        self.forward_meters = float(forward_meters)
        self.goal_xy_arg = None if goal_xy is None else [float(v) for v in goal_xy]
        self.target_joint_qpos = None if target_joint_qpos is None else [float(v) for v in target_joint_qpos]
        self.target_height = float(target_height)
        self.arrival_radius = float(arrival_radius)
        self.yaw_tol_rad = math.radians(float(yaw_tol_deg))
        self.joint_tol_rad = float(joint_tol_rad)
        self.final_leg_distance = max(float(final_leg_distance), self.arrival_radius)
        self.landing_hold_seconds = max(0.0, float(landing_hold_seconds))
        self.stall_patience = max(1, int(stall_patience))
        self.min_improve = float(min_improve)
        self.hold_seconds = max(0.0, float(hold_seconds))
        self.show_target_markers = bool(show_target_markers)
        self.target_marker_radius = float(target_marker_radius)
        self.show_plan_markers = bool(show_plan_markers)
        self.plan_marker_radius = float(plan_marker_radius)
        self.marker_z_offset = float(marker_z_offset)
        self.max_plan_distance = float(max_plan_distance)
        self.seconds_per_meter = float(seconds_per_meter)
        self.min_duration = float(min_duration)
        self.track_fraction = float(np.clip(track_fraction, 0.1, 1.0))
        self.replan_request_fraction = float(np.clip(replan_request_fraction, 0.05, 0.98))
        self.max_async_start_xy_error = max(0.0, float(max_async_start_xy_error))
        self.handoff_blend_frames = max(0, int(handoff_blend_frames))
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
        self.use_base_state_estimator = _as_bool(
            os.environ.get("ESTIMATE_BASE_STATE", use_base_state_estimator)
        )
        if strict_no_privileged_state is None:
            strict_no_privileged_state = self.use_base_state_estimator
        self.strict_no_privileged_state = _as_bool(
            os.environ.get("STRICT_NO_PRIVILEGED_STATE", strict_no_privileged_state)
        )
        env_initial_xy = _env_xy("ESTIMATOR_INITIAL_XY")
        if env_initial_xy is not None:
            estimator_initial_xy = env_initial_xy
        estimator_initial_xy = _parse_optional_xy(estimator_initial_xy, "estimator_initial_xy")
        self.estimator_initial_xy = (
            None
            if estimator_initial_xy is None
            else np.asarray(estimator_initial_xy, dtype=np.float32).reshape(2)
        )
        self.estimator_base_z = _env_float(
            "ESTIMATOR_BASE_Z",
            self.target_height if estimator_base_z is None else float(estimator_base_z),
        )
        self.estimator_contact_source = str(
            os.environ.get("ESTIMATOR_CONTACT_SOURCE", estimator_contact_source)
        ).strip().lower()
        self.estimator_kinematic_contact_z_margin = _env_float(
            "ESTIMATOR_KINEMATIC_CONTACT_Z_MARGIN", estimator_kinematic_contact_z_margin
        )
        self.estimator_contact_force_threshold = float(estimator_contact_force_threshold)
        self.estimator_contact_enter_steps = max(
            1, int(_env_float("ESTIMATOR_CONTACT_ENTER_STEPS", estimator_contact_enter_steps))
        )
        self.estimator_contact_exit_steps = max(
            1, int(_env_float("ESTIMATOR_CONTACT_EXIT_STEPS", estimator_contact_exit_steps))
        )
        self.estimator_xy_correction_alpha = float(np.clip(
            _env_float("ESTIMATOR_XY_CORRECTION_ALPHA", estimator_xy_correction_alpha),
            0.0,
            1.0,
        ))
        self.estimator_max_xy_correction_per_step = max(
            0.0,
            _env_float("ESTIMATOR_MAX_XY_CORRECTION_PER_STEP", estimator_max_xy_correction_per_step),
        )
        self.estimator_max_anchor_residual = max(
            1e-6,
            _env_float("ESTIMATOR_MAX_ANCHOR_RESIDUAL", estimator_max_anchor_residual),
        )
        self.estimator_max_yaw_rate = max(
            0.0,
            _env_float("ESTIMATOR_MAX_YAW_RATE", estimator_max_yaw_rate),
        )
        self.estimator_log_interval = max(1, int(estimator_log_interval))
        self.estimator_foot_body_names = {
            "left": str(estimator_left_foot_body),
            "right": str(estimator_right_foot_body),
        }
        self.estimator_init_from_sim = _as_bool(estimator_init_from_sim)
        if self.strict_no_privileged_state and self.estimator_init_from_sim:
            raise ValueError(
                "estimator_init_from_sim=True reads simulator root XY. "
                "Set STRICT_NO_PRIVILEGED_STATE=False for sim-only validation, or use ESTIMATOR_INITIAL_XY."
            )
        if self.strict_no_privileged_state and self.estimator_contact_source == "sim_sensor":
            raise ValueError(
                "ESTIMATOR_CONTACT_SOURCE=sim_sensor is not allowed with STRICT_NO_PRIVILEGED_STATE=True. "
                "Use kinematic contact or provide a real sensor adapter."
            )
        self._base_estimator = (
            FootOdometryBaseEstimator(
                default_height=self.target_height,
                contact_enter_steps=self.estimator_contact_enter_steps,
                contact_exit_steps=self.estimator_contact_exit_steps,
                xy_correction_alpha=self.estimator_xy_correction_alpha,
                max_xy_correction_per_update=self.estimator_max_xy_correction_per_step,
                max_anchor_residual=self.estimator_max_anchor_residual,
                max_yaw_rate=self.estimator_max_yaw_rate,
            )
            if self.use_base_state_estimator
            else None
        )
        self._estimator_body_index_cache = {}
        self._last_estimator_log_step = -1
        self._estimator_err_sum = 0.0
        self._estimator_err_count = 0
        self._estimator_yaw_err_sum = 0.0
        self._estimator_yaw_err_count = 0

        self._target_joints = np.asarray(
            self.target_joint_qpos if self.target_joint_qpos is not None else [0.0] * P.NUM_JOINTS,
            dtype=np.float32,
        )

        self._paths = P.RuntimePaths(runtime_dir).ensure()
        # eval_agent_trl sets callback.model if the attribute exists.
        self.model = None

        self._step = 0
        self._req_counter = 0
        self._replans = 0
        self._initialized = False
        self._done = False
        self._need_replan = True
        self._goal_xy = np.zeros(2, dtype=np.float32)
        self._start_xy = np.zeros(2, dtype=np.float32)
        self._goal_heading = 0.0
        # currently tracked segment
        self._phase = "walk"                 # "walk" (may be cut+replanned) | "landing" (tracked to completion)
        self._seg_end_step = 0
        self._seg_request_step = 0
        self._install_step = 0
        self._seg_total_steps = 0
        self._cur_plan_reaches_goal = False
        self._pending_plan = None
        self._tail_wait_logged = False
        # landing bookkeeping
        self._landings = 0
        self._best_score = float("inf")
        self._stall_count = 0
        # arrival hold
        self._arrived = False
        self._hold_until_step = 0
        # destination-pose markers
        self._markers = None
        self._markers_failed = False
        self._target_body_pos_w = None
        # plan-path markers
        self._plan_markers = None
        self._plan_markers_failed = False
        self._plan_marker_pos_w = None

    # -- lifecycle hooks -------------------------------------------------------
    def on_step_end(self, *args, **kwargs) -> None:
        """Called once before the eval loop. Clear stale responses + create markers."""
        for f in self._paths.responses.glob("plan_*.npz"):
            try:
                f.unlink()
            except OSError:
                pass
        self._log(f"[ardy_replan] runtime={self._paths.root} goal_mode={self.goal_mode} "
                  f"forward_meters={self.forward_meters} stop=ardy_goal_reached "
                  f"hold={self.hold_seconds:.1f}s diagnostics: pos<={self.arrival_radius}m "
                  f"yaw<={math.degrees(self.yaw_tol_rad):.0f}deg joint<={self.joint_tol_rad}rad")
        if self.use_base_state_estimator:
            self._log(
                "[ardy_replan] base-state estimator enabled "
                "(IMU orientation + encoder-FK foot positions + foot contact; "
                f"strict_no_privileged_state={self.strict_no_privileged_state} "
                f"contact_source={self.estimator_contact_source} "
                f"contact_hysteresis={self.estimator_contact_enter_steps}/{self.estimator_contact_exit_steps} "
                f"max_anchor_residual={self.estimator_max_anchor_residual:.3f}m "
                f"max_yaw_rate={math.degrees(self.estimator_max_yaw_rate):.1f}deg/s "
                f"feet={self.estimator_foot_body_names})"
            )
        # Create marker prims BEFORE the eval loop so they attach to the renderer
        # (creating markers mid-loop can leave them invisible). Positions are filled
        # in later: destination on the first eval_step, plan path on each install.
        if self.show_target_markers and self._markers is None:
            self._markers = self._make_sphere_markers(
                "/Visuals/ArdyDestinationPose", self.target_marker_radius, (0.0, 0.25, 1.0))
            self._markers_failed = self._markers is None
        if self.show_plan_markers and self._plan_markers is None:
            self._plan_markers = self._make_sphere_markers(
                "/Visuals/ArdyPlanPath", self.plan_marker_radius, (1.0, 0.45, 0.0))
            self._plan_markers_failed = self._plan_markers is None

    def _make_sphere_markers(self, prim_path: str, radius: float, color):
        try:
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
            import isaaclab.sim as sim_utils

            cfg = VisualizationMarkersCfg(
                prim_path=prim_path,
                markers={
                    "pt": sim_utils.SphereCfg(
                        radius=radius,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=color, opacity=0.9
                        ),
                    )
                },
            )
            markers = VisualizationMarkers(cfg)
            markers.set_visibility(True)
            self._log(f"[ardy_replan] marker prim created at {prim_path}")
            return markers
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            self._log(f"[ardy_replan] marker prim {prim_path} creation failed: {exc}; continuing without it.")
            return None

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

        # Refresh the destination pose (blue) and the active plan path (orange).
        self._update_target_markers(command, env_idx)
        self._update_plan_markers()

        self._step += 1

        # Holding at the destination after arrival: keep tracking (the reference tail
        # pins the exact target pose) until the hold window elapses, then stop.
        if self._arrived:
            if self._step >= self._hold_until_step:
                pos_err, yaw_err, joint_err = self._errors(cur)
                self._log(f"[ardy_replan] hold complete; stopping at destination. "
                          f"final errors: pos={pos_err:.3f}m yaw={math.degrees(yaw_err):.1f}deg "
                          f"max_joint={joint_err:.3f}rad")
                self._done = True
                return True
            return False

        if self._step >= self.max_steps:
            self._stop_with_report(cur, f"max_steps={self.max_steps} reached")
            return True

        pos_err, yaw_err, joint_err = self._errors(cur)

        try:
            pending = self._poll_pending_plan(cur_xy)
        except Exception as exc:  # noqa: BLE001
            self._log(f"[ardy_replan] planning failed: {exc}. Stopping.")
            self._done = True
            return True
        if pending is not None:
            qpos, fps, meta = pending
            try:
                self._install_plan(command, env_idx, qpos, fps, meta)
            except Exception as exc:  # noqa: BLE001
                self._log(f"[ardy_replan] plan install failed: {exc}. Stopping.")
                self._done = True
                return True
            return False

        if not self._need_replan:
            if self._phase == "landing" and self._step >= self._seg_end_step:
                # The Ardy goal-reaching plan was tracked through its exact landing.
                # Stop based on the planner/reference reaching the goal; print robot
                # tracking errors only as diagnostics.
                self._landings += 1
                self._arrived = True
                self._hold_until_step = self._step + max(1, int(self.hold_seconds * self.control_hz))
                self._log(f"[ardy_replan] ARDY PLAN REACHED GOAL (landing {self._landings}). "
                          f"robot diagnostic errors: pos={pos_err:.3f}m "
                          f"(tol {self.arrival_radius}) yaw={math.degrees(yaw_err):.1f}deg "
                          f"(tol {math.degrees(self.yaw_tol_rad):.0f}) "
                          f"max_joint={joint_err:.3f}rad (tol {self.joint_tol_rad}). "
                          f"Holding {self.hold_seconds:.1f}s, then stopping.")
                return False
            elif self._phase == "walk" and self._cur_plan_reaches_goal and pos_err <= self.final_leg_distance:
                # Close to the goal and the active plan already ends exactly on the
                # goal pose: do NOT cut it -- extend tracking through its exact-landing
                # frames plus the frozen-pose tail. One continuous walk, no correction leg.
                self._phase = "landing"
                self._seg_end_step = (self._install_step + self._seg_total_steps
                                      + int(self.landing_hold_seconds * self.control_hz))
                self._log(f"[ardy_replan] within final_leg_distance ({pos_err:.3f}<= "
                          f"{self.final_leg_distance}); tracking the current plan to its exact "
                          f"landing (until step {self._seg_end_step}, no replan).")
                return False

        if self._need_replan:
            self._log(f"[ardy_replan] step={self._step} phase={self._phase} "
                      f"cur_xy={cur_xy.round(3).tolist()} pos_err={pos_err:.3f}m "
                      f"yaw_err={math.degrees(yaw_err):.1f}deg joint_err={joint_err:.3f}rad "
                      f"replans={self._replans}")
            if self._replans >= self.max_replans:
                self._stop_with_report(cur, f"max_replans={self.max_replans} reached")
                return True
            try:
                self._plan_leg(command, env_idx, cur_xy, pos_err)
            except Exception as exc:  # noqa: BLE001
                self._log(f"[ardy_replan] planning failed: {exc}. Stopping.")
                self._done = True
                return True
            return False

        if self._phase == "walk" and not self._cur_plan_reaches_goal:
            if self._pending_plan is None and self._step >= self._seg_request_step:
                if self._replans >= self.max_replans:
                    self._stop_with_report(cur, f"max_replans={self.max_replans} reached")
                    return True
                try:
                    self._request_next_plan(cur_xy, pos_err, wait=False)
                except Exception as exc:  # noqa: BLE001
                    self._log(f"[ardy_replan] planning failed: {exc}. Stopping.")
                    self._done = True
                    return True
            if self._pending_plan is not None and self._step >= self._seg_end_step and not self._tail_wait_logged:
                self._tail_wait_logged = True
                self._log(f"[ardy_replan] step={self._step} reached current plan end while "
                          f"'{self._pending_plan['id']}' is still pending; continuing the "
                          "current reference tail instead of blocking.")
        return False

    # -- planning ----------------------------------------------------------------
    def _plan_leg(self, command, env_idx: int, cur_xy: np.ndarray, remaining: float) -> None:
        """Plan from the robot's CURRENT pose to the goal and install it.

        The plan is start-anchored at ``cur_xy`` (it visibly re-attaches to the G1)
        and, when it reaches the goal, ends exactly on the goal pose. Short legs are
        tracked straight through their landing; longer legs are prefetched
        asynchronously before the current reference ends.
        """
        self._request_next_plan(cur_xy, remaining, wait=True)
        qpos, fps, meta = self._wait_for_pending_plan()
        self._install_plan(command, env_idx, qpos, fps, meta)

    def _request_next_plan(self, cur_xy: np.ndarray, remaining: float, wait: bool):
        """Create a planner request. When ``wait`` is False, eval keeps stepping."""
        if self._pending_plan is not None:
            return self._pending_plan
        dist, heading, duration, reach_target = self._plan_params(cur_xy, remaining)
        meta = self._start_plan_request(cur_xy, heading, dist, duration, reach_target, wait=wait)
        self._pending_plan = meta
        return meta

    def _plan_params(self, cur_xy: np.ndarray, remaining: float) -> tuple[float, float, float, bool]:
        remaining = float(remaining)
        max_dist = float(self.max_plan_distance)
        if remaining <= max_dist + self.final_leg_distance:
            # If one capped step would leave only a short residual, request the
            # full remaining leg as the goal-reaching plan instead of producing
            # another near-target intermediate segment.
            dist = remaining
            reach_target = True
        else:
            dist = max_dist
            reach_target = False
        if remaining < 0.30:
            # Residual too small for atan2 to give a meaningful walk direction (and a
            # tiny leg must still END facing the goal heading): walk out the residual
            # along the goal heading itself.
            heading = self._goal_heading
        else:
            delta = self._goal_xy - cur_xy
            heading = math.atan2(float(delta[1]), float(delta[0]))
        duration = max(self.min_duration, dist * self.seconds_per_meter)
        return dist, heading, duration, reach_target

    def _install_plan(self, command, env_idx: int, qpos: np.ndarray, fps: float, meta: dict) -> None:
        first_install = self._replans == 0
        if not first_install and self.handoff_blend_frames > 0:
            cur = self._robot_qpos_mujoco(command, env_idx)
            qpos = self._blend_plan_start_from_current(cur, qpos)
        command.install_live_qpos_segment(
            torch.as_tensor(qpos, dtype=torch.float32),
            fps=int(round(fps)),
            env_ids=torch.tensor([env_idx], device=command.device),
            reset_time=first_install,
            install_at_current_time=not first_install,
        )
        self._replans += 1
        self._need_replan = False
        self._pending_plan = None
        self._tail_wait_logged = False
        self._set_plan_marker_positions(command, env_idx, qpos, fps)

        self._install_step = self._step
        duration = float(meta["duration"])
        dist = float(meta["dist"])
        reach_target = bool(meta["reach_target"])
        self._seg_total_steps = min(int(round(duration * self.control_hz)), self.placeholder_frames)
        final_leg = reach_target and dist <= self.final_leg_distance
        if final_leg:
            self._phase = "landing"
            self._seg_end_step = (self._install_step + self._seg_total_steps
                                  + int(self.landing_hold_seconds * self.control_hz))
            self._seg_request_step = self._seg_end_step
        else:
            self._phase = "walk"
            self._seg_end_step = self._install_step + self._seg_total_steps
            self._seg_request_step = (
                self._install_step
                + max(1, int(self.replan_request_fraction * self._seg_total_steps))
            )
        self._cur_plan_reaches_goal = reach_target
        self._log(f"[ardy_replan] plan installed ({self._phase}): {qpos.shape[0]}x{qpos.shape[1]} "
                  f"@ {fps:g}fps dist={dist:.3f} dur={duration:.2f}s reach_target={reach_target} "
                  f"handoff_blend_frames={0 if first_install else self.handoff_blend_frames} "
                  f"request_next_step={self._seg_request_step} track_until_step={self._seg_end_step}")

    def _blend_plan_start_from_current(self, cur_qpos: np.ndarray, qpos: np.ndarray) -> np.ndarray:
        """Replace the beginning of a replan with a smooth current-robot handoff."""
        qpos = np.asarray(qpos, dtype=np.float32).copy()
        if qpos.ndim != 2 or qpos.shape[0] < 2:
            return qpos
        n = min(self.handoff_blend_frames, qpos.shape[0])
        if n <= 0:
            return qpos

        target = qpos[n - 1].copy()
        cur = np.asarray(cur_qpos[:P.QPOS_DIM], dtype=np.float32).copy()
        start_yaw = P.yaw_from_quat_wxyz(cur[3:7])
        target_yaw = P.yaw_from_quat_wxyz(target[3:7])
        yaw_delta = _wrap_angle(target_yaw - start_yaw)

        for i in range(n):
            denom = max(1, n - 1)
            t = float(i) / float(denom)
            a = t * t * (3.0 - 2.0 * t)
            qpos[i, :3] = (1.0 - a) * cur[:3] + a * target[:3]
            yaw = start_yaw + a * yaw_delta
            qpos[i, 3:7] = [math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)]
            qpos[i, 7:36] = (1.0 - a) * cur[7:36] + a * target[7:36]
        return qpos

    # -- errors / reporting -------------------------------------------------------
    def _errors(self, cur: np.ndarray) -> tuple[float, float, float]:
        """(pos_err m, |yaw_err| rad, max |joint - target| rad) vs the goal pose."""
        pos_err = float(np.linalg.norm(self._goal_xy - cur[:2]))
        yaw_err = abs(float(_wrap_angle(P.yaw_from_quat_wxyz(cur[3:7]) - self._goal_heading)))
        joint_err = float(np.abs(_wrap_angle(cur[7:36] - self._target_joints)).max())
        return pos_err, yaw_err, joint_err

    def _stop_with_report(self, cur: np.ndarray, reason: str) -> None:
        pos_err, yaw_err, joint_err = self._errors(cur)
        self._log(f"[ardy_replan] STOPPING WITHOUT ARRIVAL ({reason}). achieved errors: "
                  f"pos={pos_err:.3f}m (tol {self.arrival_radius}) "
                  f"yaw={math.degrees(yaw_err):.1f}deg (tol {math.degrees(self.yaw_tol_rad):.0f}) "
                  f"max_joint={joint_err:.3f}rad (tol {self.joint_tol_rad})")
        self._done = True

    # -- plan-path markers -------------------------------------------------------
    def _set_plan_marker_positions(self, command, env_idx: int, qpos: np.ndarray, fps: float) -> None:
        """Store the installed plan's root path as a ground track (drawn each step)."""
        if self._plan_markers is None or self._plan_markers_failed:
            return
        try:
            stride = max(1, int(round(float(fps) / 5.0)))  # ~5 markers per plan second
            pts = np.asarray(qpos[::stride, :3], dtype=np.float32).copy()
            if (qpos.shape[0] - 1) % stride != 0:
                pts = np.vstack([pts, qpos[-1, :3]])
            pts[:, 2] = 0.05 + self.marker_z_offset  # draw the root xy path as a lifted ground track
            t = torch.as_tensor(pts, dtype=torch.float32, device=command.device)
            env_origins = getattr(command._env.scene, "env_origins", None)  # noqa: SLF001
            if env_origins is not None:
                t = t + env_origins[env_idx].to(t.device)
            self._plan_marker_pos_w = t
        except Exception as exc:  # noqa: BLE001
            self._plan_markers_failed = True
            self._log(f"[ardy_replan] plan marker update failed: {exc}; continuing without it.")

    def _update_plan_markers(self) -> None:
        if self._plan_markers is None or self._plan_markers_failed or self._plan_marker_pos_w is None:
            return
        try:
            self._plan_markers.visualize(translations=self._plan_marker_pos_w)
        except Exception as exc:  # noqa: BLE001
            self._plan_markers_failed = True
            self._log(f"[ardy_replan] plan marker visualize failed: {exc}; continuing without it.")

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
        body_pos[:, 2] += self.marker_z_offset
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

    def _start_plan_request(self, start_xy, heading, dist, duration, reach_target, wait: bool):
        req_id = f"plan_{self._req_counter:04d}"
        self._req_counter += 1
        req = P.PlanRequest(
            id=req_id,
            start_xy=[float(start_xy[0]), float(start_xy[1])],
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
        mode = "waiting for planner ..." if wait else "continuing current reference while planner runs"
        self._log(f"[ardy_replan] requested plan '{req_id}' dist={dist:.3f} dur={duration:.2f}s "
                  f"heading={math.degrees(heading):.1f}deg; {mode}")
        return {
            "id": req_id,
            "resp_path": resp_path,
            "t0": time.time(),
            "start_xy": np.asarray(start_xy, dtype=np.float32).copy(),
            "heading": float(heading),
            "dist": float(dist),
            "duration": float(duration),
            "reach_target": bool(reach_target),
            "wait": bool(wait),
        }

    def _wait_for_pending_plan(self):
        while True:
            ready = self._poll_pending_plan(cur_xy=None)
            if ready is not None:
                return ready
            time.sleep(0.1)

    def _poll_pending_plan(self, cur_xy: Optional[np.ndarray]):
        meta = self._pending_plan
        if meta is None:
            return None
        resp_path = meta["resp_path"]
        req_id = meta["id"]
        while not resp_path.exists():
            if time.time() - meta["t0"] > self.plan_timeout_s:
                raise TimeoutError(f"planner did not respond within {self.plan_timeout_s}s for '{req_id}'")
            return None
        resp = P.read_response(resp_path)
        if not resp["ok"]:
            raise RuntimeError(f"planner error for '{req_id}': {resp['error']}")
        qpos, fps = self._validate_plan_response(req_id, resp, meta, cur_xy)
        if qpos is None:
            self._pending_plan = None
            return None
        self._pending_plan = None
        self._log(f"[ardy_replan] received valid plan '{req_id}' in {time.time()-meta['t0']:.1f}s "
                  f"frames={qpos.shape[0]} final_xy={resp['final_root_xy'].round(3).tolist()}")
        return qpos, fps, meta

    def _validate_plan_response(self, req_id: str, resp: dict, meta: dict, cur_xy: Optional[np.ndarray]):
        qpos = resp["qpos"]
        fps = float(resp["fps"])
        if qpos.ndim != 2 or qpos.shape[1] < P.QPOS_DIM or qpos.shape[0] < 2:
            raise ValueError(f"planner returned bad qpos shape {qpos.shape}")
        if not np.isfinite(qpos[:, :P.QPOS_DIM]).all():
            raise ValueError(f"planner returned non-finite qpos for '{req_id}'")
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"planner returned bad fps={fps} for '{req_id}'")

        if cur_xy is not None:
            start_err = float(np.linalg.norm(qpos[0, :2] - np.asarray(cur_xy, dtype=np.float32)))
            if start_err > self.max_async_start_xy_error:
                self._log(f"[ardy_replan] discarded stale async plan '{req_id}': "
                          f"start_xy_error={start_err:.3f}m > {self.max_async_start_xy_error:.3f}m")
                return None, None

        if meta["reach_target"]:
            final_xy = qpos[-1, :2]
            goal_err = float(np.linalg.norm(final_xy - self._goal_xy))
            final_yaw = P.yaw_from_quat_wxyz(qpos[-1, 3:7])
            yaw_err = abs(float(_wrap_angle(final_yaw - self._goal_heading)))
            joint_err = float(np.abs(_wrap_angle(qpos[-1, 7:36] - self._target_joints)).max())
            if goal_err > max(self.arrival_radius, 0.25):
                raise ValueError(f"goal-reaching plan '{req_id}' final_xy is {goal_err:.3f}m from goal")
            if yaw_err > max(self.yaw_tol_rad, math.radians(20.0)):
                raise ValueError(f"goal-reaching plan '{req_id}' final yaw error is {math.degrees(yaw_err):.1f}deg")
            if joint_err > max(self.joint_tol_rad, 0.50):
                raise ValueError(f"goal-reaching plan '{req_id}' final joint error is {joint_err:.3f}rad")
        return qpos, fps

    def _resolve_estimator_foot_body_indices(self, body_names, source: str) -> dict[str, int]:
        key = (source, tuple(body_names))
        if key in self._estimator_body_index_cache:
            return self._estimator_body_index_cache[key]
        indices = {}
        for side, name in self.estimator_foot_body_names.items():
            if name not in body_names:
                raise ValueError(f"estimator foot body '{name}' not found in {source} body names")
            indices[side] = body_names.index(name)
        self._estimator_body_index_cache[key] = indices
        return indices

    @staticmethod
    def _quat_apply_inverse_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        w, x, y, z = [float(a) for a in q]
        qv = np.asarray([x, y, z], dtype=np.float32)
        vec = np.asarray(v, dtype=np.float32)
        # q^-1 * v * q for unit q, expanded without building matrices.
        t = 2.0 * np.cross(-qv, vec)
        return vec + w * t + np.cross(-qv, t)

    def _foot_positions_in_base_from_sim_fk(self, command, env_idx: int, root_quat_wxyz: np.ndarray) -> dict[str, np.ndarray]:
        """Foot positions in base frame.

        In sim this uses Isaac's already-computed link transforms as a FK oracle.
        A real G1 adapter should replace this boundary with encoder-based FK.
        """
        if self.strict_no_privileged_state:
            raise RuntimeError("_foot_positions_in_base_from_sim_fk requires privileged simulator body poses")
        indices = self._resolve_estimator_foot_body_indices(
            getattr(command.robot, "body_names", []), "robot"
        )
        root_pos_w = command.robot.data.root_pos_w[env_idx].detach().cpu().numpy().astype(np.float32)
        body_pos_w = command.robot.data.body_pos_w[env_idx].detach().cpu().numpy().astype(np.float32)
        foot_pos_base = {}
        for side, body_idx in indices.items():
            foot_pos_base[side] = self._quat_apply_inverse_wxyz(
                root_quat_wxyz, body_pos_w[body_idx] - root_pos_w
            ).astype(np.float32)
        return foot_pos_base

    def _foot_positions_in_base_from_encoder_fk(self, command, qpos: np.ndarray) -> dict[str, np.ndarray]:
        """Foot positions from current joint angles, with the root placed at the base origin."""
        parser = getattr(getattr(command, "motion_lib", None), "mesh_parsers", None)
        if parser is None:
            raise ValueError("encoder-FK estimator requires command.motion_lib.mesh_parsers")
        body_names = getattr(parser, "body_names", None)
        if body_names is None:
            raise ValueError("encoder-FK estimator requires mesh parser body_names")
        indices = self._resolve_estimator_foot_body_indices(body_names, "motion-lib")
        qpos_fk = np.asarray(qpos, dtype=np.float32).copy()
        qpos_fk[:3] = 0.0
        qpos_fk[3:7] = [1.0, 0.0, 0.0, 0.0]
        qpos_t = torch.as_tensor(qpos_fk, dtype=torch.float32, device="cpu")
        body_pos, _ = parser.qpos_to_global_transforms(qpos_t, root_quat_wxyz=True)
        body_pos_np = body_pos.detach().cpu().numpy().astype(np.float32)
        return {side: body_pos_np[body_idx].copy() for side, body_idx in indices.items()}

    def _estimate_contacts_from_kinematics(self, foot_pos_base: dict[str, np.ndarray]) -> dict[str, bool]:
        if not foot_pos_base:
            raise ValueError("kinematic contact estimation needs at least one foot position")
        foot_z = {side: float(np.asarray(pos).reshape(-1)[2]) for side, pos in foot_pos_base.items()}
        min_z = min(foot_z.values())
        margin = max(0.0, float(self.estimator_kinematic_contact_z_margin))
        return {side: z <= min_z + margin for side, z in foot_z.items()}

    def _read_sim_foot_contacts(self, command, env_idx: int) -> dict[str, bool]:
        if self.strict_no_privileged_state:
            raise RuntimeError("_read_sim_foot_contacts requires privileged simulator contact/body data")
        indices = self._resolve_estimator_foot_body_indices(
            getattr(command.robot, "body_names", []), "robot"
        )
        contacts = {side: False for side in indices}
        sensor = None
        scene = getattr(getattr(command, "_env", None), "scene", None)
        if scene is not None:
            sensors = getattr(scene, "sensors", None)
            if sensors is not None and "contact_forces" in sensors:
                sensor = sensors["contact_forces"]
            if sensor is None:
                try:
                    sensor = scene["contact_forces"]
                except Exception:  # noqa: BLE001
                    sensor = None

        if sensor is not None:
            data = getattr(sensor, "data", None)
            forces = getattr(data, "net_forces_w", None) if data is not None else None
            if forces is None and data is not None:
                forces = getattr(data, "force_matrix_w", None)
            if forces is not None:
                forces = forces.detach()
                sensor_body_names = getattr(sensor, "body_names", None)
                for side, robot_body_idx in indices.items():
                    sensor_body_idx = robot_body_idx
                    body_name = self.estimator_foot_body_names[side]
                    if sensor_body_names is not None and body_name in sensor_body_names:
                        sensor_body_idx = sensor_body_names.index(body_name)
                    if forces.ndim == 3 and sensor_body_idx < forces.shape[1]:
                        force_norm = torch.linalg.norm(forces[env_idx, sensor_body_idx]).item()
                    elif forces.ndim == 4 and sensor_body_idx < forces.shape[1]:
                        force_norm = torch.linalg.norm(forces[env_idx, sensor_body_idx].reshape(-1, 3), dim=-1).max().item()
                    else:
                        continue
                    contacts[side] = force_norm >= self.estimator_contact_force_threshold
                return contacts

        # Fallback when the ContactSensor layout is unavailable: treat the lowest
        # ankle-roll link(s) as contact candidates. This is for sim bring-up only.
        body_pos_w = command.robot.data.body_pos_w[env_idx].detach().cpu()
        foot_z = {side: float(body_pos_w[body_idx, 2]) for side, body_idx in indices.items()}
        min_z = min(foot_z.values())
        return {side: z <= min_z + 0.035 for side, z in foot_z.items()}

    def _apply_base_state_estimator(self, command, env_idx: int, qpos: np.ndarray) -> np.ndarray:
        if self._base_estimator is None:
            return qpos
        root_quat_wxyz = np.asarray(qpos[3:7], dtype=np.float32)
        if self.estimator_contact_source == "sim_sensor":
            foot_pos_base = self._foot_positions_in_base_from_sim_fk(command, env_idx, root_quat_wxyz)
            contacts = self._read_sim_foot_contacts(command, env_idx)
        elif self.estimator_contact_source == "kinematic":
            foot_pos_base = self._foot_positions_in_base_from_encoder_fk(command, qpos)
            contacts = self._estimate_contacts_from_kinematics(foot_pos_base)
        else:
            raise ValueError(
                f"unknown ESTIMATOR_CONTACT_SOURCE={self.estimator_contact_source!r}; "
                "expected 'kinematic' or 'sim_sensor'"
            )
        if not self._base_estimator.initialized and self.estimator_initial_xy is not None:
            initial_xy = self.estimator_initial_xy
        elif self.estimator_init_from_sim and not self._base_estimator.initialized:
            if self.strict_no_privileged_state:
                raise RuntimeError("estimator_init_from_sim is not allowed in strict no-privileged mode")
            initial_xy = qpos[:2]
        else:
            initial_xy = None
        estimate = self._base_estimator.update(
            root_quat_wxyz=root_quat_wxyz,
            foot_pos_base=foot_pos_base,
            contacts=contacts,
            base_z=float(self.estimator_base_z),
            initial_xy=initial_xy,
            dt=1.0 / self.control_hz if self.control_hz > 0.0 else None,
        )
        out = qpos.copy()
        out[:3] = estimate.root_pos
        out[3:7] = estimate.root_quat_wxyz

        if self._step - self._last_estimator_log_step >= self.estimator_log_interval:
            self._last_estimator_log_step = self._step
            est_yaw = P.yaw_from_quat_wxyz(out[3:7])
            foot_z = {
                side: round(float(np.asarray(pos).reshape(-1)[2]), 3)
                for side, pos in foot_pos_base.items()
            }
            msg = (
                "[base_estimator] "
                f"step={self._step} source={self.estimator_contact_source} "
                f"est_xy={out[:2].round(3).tolist()} "
                f"est_yaw={math.degrees(est_yaw):.1f}deg "
                f"contacts={estimate.contacts} "
                f"conf={{{', '.join(f'{k}: {v:.2f}' for k, v in estimate.contact_confidence.items())}}} "
                f"slip={estimate.slip_feet} "
                f"foot_z={foot_z}"
            )
            if not self.strict_no_privileged_state:
                sim_state = self._sim_root_state_for_diagnostics(command, env_idx)
                if sim_state is not None:
                    sim_xy, sim_yaw = sim_state
                    err = float(np.linalg.norm(out[:2] - sim_xy))
                    yaw_err = abs(float(_wrap_angle(est_yaw - sim_yaw)))
                    self._estimator_err_sum += err
                    self._estimator_err_count += 1
                    self._estimator_yaw_err_sum += yaw_err
                    self._estimator_yaw_err_count += 1
                    mean_err = self._estimator_err_sum / max(1, self._estimator_err_count)
                    mean_yaw_err = (
                        self._estimator_yaw_err_sum
                        / max(1, self._estimator_yaw_err_count)
                    )
                    msg += (
                        f" sim_xy={sim_xy.round(3).tolist()} err={err:.3f}m "
                        f"mean_err={mean_err:.3f}m "
                        f"sim_yaw={math.degrees(sim_yaw):.1f}deg "
                        f"yaw_err={math.degrees(yaw_err):.1f}deg "
                        f"mean_yaw_err={math.degrees(mean_yaw_err):.1f}deg"
                    )
            self._log(msg)
        return out

    def _sim_root_state_for_diagnostics(self, command, env_idx: int):
        try:
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
            sim_xy = root_pos[:2].detach().cpu().numpy().astype(np.float32)
            sim_yaw = P.yaw_from_quat_wxyz(root_quat_wxyz.detach().cpu().numpy())
            return sim_xy, sim_yaw
        except Exception:  # noqa: BLE001
            return None

    def _robot_qpos_mujoco(self, command, env_idx: int) -> np.ndarray:
        """Current robot state as MuJoCo qpos [root xyz(env-local), quat wxyz, 29 joints]."""
        if self.use_base_state_estimator:
            return self._estimated_robot_qpos_mujoco(command, env_idx)
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

    def _estimated_robot_qpos_mujoco(self, command, env_idx: int) -> np.ndarray:
        """Current qpos for replanning without simulator global root position."""
        root_quat_wxyz = command.robot.data.root_quat_w[env_idx].detach().clone()
        try:
            from gear_sonic.isaac_utils import quaternion_adapter as quat_adapter

            root_quat_wxyz = quat_adapter.isaaclab_to_wxyz(root_quat_wxyz)
        except Exception:  # noqa: BLE001
            pass
        joint_pos_isaac = command.robot.data.joint_pos[env_idx].detach()
        joint_pos_mujoco = joint_pos_isaac[command.isaaclab_to_mujoco_dof]
        root_pos = torch.tensor(
            [0.0, 0.0, float(self.estimator_base_z)],
            dtype=joint_pos_mujoco.dtype,
            device=joint_pos_mujoco.device,
        )
        qpos = torch.cat([root_pos, root_quat_wxyz, joint_pos_mujoco[:29]], dim=0)
        return self._apply_base_state_estimator(
            command, env_idx, qpos.detach().cpu().numpy().astype(np.float32)
        )

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)
