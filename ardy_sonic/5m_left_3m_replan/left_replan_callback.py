"""Two-stage Ardy replanning experiment: 5 m forward, then 3 m left.

This module is loaded as a top-level module by putting this directory on
PYTHONPATH. It subclasses the existing ardy_sonic callback so the planner server,
runtime protocol, markers, body-tracking recording, and SONIC launch path stay the
same as the normal 5 m test.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np

from ardy_sonic.ardy_replan_callback import ArdyReplanCallback


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return float(default)
    return float(value)


class FiveMeterThenLeftThreeMeterReplanCallback(ArdyReplanCallback):
    """Switch the goal to the left once the robot gets close to the 5 m target."""

    def __init__(
        self,
        *args,
        left_meters: Optional[float] = None,
        switch_radius: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.left_meters = (
            _env_float("LEFT_METERS", 3.0) if left_meters is None else float(left_meters)
        )
        self.switch_radius = (
            _env_float("LEFT_SWITCH_RADIUS", 0.2)
            if switch_radius is None
            else float(switch_radius)
        )
        self._mission_stage = "forward_5m"
        self._first_goal_xy = None
        self._first_goal_heading = None
        self._second_goal_xy = None

    def on_step_end(self, *args, **kwargs) -> None:
        super().on_step_end(*args, **kwargs)
        self._log(
            "[5m_left_3m] mission=5m_forward_then_left "
            f"switch_radius={self.switch_radius:.3f}m left_meters={self.left_meters:.3f}m"
        )

    def eval_step(self, env, _results) -> bool:
        if self._done:
            return True

        if self._initialized and self._mission_stage == "forward_5m":
            command = env.motion_command
            env_idx = min(self.env_index, env.num_envs - 1)
            cur = self._robot_qpos_mujoco(command, env_idx)
            cur_xy = cur[:2]
            pos_err = float(np.linalg.norm(self._goal_xy - cur_xy))

            if pos_err <= self.switch_radius:
                self._switch_to_left_goal(cur_xy, pos_err)
            elif self._arrived or (
                self._phase == "landing" and self._step >= self._seg_end_step
            ):
                # The base callback treats a completed goal-reaching reference as
                # arrival even if the physical robot is still outside this
                # experiment's 0.2 m switching radius. Keep correcting the first
                # goal until the robot itself enters the switch radius.
                self._arrived = False
                self._need_replan = True
                self._phase = "walk"
                self._cur_plan_reaches_goal = False
                self._tail_wait_logged = False
                self._log(
                    "[5m_left_3m] first goal reference ended but robot is still "
                    f"{pos_err:.3f}m from the 5m target; requesting another correction plan."
                )

        return super().eval_step(env, _results)

    def _switch_to_left_goal(self, cur_xy: np.ndarray, first_goal_err: float) -> None:
        first_goal_xy = self._goal_xy.copy()
        first_heading = float(self._goal_heading)
        left_vec = np.array([-math.sin(first_heading), math.cos(first_heading)], dtype=np.float32)
        second_goal_xy = (first_goal_xy + self.left_meters * left_vec).astype(np.float32)
        second_heading = math.atan2(float(left_vec[1]), float(left_vec[0]))

        self._mission_stage = "left_3m"
        self._first_goal_xy = first_goal_xy
        self._first_goal_heading = first_heading
        self._second_goal_xy = second_goal_xy.copy()
        self._goal_xy = second_goal_xy
        self._goal_heading = second_heading

        # Drop any in-flight old-goal request and force an immediate request for
        # the new left-side goal on the next base-callback pass.
        old_pending = self._pending_plan["id"] if self._pending_plan is not None else None
        self._pending_plan = None
        self._need_replan = True
        self._phase = "walk"
        self._cur_plan_reaches_goal = False
        self._tail_wait_logged = False
        self._arrived = False
        self._target_body_pos_w = None

        remaining = float(np.linalg.norm(self._goal_xy - cur_xy))
        msg = (
            "[5m_left_3m] first target reached within switch radius "
            f"({first_goal_err:.3f}m <= {self.switch_radius:.3f}m). "
            f"new_goal_xy={self._goal_xy.round(3).tolist()} "
            f"left_of_heading={math.degrees(first_heading):.1f}deg "
            f"new_heading={math.degrees(second_heading):.1f}deg "
            f"remaining={remaining:.3f}m"
        )
        if old_pending is not None:
            msg += f" dropped_pending={old_pending}"
        self._log(msg)
