"""Mid-course Ardy replanning experiment: switch at 2.5 m to a left-shifted goal."""

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


class TwoPointFiveMeterThenLeftGoalReplanCallback(ArdyReplanCallback):
    """Keep the original 5 m plan until 2.5 m progress, then move the goal left."""

    def __init__(
        self,
        *args,
        switch_progress_meters: Optional[float] = None,
        left_meters: Optional[float] = None,
        post_switch_plan_distance: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.switch_progress_meters = (
            _env_float("SWITCH_PROGRESS_METERS", 2.5)
            if switch_progress_meters is None
            else float(switch_progress_meters)
        )
        self.left_meters = (
            _env_float("LEFT_METERS", 3.0) if left_meters is None else float(left_meters)
        )
        self.post_switch_plan_distance = (
            _env_float("POST_SWITCH_PLAN_DISTANCE", 1.2)
            if post_switch_plan_distance is None
            else float(post_switch_plan_distance)
        )
        self._mission_stage = "forward_until_switch"
        self._original_goal_xy = None
        self._original_goal_heading = None
        self._left_shifted_goal_xy = None

    def on_step_end(self, *args, **kwargs) -> None:
        super().on_step_end(*args, **kwargs)
        self._log(
            "[2_5m_left_3m] mission=forward_then_midcourse_left_goal "
            f"switch_progress={self.switch_progress_meters:.3f}m "
            f"left_meters={self.left_meters:.3f}m "
            f"post_switch_plan_distance={self.post_switch_plan_distance:.3f}m"
        )

    def eval_step(self, env, _results) -> bool:
        if self._done:
            return True

        if self._initialized and self._mission_stage == "forward_until_switch":
            command = env.motion_command
            env_idx = min(self.env_index, env.num_envs - 1)
            cur = self._robot_qpos_mujoco(command, env_idx)
            cur_xy = cur[:2]
            progress = self._progress_along_original_goal(cur_xy)
            if progress >= self.switch_progress_meters:
                self._switch_to_left_shifted_goal(cur_xy, progress)

        return super().eval_step(env, _results)

    def _progress_along_original_goal(self, cur_xy: np.ndarray) -> float:
        delta = self._goal_xy - self._start_xy
        length = float(np.linalg.norm(delta))
        if length <= 1e-6:
            return 0.0
        direction = delta / length
        return float(np.dot(cur_xy - self._start_xy, direction))

    def _plan_params(self, cur_xy: np.ndarray, remaining: float) -> tuple[float, float, float, bool]:
        if self._mission_stage != "left_shifted_goal":
            return super()._plan_params(cur_xy, remaining)

        local_max = max(0.05, float(self.post_switch_plan_distance))
        dist = min(float(remaining), local_max)
        reach_target = dist >= float(remaining) - 1e-6
        if remaining < 0.30:
            heading = self._goal_heading
        else:
            delta = self._goal_xy - cur_xy
            heading = math.atan2(float(delta[1]), float(delta[0]))
            self._goal_heading = heading
        duration = max(self.min_duration, dist * self.seconds_per_meter)
        return dist, heading, duration, reach_target

    def _switch_to_left_shifted_goal(self, cur_xy: np.ndarray, progress: float) -> None:
        original_goal_xy = self._goal_xy.copy()
        original_heading = float(self._goal_heading)
        left_vec = np.array([-math.sin(original_heading), math.cos(original_heading)], dtype=np.float32)
        shifted_goal_xy = (original_goal_xy + self.left_meters * left_vec).astype(np.float32)

        # After the sudden target change, make the terminal heading face the new
        # target from the switch point. The base callback will use this same goal
        # heading for final validation and tiny residual legs.
        to_shifted_goal = shifted_goal_xy - cur_xy
        new_heading = math.atan2(float(to_shifted_goal[1]), float(to_shifted_goal[0]))

        self._mission_stage = "left_shifted_goal"
        self._original_goal_xy = original_goal_xy
        self._original_goal_heading = original_heading
        self._left_shifted_goal_xy = shifted_goal_xy.copy()
        self._goal_xy = shifted_goal_xy
        self._goal_heading = new_heading

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
            "[2_5m_left_3m] switch progress reached "
            f"({progress:.3f}m >= {self.switch_progress_meters:.3f}m). "
            f"old_goal_xy={original_goal_xy.round(3).tolist()} "
            f"new_goal_xy={self._goal_xy.round(3).tolist()} "
            f"left_of_original_heading={math.degrees(original_heading):.1f}deg "
            f"new_heading={math.degrees(new_heading):.1f}deg "
            f"remaining={remaining:.3f}m"
        )
        if old_pending is not None:
            msg += f" dropped_pending={old_pending}"
        self._log(msg)
