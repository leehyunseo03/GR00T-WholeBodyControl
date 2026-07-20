"""Interactive goal callback: keep local Ardy replanning toward commanded goals.

This is the general version of the 2.5m-left experiment's useful behavior:
after a goal is selected, every non-final Ardy request is capped to a short
local segment (default 1.2 m), so each response re-attaches to the robot's
current achieved pose and continuously corrects tracking drift.

A separate command terminal writes JSON goals into the shared runtime directory.
The callback watches that file and switches goals immediately when a newer
command appears.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ardy_sonic.ardy_replan_callback import ArdyReplanCallback


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return float(default)
    return float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


class LiveGoalReplanCallback(ArdyReplanCallback):
    """Follow live-updated XY commands using short local Ardy plans."""

    def __init__(
        self,
        *args,
        local_plan_distance: Optional[float] = None,
        goal_command_file: Optional[str] = None,
        accept_existing_goal_command: Optional[bool] = None,
        keep_alive_after_arrival: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.local_plan_distance = (
            _env_float("LOCAL_PLAN_DISTANCE", 1.2)
            if local_plan_distance is None
            else float(local_plan_distance)
        )
        self.goal_command_file_arg = goal_command_file or os.environ.get("GOAL_COMMAND_FILE")
        self.accept_existing_goal_command = (
            _env_bool("ACCEPT_EXISTING_GOAL_COMMAND", False)
            if accept_existing_goal_command is None
            else bool(accept_existing_goal_command)
        )
        self.keep_alive_after_arrival = (
            _env_bool("KEEP_ALIVE_AFTER_ARRIVAL", True)
            if keep_alive_after_arrival is None
            else bool(keep_alive_after_arrival)
        )

        self._goal_command_file: Optional[Path] = None
        self._last_goal_command_seq = None
        self._launch_time = time.time()
        self._live_goal_count = 0

    def on_step_end(self, *args, **kwargs) -> None:
        super().on_step_end(*args, **kwargs)
        self._goal_command_file = self._resolve_goal_command_file()
        self._log(
            "[general] live_goal mode enabled "
            f"command_file={self._goal_command_file} "
            f"local_plan_distance={self.local_plan_distance:.3f}m "
            f"keep_alive_after_arrival={self.keep_alive_after_arrival}"
        )
        self._log("[general] command format: absolute env-local/global goal by default, e.g. `0 3`")

    def eval_step(self, env, _results) -> bool:
        if self._done:
            return True

        if self._initialized:
            command = env.motion_command
            env_idx = min(self.env_index, env.num_envs - 1)
            cur = self._robot_qpos_mujoco(command, env_idx)
            cur_xy = cur[:2]
            self._maybe_apply_goal_command(cur_xy)

            # The normal callback stops after the planner-goal hold completes.
            # In interactive mode, stay alive at the frozen reference tail so a
            # later command can wake the loop and force a new local plan.
            if self.keep_alive_after_arrival and self._arrived:
                self._step += 1
                self._update_target_markers(command, env_idx)
                self._update_plan_markers()
                if self._step >= self.max_steps:
                    self._stop_with_report(cur, f"max_steps={self.max_steps} reached while waiting for a new goal")
                    return True
                if self._step == self._hold_until_step:
                    self._log("[general] goal hold complete; waiting for the next command without stopping.")
                return False

        return super().eval_step(env, _results)

    def _resolve_goal_command_file(self) -> Path:
        if self.goal_command_file_arg:
            return Path(self.goal_command_file_arg)
        return self._paths.root / "general_goal_command.json"

    def _maybe_apply_goal_command(self, cur_xy: np.ndarray) -> None:
        path = self._goal_command_file or self._resolve_goal_command_file()
        try:
            stat = path.stat()
        except FileNotFoundError:
            return
        except OSError as exc:
            self._log(f"[general] could not stat goal command file {path}: {exc}")
            return

        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            return  # writer may be mid-atomic-replace on a non-POSIX filesystem
        except OSError as exc:
            self._log(f"[general] could not read goal command file {path}: {exc}")
            return

        seq = data.get("seq", stat.st_mtime_ns)
        if seq == self._last_goal_command_seq:
            return
        created_at = float(data.get("created_at", stat.st_mtime))
        if (
            self._last_goal_command_seq is None
            and not self.accept_existing_goal_command
            and created_at < self._launch_time - 1.0
        ):
            self._last_goal_command_seq = seq
            self._log(f"[general] ignoring stale pre-launch goal command seq={seq}; enter a new command.")
            return

        try:
            goal_xy, command_desc = self._resolve_command_goal_xy(data, cur_xy)
        except Exception as exc:  # noqa: BLE001
            self._last_goal_command_seq = seq
            self._log(f"[general] invalid goal command seq={seq}: {exc}")
            return
        if goal_xy.shape != (2,) or not np.isfinite(goal_xy).all():
            self._last_goal_command_seq = seq
            self._log(f"[general] invalid goal_xy in command seq={seq}: {goal_xy!r}")
            return

        self._last_goal_command_seq = seq
        self._switch_to_commanded_goal(goal_xy, cur_xy, seq, command_desc=command_desc)

    def _resolve_command_goal_xy(self, data: dict, cur_xy: np.ndarray) -> tuple[np.ndarray, str]:
        mode = str(data.get("command_mode", "")).strip().lower()
        if "goal_delta_xy" in data or mode == "relative":
            delta = np.asarray(data["goal_delta_xy"], dtype=np.float32)
            if delta.shape != (2,):
                raise ValueError(f"goal_delta_xy must have shape (2,), got {delta!r}")
            return cur_xy.astype(np.float32) + delta, f"relative_delta={delta.round(3).tolist()}"
        if "goal_xy" in data or mode in {"", "absolute"}:
            goal = np.asarray(data["goal_xy"], dtype=np.float32)
            if goal.shape != (2,):
                raise ValueError(f"goal_xy must have shape (2,), got {goal!r}")
            return goal, "absolute_goal"
        raise ValueError("expected goal_delta_xy or goal_xy")

    def _switch_to_commanded_goal(
        self,
        goal_xy: np.ndarray,
        cur_xy: np.ndarray,
        seq,
        command_desc: str = "absolute_goal",
    ) -> None:
        delta = goal_xy - cur_xy
        remaining = float(np.linalg.norm(delta))
        if remaining > 1e-6:
            heading = math.atan2(float(delta[1]), float(delta[0]))
        else:
            heading = float(self._goal_heading)

        old_pending = self._pending_plan["id"] if self._pending_plan is not None else None
        if old_pending is not None:
            try:
                self._paths.request_path(old_pending).unlink()
            except OSError:
                pass

        self._live_goal_count += 1
        self._goal_xy = goal_xy.astype(np.float32)
        self._goal_heading = heading
        self._pending_plan = None
        self._need_replan = True
        self._phase = "walk"
        self._cur_plan_reaches_goal = False
        self._tail_wait_logged = False
        self._arrived = False
        self._target_body_pos_w = None

        msg = (
            "[general] new commanded goal "
            f"seq={seq} count={self._live_goal_count} "
            f"{command_desc} "
            f"cur_xy={cur_xy.round(3).tolist()} "
            f"goal_xy={self._goal_xy.round(3).tolist()} "
            f"heading_to_goal={math.degrees(self._goal_heading):.1f}deg "
            f"remaining={remaining:.3f}m"
        )
        if old_pending is not None:
            msg += f" dropped_pending={old_pending}"
        self._log(msg)

    def _plan_params(self, cur_xy: np.ndarray, remaining: float) -> tuple[float, float, float, bool]:
        local_max = max(0.05, float(self.local_plan_distance))
        remaining = float(remaining)
        if remaining <= local_max + self.final_leg_distance:
            # Avoid leaving a tiny residual plan near the goal. Once the capped
            # local step would enter the final-leg radius, make this request the
            # exact goal-reaching plan.
            dist = remaining
            reach_target = True
        else:
            dist = local_max
            reach_target = False
        if remaining < 0.30:
            heading = self._goal_heading
        else:
            delta = self._goal_xy - cur_xy
            heading = math.atan2(float(delta[1]), float(delta[0]))
            self._goal_heading = heading
        duration = max(self.min_duration, dist * self.seconds_per_meter)
        return dist, heading, duration, reach_target
