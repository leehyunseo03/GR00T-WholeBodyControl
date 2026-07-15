"""GEAR-Sonic eval callback that keeps MotionBricks on native position goals."""

from __future__ import annotations

import logging
import numpy as np
import torch

from motion_sonic.live_motionbricks_callback import MotionBricksRecedingHorizonCallback

LOGGER = logging.getLogger(__name__)


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class MotionBricksNativeStopCallback(MotionBricksRecedingHorizonCallback):
    """Receding-horizon callback with pose conditioning disabled by default.

    MotionBricks plans toward root position/heading only until the robot enters
    ``final_approach_radius``. At that point the callback installs one terminal
    qpos segment and stops replanning, so GEAR-Sonic can actually play through
    the final 29-DOF pose instead of having the reference time reset every
    replan.
    """

    def __init__(
        self,
        *,
        target_pose_condition: bool = False,
        stop_at_target: bool = False,
        arrival_radius: float = 0.15,
        final_approach_radius: float = 0.80,
        final_snap_frames: int = 60,
        final_hold_frames: int = 120,
        debug_print_interval_steps: int = 25,
        **kwargs,
    ):
        super().__init__(
            target_pose_condition=_as_bool(target_pose_condition),
            stop_at_target=_as_bool(stop_at_target),
            arrival_radius=arrival_radius,
            final_approach_radius=final_approach_radius,
            final_snap_frames=final_snap_frames,
            final_hold_frames=final_hold_frames,
            **kwargs,
        )
        self.debug_print_interval_steps = max(1, int(debug_print_interval_steps))
        self._terminal_pose_installed = False
        self._terminal_pose_steps = 0
        self._arrival_printed = False
        self._final_approach_printed = False
        self._startup_printed = False

    def eval_step(self, env, _results) -> bool:
        command = env.motion_command
        env_idx = min(self.env_index, env.num_envs - 1)
        self._ensure_motion_lib_final_target(command, env_idx)
        current_qpos = self._robot_qpos_mujoco(command, env_idx)
        self._qpos_history.append(current_qpos)

        target_qpos = self._active_final_target_qpos()
        remaining = float(np.linalg.norm(current_qpos[:2] - target_qpos[:2]))
        dof_rmse = float(np.linalg.norm(current_qpos[7:36] - target_qpos[7:36]) / np.sqrt(29))
        self._print_startup_once(target_qpos)

        if self._terminal_pose_installed:
            self._terminal_pose_steps += 1
            self._step += 1
            if self._terminal_pose_steps % self.debug_print_interval_steps == 0:
                self._emit(
                    "[MotionBricksNativeStop] terminal_pose "
                    f"step={self._step} terminal_step={self._terminal_pose_steps} "
                    f"root_xy=({current_qpos[0]:.3f},{current_qpos[1]:.3f}) "
                    f"target_xy=({target_qpos[0]:.3f},{target_qpos[1]:.3f}) "
                    f"remaining={remaining:.3f}m dof_rmse={dof_rmse:.4f}"
                )
            if remaining <= self.arrival_radius:
                self._print_arrived(current_qpos, target_qpos, remaining, dof_rmse)
            return bool(self.stop_at_target and remaining <= self.arrival_radius and dof_rmse <= 0.05)

        if remaining <= self.final_approach_radius:
            segment = self._terminal_pose_segment(current_qpos, target_qpos)
            command.install_live_qpos_segment(
                segment,
                fps=self.fps,
                env_ids=torch.tensor([env_idx], device=command.device),
                reset_time=True,
            )
            self._terminal_pose_installed = True
            self._installed_once = True
            self._print_final_approach(current_qpos, target_qpos, remaining, dof_rmse)
            if remaining <= self.arrival_radius:
                self._print_arrived(current_qpos, target_qpos, remaining, dof_rmse)
            self._step += 1
            return False

        should_replan = (
            (not self._installed_once and self.install_on_first_step)
            or self._step % self.replan_interval_steps == 0
        )
        if should_replan:
            segment = self._plan_segment(current_qpos)
            command.install_live_qpos_segment(
                segment,
                fps=self.fps,
                env_ids=torch.tensor([env_idx], device=command.device),
                reset_time=True,
            )
            self._installed_once = True
            self._emit(
                "[MotionBricksNativeStop] replan "
                f"step={self._step} root_xy=({current_qpos[0]:.3f},{current_qpos[1]:.3f}) "
                f"target_xy=({target_qpos[0]:.3f},{target_qpos[1]:.3f}) "
                f"remaining={remaining:.3f}m lookahead={self.lookahead_meters:.3f}m "
                f"target_pose_condition={self.target_pose_condition}"
            )
        elif self._step % self.debug_print_interval_steps == 0:
            self._emit(
                "[MotionBricksNativeStop] tracking "
                f"step={self._step} root_xy=({current_qpos[0]:.3f},{current_qpos[1]:.3f}) "
                f"target_xy=({target_qpos[0]:.3f},{target_qpos[1]:.3f}) "
                f"remaining={remaining:.3f}m"
            )

        self._step += 1
        return False

    def _lookahead_target(self, current_qpos: np.ndarray) -> tuple[np.ndarray, str, bool]:
        final_target = self._active_final_target_qpos()
        target = final_target.copy()
        delta = target[:2] - current_qpos[:2]
        dist = float(np.linalg.norm(delta))
        if dist > self.lookahead_meters:
            target[:2] = current_qpos[:2] + delta / max(dist, 1e-6) * self.lookahead_meters
        return target, self.mode, False

    def _terminal_pose_segment(self, current_qpos: np.ndarray, target_qpos: np.ndarray) -> np.ndarray:
        pieces = []
        blend_frames = max(1, int(self.final_snap_frames))
        start = current_qpos.astype(np.float32).copy()
        target = target_qpos.astype(np.float32).copy()
        blend = []
        for idx in range(blend_frames):
            alpha = (idx + 1) / float(blend_frames)
            frame = (1.0 - alpha) * start + alpha * target
            frame[3:7] = target[3:7]
            blend.append(frame.astype(np.float32))
        pieces.append(np.asarray(blend, dtype=np.float32))
        if self.final_hold_frames > 0:
            pieces.append(np.repeat(target[None, :], self.final_hold_frames, axis=0).astype(np.float32))
        return np.concatenate(pieces, axis=0)

    def _print_startup_once(self, target_qpos: np.ndarray) -> None:
        if self._startup_printed:
            return
        self._startup_printed = True
        self._emit(
            "[MotionBricksNativeStop] start "
            f"target_xy=({target_qpos[0]:.3f},{target_qpos[1]:.3f}) "
            f"arrival_radius={self.arrival_radius:.3f}m "
            f"final_approach_radius={self.final_approach_radius:.3f}m "
            f"replan_interval_steps={self.replan_interval_steps} "
            f"segment_frames={self.segment_frames} "
            f"final_snap_frames={self.final_snap_frames} "
            f"final_hold_frames={self.final_hold_frames}"
        )

    def _print_final_approach(
        self,
        current_qpos: np.ndarray,
        target_qpos: np.ndarray,
        remaining: float,
        dof_rmse: float,
    ) -> None:
        if self._final_approach_printed:
            return
        self._final_approach_printed = True
        self._emit(
            "[MotionBricksNativeStop] FINAL_APPROACH "
            f"step={self._step} root_xy=({current_qpos[0]:.3f},{current_qpos[1]:.3f}) "
            f"target_xy=({target_qpos[0]:.3f},{target_qpos[1]:.3f}) "
            f"remaining={remaining:.3f}m dof_rmse_before_terminal={dof_rmse:.4f}; "
            "disabled replanning and installed terminal pose segment."
        )

    def _print_arrived(
        self,
        current_qpos: np.ndarray,
        target_qpos: np.ndarray,
        remaining: float,
        dof_rmse: float,
    ) -> None:
        if self._arrival_printed:
            return
        self._arrival_printed = True
        self._emit(
            "[MotionBricksNativeStop] ARRIVED "
            f"step={self._step} root_xy=({current_qpos[0]:.3f},{current_qpos[1]:.3f}) "
            f"target_xy=({target_qpos[0]:.3f},{target_qpos[1]:.3f}) "
            f"remaining={remaining:.3f}m dof_rmse_before_terminal={dof_rmse:.4f}; "
            "within arrival radius."
        )

    @staticmethod
    def _emit(message: str) -> None:
        print(message, flush=True)
        LOGGER.info(message)
