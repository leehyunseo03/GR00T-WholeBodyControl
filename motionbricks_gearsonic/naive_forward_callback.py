"""Naive forward MotionBricks replanning callback for GEAR-Sonic eval."""

from __future__ import annotations

import logging

import numpy as np
import torch

from motion_sonic.live_motionbricks_callback import MotionBricksRecedingHorizonCallback

LOGGER = logging.getLogger(__name__)


class MotionBricksNaiveForwardCallback(MotionBricksRecedingHorizonCallback):
    """Continuously replan a forward-walking MotionBricks segment.

    This callback intentionally has no target position, arrival condition, or
    terminal pose. It reads the current robot qpos, asks MotionBricks for a
    short forward walking segment, and installs that segment into the active
    GEAR-Sonic motion reference.
    """

    def __init__(
        self,
        *,
        mode: str = "walk",
        target_vel: float = 0.40,
        replan_interval_steps: int = 25,
        segment_frames: int = 72,
        debug_print_interval_steps: int = 25,
        **kwargs,
    ):
        super().__init__(
            mode=mode,
            target_vel=target_vel,
            replan_interval_steps=replan_interval_steps,
            segment_frames=segment_frames,
            stop_at_target=False,
            use_motion_lib_final_target=False,
            target_pose_condition=False,
            **kwargs,
        )
        self.debug_print_interval_steps = max(1, int(debug_print_interval_steps))
        self._startup_printed = False

    def eval_step(self, env, _results) -> bool:
        command = env.motion_command
        env_idx = min(self.env_index, env.num_envs - 1)
        current_qpos = self._robot_qpos_mujoco(command, env_idx)
        self._qpos_history.append(current_qpos)
        self._print_startup_once()

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
                "[MotionBricksNaiveForward] replan "
                f"step={self._step} root_xy=({current_qpos[0]:.3f},{current_qpos[1]:.3f}) "
                f"mode={self.mode} target_vel={self.target_vel:.3f} "
                f"segment_frames={self.segment_frames}"
            )
        elif self._step % self.debug_print_interval_steps == 0:
            self._emit(
                "[MotionBricksNaiveForward] tracking "
                f"step={self._step} root_xy=({current_qpos[0]:.3f},{current_qpos[1]:.3f})"
            )

        self._step += 1
        return False

    def _plan_segment(self, current_qpos: np.ndarray) -> np.ndarray:
        self._ensure_motionbricks()
        assert self._demo is not None

        context = self._context_tensor(current_qpos)
        control = self._control_signals(context)
        with torch.no_grad():
            self._demo.full_agent.generate_new_frames(
                control,
                self._demo.controller.get_controller_dt() * self.generate_dt,
                force_generation=True,
            )

        planned = self._demo.full_agent.frames["mujoco_qpos"][0].detach().cpu().numpy()
        planned = planned[: self.segment_frames].astype(np.float32)
        if planned.shape[0] < self.segment_frames:
            pad = np.repeat(planned[-1:], self.segment_frames - planned.shape[0], axis=0)
            planned = np.concatenate([planned, pad], axis=0)
        return planned

    def _control_signals(self, context_mujoco_qpos: torch.Tensor) -> dict[str, torch.Tensor]:
        assert self._demo is not None
        clip_names = list(self._demo.full_agent._clip_holder.CLIPS.keys())
        if self.mode not in clip_names:
            raise ValueError(f"Unknown MotionBricks mode {self.mode!r}; available={clip_names}")
        mode_id = clip_names.index(self.mode)
        control = {
            "movement_direction": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            "facing_direction": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            "mode": torch.tensor([[mode_id]], dtype=torch.long),
            "context_mujoco_qpos": context_mujoco_qpos,
            "random_seed": torch.tensor([self.random_seed + self._step], dtype=torch.long),
            "allowed_pred_num_tokens": self._demo.controller.get_default_allowed_pred_num_tokens(mode_id),
        }
        if self.target_vel > 0.0:
            control["target_vel"] = torch.tensor([[self.target_vel]], dtype=torch.float32)
        return control

    def _print_startup_once(self) -> None:
        if self._startup_printed:
            return
        self._startup_printed = True
        self._emit(
            "[MotionBricksNaiveForward] start "
            f"mode={self.mode} target_vel={self.target_vel:.3f} "
            f"replan_interval_steps={self.replan_interval_steps} "
            f"segment_frames={self.segment_frames}"
        )

    @staticmethod
    def _emit(message: str) -> None:
        print(message, flush=True)
        LOGGER.info(message)

