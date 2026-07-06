"""Receding-horizon MotionBricks planner callback for GEAR-Sonic eval."""

from __future__ import annotations

from collections import deque
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MOTIONBRICKS_ROOT = REPO_ROOT / "motionbricks"
for path in (REPO_ROOT, MOTIONBRICKS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from motion_sonic import motionbricks_trajectory as mb_traj  # noqa: E402


class MotionBricksRecedingHorizonCallback:
    """Generate short MotionBricks plans and inject them into SONIC tracking.

    The callback runs inside ``gear_sonic/eval_agent_trl.py``. Every
    ``replan_interval_steps`` simulation steps it:

    1. Reads the current robot state.
    2. Builds a short MuJoCo qpos history in MotionBricks ordering.
    3. Generates a fresh future kinematic segment toward a lookahead target.
    4. Replaces the active ``TrackingCommand`` reference segment in-place.
    """

    def __init__(
        self,
        forward_meters: float = 5.0,
        target_height: float = 0.78,
        mode: str = "walk",
        arrival_mode: str = "idle",
        target_vel: float = 0.20,
        lookahead_meters: float = 0.35,
        arrival_radius: float = 0.15,
        replan_interval_steps: int = 5,
        segment_frames: int = 90,
        fps: int = 30,
        env_index: int = 0,
        random_seed: int = 1234,
        generate_dt: float = 2.0,
        warm_start_context: int = 4,
        install_on_first_step: bool = True,
        stop_at_target: bool = False,
        use_motion_lib_final_target: bool = True,
        final_approach_radius: float = 0.80,
        final_snap_frames: int = 12,
        final_hold_frames: int = 24,
        result_dir: str | None = None,
        data_root: str | None = None,
        humanoid_xml: str | None = None,
        skeleton_xml: str | None = None,
    ):
        self.forward_meters = float(forward_meters)
        self.target_height = float(target_height)
        self.mode = mode
        self.arrival_mode = arrival_mode
        self.target_vel = float(target_vel)
        self.lookahead_meters = float(lookahead_meters)
        self.arrival_radius = float(arrival_radius)
        self.replan_interval_steps = max(1, int(replan_interval_steps))
        self.segment_frames = max(8, int(segment_frames))
        self.fps = int(fps)
        self.env_index = int(env_index)
        self.random_seed = int(random_seed)
        self.generate_dt = float(generate_dt)
        self.warm_start_context = max(4, int(warm_start_context))
        self.install_on_first_step = bool(install_on_first_step)
        self.stop_at_target = bool(stop_at_target)
        self.use_motion_lib_final_target = bool(use_motion_lib_final_target)
        self.final_approach_radius = max(0.0, float(final_approach_radius))
        self.final_snap_frames = max(0, int(final_snap_frames))
        self.final_hold_frames = max(0, int(final_hold_frames))
        self.result_dir = result_dir
        self.data_root = data_root
        self.humanoid_xml = humanoid_xml
        self.skeleton_xml = skeleton_xml

        self._demo = None
        self._qpos_history: deque[np.ndarray] = deque(maxlen=self.warm_start_context)
        self._step = 0
        self._installed_once = False
        self._target_qpos = mb_traj.make_forward_target(self.forward_meters, self.target_height)
        self._motion_lib_final_target_qpos: np.ndarray | None = None
        self._ending_active = False

    def on_step_end(self, *_args, **_kwargs) -> None:
        """Compatibility hook called once before the eval loop."""
        self._ensure_motionbricks()

    def eval_step(self, env, _results) -> bool:
        command = env.motion_command
        env_idx = min(self.env_index, env.num_envs - 1)
        self._ensure_motion_lib_final_target(command, env_idx)
        current_qpos = self._robot_qpos_mujoco(command, env_idx)
        self._qpos_history.append(current_qpos)

        should_replan = (
            (not self._installed_once and self.install_on_first_step)
            or self._step % self.replan_interval_steps == 0
        )
        self._step += 1
        if not should_replan:
            return False

        segment = self._plan_segment(current_qpos)
        command.install_live_qpos_segment(
            segment,
            fps=self.fps,
            env_ids=torch.tensor([env_idx], device=command.device),
            reset_time=True,
        )
        self._installed_once = True

        remaining = float(np.linalg.norm(current_qpos[:2] - self._active_final_target_qpos()[:2]))
        return bool(self.stop_at_target and remaining <= self.arrival_radius)

    def _ensure_motionbricks(self) -> None:
        if self._demo is not None:
            return
        args = mb_traj.parse_args_for_live_defaults()
        args.mode = self.mode
        args.target_vel = self.target_vel
        args.random_seed = self.random_seed
        args.generate_dt = self.generate_dt
        args.result_dir = self.result_dir or str(MOTIONBRICKS_ROOT / "out")
        args.data_root = self.data_root or str(MOTIONBRICKS_ROOT / "datasets")
        args.humanoid_xml = self.humanoid_xml or str(mb_traj.DEFAULT_HUMANOID_XML)
        args.skeleton_xml = self.skeleton_xml or str(mb_traj.DEFAULT_SKELETON_XML)
        args.lookat_movement_direction = 1
        args.speed_scale = "1.0,1.0"
        args.force_generation = True
        self._demo = mb_traj.build_motionbricks_demo(args)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        self._demo.full_agent.reset()

    def _plan_segment(self, current_qpos: np.ndarray) -> np.ndarray:
        self._ensure_motionbricks()
        assert self._demo is not None

        target_qpos, mode_name, is_final_approach = self._lookahead_target(current_qpos)
        context = self._context_tensor(current_qpos)
        control = self._control_signals(context, current_qpos, target_qpos, mode_name)
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
        if is_final_approach:
            planned = self._append_final_target_transition(planned, target_qpos)
        return planned

    def _active_final_target_qpos(self) -> np.ndarray:
        if self.use_motion_lib_final_target and self._motion_lib_final_target_qpos is not None:
            return self._motion_lib_final_target_qpos
        return self._target_qpos

    def _lookahead_target(self, current_qpos: np.ndarray) -> tuple[np.ndarray, str, bool]:
        final_target = self._active_final_target_qpos()
        target = final_target.copy()
        delta = target[:2] - current_qpos[:2]
        dist = float(np.linalg.norm(delta))
        if dist <= self.final_approach_radius:
            self._ending_active = True
            mode_name = self.arrival_mode if dist <= self.arrival_radius else self.mode
            return target, mode_name, True
        if dist > self.lookahead_meters:
            target[:2] = current_qpos[:2] + delta / max(dist, 1e-6) * self.lookahead_meters
        return target, self.mode, False

    def _append_final_target_transition(self, planned: np.ndarray, target_qpos: np.ndarray) -> np.ndarray:
        pieces = [planned]
        if self.final_snap_frames > 0:
            start = planned[-1].copy()
            transition = []
            for idx in range(1, self.final_snap_frames + 1):
                alpha = idx / float(self.final_snap_frames)
                frame = (1.0 - alpha) * start + alpha * target_qpos
                frame[3:7] = target_qpos[3:7]
                transition.append(frame.astype(np.float32))
            pieces.append(np.asarray(transition, dtype=np.float32))
        if self.final_hold_frames > 0:
            pieces.append(
                np.repeat(target_qpos[None, :], self.final_hold_frames, axis=0).astype(np.float32)
            )
        return np.concatenate(pieces, axis=0)

    def _ensure_motion_lib_final_target(self, command, env_idx: int) -> None:
        if not self.use_motion_lib_final_target or self._motion_lib_final_target_qpos is not None:
            return

        motion_id = int(command.motion_ids[env_idx].item())
        num_frames = int(command.motion_lib._motion_num_frames[motion_id].item())  # noqa: SLF001
        final_step = max(0, num_frames - 1)
        motion_ids = torch.tensor([motion_id], device=command.device, dtype=torch.long)
        motion_steps = torch.tensor([final_step], device=command.device, dtype=torch.long)

        root_pos = command.motion_lib.get_root_pos_w(motion_ids, motion_steps)[0]
        root_quat_wxyz = command.motion_lib.get_root_quat_w(motion_ids, motion_steps)[0]
        dof_isaaclab = command.motion_lib.get_dof_pos(motion_ids, motion_steps)[0]
        dof_mujoco = dof_isaaclab[command.isaaclab_to_mujoco_dof][:29]

        qpos = torch.cat([root_pos, root_quat_wxyz, dof_mujoco], dim=0)
        qpos_np = qpos.detach().cpu().numpy().astype(np.float32)
        if qpos_np.shape[0] < 36:
            padded = np.zeros(36, dtype=np.float32)
            padded[: qpos_np.shape[0]] = qpos_np
            qpos_np = padded
        self._motion_lib_final_target_qpos = qpos_np[:36]
        self._target_qpos = self._motion_lib_final_target_qpos.copy()

    def _context_tensor(self, current_qpos: np.ndarray) -> torch.Tensor:
        while len(self._qpos_history) < self.warm_start_context:
            self._qpos_history.appendleft(current_qpos.copy())
        history = np.asarray(list(self._qpos_history)[-4:], dtype=np.float32)
        return torch.from_numpy(history).view(1, 4, 36)

    def _control_signals(
        self,
        context_mujoco_qpos: torch.Tensor,
        current_qpos: np.ndarray,
        target_qpos: np.ndarray,
        mode_name: str,
    ) -> dict[str, torch.Tensor]:
        assert self._demo is not None
        clip_names = list(self._demo.full_agent._clip_holder.CLIPS.keys())
        if mode_name not in clip_names:
            raise ValueError(f"Unknown MotionBricks mode {mode_name!r}; available={clip_names}")
        mode_id = clip_names.index(mode_name)
        delta = target_qpos[:2] - current_qpos[:2]
        norm = float(np.linalg.norm(delta))
        if norm > 1e-6:
            direction = delta / norm
        else:
            direction = np.asarray([1.0, 0.0], dtype=np.float32)
        facing = np.asarray(
            [math.cos(mb_traj.quat_wxyz_to_yaw(target_qpos[3:7])), math.sin(mb_traj.quat_wxyz_to_yaw(target_qpos[3:7]))],
            dtype=np.float32,
        )
        mode = torch.tensor([[mode_id]], dtype=torch.long)
        return {
            "movement_direction": torch.tensor([[direction[0], direction[1], 0.0]], dtype=torch.float32),
            "facing_direction": torch.tensor([[facing[0], facing[1], 0.0]], dtype=torch.float32),
            "mode": mode,
            "context_mujoco_qpos": context_mujoco_qpos,
            "target_vel": torch.tensor([self.target_vel], dtype=torch.float32),
            "specific_target_positions": torch.from_numpy(target_qpos[:3].astype(np.float32)).view(1, 1, 3),
            "specific_target_headings": torch.tensor([[mb_traj.quat_wxyz_to_yaw(target_qpos[3:7])]], dtype=torch.float32),
            "has_specific_target": torch.ones((1, 1), dtype=torch.int32),
            "random_seed": torch.tensor([self.random_seed + self._step], dtype=torch.long),
            "allowed_pred_num_tokens": self._demo.controller.get_default_allowed_pred_num_tokens(mode_id),
        }

    def _robot_qpos_mujoco(self, command, env_idx: int) -> np.ndarray:
        root_pos = command.robot.data.root_pos_w[env_idx].detach().clone()
        env_origins = getattr(command._env.scene, "env_origins", None)  # noqa: SLF001
        if env_origins is not None:
            root_pos = root_pos - env_origins[env_idx].to(root_pos.device)
        root_quat_wxyz = command.robot.data.root_quat_w[env_idx].detach().clone()
        try:
            from gear_sonic.isaac_utils import quaternion_adapter as quat_adapter

            root_quat_wxyz = quat_adapter.isaaclab_to_wxyz(root_quat_wxyz)
        except Exception:
            pass

        joint_pos_isaac = command.robot.data.joint_pos[env_idx].detach()
        joint_pos_mujoco = joint_pos_isaac[command.isaaclab_to_mujoco_dof]
        qpos = torch.cat([root_pos, root_quat_wxyz, joint_pos_mujoco[:29]], dim=0)
        return qpos.detach().cpu().numpy().astype(np.float32)
