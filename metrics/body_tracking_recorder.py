"""Recorder for first-episode body tracking errors.

This recorder is intended to be attached as ``manager_env.recorders.trajectory`` so
the existing ``end_render_results()`` hook closes and saves it when ``run_once``
finishes.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from isaaclab.managers import manager_term_cfg, recorder_manager
from isaaclab.utils import configclass
import numpy as np
import torch

if TYPE_CHECKING:
    from isaaclab import envs


class BodyTrackingRecorderTerm(recorder_manager.RecorderTerm):
    """Save reference-vs-robot body tracking data for one rollout."""

    cfg: "BodyTrackingRecorderCfg"

    def __init__(self, cfg: "BodyTrackingRecorderCfg", env: "envs.ManagerBasedEnv"):
        super().__init__(cfg, env)
        self.cfg = cfg
        self.env = env
        self.save_dir = cfg.save_path
        os.makedirs(self.save_dir, exist_ok=True)
        self.env_index = int(cfg.env_index)
        self._closed = False
        self._body_names: list[str] | None = None
        self._frames: dict[str, list] = {
            "time_step": [],
            "motion_id": [],
            "ref_body_pos_w": [],
            "robot_body_pos_w": [],
            "body_error": [],
            "ref_joint_pos": [],
            "robot_joint_pos": [],
            "joint_error": [],
            "body_error_mean": [],
            "body_error_max": [],
            "joint_error_mean": [],
            "joint_error_max": [],
            "anchor_pos_error": [],
            "vr3_error_mean": [],
            "foot_error_mean": [],
        }
        self._command_metric_frames: dict[str, list] = {}
        self._joint_names: list[str] | None = None

    def _motion_command(self):
        return self.env.command_manager.get_term("motion")

    def _body_indices(self, names: list[str]) -> list[int]:
        assert self._body_names is not None
        return [self._body_names.index(name) for name in names if name in self._body_names]

    @staticmethod
    def _mean_for_indices(values: np.ndarray, indices: list[int]) -> float:
        if not indices:
            return float("nan")
        return float(np.mean(values[indices]))

    def record_post_step(self) -> tuple[str | None, torch.Tensor | dict | None]:
        """Record after physics/reward computation and before reset."""
        if self._closed:
            return "body_tracking_record", torch.ones(self.env.num_envs, 1, device=self.env.device)

        command = self._motion_command()
        env_idx = min(self.env_index, self.env.num_envs - 1)
        if self._body_names is None:
            self._body_names = list(command.cfg.body_names)

        ref_joint = command.joint_pos[env_idx].detach().cpu().numpy()
        if getattr(command, "has_dof_mismatch", False):
            joint_indices = command.body_joint_indices
            robot_joint = command.robot_joint_pos[env_idx, joint_indices].detach().cpu().numpy()
            if self._joint_names is None:
                self._joint_names = [command.robot.joint_names[int(i)] for i in joint_indices]
        else:
            robot_joint = command.robot_joint_pos[env_idx].detach().cpu().numpy()
            if self._joint_names is None:
                self._joint_names = list(command.robot.joint_names)
        joint_count = min(ref_joint.shape[-1], robot_joint.shape[-1])
        ref_joint = ref_joint[:joint_count]
        robot_joint = robot_joint[:joint_count]
        joint_error = robot_joint - ref_joint
        if self._joint_names is not None:
            self._joint_names = self._joint_names[:joint_count]

        ref_body = command.body_pos_w[env_idx].detach().cpu().numpy()
        robot_body = command.robot_body_pos_w[env_idx].detach().cpu().numpy()
        body_error = np.linalg.norm(ref_body - robot_body, axis=-1)

        anchor_error = torch.norm(
            command.anchor_pos_w[env_idx] - command.robot_anchor_pos_w[env_idx]
        ).item()

        vr3_indices = self._body_indices(["torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link"])
        foot_indices = self._body_indices(["left_ankle_roll_link", "right_ankle_roll_link"])

        self._frames["time_step"].append(int(command.time_steps[env_idx].detach().cpu().item()))
        self._frames["motion_id"].append(int(command.motion_ids[env_idx].detach().cpu().item()))
        self._frames["ref_body_pos_w"].append(ref_body)
        self._frames["robot_body_pos_w"].append(robot_body)
        self._frames["body_error"].append(body_error)
        self._frames["ref_joint_pos"].append(ref_joint)
        self._frames["robot_joint_pos"].append(robot_joint)
        self._frames["joint_error"].append(joint_error)
        self._frames["body_error_mean"].append(float(np.mean(body_error)))
        self._frames["body_error_max"].append(float(np.max(body_error)))
        self._frames["joint_error_mean"].append(float(np.mean(np.abs(joint_error))))
        self._frames["joint_error_max"].append(float(np.max(np.abs(joint_error))))
        self._frames["anchor_pos_error"].append(float(anchor_error))
        self._frames["vr3_error_mean"].append(self._mean_for_indices(body_error, vr3_indices))
        self._frames["foot_error_mean"].append(self._mean_for_indices(body_error, foot_indices))

        try:
            command._update_metrics()  # noqa: SLF001
            for key, value in command.metrics.items():
                if isinstance(value, torch.Tensor) and value.numel() >= self.env.num_envs:
                    self._command_metric_frames.setdefault(key, []).append(
                        float(value[env_idx].detach().cpu().item())
                    )
        except Exception:
            pass

        return "body_tracking_record", torch.ones(self.env.num_envs, 1, device=self.env.device)

    def close_writers(self) -> None:
        """Save arrays and a compact JSON summary."""
        if self._closed:
            return
        self._closed = True
        if not self._frames["body_error"]:
            return

        arrays = {key: np.asarray(value) for key, value in self._frames.items()}
        for key, value in self._command_metric_frames.items():
            arrays[f"command_metric__{key}"] = np.asarray(value, dtype=np.float64)

        npz_path = os.path.join(self.save_dir, "first_episode_body_tracking.npz")
        np.savez_compressed(
            npz_path,
            **arrays,
            body_names=np.asarray(self._body_names or [], dtype=object),
            joint_names=np.asarray(self._joint_names or [], dtype=object),
        )

        summary = {
            "npz_file": os.path.basename(npz_path),
            "num_frames": int(len(self._frames["body_error"])),
            "env_index": int(self.env_index),
            "motion_id": int(self._frames["motion_id"][0]),
            "body_error_mean": float(np.mean(arrays["body_error_mean"])),
            "body_error_max": float(np.max(arrays["body_error_max"])),
            "joint_error_mean": float(np.mean(arrays["joint_error_mean"])),
            "joint_error_max": float(np.max(arrays["joint_error_max"])),
            "anchor_pos_error_mean": float(np.mean(arrays["anchor_pos_error"])),
            "vr3_error_mean": float(np.nanmean(arrays["vr3_error_mean"])),
            "foot_error_mean": float(np.nanmean(arrays["foot_error_mean"])),
        }
        for key, value in self._command_metric_frames.items():
            if value:
                summary[f"{key}_mean"] = float(np.mean(value))

        with open(os.path.join(self.save_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def __del__(self):
        self.close_writers()


@configclass
class BodyTrackingRecorderCfg(manager_term_cfg.RecorderTermCfg):
    """Configuration for first-episode body tracking recording."""

    class_type = BodyTrackingRecorderTerm
    save_path: str = "/workspace/GR00T-WholeBodyControl/metrics/first_episode"
    env_index: int = 0
