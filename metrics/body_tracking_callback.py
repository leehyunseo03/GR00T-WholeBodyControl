"""Eval-step callback that saves first-episode body tracking data."""

from __future__ import annotations

import atexit
import json
import os
from typing import Any

import numpy as np
import torch


class BodyTrackingCallback:
    """Capture reference-vs-robot body tracking until the first reset/done."""

    def __init__(
        self,
        save_path: str = "/workspace/GR00T-WholeBodyControl/metrics/first_episode",
        env_index: int = 0,
        max_steps: int = 0,
        stop_on_done: bool = True,
    ):
        self.save_path = save_path
        self.env_index = int(env_index)
        self.max_steps = int(max_steps)
        self.stop_on_done = bool(stop_on_done)
        self._saved = False
        self._done_logged = False
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
            "done": [],
        }
        self._command_metrics: dict[str, list] = {}
        self._joint_names: list[str] | None = None
        atexit.register(self._save)

    def on_step_end(self, *_args, **_kwargs) -> None:
        """Compatibility hook called once before the eval loop."""
        os.makedirs(self.save_path, exist_ok=True)

    @staticmethod
    def _as_done_mask(dones: torch.Tensor | np.ndarray | Any) -> np.ndarray:
        if isinstance(dones, torch.Tensor):
            arr = dones.detach().cpu().numpy()
        else:
            arr = np.asarray(dones)
        return arr.astype(bool).reshape(-1)

    @staticmethod
    def _scalar_for_env(value: Any, env_idx: int) -> float | None:
        try:
            if isinstance(value, torch.Tensor):
                arr = value.detach().cpu().reshape(-1)
                if arr.numel() == 0:
                    return None
                return float(arr[min(env_idx, arr.numel() - 1)].item())
            arr = np.asarray(value).reshape(-1)
            if arr.size == 0:
                return None
            return float(arr[min(env_idx, arr.size - 1)])
        except Exception:
            return None

    def _done_reason(self, env, extras: dict[str, Any], env_idx: int) -> str:
        parts: list[str] = []

        base_env = getattr(env, "env", env)
        terminated = self._scalar_for_env(getattr(base_env, "reset_terminated", None), env_idx)
        truncated = self._scalar_for_env(getattr(base_env, "reset_time_outs", None), env_idx)
        if terminated is not None:
            parts.append(f"terminated={bool(terminated)}")
        if truncated is not None:
            parts.append(f"truncated={bool(truncated)}")

        time_outs = self._scalar_for_env(extras.get("time_outs"), env_idx)
        if time_outs is not None and truncated is None:
            parts.append(f"truncated={bool(time_outs)}")

        logs = extras.get("to_log") or extras.get("log") or {}
        fired_terms = []
        for key, value in logs.items():
            if not str(key).startswith("Episode_Termination/"):
                continue
            scalar = self._scalar_for_env(value, 0)
            if scalar is not None and scalar > 0.0:
                fired_terms.append(f"{key.split('/', 1)[1]}={scalar:g}")
        if fired_terms:
            parts.append("episode_terms=" + ",".join(fired_terms))

        manager = getattr(base_env, "termination_manager", None)
        term_names = getattr(manager, "_term_names", [])
        live_terms = []
        for term_name in term_names:
            try:
                scalar = self._scalar_for_env(manager.get_term(term_name), env_idx)
            except Exception:
                continue
            if scalar is not None and scalar > 0.0:
                live_terms.append(term_name)
        if live_terms:
            parts.append("live_terms=" + ",".join(live_terms))

        if hasattr(base_env, "episode_length_buf"):
            ep_step = self._scalar_for_env(base_env.episode_length_buf, env_idx)
            max_len = getattr(base_env, "max_episode_length", None)
            if ep_step is not None and max_len is not None:
                try:
                    parts.append(f"episode_step={int(ep_step)}/{int(max_len)}")
                except Exception:
                    pass

        return "; ".join(parts) if parts else "reason unavailable"

    def _body_indices(self, names: list[str]) -> list[int]:
        assert self._body_names is not None
        return [self._body_names.index(name) for name in names if name in self._body_names]

    @staticmethod
    def _mean_for_indices(values: np.ndarray, indices: list[int]) -> float:
        if not indices:
            return float("nan")
        return float(np.mean(values[indices]))

    def _record_frame(self, env, done: bool) -> None:
        command = env.motion_command
        env_idx = min(self.env_index, env.num_envs - 1)
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
        self._frames["done"].append(bool(done))

        try:
            command._update_metrics()  # noqa: SLF001
            for key, value in command.metrics.items():
                if isinstance(value, torch.Tensor) and value.numel() >= env.num_envs:
                    self._command_metrics.setdefault(key, []).append(
                        float(value[env_idx].detach().cpu().item())
                    )
        except Exception:
            pass

    def _save(self) -> None:
        if self._saved:
            return
        self._saved = True
        os.makedirs(self.save_path, exist_ok=True)

        if not self._frames["body_error"]:
            summary = {"num_frames": 0, "message": "No frames were recorded before done/max_steps."}
            with open(os.path.join(self.save_path, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            return

        arrays = {key: np.asarray(value) for key, value in self._frames.items()}
        for key, value in self._command_metrics.items():
            arrays[f"command_metric__{key}"] = np.asarray(value, dtype=np.float64)

        npz_path = os.path.join(self.save_path, "first_episode_body_tracking.npz")
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
        for key, value in self._command_metrics.items():
            if value:
                summary[f"{key}_mean"] = float(np.mean(value))

        with open(os.path.join(self.save_path, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def eval_step(self, env, results) -> bool:
        """Record this step and return True when the eval loop should stop."""
        if self._saved:
            return True

        dones = self._as_done_mask(results[2])
        extras = results[3] if len(results) > 3 and isinstance(results[3], dict) else {}
        env_idx = min(self.env_index, len(dones) - 1)
        done = bool(dones[env_idx])

        if done and not self._done_logged:
            self._done_logged = True
            reason = self._done_reason(env, extras, env_idx)
            print(
                "[body_tracking] env done observed "
                f"(env_index={env_idx}, stop_on_done={self.stop_on_done}, "
                f"recorded_frames={len(self._frames['body_error'])}, {reason}).",
                flush=True,
            )

        # In IsaacLab, done envs may already have reset by the time callbacks run.
        # So skip the done frame and save what we collected before reset.
        if done and self.stop_on_done:
            self._save()
            return True

        self._record_frame(env, done=done)

        if self.max_steps > 0 and len(self._frames["body_error"]) >= self.max_steps:
            self._save()
            return True

        return False

    def on_eval_end(self, *_args, **_kwargs) -> None:
        """Flush partial recordings when another eval callback stops the loop."""
        self._save()
