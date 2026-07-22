"""Legged base pose estimator for real-robot-style ARDY replanning.

The estimator keeps a map-frame base pose using the same ingredients a deploy
adapter should provide on a G1:

* IMU orientation, expressed as a map-frame base quaternion.
* Joint-encoder FK foot positions in the base frame.
* Binary foot contact.

It intentionally does not require simulator global state. While one or both feet
are trusted as stance feet, it treats each stance foot as a fixed map-frame
anchor and solves the base XY that makes the current FK foot position land on
that anchor. A small amount of contact hysteresis, slip/outlier gating, XY
correction clamping, and yaw jump gating keeps noisy contact guesses from
instantly moving the odometry frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Optional, Sequence

import numpy as np


def yaw_from_quat_wxyz(q: Sequence[float]) -> float:
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_wxyz_from_yaw(yaw: float) -> np.ndarray:
    return np.asarray(
        [math.cos(0.5 * float(yaw)), 0.0, 0.0, math.sin(0.5 * float(yaw))],
        dtype=np.float32,
    )


def wrap_angle(a: float) -> float:
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def rotate_yaw_xy(yaw: float, xy: Sequence[float]) -> np.ndarray:
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    x, y = float(xy[0]), float(xy[1])
    return np.asarray([c * x - s * y, s * x + c * y], dtype=np.float32)


def _clamp_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    max_norm = float(max_norm)
    if max_norm <= 0.0 or not math.isfinite(max_norm):
        return v
    n = float(np.linalg.norm(v))
    if n <= max_norm or n <= 1e-8:
        return v
    return (v * (max_norm / n)).astype(np.float32)


@dataclass
class BaseStateEstimate:
    """Estimated map-frame base state."""

    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    contacts: dict[str, bool]
    num_contact_feet: int
    initialized: bool
    used_measurement_xy: bool = False
    yaw: float = 0.0
    contact_confidence: dict[str, float] = field(default_factory=dict)
    slip_feet: dict[str, bool] = field(default_factory=dict)


@dataclass
class FootOdometryBaseEstimator:
    """Strict-signal legged odometry estimator for planar base XY and yaw.

    This remains local odometry, not global localization. The added filters only
    make bad contact/anchor events less destructive; an external map-frame
    localization source is still needed for long-horizon absolute XY/yaw.
    """

    default_height: float = 0.72
    foot_weights: Mapping[str, float] = field(default_factory=lambda: {"left": 1.0, "right": 1.0})
    contact_enter_steps: int = 2
    contact_exit_steps: int = 2
    xy_correction_alpha: float = 0.75
    max_xy_correction_per_update: float = 0.08
    max_anchor_residual: float = 0.18
    reanchor_on_slip: bool = True
    yaw_correction_alpha: float = 1.0
    max_yaw_rate: float = 3.5

    initialized: bool = False
    base_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    base_z: float = 0.72
    root_quat_wxyz: np.ndarray = field(
        default_factory=lambda: np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    )
    foot_anchors_xy: dict[str, np.ndarray] = field(default_factory=dict)
    prev_contacts: dict[str, bool] = field(default_factory=dict)
    contact_hits: dict[str, int] = field(default_factory=dict)
    contact_misses: dict[str, int] = field(default_factory=dict)
    filtered_yaw: float = 0.0

    def reset(
        self,
        initial_xy: Sequence[float] | None = None,
        initial_quat_wxyz: Sequence[float] | None = None,
        initial_z: float | None = None,
    ) -> None:
        self.initialized = False
        self.foot_anchors_xy.clear()
        self.prev_contacts.clear()
        self.contact_hits.clear()
        self.contact_misses.clear()
        if initial_xy is not None:
            self.base_xy = np.asarray(initial_xy, dtype=np.float32).reshape(2)
        if initial_quat_wxyz is not None:
            self.root_quat_wxyz = np.asarray(initial_quat_wxyz, dtype=np.float32).reshape(4)
            self.filtered_yaw = yaw_from_quat_wxyz(self.root_quat_wxyz)
        self.base_z = float(self.default_height if initial_z is None else initial_z)

    def _filtered_contact_state(self, contacts: Mapping[str, bool]) -> dict[str, bool]:
        names = set(self.prev_contacts) | set(contacts)
        state: dict[str, bool] = {}
        enter_steps = max(1, int(self.contact_enter_steps))
        exit_steps = max(1, int(self.contact_exit_steps))
        for name in names:
            raw = bool(contacts.get(name, False))
            if raw:
                self.contact_hits[name] = self.contact_hits.get(name, 0) + 1
                self.contact_misses[name] = 0
            else:
                self.contact_misses[name] = self.contact_misses.get(name, 0) + 1
                self.contact_hits[name] = 0

            was_active = bool(self.prev_contacts.get(name, False))
            if was_active:
                active = self.contact_misses.get(name, 0) < exit_steps
            else:
                active = self.contact_hits.get(name, 0) >= enter_steps
            state[name] = active
        return state

    def _update_yaw(self, root_quat_wxyz: Sequence[float], dt: Optional[float]) -> float:
        raw_yaw = yaw_from_quat_wxyz(root_quat_wxyz)
        if not self.initialized:
            self.filtered_yaw = raw_yaw
            return raw_yaw
        dyaw = wrap_angle(raw_yaw - self.filtered_yaw)
        if dt is not None and math.isfinite(float(dt)) and float(dt) > 0.0:
            max_step = max(0.0, float(self.max_yaw_rate)) * float(dt)
            dyaw = float(np.clip(dyaw, -max_step, max_step))
        alpha = float(np.clip(self.yaw_correction_alpha, 0.0, 1.0))
        self.filtered_yaw = wrap_angle(self.filtered_yaw + alpha * dyaw)
        return self.filtered_yaw

    def update(
        self,
        root_quat_wxyz: Sequence[float],
        foot_pos_base: Mapping[str, Sequence[float]],
        contacts: Mapping[str, bool],
        base_z: Optional[float] = None,
        initial_xy: Optional[Sequence[float]] = None,
        dt: Optional[float] = None,
    ) -> BaseStateEstimate:
        """Advance the estimate by one sensor sample.

        Args:
            root_quat_wxyz: Map-frame base orientation from the IMU/state estimator.
            foot_pos_base: FK foot positions relative to the base frame.
            contacts: Raw binary stance flags keyed like ``{"left": bool, "right": bool}``.
            base_z: Optional root height estimate.
            initial_xy: Optional one-shot map origin for initialization.
            dt: Optional sample period used for yaw jump gating.
        """
        self.root_quat_wxyz = np.asarray(root_quat_wxyz, dtype=np.float32).reshape(4)
        self.base_z = float(self.default_height if base_z is None else base_z)
        yaw = self._update_yaw(self.root_quat_wxyz, dt)
        self.root_quat_wxyz = quat_wxyz_from_yaw(yaw)

        used_measurement_xy = False
        if not self.initialized:
            if initial_xy is not None:
                self.base_xy = np.asarray(initial_xy, dtype=np.float32).reshape(2)
                used_measurement_xy = True
            self.initialized = True

        contact_dict = self._filtered_contact_state(contacts)

        # Continuing stance feet constrain the base position through no-slip
        # kinematics. New stance feet become map anchors after this correction,
        # so a double-support transition transfers the map anchor without adding
        # a one-step lag from the swing foot.
        candidates: list[np.ndarray] = []
        weights: list[float] = []
        contact_confidence: dict[str, float] = {name: 0.0 for name in contact_dict}
        slip_feet: dict[str, bool] = {name: False for name in contact_dict}
        for name, in_contact in contact_dict.items():
            if name not in foot_pos_base:
                continue
            foot_xy_base = np.asarray(foot_pos_base[name], dtype=np.float32).reshape(-1)[:2]
            if in_contact and self.prev_contacts.get(name, False) and name in self.foot_anchors_xy:
                candidate = self.foot_anchors_xy[name] - rotate_yaw_xy(yaw, foot_xy_base)
                residual = float(np.linalg.norm(candidate - self.base_xy))
                if residual <= self.max_anchor_residual:
                    confidence = 1.0 - residual / max(float(self.max_anchor_residual), 1e-6)
                    contact_confidence[name] = float(np.clip(confidence, 0.05, 1.0))
                    candidates.append(candidate.astype(np.float32))
                    weights.append(float(self.foot_weights.get(name, 1.0)) * contact_confidence[name])
                else:
                    slip_feet[name] = True
                    contact_confidence[name] = 0.0
                    if self.reanchor_on_slip:
                        self.foot_anchors_xy[name] = (
                            self.base_xy + rotate_yaw_xy(yaw, foot_xy_base)
                        ).astype(np.float32)

        if candidates:
            w = np.asarray(weights, dtype=np.float32)
            w = w / max(float(w.sum()), 1e-6)
            stacked = np.stack(candidates, axis=0)
            measured_xy = (stacked * w[:, None]).sum(axis=0).astype(np.float32)
            delta_xy = measured_xy - self.base_xy
            delta_xy = _clamp_norm(delta_xy, self.max_xy_correction_per_update)
            alpha = float(np.clip(self.xy_correction_alpha, 0.0, 1.0))
            self.base_xy = (self.base_xy + alpha * delta_xy).astype(np.float32)

        for name, in_contact in contact_dict.items():
            if name not in foot_pos_base:
                continue
            foot_xy_base = np.asarray(foot_pos_base[name], dtype=np.float32).reshape(-1)[:2]
            if in_contact and not self.prev_contacts.get(name, False):
                self.foot_anchors_xy[name] = (
                    self.base_xy + rotate_yaw_xy(yaw, foot_xy_base)
                ).astype(np.float32)
                contact_confidence[name] = max(contact_confidence.get(name, 0.0), 0.5)
            if not in_contact:
                self.foot_anchors_xy.pop(name, None)

        self.prev_contacts = contact_dict
        root_pos = np.asarray([self.base_xy[0], self.base_xy[1], self.base_z], dtype=np.float32)
        return BaseStateEstimate(
            root_pos=root_pos,
            root_quat_wxyz=self.root_quat_wxyz.copy(),
            contacts=contact_dict,
            num_contact_feet=sum(1 for value in contact_dict.values() if value),
            initialized=self.initialized,
            used_measurement_xy=used_measurement_xy,
            yaw=float(yaw),
            contact_confidence=contact_confidence,
            slip_feet=slip_feet,
        )
