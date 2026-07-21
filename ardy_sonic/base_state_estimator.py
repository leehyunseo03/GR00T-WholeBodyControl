"""Lightweight base pose estimator for real-robot-style ARDY replanning.

The estimator keeps a map-frame base pose using the same ingredients a deploy
adapter should provide on a G1:

* IMU orientation, expressed as a map-frame base quaternion.
* Joint-encoder FK foot positions in the base frame.
* Binary foot contact.

It intentionally does not integrate IMU acceleration. While one or both feet are
in contact, it treats the stance foot as a fixed map-frame anchor and solves the
base XY that makes the current FK foot position land on that anchor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Optional, Sequence

import numpy as np


def yaw_from_quat_wxyz(q: Sequence[float]) -> float:
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rotate_yaw_xy(yaw: float, xy: Sequence[float]) -> np.ndarray:
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    x, y = float(xy[0]), float(xy[1])
    return np.asarray([c * x - s * y, s * x + c * y], dtype=np.float32)


@dataclass
class BaseStateEstimate:
    """Estimated map-frame base state."""

    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    contacts: dict[str, bool]
    num_contact_feet: int
    initialized: bool
    used_measurement_xy: bool = False


@dataclass
class FootOdometryBaseEstimator:
    """No-slip foot odometry estimator for planar base XY."""

    default_height: float = 0.72
    foot_weights: Mapping[str, float] = field(default_factory=lambda: {"left": 1.0, "right": 1.0})

    initialized: bool = False
    base_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    base_z: float = 0.72
    root_quat_wxyz: np.ndarray = field(
        default_factory=lambda: np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    )
    foot_anchors_xy: dict[str, np.ndarray] = field(default_factory=dict)
    prev_contacts: dict[str, bool] = field(default_factory=dict)

    def reset(
        self,
        initial_xy: Sequence[float] | None = None,
        initial_quat_wxyz: Sequence[float] | None = None,
        initial_z: float | None = None,
    ) -> None:
        self.initialized = False
        self.foot_anchors_xy.clear()
        self.prev_contacts.clear()
        if initial_xy is not None:
            self.base_xy = np.asarray(initial_xy, dtype=np.float32).reshape(2)
        if initial_quat_wxyz is not None:
            self.root_quat_wxyz = np.asarray(initial_quat_wxyz, dtype=np.float32).reshape(4)
        self.base_z = float(self.default_height if initial_z is None else initial_z)

    def update(
        self,
        root_quat_wxyz: Sequence[float],
        foot_pos_base: Mapping[str, Sequence[float]],
        contacts: Mapping[str, bool],
        base_z: Optional[float] = None,
        initial_xy: Optional[Sequence[float]] = None,
    ) -> BaseStateEstimate:
        """Advance the estimate by one sensor sample.

        Args:
            root_quat_wxyz: Map-frame base orientation from the IMU/state estimator.
            foot_pos_base: FK foot positions relative to the base frame.
            contacts: Binary stance flags keyed like ``{"left": bool, "right": bool}``.
            base_z: Optional root height estimate.
            initial_xy: Optional one-shot map origin for initialization.
        """
        self.root_quat_wxyz = np.asarray(root_quat_wxyz, dtype=np.float32).reshape(4)
        self.base_z = float(self.default_height if base_z is None else base_z)

        used_measurement_xy = False
        if not self.initialized:
            if initial_xy is not None:
                self.base_xy = np.asarray(initial_xy, dtype=np.float32).reshape(2)
                used_measurement_xy = True
            self.initialized = True

        yaw = yaw_from_quat_wxyz(self.root_quat_wxyz)
        contact_dict = {name: bool(value) for name, value in contacts.items()}

        # Continuing stance feet constrain the base position through no-slip
        # kinematics. New stance feet become map anchors after this correction,
        # so a double-support transition transfers the map anchor without adding
        # a one-step lag from the swing foot.
        candidates: list[np.ndarray] = []
        weights: list[float] = []
        for name, in_contact in contact_dict.items():
            if name not in foot_pos_base:
                continue
            foot_xy_base = np.asarray(foot_pos_base[name], dtype=np.float32).reshape(-1)[:2]
            if in_contact and self.prev_contacts.get(name, False) and name in self.foot_anchors_xy:
                candidates.append(self.foot_anchors_xy[name] - rotate_yaw_xy(yaw, foot_xy_base))
                weights.append(float(self.foot_weights.get(name, 1.0)))

        if candidates:
            w = np.asarray(weights, dtype=np.float32)
            w = w / max(float(w.sum()), 1e-6)
            stacked = np.stack(candidates, axis=0)
            self.base_xy = (stacked * w[:, None]).sum(axis=0).astype(np.float32)

        for name, in_contact in contact_dict.items():
            if name not in foot_pos_base:
                continue
            foot_xy_base = np.asarray(foot_pos_base[name], dtype=np.float32).reshape(-1)[:2]
            if in_contact and not self.prev_contacts.get(name, False):
                self.foot_anchors_xy[name] = (
                    self.base_xy + rotate_yaw_xy(yaw, foot_xy_base)
                ).astype(np.float32)
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
        )
