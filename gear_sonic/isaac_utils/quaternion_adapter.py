"""Quaternion convention adapter for IsaacLab boundaries.

SONIC stores and computes quaternions as wxyz.  IsaacLab 3.x exposes simulator
state quaternions as xyzw, so all simulator reads/writes should pass through
this module instead of changing SONIC internals.
"""

from __future__ import annotations

import os
from typing import TypeVar

import numpy as np
import torch

QuatArray = TypeVar("QuatArray", torch.Tensor, np.ndarray)

_FALSE_VALUES = {"0", "false", "no", "off"}


def _validate_quat(q: QuatArray) -> QuatArray:
    if q.shape[-1] != 4:
        raise ValueError(f"Quaternion last dimension must be 4, got shape {tuple(q.shape)}")
    return q


def xyzw_to_wxyz(q: QuatArray) -> QuatArray:
    """Convert quaternion(s) from xyzw to wxyz order."""
    q = _validate_quat(q)
    return q[..., [3, 0, 1, 2]]


def wxyz_to_xyzw(q: QuatArray) -> QuatArray:
    """Convert quaternion(s) from wxyz to xyzw order."""
    q = _validate_quat(q)
    return q[..., [1, 2, 3, 0]]


def isaaclab_uses_xyzw() -> bool:
    """Return whether IsaacLab boundary quaternions should be treated as xyzw.

    Defaults to enabled for the IsaacLab 3.0 beta deployment.  Set
    ``GEAR_SONIC_ISAACLAB_XYZW=0`` to run against an older wxyz boundary.
    """
    value = os.environ.get("GEAR_SONIC_ISAACLAB_XYZW", "1")
    return value.strip().lower() not in _FALSE_VALUES


def isaaclab_to_wxyz(q: QuatArray) -> QuatArray:
    """Convert an IsaacLab-read quaternion to SONIC's internal wxyz order."""
    q = _validate_quat(q)
    return xyzw_to_wxyz(q) if isaaclab_uses_xyzw() else q


def wxyz_to_isaaclab(q: QuatArray) -> QuatArray:
    """Convert a SONIC internal wxyz quaternion for an IsaacLab write call."""
    q = _validate_quat(q)
    return wxyz_to_xyzw(q) if isaaclab_uses_xyzw() else q
