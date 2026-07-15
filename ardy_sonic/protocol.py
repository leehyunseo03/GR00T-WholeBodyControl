#!/usr/bin/env python3
"""Shared plan request/response protocol for the Ardy <-> GEAR-SONIC bridge.

Ardy runs on the *host* (``conda activate ardy``) and GEAR-SONIC runs inside the
``gear-sonic-base`` container (``/workspace/isaaclab/isaaclab.sh -p``). The two
processes cannot share a Python interpreter, but they *do* share a filesystem:
the host path ``/home/hslee/IsaacLab_ws`` is bind-mounted to ``/workspace`` in the
container. This module defines a tiny file-based request/response protocol over a
shared ``runtime/`` directory plus the SE(2) transform that places a canonical
Ardy plan into the robot's current world pose.

Only depends on numpy so it imports cleanly in *both* the Ardy conda env and the
Isaac Lab container python.

Request  (JSON, written atomically to ``runtime/requests/<id>.json``):
    id                : str
    start_xy          : [x, y]      robot current env-local root xy (m)
    heading           : float       direction to walk (rad), = atan2(goal-cur)
    distance          : float       forward distance for this plan (m)
    duration          : float       plan duration (s)
    target_joint_qpos : [29] | null terminal 29-DOF pose (null -> zeros)
    reach_target_pose : bool         land exactly on target pose at the end
    seed              : int
    prompt            : str
    cfg_weight        : [text, constraint]
    target_height     : float
    goal_xy           : [x, y] | null world goal root xy; required when anchor == "goal"
    anchor            : "start" | "goal"
                        "start": plan frame 0 sits at start_xy (continuous with the
                                 robot; endpoint inherits Ardy's small canonical
                                 start offset).
                        "goal" : plan LAST frame sits exactly at goal_xy (used for
                                 settle plans so the reference terminal root xy --
                                 and with reach_target_pose the terminal heading +
                                 29-DOF pose -- is exact).

Response (NPZ, written atomically to ``runtime/responses/<id>.npz``):
    qpos          : (T, 36) float32  world-frame MuJoCo qpos
                    [root xyz(env-local), root quat wxyz, 29 joints (MuJoCo order)]
    fps           : float
    ok            : bool
    error         : str
    final_root_xy : (2,) float32
    id            : str
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# ARDY G1 qpos column layout (matches GEAR-SONIC MuJoCo qpos exactly).
QPOS_DIM = 36
NUM_JOINTS = 29


# --------------------------------------------------------------------------- #
# Runtime directory layout                                                    #
# --------------------------------------------------------------------------- #
def default_runtime_dir() -> Path:
    """Shared runtime dir; overridable with ARDY_SONIC_RUNTIME.

    Defaults to ``<this package>/runtime``. Because the package lives under the
    bind-mounted workspace, the same absolute-ish location resolves on host and
    in the container (host ``.../ardy_sonic/runtime`` == container
    ``/workspace/GR00T-WholeBodyControl/ardy_sonic/runtime``).
    """
    env = os.environ.get("ARDY_SONIC_RUNTIME")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent / "runtime"


class RuntimePaths:
    def __init__(self, runtime_dir: Optional[Path | str] = None):
        self.root = Path(runtime_dir) if runtime_dir is not None else default_runtime_dir()
        self.requests = self.root / "requests"
        self.responses = self.root / "responses"
        self.plans = self.root / "plans"  # debug copies of world-frame plans

    def ensure(self) -> "RuntimePaths":
        for d in (self.requests, self.responses, self.plans):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def request_path(self, req_id: str) -> Path:
        return self.requests / f"{req_id}.json"

    def response_path(self, req_id: str) -> Path:
        return self.responses / f"{req_id}.npz"


# --------------------------------------------------------------------------- #
# Request / response dataclasses                                              #
# --------------------------------------------------------------------------- #
@dataclass
class PlanRequest:
    id: str
    start_xy: Sequence[float]
    heading: float
    distance: float
    duration: float
    target_joint_qpos: Optional[Sequence[float]] = None
    reach_target_pose: bool = True
    seed: int = 0
    prompt: str = "A person walks forward at a steady natural pace and comes to a stop."
    cfg_weight: Sequence[float] = field(default_factory=lambda: [2.0, 3.0])
    target_height: float = 0.72
    goal_xy: Optional[Sequence[float]] = None
    anchor: str = "start"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(text: str) -> "PlanRequest":
        d = json.loads(text)
        return PlanRequest(**d)


# --------------------------------------------------------------------------- #
# Atomic file IO (write to a temp name, then rename == atomic on POSIX)        #
# --------------------------------------------------------------------------- #
def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_write_npz(path: Path, **arrays) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    np.savez(tmp, **arrays)
    # np.savez appends .npz to a path without that suffix; normalize.
    tmp_written = tmp if tmp.exists() else tmp.with_name(tmp.name + ".npz")
    os.replace(tmp_written, path)


def write_response(
    paths: RuntimePaths,
    req_id: str,
    qpos: Optional[np.ndarray],
    fps: float,
    ok: bool,
    error: str = "",
    final_root_xy: Optional[Sequence[float]] = None,
) -> Path:
    if qpos is None:
        qpos = np.zeros((0, QPOS_DIM), dtype=np.float32)
    if final_root_xy is None:
        final_root_xy = qpos[-1, :2] if qpos.shape[0] else np.zeros(2, dtype=np.float32)
    out = paths.response_path(req_id)
    atomic_write_npz(
        out,
        qpos=np.asarray(qpos, dtype=np.float32),
        fps=np.asarray(float(fps), dtype=np.float32),
        ok=np.asarray(bool(ok)),
        error=np.asarray(str(error)),
        final_root_xy=np.asarray(final_root_xy, dtype=np.float32),
        id=np.asarray(str(req_id)),
    )
    return out


def read_response(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as d:
        return {
            "qpos": np.asarray(d["qpos"], dtype=np.float32),
            "fps": float(d["fps"]),
            "ok": bool(d["ok"]),
            "error": str(d["error"]),
            "final_root_xy": np.asarray(d["final_root_xy"], dtype=np.float32),
            "id": str(d["id"]),
        }


# --------------------------------------------------------------------------- #
# Geometry: yaw <-> wxyz quaternion, SE(2) transform of a qpos trajectory      #
# --------------------------------------------------------------------------- #
def yaw_from_quat_wxyz(q: Sequence[float]) -> float:
    """Extract the z-axis yaw (rad) from a wxyz quaternion."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def transform_qpos_traj_se2(
    qpos: np.ndarray,
    origin_xy: Sequence[float],
    heading: float,
    anchor_start: bool = True,
    anchor_end_xy: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Rotate a canonical Ardy qpos trajectory by ``heading`` about +z and place it.

    Ardy generates in a canonical frame: the root starts near (0,0) facing +x and
    walks toward +x. This rotates the whole trajectory so +x maps to ``heading``
    and translates it so (with ``anchor_start``) the plan's first frame sits exactly
    at ``origin_xy`` -- i.e. the robot's current planar position. Root z and the 29
    joint angles are untouched; the root wxyz quaternion is pre-rotated about +z.

    ``anchor_end_xy`` (overrides ``anchor_start``): translate so the plan's LAST
    frame sits exactly at that world xy instead. After the terminal landing pins the
    canonical last frame to the exact target pose, this makes the world-frame
    reference end exactly at the goal root xy -- Ardy's small canonical start offset
    lands on the (re-plannable) start side instead of the goal side.
    """
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] < QPOS_DIM:
        raise ValueError(f"expected (T,>={QPOS_DIM}) qpos, got {qpos.shape}")
    out = qpos.copy()
    c, s = math.cos(heading), math.sin(heading)
    if anchor_end_xy is not None:
        x0, y0 = float(qpos[-1, 0]), float(qpos[-1, 1])
        ox, oy = float(anchor_end_xy[0]), float(anchor_end_xy[1])
    elif anchor_start:
        x0, y0 = float(qpos[0, 0]), float(qpos[0, 1])
        ox, oy = float(origin_xy[0]), float(origin_xy[1])
    else:
        x0, y0 = 0.0, 0.0
        ox, oy = float(origin_xy[0]), float(origin_xy[1])
    x = qpos[:, 0] - x0
    y = qpos[:, 1] - y0
    out[:, 0] = c * x - s * y + ox
    out[:, 1] = s * x + c * y + oy
    # z (col 2) unchanged.

    # Pre-multiply root quat (wxyz) by yaw rotation q_yaw = (cos(h/2),0,0,sin(h/2)).
    cw, sw = math.cos(heading * 0.5), math.sin(heading * 0.5)
    qw, qx, qy, qz = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
    out[:, 3] = cw * qw - sw * qz
    out[:, 4] = cw * qx - sw * qy
    out[:, 5] = cw * qy + sw * qx
    out[:, 6] = cw * qz + sw * qw
    return out
