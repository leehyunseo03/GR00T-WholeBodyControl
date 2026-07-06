#!/usr/bin/env python3
"""Visual WebRTC metrics viewer for a MotionBricks 5 m target run.

Run through Isaac Lab so AppLauncher can start Isaac Sim/WebRTC:

    LIVESTREAM=2 /workspace/isaaclab/isaaclab.sh -p motion_sonic/motionbricks_metrics.py --livestream 2

The default inputs point at ``motion_sonic/motion/forward_5m_target``.
It displays the existing GEAR-Sonic G1 29DOF robot, the target reference frame,
the MotionBricks root path, a moving current-position marker, and live numeric
metrics/29DOF panels.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import isaaclab  # noqa: F401
except ImportError:
    print(
        "\nERROR: Isaac Lab is required. Run this with Isaac Lab's Python, for example:\n"
        "  LIVESTREAM=2 /workspace/isaaclab/isaaclab.sh -p motion_sonic/motionbricks_metrics.py --livestream 2\n",
        file=sys.stderr,
    )
    sys.exit(1)

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_RUN_DIR = REPO_ROOT / "motion_sonic" / "motion" 
DEFAULT_QPOS = DEFAULT_RUN_DIR / "qpos" / "motionbricks_to_target_forward_5m_target.npy"
DEFAULT_TARGET = DEFAULT_RUN_DIR / "target_reference" / "forward_5m_target.npz"
DEFAULT_MARKERS = (
    DEFAULT_RUN_DIR
    / "visualization"
    / "motionbricks_to_target_forward_5m_target_trajectory_markers.json"
)
DEFAULT_G1_USD = (
    REPO_ROOT
    / "gear_sonic"
    / "data"
    / "robots"
    / "g1"
    / "g1_29dof_rev_1_0"
    / "g1_29dof_rev_1_0.usd"
)

MUJOCO_DOF_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def _repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _load_pickle(path: Path) -> Any:
    try:
        import joblib

        return joblib.load(path)
    except Exception:
        with path.open("rb") as f:
            return pickle.load(f)


def _select_motion_dict(data: Any, motion_key: str | None = None) -> Any:
    if motion_key is not None:
        if not isinstance(data, dict) or motion_key not in data:
            raise KeyError(f"motion_key={motion_key!r} not found in {list(data) if isinstance(data, dict) else type(data)}")
        return data[motion_key]

    if isinstance(data, dict):
        qpos_like = {"qpos", "target_qpos", "root_trans_offset", "root_pos", "root_xyz"}
        if any(key in data for key in qpos_like):
            return data
        dict_items = [(key, value) for key, value in data.items() if isinstance(value, dict)]
        if len(dict_items) == 1:
            return dict_items[0][1]
    return data


def load_qpos_sequence(path: str | Path, qpos_key: str = "qpos", motion_key: str | None = None) -> np.ndarray:
    path = _repo_path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".npy":
        qpos = np.load(path, allow_pickle=True)
        if qpos.dtype == object and qpos.shape == ():
            qpos = _select_motion_dict(qpos.item(), motion_key=motion_key)
    elif path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        if qpos_key not in data:
            raise KeyError(f"{qpos_key!r} not found in {path}; keys={data.files}")
        qpos = data[qpos_key]
    elif path.suffix == ".pkl":
        data = _select_motion_dict(_load_pickle(path), motion_key=motion_key)
        if not isinstance(data, dict) or qpos_key not in data:
            raise KeyError(f"{qpos_key!r} not found in {path}")
        qpos = data[qpos_key]
    else:
        raise ValueError(f"Unsupported qpos extension: {path.suffix}")

    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] < 3:
        raise ValueError(f"Expected qpos shape (T, >=3), got {qpos.shape}")
    if qpos.shape[1] < 36:
        padded = np.zeros((qpos.shape[0], 36), dtype=np.float32)
        padded[:, : qpos.shape[1]] = qpos
        qpos = padded
    return qpos[:, :36]


def _qpos_from_root_fields(data: dict[str, Any], frame: int) -> np.ndarray:
    root_key = next((key for key in ("root_trans_offset", "root_pos", "root_xyz") if key in data), None)
    quat_key = next((key for key in ("root_rot", "root_quat", "root_quat_wxyz") if key in data), None)
    dof_key = next((key for key in ("dof", "joint_pos") if key in data), None)
    if root_key is None:
        raise KeyError(f"Could not find root position field in keys={list(data)}")

    root = np.asarray(data[root_key], dtype=np.float32)
    root = root[frame] if root.ndim > 1 else root

    qpos = np.zeros(36, dtype=np.float32)
    qpos[:3] = root[:3]
    qpos[3] = 1.0

    if quat_key is not None:
        quat = np.asarray(data[quat_key], dtype=np.float32)
        quat = quat[frame] if quat.ndim > 1 else quat
        if quat_key == "root_rot":
            qpos[3:7] = quat[[3, 0, 1, 2]]
        else:
            qpos[3:7] = quat[:4]
    if dof_key is not None:
        dof = np.asarray(data[dof_key], dtype=np.float32)
        dof = dof[frame] if dof.ndim > 1 else dof
        qpos[7 : 7 + min(29, dof.shape[0])] = dof[:29]
    return qpos


def load_target_qpos(
    path: str | Path,
    frame: int = -1,
    qpos_key: str = "target_qpos",
    motion_key: str | None = None,
) -> np.ndarray:
    path = _repo_path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        if qpos_key in data:
            qpos = data[qpos_key]
        elif "qpos" in data:
            qpos = data["qpos"][frame]
        else:
            raise KeyError(f"Could not find {qpos_key!r} or 'qpos' in {path}; keys={data.files}")
    elif path.suffix == ".npy":
        arr = np.load(path, allow_pickle=True)
        qpos = arr[frame] if arr.ndim > 1 else arr
    elif path.suffix == ".pkl":
        data = _select_motion_dict(_load_pickle(path), motion_key=motion_key)
        if isinstance(data, dict) and qpos_key in data:
            qpos = np.asarray(data[qpos_key], dtype=np.float32)
        elif isinstance(data, dict):
            qpos = _qpos_from_root_fields(data, frame)
        else:
            qpos = np.asarray(data, dtype=np.float32)
            qpos = qpos[frame] if qpos.ndim > 1 else qpos
    else:
        raise ValueError(f"Unsupported target extension: {path.suffix}")

    qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
    if qpos.shape[0] < 36:
        padded = np.zeros(36, dtype=np.float32)
        padded[: qpos.shape[0]] = qpos
        if padded[3:7].sum() == 0:
            padded[3] = 1.0
        qpos = padded
    return qpos[:36]


def load_marker_path(path: str | Path | None) -> np.ndarray | None:
    if path is None:
        return None
    path = _repo_path(path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    markers = payload.get("intermediate_xyz")
    if not markers:
        return None
    return np.asarray(markers, dtype=np.float32)


def strip_target_hold(qpos: np.ndarray, target_qpos: np.ndarray, eps: float) -> np.ndarray:
    """Remove repeated terminal frames that exactly hold the target pose."""
    if qpos.shape[0] <= 2:
        return qpos
    target_xyz = target_qpos[:3]
    tail_matches = np.linalg.norm(qpos[:, :3] - target_xyz[None, :], axis=1) <= eps
    if not bool(tail_matches[-1]):
        return qpos
    first_tail = qpos.shape[0] - 1
    while first_tail > 0 and tail_matches[first_tail - 1]:
        first_tail -= 1
    if first_tail == 0:
        return qpos
    return qpos[:first_tail]


def compute_metrics(qpos: np.ndarray, motion_qpos: np.ndarray, target_qpos: np.ndarray, fps: float) -> dict[str, float]:
    xy = motion_qpos[:, :2]
    target_xy = target_qpos[:2]
    dist_xy = np.linalg.norm(xy - target_xy[None, :], axis=1)
    step_delta = np.diff(motion_qpos[:, :3], axis=0)
    path_length = float(np.linalg.norm(step_delta, axis=1).sum()) if len(motion_qpos) > 1 else 0.0
    displacement = float(np.linalg.norm(motion_qpos[-1, :2] - motion_qpos[0, :2]))
    duration = float(max(len(motion_qpos) - 1, 0) / max(fps, 1e-6))
    speed = path_length / duration if duration > 1e-6 else 0.0
    closest_idx = int(dist_xy.argmin()) if len(dist_xy) else 0
    return {
        "frames_total": float(len(qpos)),
        "frames_motion": float(len(motion_qpos)),
        "duration_sec": duration,
        "path_length_m": path_length,
        "xy_displacement_m": displacement,
        "mean_speed_mps": speed,
        "endpoint_xy_error_m": float(dist_xy[-1]) if len(dist_xy) else 0.0,
        "closest_xy_error_m": float(dist_xy[closest_idx]) if len(dist_xy) else 0.0,
        "closest_frame": float(closest_idx),
        "target_x": float(target_qpos[0]),
        "target_y": float(target_qpos[1]),
        "target_z": float(target_qpos[2]),
    }


def _append_kit_arg(args: argparse.Namespace, setting: str) -> None:
    kit_args = getattr(args, "kit_args", "") or ""
    if setting not in kit_args:
        args.kit_args = f"{kit_args} {setting}".strip()


def configure_livestream_defaults(args: argparse.Namespace) -> None:
    livestream = getattr(args, "livestream", -1)
    if livestream < 0:
        livestream = int(os.environ.get("LIVESTREAM", "2"))
        args.livestream = livestream
    os.environ.setdefault("LIVESTREAM", str(livestream))

    if livestream > 0:
        if hasattr(args, "visualizer") and not getattr(args, "visualizer"):
            args.visualizer = ["kit"]
        if hasattr(args, "headless"):
            args.headless = True
        _append_kit_arg(args, "--/exts/omni.kit.livestream.app/primaryStream/allowDynamicResize=true")


def create_g1_scene(args: argparse.Namespace):
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation

    from gear_sonic.envs.manager_env.robots.g1 import (
        G1_CYLINDER_MODEL_12_DEX_CFG,
        G1_MUJOCO_TO_ISAACLAB_DOF,
    )

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / float(args.sim_fps), device=args.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    ground_cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        )
    )
    ground_cfg.func("/World/ground", ground_cfg)

    light_cfg = sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0)
    light_cfg.func("/World/light", light_cfg)
    sky_cfg = sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0)
    sky_cfg.func("/World/skyLight", sky_cfg)

    robot_cfg = G1_CYLINDER_MODEL_12_DEX_CFG.replace(prim_path=args.robot_prim_path)
    robot_cfg.spawn.usd_path = str(_repo_path(args.g1_usd))
    robot = Articulation(robot_cfg)

    mapping = torch.tensor(G1_MUJOCO_TO_ISAACLAB_DOF, dtype=torch.long, device=sim.device)
    return sim, robot, mapping


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat / norm


def qpos_to_isaaclab_state(qpos_frame: np.ndarray, dof_mapping, device: str):
    import torch

    from gear_sonic.isaac_utils.quaternion_adapter import wxyz_to_isaaclab

    root_pos = torch.tensor(qpos_frame[:3], dtype=torch.float32, device=device).view(1, 3)
    root_quat_wxyz = torch.tensor(_normalize_quat_wxyz(qpos_frame[3:7]), dtype=torch.float32, device=device).view(1, 4)
    root_quat = wxyz_to_isaaclab(root_quat_wxyz)
    root_pose = torch.cat([root_pos, root_quat], dim=-1)

    dof_mujoco = torch.tensor(qpos_frame[7:36], dtype=torch.float32, device=device)
    joint_pos = dof_mujoco[dof_mapping].view(1, -1)
    joint_vel = torch.zeros_like(joint_pos)
    root_vel = torch.zeros((1, 6), dtype=torch.float32, device=device)
    return root_pose, root_vel, joint_pos, joint_vel


def write_g1_qpos_frame(robot, qpos_frame: np.ndarray, dof_mapping, device: str) -> None:
    root_pose, root_vel, joint_pos, joint_vel = qpos_to_isaaclab_state(qpos_frame, dof_mapping, device)
    robot.write_root_pose_to_sim_index(root_pose=root_pose)
    robot.write_root_velocity_to_sim_index(root_velocity=root_vel)
    robot.write_joint_position_to_sim_index(position=joint_pos)
    robot.write_joint_velocity_to_sim_index(velocity=joint_vel)
    robot.set_joint_position_target_index(target=joint_pos)
    robot.write_data_to_sim()


def vec3(value: np.ndarray | list[float] | tuple[float, float, float]):
    from pxr import Gf

    return Gf.Vec3f(float(value[0]), float(value[1]), float(value[2]))


def make_material(stage, path: str, color: tuple[float, float, float], opacity: float = 1.0):
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(0.15 * c for c in color)))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def bind_material(prim, material) -> None:
    from pxr import UsdShade

    prim.ApplyAPI(UsdShade.MaterialBindingAPI)
    UsdShade.MaterialBindingAPI(prim).Bind(material)


def add_sphere(stage, path: str, xyz: np.ndarray, radius: float, material):
    from pxr import UsdGeom

    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(float(radius))
    xform = UsdGeom.Xformable(sphere.GetPrim())
    xform.ClearXformOpOrder()
    translate_op = xform.AddTranslateOp()
    translate_op.Set(vec3(xyz))
    bind_material(sphere.GetPrim(), material)
    return translate_op


def add_curve(stage, path: str, points: np.ndarray, width: float, material) -> None:
    from pxr import UsdGeom, Vt

    if len(points) < 2:
        return
    curve = UsdGeom.BasisCurves.Define(stage, path)
    curve.CreateTypeAttr("linear")
    curve.CreateCurveVertexCountsAttr(Vt.IntArray([len(points)]))
    curve.CreatePointsAttr(Vt.Vec3fArray([vec3(p) for p in points]))
    curve.CreateWidthsAttr(Vt.FloatArray([float(width)]))
    bind_material(curve.GetPrim(), material)


def add_axis_frame(stage, root_path: str, origin: np.ndarray, yaw: float, length: float, width: float, materials: dict[str, Any]):
    x_axis = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)
    y_axis = np.array([-math.sin(yaw), math.cos(yaw), 0.0], dtype=np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    axes = {
        "x": (x_axis, materials["red"]),
        "y": (y_axis, materials["green"]),
        "z": (z_axis, materials["blue"]),
    }
    for name, (direction, material) in axes.items():
        pts = np.stack([origin, origin + direction * length], axis=0)
        add_curve(stage, f"{root_path}/{name}_axis", pts, width, material)
        add_sphere(stage, f"{root_path}/{name}_tip", pts[-1], width * 4.0, material)


def quat_yaw_wxyz(q: np.ndarray) -> float:
    q = np.asarray(q, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        return 0.0
    w, x, y, z = q / norm
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def setup_scene(qpos: np.ndarray, motion_qpos: np.ndarray, target_qpos: np.ndarray, marker_path: np.ndarray | None, args):
    import omni.usd
    from pxr import UsdGeom

    context = omni.usd.get_context()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    materials = {
        "path": make_material(stage, "/World/Materials/path_yellow", (1.0, 0.78, 0.02), 0.9),
        "target": make_material(stage, "/World/Materials/target_blue", (0.05, 0.32, 1.0), 1.0),
        "current": make_material(stage, "/World/Materials/current_red", (1.0, 0.08, 0.03), 1.0),
        "start": make_material(stage, "/World/Materials/start_green", (0.1, 0.8, 0.28), 1.0),
        "red": make_material(stage, "/World/Materials/axis_red", (1.0, 0.05, 0.04), 1.0),
        "green": make_material(stage, "/World/Materials/axis_green", (0.0, 0.85, 0.2), 1.0),
        "blue": make_material(stage, "/World/Materials/axis_blue", (0.05, 0.32, 1.0), 1.0),
    }

    path_points = marker_path if marker_path is not None else motion_qpos[:: max(1, args.marker_stride), :3]
    all_points = np.concatenate([path_points[:, :3], target_qpos[None, :3]], axis=0)
    center = all_points.mean(axis=0)

    add_curve(stage, "/World/MotionBricks/RootPathLine", path_points[:, :3], args.path_width, materials["path"])
    for idx, xyz in enumerate(path_points[:, :3]):
        add_sphere(stage, f"/World/MotionBricks/PathMarker_{idx:04d}", xyz, args.marker_radius, materials["path"])

    add_sphere(stage, "/World/MotionBricks/Start", motion_qpos[0, :3], args.target_radius * 0.8, materials["start"])
    add_sphere(stage, "/World/MotionBricks/Target", target_qpos[:3], args.target_radius, materials["target"])
    current_op = add_sphere(stage, "/World/MotionBricks/CurrentPosition", qpos[0, :3], args.current_radius, materials["current"])

    add_axis_frame(
        stage,
        "/World/MotionBricks/TargetReferenceFrame",
        target_qpos[:3].astype(np.float32),
        quat_yaw_wxyz(target_qpos[3:7]),
        args.frame_axis_length,
        args.frame_axis_width,
        materials,
    )

    return stage, current_op, center


class MetricsPanel:
    def __init__(self, metrics: dict[str, float], target_qpos: np.ndarray, endpoint_qpos: np.ndarray):
        import omni.ui as ui

        self._ui = ui
        self.window = ui.Window("MotionBricks 5m Metrics", width=460, height=260)
        with self.window.frame:
            with ui.VStack(spacing=5):
                ui.Label("MotionBricks 5m run", style={"font_size": 18})
                self.current = ui.Label("")
                self.error = ui.Label("")
                ui.Label(
                    "target xyz: "
                    f"({target_qpos[0]:.3f}, {target_qpos[1]:.3f}, {target_qpos[2]:.3f}) m"
                )
                ui.Label(
                    "generated endpoint xyz: "
                    f"({endpoint_qpos[0]:.3f}, {endpoint_qpos[1]:.3f}, {endpoint_qpos[2]:.3f}) m"
                )
                ui.Label(f"endpoint xy error: {metrics['endpoint_xy_error_m']:.3f} m")
                ui.Label(f"closest xy error: {metrics['closest_xy_error_m']:.3f} m @ frame {int(metrics['closest_frame'])}")
                ui.Label(f"path length: {metrics['path_length_m']:.3f} m")
                ui.Label(f"mean root speed: {metrics['mean_speed_mps']:.3f} m/s")
        self.update(0, np.zeros(3, dtype=np.float32), 0.0, 0.0)

    def update(self, frame_idx: int, xyz: np.ndarray, target_error: float, progress: float) -> None:
        self.current.text = f"current frame {frame_idx:04d}: ({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}) m"
        self.error.text = f"current xy error: {target_error:.3f} m | playback: {100.0 * progress:.1f}%"


class DofPanel:
    def __init__(self):
        import omni.ui as ui

        self.window = ui.Window("G1 29DOF qpos", width=520, height=780)
        self.labels = []
        with self.window.frame:
            with ui.VStack(spacing=2):
                ui.Label("MotionBricks qpos[7:36] in MuJoCo order", style={"font_size": 16})
                for idx, name in enumerate(MUJOCO_DOF_NAMES):
                    label = ui.Label(f"{idx:02d} {name}: +0.0000 rad", style={"font_size": 11})
                    self.labels.append(label)

    def update(self, dof_mujoco: np.ndarray) -> None:
        for idx, (label, name) in enumerate(zip(self.labels, MUJOCO_DOF_NAMES, strict=True)):
            label.text = f"{idx:02d} {name}: {float(dof_mujoco[idx]):+.4f} rad"


def set_viewport_camera(center: np.ndarray, target_qpos: np.ndarray, args: argparse.Namespace) -> None:
    try:
        from omni.kit.viewport.utility import get_active_viewport_window
        from omni.kit.viewport.utility.camera_state import ViewportCameraState
        from pxr import Gf

        viewport_window = get_active_viewport_window()
        if viewport_window is None:
            return
        viewport_api = viewport_window.viewport_api
        camera_path = viewport_api.get_active_camera() or "/OmniverseKit_Persp"
        camera_state = ViewportCameraState(camera_path, viewport_api)
        target = Gf.Vec3d(float(center[0]), float(center[1]), float(max(center[2], target_qpos[2])))
        eye = Gf.Vec3d(
            float(center[0] + args.camera_back),
            float(center[1] - args.camera_side),
            float(args.camera_height),
        )
        camera_state.set_position_world(eye, False)
        camera_state.set_target_world(target, True)
    except Exception as exc:
        print(f"[WARN] Could not set viewport camera: {exc}", flush=True)


def run_viewer(args: argparse.Namespace) -> None:
    configure_livestream_defaults(args)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    qpos = load_qpos_sequence(args.qpos, qpos_key=args.qpos_key, motion_key=args.motion_key)
    target_qpos = load_target_qpos(
        args.target,
        frame=args.target_frame,
        qpos_key=args.target_qpos_key,
        motion_key=args.target_motion_key,
    )
    motion_qpos = strip_target_hold(qpos, target_qpos, args.hold_epsilon) if args.strip_target_hold else qpos
    marker_path = load_marker_path(args.marker_json)
    metrics = compute_metrics(qpos, motion_qpos, target_qpos, args.fps)

    sim, robot, dof_mapping = create_g1_scene(args)
    _, current_op, center = setup_scene(qpos, motion_qpos, target_qpos, marker_path, args)
    panel = MetricsPanel(metrics, target_qpos, motion_qpos[-1])
    dof_panel = DofPanel() if args.show_dof_panel else None
    set_viewport_camera(center, target_qpos, args)
    sim.set_camera_view(
        eye=[float(center[0] + args.camera_back), float(center[1] - args.camera_side), float(args.camera_height)],
        target=[float(center[0]), float(center[1]), float(max(center[2], target_qpos[2]))],
    )

    sim.reset()
    robot.reset()
    write_g1_qpos_frame(robot, qpos[0], dof_mapping, sim.device)
    if dof_panel is not None:
        dof_panel.update(qpos[0, 7:36])

    print("\nMotionBricks 5m metrics viewer is running.", flush=True)
    print(f"qpos={_repo_path(args.qpos)}", flush=True)
    print(f"target={_repo_path(args.target)}", flush=True)
    print(f"g1_usd={_repo_path(args.g1_usd)}", flush=True)
    print(f"target_xyz=({target_qpos[0]:.3f}, {target_qpos[1]:.3f}, {target_qpos[2]:.3f}) m", flush=True)
    print(f"endpoint_xy_error={metrics['endpoint_xy_error_m']:.4f} m", flush=True)
    print(f"closest_xy_error={metrics['closest_xy_error_m']:.4f} m at frame {int(metrics['closest_frame'])}", flush=True)
    if getattr(args, "livestream", 0) > 0:
        print("WebRTC: open http://<server-ip>:8211/streaming/client/ if your Isaac Sim container exposes that port.", flush=True)

    sim_dt = sim.get_physics_dt()
    for _ in range(max(0, args.warmup_frames)):
        sim.step()
        robot.update(sim_dt)
    print("Warmed up livestream viewport.", flush=True)

    frame_idx = 0
    render_steps = 0
    last_advance = time.time()
    frame_period = 1.0 / max(args.fps * args.playback_speed, 1e-6)

    try:
        while simulation_app.is_running():
            now = time.time()
            if now - last_advance >= frame_period:
                last_advance = now
                frame_idx += 1
                if frame_idx >= len(qpos):
                    if args.loop:
                        frame_idx = 0
                    else:
                        frame_idx = len(qpos) - 1

                xyz = qpos[frame_idx, :3]
                current_op.Set(vec3(xyz))
                write_g1_qpos_frame(robot, qpos[frame_idx], dof_mapping, sim.device)
                error = float(np.linalg.norm(xyz[:2] - target_qpos[:2]))
                progress = frame_idx / max(len(qpos) - 1, 1)
                panel.update(frame_idx, xyz, error, progress)
                if dof_panel is not None:
                    dof_panel.update(qpos[frame_idx, 7:36])

            sim.step()
            robot.update(sim_dt)
            render_steps += 1
            if args.max_render_steps > 0 and render_steps >= args.max_render_steps:
                print(f"Reached max_render_steps={args.max_render_steps}. Exiting.", flush=True)
                break
    except KeyboardInterrupt:
        print("Interrupted by user.", flush=True)
    finally:
        simulation_app.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Isaac Sim WebRTC viewer for MotionBricks 5 m trajectory metrics.")

    parser.add_argument("--qpos", type=str, default=str(DEFAULT_QPOS), help="MotionBricks qpos .npy/.npz/.pkl path.")
    parser.add_argument("--target", type=str, default=str(DEFAULT_TARGET), help="Target reference .npz/.npy/.pkl path.")
    parser.add_argument("--marker_json", type=str, default=str(DEFAULT_MARKERS), help="Optional trajectory marker JSON.")
    parser.add_argument("--g1_usd", type=str, default=str(DEFAULT_G1_USD), help="GEAR-Sonic G1 29DOF USD path.")
    parser.add_argument("--robot_prim_path", type=str, default="/World/envs/env_0/Robot")
    parser.add_argument("--qpos_key", type=str, default="qpos")
    parser.add_argument("--target_qpos_key", type=str, default="target_qpos")
    parser.add_argument("--motion_key", type=str, default=None)
    parser.add_argument("--target_motion_key", type=str, default=None)
    parser.add_argument("--target_frame", type=int, default=-1)

    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--sim_fps", type=float, default=200.0)
    parser.add_argument("--playback_speed", type=float, default=1.0)
    parser.add_argument("--loop", type=int, default=1)
    parser.add_argument("--max_render_steps", type=int, default=0, help="0 means run until Ctrl+C/window close.")
    parser.add_argument("--warmup_frames", type=int, default=30)
    parser.add_argument("--strip_target_hold", type=int, default=1)
    parser.add_argument("--hold_epsilon", type=float, default=1e-4)
    parser.add_argument("--show_dof_panel", type=int, default=1)

    parser.add_argument("--marker_stride", type=int, default=10)
    parser.add_argument("--marker_radius", type=float, default=0.045)
    parser.add_argument("--target_radius", type=float, default=0.13)
    parser.add_argument("--current_radius", type=float, default=0.11)
    parser.add_argument("--path_width", type=float, default=0.035)
    parser.add_argument("--frame_axis_length", type=float, default=0.55)
    parser.add_argument("--frame_axis_width", type=float, default=0.018)
    parser.add_argument("--camera_back", type=float, default=-1.0)
    parser.add_argument("--camera_side", type=float, default=5.8)
    parser.add_argument("--camera_height", type=float, default=4.4)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_viewer(args)


if __name__ == "__main__":
    main()
