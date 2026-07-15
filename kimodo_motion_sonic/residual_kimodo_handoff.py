#!/usr/bin/env python3
"""Generate a short Kimodo residual segment after MotionBricks arrival.

The intended handoff is:

1. MotionBricks/GEAR-Sonic walks close to the final target.
2. The current robot MuJoCo qpos is captured at the stop/settle moment.
3. This script generates a short Kimodo segment from that qpos to the final
   target qpos and exports a normal GEAR-Sonic motion_lib PKL.

Use ``--serve`` to keep the Kimodo model loaded and process JSONL requests with
minimal per-request model-startup latency.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import joblib
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


REPO_ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_WS = REPO_ROOT.parent
KIMODO_ROOT = ISAACLAB_WS / "kimodo"

for path in (REPO_ROOT, KIMODO_ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


MUJOCO_DOF_AXES = np.asarray(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class HandoffSpec:
    current_qpos: str
    target_qpos: str | None
    output_dir: str
    motion_name: str
    target_name: str
    prompt: str
    duration: float
    root_waypoints: int
    fps: int
    diffusion_steps: int
    seed: int | None
    num_transition_frames: int
    normalize_root: bool
    prepend_start_hold: int
    snap_to_target_frames: int
    append_target_hold: int
    marker_stride: int
    qpos_key: str
    target_qpos_key: str
    current_frame: int
    target_frame: int
    fallback_forward_meters: float
    target_height: float
    target_yaw: float


def resolve_path(path_text: str | Path, root: Path = REPO_ROOT) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return (root / path).resolve()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ISAACLAB_WS))
    except ValueError:
        return str(path)


def container_workspace_path(path: Path) -> str:
    try:
        return f"/workspace/{path.resolve().relative_to(ISAACLAB_WS)}"
    except ValueError:
        return str(path)


def yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    quat_xyzw = Rotation.from_euler("z", yaw).as_quat()
    return quat_xyzw[[3, 0, 1, 2]].astype(np.float32)


def quat_wxyz_to_yaw(quat_wxyz: np.ndarray) -> float:
    quat_xyzw = np.asarray(quat_wxyz, dtype=np.float64)[[1, 2, 3, 0]]
    return float(Rotation.from_quat(quat_xyzw).as_euler("xyz")[2])


def normalize_quat_wxyz(qpos: np.ndarray) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float32).copy()
    q = qpos[..., 3:7]
    qpos[..., 3:7] = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    return qpos


def qpos_to_pose_aa(qpos: np.ndarray) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float32)
    pose_aa = np.zeros((qpos.shape[0], 30, 3), dtype=np.float32)
    quat_xyzw = qpos[:, 3:7][:, [1, 2, 3, 0]]
    pose_aa[:, 0, :] = Rotation.from_quat(quat_xyzw).as_rotvec().astype(np.float32)
    pose_aa[:, 1:, :] = qpos[:, 7:36, None] * MUJOCO_DOF_AXES[None, :, :]
    return pose_aa


def make_forward_target_qpos(forward_meters: float, target_height: float, yaw: float) -> np.ndarray:
    qpos = np.zeros(36, dtype=np.float32)
    qpos[:3] = [forward_meters, 0.0, target_height]
    qpos[3:7] = yaw_to_quat_wxyz(yaw)
    return qpos


def load_qpos_any(path: Path, qpos_key: str, frame: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".csv":
        data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    elif path.suffix == ".npy":
        data = np.load(path, allow_pickle=True)
    elif path.suffix == ".npz":
        npz = np.load(path, allow_pickle=True)
        if qpos_key in npz:
            data = npz[qpos_key]
        elif len(npz.files) == 1:
            data = npz[npz.files[0]]
        else:
            raise KeyError(f"{qpos_key!r} not found in {path}; keys={npz.files}")
    elif path.suffix == ".pkl":
        payload = joblib.load(path)
        if isinstance(payload, dict) and qpos_key in payload:
            data = payload[qpos_key]
        else:
            raise KeyError(f"{qpos_key!r} not found in {path}")
    else:
        raise ValueError(f"Unsupported qpos file {path}; use .npy, .npz, .csv, or .pkl")

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        if arr.shape[0] < 36:
            raise ValueError(f"Expected at least 36 qpos values, got {arr.shape}")
        qpos = arr[:36]
    elif arr.ndim >= 2:
        if arr.shape[-1] < 36:
            raise ValueError(f"Expected qpos last dimension >=36, got {arr.shape}")
        qpos = arr.reshape(-1, arr.shape[-1])[frame, :36]
    else:
        raise ValueError(f"Cannot read qpos from shape {arr.shape}")
    return normalize_quat_wxyz(qpos)


def interpolate_qpos(start_qpos: np.ndarray, target_qpos: np.ndarray, num_frames: int) -> np.ndarray:
    num_frames = max(2, int(num_frames))
    alpha = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    qpos = (1.0 - alpha[:, None]) * start_qpos[None, :] + alpha[:, None] * target_qpos[None, :]
    rotations = Rotation.from_quat(np.stack([start_qpos[3:7][[1, 2, 3, 0]], target_qpos[3:7][[1, 2, 3, 0]]]))
    slerp = Slerp([0.0, 1.0], rotations)
    qpos[:, 3:7] = slerp(alpha).as_quat()[:, [3, 0, 1, 2]].astype(np.float32)
    return normalize_quat_wxyz(qpos)


def root_progress(qpos: np.ndarray) -> np.ndarray:
    start_xy = qpos[0, :2]
    end_xy = qpos[-1, :2]
    delta = end_xy - start_xy
    denom = float(np.dot(delta, delta))
    if denom < 1e-8:
        return np.linspace(0.0, 1.0, qpos.shape[0], dtype=np.float32)
    progress = ((qpos[:, :2] - start_xy[None, :]) @ delta) / denom
    progress = np.clip(progress.astype(np.float32), 0.0, 1.0)
    return np.maximum.accumulate(progress)


def normalize_root_to_start_end(qpos: np.ndarray, start_qpos: np.ndarray, target_qpos: np.ndarray, enabled: bool) -> np.ndarray:
    out = normalize_quat_wxyz(qpos)
    if not enabled:
        return out
    progress = root_progress(out)
    raw_start = out[0, :2].copy()
    raw_end = out[-1, :2].copy()
    raw_delta = raw_end - raw_start
    denom = float(np.dot(raw_delta, raw_delta))
    if denom < 1e-8:
        residual = np.zeros_like(out[:, :2])
    else:
        residual = out[:, :2] - (raw_start[None, :] + progress[:, None] * raw_delta[None, :])
    residual *= np.sin(np.pi * progress)[:, None] * 0.5
    target_xy = start_qpos[:2][None, :] + progress[:, None] * (target_qpos[:2] - start_qpos[:2])[None, :]
    out[:, :2] = target_xy + residual
    out[:, 2] = (1.0 - progress) * start_qpos[2] + progress * target_qpos[2]
    return out.astype(np.float32)


def append_terminal_segments(
    qpos: np.ndarray,
    start_qpos: np.ndarray,
    target_qpos: np.ndarray,
    spec: HandoffSpec,
) -> np.ndarray:
    pieces = []
    if spec.prepend_start_hold > 0:
        pieces.append(np.repeat(start_qpos[None, :], spec.prepend_start_hold, axis=0).astype(np.float32))

    body = np.asarray(qpos, dtype=np.float32).copy()
    body[0] = start_qpos
    pieces.append(body)

    if spec.snap_to_target_frames > 0:
        pieces.append(interpolate_qpos(body[-1], target_qpos, spec.snap_to_target_frames + 1)[1:])
    if spec.append_target_hold > 0:
        pieces.append(np.repeat(target_qpos[None, :], spec.append_target_hold, axis=0).astype(np.float32))
    out = np.concatenate(pieces, axis=0).astype(np.float32)
    out[0] = start_qpos
    out[-1] = target_qpos
    return normalize_quat_wxyz(out)


def write_markers(json_path: Path, xml_path: Path, root_xyz: np.ndarray, target_xyz: np.ndarray, stride: int) -> None:
    stride = max(1, int(stride))
    sampled = root_xyz[::stride]
    payload = {
        "motion_name": json_path.stem.replace("_trajectory_markers", ""),
        "coordinate_frame": "MuJoCo qpos root translation: x forward, y left, z up",
        "marker_stride": stride,
        "intermediate_xyz": sampled.astype(float).round(6).tolist(),
        "target_xyz": np.asarray(target_xyz, dtype=float).round(6).tolist(),
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    geoms = [
        f'    <geom name="hybrid_path_{idx:04d}" type="sphere" size="0.035" '
        f'pos="{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}" rgba="0.95 0.72 0.05 0.85"/>'
        for idx, xyz in enumerate(sampled)
    ]
    geoms.append(
        f'    <geom name="hybrid_target" type="sphere" size="0.07" '
        f'pos="{target_xyz[0]:.6f} {target_xyz[1]:.6f} {target_xyz[2]:.6f}" rgba="0.1 0.35 1 1"/>'
    )
    xml_path.write_text(
        '<mujoco model="kimodo motion sonic trajectory markers">\n  <worldbody>\n'
        + "\n".join(geoms)
        + "\n  </worldbody>\n</mujoco>\n",
        encoding="utf-8",
    )


def export_motion_lib(
    spec: HandoffSpec,
    qpos: np.ndarray,
    start_qpos: np.ndarray,
    target_qpos: np.ndarray,
    constraints_path: Path,
    raw_csv_path: Path,
    elapsed_sec: float,
) -> dict[str, Any]:
    output_dir = resolve_path(spec.output_dir)
    qpos_dir = output_dir / "qpos"
    robot_dir = output_dir / "robot_filtered" / "kimodo_motion_sonic"
    target_dir = output_dir / "target_reference"
    viz_dir = output_dir / "visualization"
    for directory in (qpos_dir, robot_dir, target_dir, viz_dir):
        directory.mkdir(parents=True, exist_ok=True)

    qpos_npy = qpos_dir / f"{spec.motion_name}.npy"
    qpos_npz = qpos_dir / f"{spec.motion_name}.npz"
    robot_pkl = robot_dir / f"{spec.motion_name}.pkl"
    target_npz = target_dir / f"{spec.target_name}.npz"
    markers_json = viz_dir / f"{spec.motion_name}_trajectory_markers.json"
    markers_xml = viz_dir / f"{spec.motion_name}_trajectory_markers.xml"

    np.save(qpos_npy, qpos)
    np.savez_compressed(
        qpos_npz,
        qpos=qpos,
        start_qpos=start_qpos,
        target_qpos=target_qpos,
        root_xyz=qpos[:, :3],
        dof=qpos[:, 7:36],
        fps=np.asarray(spec.fps, dtype=np.int32),
    )
    np.savez_compressed(
        target_npz,
        qpos=target_qpos[None, :],
        target_qpos=target_qpos,
        start_qpos=start_qpos,
        fps=np.asarray(spec.fps, dtype=np.int32),
    )

    entry = {
        "root_trans_offset": qpos[:, :3].astype(np.float32),
        "root_rot": qpos[:, 3:7][:, [1, 2, 3, 0]].astype(np.float32),
        "dof": qpos[:, 7:36].astype(np.float32),
        "pose_aa": qpos_to_pose_aa(qpos),
        "smpl_joints": np.zeros((qpos.shape[0], 24, 3), dtype=np.float32),
        "fps": int(spec.fps),
    }
    joblib.dump({spec.motion_name: entry}, robot_pkl, compress=True)
    write_markers(markers_json, markers_xml, qpos[:, :3], target_qpos[:3], spec.marker_stride)

    final_root_xy_error = float(np.linalg.norm(qpos[-1, :2] - target_qpos[:2]))
    final_dof_rmse = float(np.linalg.norm(qpos[-1, 7:36] - target_qpos[7:36]) / math.sqrt(29))
    start_jump_rmse = float(np.linalg.norm(qpos[0, 7:36] - start_qpos[7:36]) / math.sqrt(29))
    residual_xy_m = float(np.linalg.norm(target_qpos[:2] - start_qpos[:2]))

    manifest = {
        "created_unix": time.time(),
        "elapsed_sec": elapsed_sec,
        "mode": "motionbricks_to_kimodo_residual",
        "spec": asdict(spec),
        "residual_xy_m": residual_xy_m,
        "start_root_xyz": start_qpos[:3].round(6).tolist(),
        "target_root_xyz": target_qpos[:3].round(6).tolist(),
        "frames": int(qpos.shape[0]),
        "final_root_xy_error": final_root_xy_error,
        "final_dof_rmse": final_dof_rmse,
        "start_dof_rmse": start_jump_rmse,
        "constraints": display_path(constraints_path),
        "raw_csv": display_path(raw_csv_path),
        "robot_pkl": display_path(robot_pkl),
        "qpos_npy": display_path(qpos_npy),
        "qpos_npz": display_path(qpos_npz),
        "target_npz": display_path(target_npz),
        "trajectory_markers_json": display_path(markers_json),
        "trajectory_markers_xml": display_path(markers_xml),
        "gearsonic_motion_file": container_workspace_path(robot_pkl),
        "smpl_motion_file": "dummy",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {display_path(robot_pkl)}", flush=True)
    print(f"wrote {display_path(manifest_path)}", flush=True)
    return manifest


class ResidualKimodoHandoffGenerator:
    """Kimodo model wrapper that stays warm across residual requests."""

    def __init__(self, model_name: str, text_encoder_device: str | None, device: str | None = None) -> None:
        import torch
        from kimodo import load_model
        from kimodo.exports.mujoco import MujocoQposConverter
        from kimodo.model.registry import get_model_info

        if text_encoder_device:
            os.environ["TEXT_ENCODER_DEVICE"] = text_encoder_device
        self.torch = torch
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        started = time.time()
        print(f"loading Kimodo once on {self.device}...", flush=True)
        self.model, self.resolved_model = load_model(
            model_name,
            device=self.device,
            default_family="Kimodo",
            return_resolved_name=True,
        )
        info = get_model_info(self.resolved_model)
        display = info.display_name if info else self.resolved_model
        if "g1" not in self.resolved_model.lower():
            raise ValueError(f"Expected a G1 Kimodo model, got {self.resolved_model!r}")
        self.converter = MujocoQposConverter(self.model.skeleton)
        print(f"loaded Kimodo model: {display} ({self.resolved_model}) in {time.time() - started:.2f}s", flush=True)

    def _qpos_constraint_payload(self, qpos_seq: np.ndarray, frame_indices: list[int]) -> dict[str, Any]:
        from kimodo.geometry import matrix_to_axis_angle

        motion_dict = self.converter.qpos_to_motion_dict(qpos_seq, source_fps=float(self.model.fps))
        root_positions = motion_dict["root_positions"].detach().cpu().numpy()
        local_aa = matrix_to_axis_angle(motion_dict["local_rot_mats"]).detach().cpu().numpy()
        selected_root = root_positions[frame_indices]
        return {
            "type": "fullbody",
            "frame_indices": frame_indices,
            "root_positions": selected_root.astype(float).round(6).tolist(),
            "smooth_root_2d": selected_root[:, [0, 2]].astype(float).round(6).tolist(),
            "local_joints_rot": local_aa[frame_indices].astype(float).round(6).tolist(),
        }

    def write_constraints(self, spec: HandoffSpec, start_qpos: np.ndarray, target_qpos: np.ndarray) -> Path:
        num_frames = max(2, int(round(float(spec.duration) * spec.fps)))
        waypoint_count = max(2, int(spec.root_waypoints))
        base_qpos = interpolate_qpos(start_qpos, target_qpos, max(5, waypoint_count))
        motion_dict = self.converter.qpos_to_motion_dict(base_qpos, source_fps=float(spec.fps))
        root_positions = motion_dict["root_positions"].detach().cpu().numpy()
        root2d_full = root_positions[:, [0, 2]]

        frame_indices = np.linspace(0, num_frames - 1, waypoint_count).round().astype(int)
        source_idx = np.linspace(0, len(root2d_full) - 1, waypoint_count).round().astype(int)
        constraints = [
            {
                "type": "root2d",
                "frame_indices": frame_indices.tolist(),
                "smooth_root_2d": root2d_full[source_idx].astype(float).round(6).tolist(),
            },
            self._qpos_constraint_payload(base_qpos, [0, len(base_qpos) - 1])
            | {"frame_indices": [0, num_frames - 1]},
        ]

        path = resolve_path(spec.output_dir) / "constraints" / f"{spec.motion_name}_constraints.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(constraints, indent=2) + "\n", encoding="utf-8")
        return path

    def generate(self, spec: HandoffSpec) -> dict[str, Any]:
        from kimodo.constraints import load_constraints_lst
        from kimodo.tools import seed_everything

        started = time.time()
        current_path = resolve_path(spec.current_qpos)
        start_qpos = load_qpos_any(current_path, spec.qpos_key, spec.current_frame)
        if spec.target_qpos:
            target_qpos = load_qpos_any(resolve_path(spec.target_qpos), spec.target_qpos_key, spec.target_frame)
        else:
            target_qpos = make_forward_target_qpos(spec.fallback_forward_meters, spec.target_height, spec.target_yaw)
        constraints_path = self.write_constraints(spec, start_qpos, target_qpos)
        constraints = load_constraints_lst(str(constraints_path), self.model.skeleton)
        if spec.seed is not None:
            seed_everything(spec.seed)

        num_frames = [max(2, int(round(float(spec.duration) * self.model.fps)))]
        print(
            f"generating residual: residual_xy={np.linalg.norm(target_qpos[:2] - start_qpos[:2]):.3f}m "
            f"frames={num_frames[0]} prompt={spec.prompt!r}",
            flush=True,
        )
        output = self.model(
            [spec.prompt],
            num_frames,
            constraint_lst=constraints,
            num_denoising_steps=spec.diffusion_steps,
            num_samples=1,
            multi_prompt=True,
            num_transition_frames=spec.num_transition_frames,
            post_processing=False,
            return_numpy=True,
        )
        qpos = self.converter.dict_to_qpos(output, self.device)
        qpos = np.asarray(qpos, dtype=np.float32)
        if qpos.ndim == 3:
            if qpos.shape[0] != 1:
                raise ValueError(f"Expected one Kimodo sample, got qpos shape {qpos.shape}")
            qpos = qpos[0]
        if qpos.ndim != 2 or qpos.shape[1] != 36:
            raise ValueError(f"Expected qpos shape (T, 36), got {qpos.shape}")

        output_dir = resolve_path(spec.output_dir)
        raw_dir = output_dir / "kimodo_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_csv = raw_dir / f"{spec.motion_name}.csv"
        self.converter.save_csv(qpos, str(raw_csv))

        qpos = normalize_root_to_start_end(qpos, start_qpos, target_qpos, spec.normalize_root)
        qpos = append_terminal_segments(qpos, start_qpos, target_qpos, spec)
        return export_motion_lib(spec, qpos, start_qpos, target_qpos, constraints_path, raw_csv, time.time() - started)


def spec_from_args(args: argparse.Namespace) -> HandoffSpec:
    return HandoffSpec(
        current_qpos=args.current_qpos,
        target_qpos=args.target_qpos,
        output_dir=args.output_dir,
        motion_name=args.motion_name,
        target_name=args.target_name,
        prompt=args.prompt,
        duration=args.duration,
        root_waypoints=args.root_waypoints,
        fps=args.fps,
        diffusion_steps=args.diffusion_steps,
        seed=args.seed,
        num_transition_frames=args.num_transition_frames,
        normalize_root=bool(args.normalize_root),
        prepend_start_hold=args.prepend_start_hold,
        snap_to_target_frames=args.snap_to_target_frames,
        append_target_hold=args.append_target_hold,
        marker_stride=args.marker_stride,
        qpos_key=args.qpos_key,
        target_qpos_key=args.target_qpos_key,
        current_frame=args.current_frame,
        target_frame=args.target_frame,
        fallback_forward_meters=args.fallback_forward_meters,
        target_height=args.target_height,
        target_yaw=args.target_yaw,
    )


def serve_requests(generator: ResidualKimodoHandoffGenerator, base_spec: HandoffSpec, args: argparse.Namespace) -> None:
    request_path = resolve_path(args.request_jsonl)
    response_path = resolve_path(args.response_jsonl)
    request_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.touch(exist_ok=True)
    print(f"serving requests from {request_path}", flush=True)
    print(f"writing responses to {response_path}", flush=True)

    offset = 0
    while True:
        with request_path.open("r", encoding="utf-8") as file:
            file.seek(offset)
            lines = file.readlines()
            offset = file.tell()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            started = time.time()
            try:
                payload = json.loads(line)
                request_id = payload.pop("request_id", f"req_{int(started * 1000)}")
                spec = replace(base_spec, **payload)
                manifest = generator.generate(spec)
                response = {
                    "request_id": request_id,
                    "ok": True,
                    "elapsed_sec": time.time() - started,
                    "manifest": manifest,
                }
            except Exception as exc:  # noqa: BLE001 - server must report request failures and continue.
                response = {
                    "request_id": locals().get("request_id", None),
                    "ok": False,
                    "elapsed_sec": time.time() - started,
                    "error": repr(exc),
                }
            with response_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(response) + "\n")
                file.flush()
                os.fsync(file.fileno())
        time.sleep(args.poll_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Warm Kimodo residual handoff generator for MotionBricks arrivals.")
    parser.add_argument("--current_qpos", type=str, required=True, help="Current robot qpos at MotionBricks handoff.")
    parser.add_argument("--target_qpos", type=str, default=None, help="Final target qpos. If omitted, use fallback forward target.")
    parser.add_argument("--qpos_key", type=str, default="qpos")
    parser.add_argument("--target_qpos_key", type=str, default="target_qpos")
    parser.add_argument("--current_frame", type=int, default=-1)
    parser.add_argument("--target_frame", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, default=str(REPO_ROOT / "kimodo_motion_sonic" / "motion"))
    parser.add_argument("--motion_name", type=str, default="kimodo_motion_sonic_residual")
    parser.add_argument("--target_name", type=str, default="kimodo_motion_sonic_target")
    parser.add_argument("--prompt", type=str, default="A humanoid robot makes a short careful step and settles into a stable final pose.")
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--root_waypoints", type=int, default=5)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--diffusion_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_transition_frames", type=int, default=5)
    parser.add_argument("--normalize_root", type=int, default=1)
    parser.add_argument("--prepend_start_hold", type=int, default=8)
    parser.add_argument("--snap_to_target_frames", type=int, default=20)
    parser.add_argument("--append_target_hold", type=int, default=60)
    parser.add_argument("--marker_stride", type=int, default=5)
    parser.add_argument("--fallback_forward_meters", type=float, default=5.0)
    parser.add_argument("--target_height", type=float, default=0.78)
    parser.add_argument("--target_yaw", type=float, default=0.0)
    parser.add_argument("--model", type=str, default="Kimodo-G1-RP-v1")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--text_encoder_device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--serve", action="store_true", help="Keep model loaded and process JSONL requests.")
    parser.add_argument("--request_jsonl", type=str, default="/tmp/kimodo_motion_sonic_requests.jsonl")
    parser.add_argument("--response_jsonl", type=str, default="/tmp/kimodo_motion_sonic_responses.jsonl")
    parser.add_argument("--poll_seconds", type=float, default=0.25)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generator = ResidualKimodoHandoffGenerator(args.model, args.text_encoder_device, args.device)
    base_spec = spec_from_args(args)
    if args.serve:
        serve_requests(generator, base_spec, args)
    else:
        generator.generate(base_spec)


if __name__ == "__main__":
    main()
