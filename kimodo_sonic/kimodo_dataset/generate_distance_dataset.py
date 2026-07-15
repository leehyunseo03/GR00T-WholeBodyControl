#!/usr/bin/env python3
"""Build a distance-randomized Kimodo-to-GEAR-Sonic motion dataset.

Each sample calls ``kimodo_sonic/generate_to_target_motion_lib.py`` with a
different forward distance, then stores the usual GEAR-Sonic motion-lib layout
under this directory:

  kimodo_dataset/datasets/<dataset_name>/
    samples/<split>/<sample_id>/
      qpos/
      target_reference/
      robot_filtered/kimodo_target/
      visualization/
      manifest.json
    motion_lib/{all,train,val}.pkl
    {index,train,val}.jsonl
    dataset_config.json

The combined PKLs are joblib dictionaries whose keys are the generated motion
names. They can be passed to GEAR-Sonic as ``motion_lib_cfg.motion_file``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import joblib
import numpy as np


KIMODO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
GENERATOR = KIMODO_ROOT / "kimodo_sonic" / "generate_to_target_motion_lib.py"
DEFAULT_DATASETS_ROOT = SCRIPT_ROOT / "datasets"

if str(KIMODO_ROOT) not in sys.path:
    sys.path.insert(0, str(KIMODO_ROOT))


@dataclass(frozen=True)
class SampleSpec:
    index: int
    split: str
    sample_id: str
    motion_name: str
    target_name: str
    output_dir: str
    forward_meters: float
    duration: float
    root_waypoints: int
    kimodo_seed: int
    speed_mps: float
    bin_min: float | None
    bin_max: float | None


def parse_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(values) < 2:
        raise argparse.ArgumentTypeError("Expected at least two comma-separated values.")
    if any(b <= a for a, b in zip(values, values[1:])):
        raise argparse.ArgumentTypeError("Values must be strictly increasing.")
    return values


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def write_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, sort_keys=True) + "\n")
        file.flush()
        os.fsync(file.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compact_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the last status per sample id while preserving sample order."""
    by_sample: dict[str, dict[str, Any]] = {}
    for record in records:
        by_sample[record["sample_id"]] = record
    return sorted(by_sample.values(), key=lambda row: int(row["index"]))


def sample_distance(
    rng: random.Random,
    args: argparse.Namespace,
    index: int,
) -> tuple[float, float | None, float | None]:
    if args.sampling == "uniform":
        return rng.uniform(args.min_distance, args.max_distance), None, None

    bins = args.distance_bins
    bin_index = index % (len(bins) - 1)
    low = bins[bin_index]
    high = bins[bin_index + 1]
    return rng.uniform(low, high), low, high


def split_for_index(index: int, val_count: int, val_indices: set[int]) -> str:
    if val_count <= 0:
        return "train"
    return "val" if index in val_indices else "train"


def make_sample_specs(args: argparse.Namespace, dataset_dir: Path) -> list[SampleSpec]:
    rng = random.Random(args.seed)
    val_count = int(round(args.num_samples * args.val_fraction))
    val_count = max(0, min(args.num_samples, val_count))
    shuffled_indices = list(range(args.num_samples))
    rng.shuffle(shuffled_indices)
    val_indices = set(shuffled_indices[:val_count])

    specs: list[SampleSpec] = []
    for index in range(args.num_samples):
        distance, bin_min, bin_max = sample_distance(rng, args, index)
        speed = rng.uniform(args.speed_min, args.speed_max)
        duration = max(args.min_duration, min(args.max_duration, distance / max(speed, 1e-6)))
        root_waypoints = int(round(duration * args.waypoint_hz)) + 1
        root_waypoints = max(args.min_root_waypoints, min(args.max_root_waypoints, root_waypoints))

        split = split_for_index(index, val_count, val_indices)
        kimodo_seed = args.kimodo_seed_base + index
        distance_cm = int(round(distance * 100.0))
        sample_id = f"{index:05d}_{distance_cm:04d}cm_seed{kimodo_seed}"
        motion_name = f"{args.motion_prefix}_{sample_id}"
        target_name = f"{args.target_prefix}_{sample_id}"
        output_dir = dataset_dir / "samples" / split / sample_id

        specs.append(
            SampleSpec(
                index=index,
                split=split,
                sample_id=sample_id,
                motion_name=motion_name,
                target_name=target_name,
                output_dir=str(output_dir),
                forward_meters=round(distance, 6),
                duration=round(duration, 6),
                root_waypoints=root_waypoints,
                kimodo_seed=kimodo_seed,
                speed_mps=round(speed, 6),
                bin_min=bin_min,
                bin_max=bin_max,
            )
        )

    return specs


def build_generator_command(spec: SampleSpec, args: argparse.Namespace) -> list[str]:
    prompt = args.prompt_template.format(distance=spec.forward_meters)
    cmd = [
        sys.executable,
        str(GENERATOR),
        "--output_dir",
        spec.output_dir,
        "--motion_name",
        spec.motion_name,
        "--forward_target_name",
        spec.target_name,
        "--forward_meters",
        f"{spec.forward_meters:.6f}",
        "--duration",
        f"{spec.duration:.6f}",
        "--root_waypoints",
        str(spec.root_waypoints),
        "--fps",
        str(args.fps),
        "--target_height",
        f"{args.target_height:.6f}",
        "--prompt",
        prompt,
        "--model",
        args.model,
        "--diffusion_steps",
        str(args.diffusion_steps),
        "--seed",
        str(spec.kimodo_seed),
        "--normalize_root_to_target",
        str(int(args.normalize_root_to_target)),
        "--snap_to_target_frames",
        str(args.snap_to_target_frames),
        "--append_target_hold",
        str(args.append_target_hold),
        "--marker_stride",
        str(args.marker_stride),
    ]
    if args.text_encoder_device:
        cmd.extend(["--text_encoder_device", args.text_encoder_device])
    return cmd


class PersistentKimodoGenerator:
    """Generate samples without reloading the Kimodo model for every clip."""

    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from kimodo import load_model
        from kimodo.exports.mujoco import MujocoQposConverter
        from kimodo.model.registry import get_model_info

        self.args = args
        self.torch = torch
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if args.text_encoder_device:
            os.environ["TEXT_ENCODER_DEVICE"] = args.text_encoder_device
        print(f"Using device: {self.device}", flush=True)
        self.model, self.resolved_model = load_model(
            args.model,
            device=self.device,
            default_family="Kimodo",
            return_resolved_name=True,
        )
        info = get_model_info(self.resolved_model)
        display = info.display_name if info else self.resolved_model
        print(f"Loaded Kimodo model once: {display} ({self.resolved_model})", flush=True)
        if "g1" not in self.resolved_model:
            raise ValueError(f"Expected a G1 Kimodo model, got {self.resolved_model!r}")
        self.converter = MujocoQposConverter(self.model.skeleton)

    def generate_qpos(self, spec: SampleSpec) -> tuple[np.ndarray, Path]:
        from kimodo.constraints import load_constraints_lst
        from kimodo.tools import seed_everything
        from kimodo_sonic import generate_to_target_motion_lib as exporter

        work_dir = Path(spec.output_dir)
        prompt = self.args.prompt_template.format(distance=spec.forward_meters)
        constraints_path = exporter.write_constraints(
            argparse.Namespace(
                duration=spec.duration,
                fps=self.args.fps,
                root_waypoints=spec.root_waypoints,
                forward_meters=spec.forward_meters,
                target_height=self.args.target_height,
            ),
            work_dir / "constraints" / f"{spec.motion_name}_constraints.json",
        )
        constraint_lst = load_constraints_lst(str(constraints_path), self.model.skeleton)
        seed_everything(spec.kimodo_seed)
        num_frames = [max(2, int(float(spec.duration) * self.model.fps))]
        output = self.model(
            [prompt],
            num_frames,
            constraint_lst=constraint_lst,
            num_denoising_steps=self.args.diffusion_steps,
            num_samples=1,
            multi_prompt=True,
            num_transition_frames=self.args.num_transition_frames,
            post_processing=False,
            return_numpy=True,
        )
        qpos = self.converter.dict_to_qpos(output, self.device)
        qpos = np.asarray(qpos, dtype=np.float32)
        if qpos.ndim == 3:
            if qpos.shape[0] != 1:
                raise ValueError(f"Expected one generated qpos sample, got shape {qpos.shape}")
            qpos = qpos[0]
        if qpos.ndim != 2 or qpos.shape[1] != 36:
            raise ValueError(f"Expected generated qpos shape (T, 36), got {qpos.shape}")

        raw_dir = work_dir / "kimodo_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        csv_path = raw_dir / f"{spec.motion_name}.csv"
        self.converter.save_csv(qpos, str(csv_path))
        return qpos, csv_path


def export_persistent_sample(
    generator: PersistentKimodoGenerator,
    spec: SampleSpec,
    args: argparse.Namespace,
) -> None:
    from kimodo_sonic import generate_to_target_motion_lib as exporter

    qpos, source_path = generator.generate_qpos(spec)
    target_qpos = exporter.make_target_qpos(spec.forward_meters, args.target_height)
    qpos = exporter.normalize_root_to_target(qpos, target_qpos, bool(args.normalize_root_to_target))
    qpos = exporter.append_target_settle(
        qpos,
        target_qpos,
        argparse.Namespace(
            snap_to_target_frames=args.snap_to_target_frames,
            append_target_hold=args.append_target_hold,
        ),
    )
    exporter.export_outputs(
        argparse.Namespace(
            output_dir=spec.output_dir,
            motion_name=spec.motion_name,
            forward_target_name=spec.target_name,
            forward_meters=spec.forward_meters,
            target_height=args.target_height,
            fps=args.fps,
            append_target_hold=args.append_target_hold,
            snap_to_target_frames=args.snap_to_target_frames,
            marker_stride=args.marker_stride,
        ),
        qpos,
        target_qpos,
        source_path,
    )


def sample_manifest_path(spec: SampleSpec) -> Path:
    return Path(spec.output_dir) / "manifest.json"


def generated_robot_pkl_path(spec: SampleSpec) -> Path:
    return Path(spec.output_dir) / "robot_filtered" / "kimodo_target" / f"{spec.motion_name}.pkl"


def run_generation(
    specs: list[SampleSpec],
    args: argparse.Namespace,
    dataset_dir: Path,
) -> list[dict[str, Any]]:
    progress_path = dataset_dir / "progress.jsonl"
    previous_records = compact_records(read_jsonl(progress_path))
    previous_by_sample = {record["sample_id"]: record for record in previous_records}
    records: list[dict[str, Any]] = []
    total = len(specs)
    persistent_generator = None
    for offset, spec in enumerate(specs, start=1):
        cmd = build_generator_command(spec, args)
        manifest_path = sample_manifest_path(spec)
        record = asdict(spec)
        record["command"] = cmd
        record["manifest"] = str(manifest_path)
        record["robot_pkl"] = str(generated_robot_pkl_path(spec))

        if args.dry_run:
            record["status"] = "dry_run"
            records.append(record)
            print(f"[{offset}/{total}] dry-run {spec.sample_id} d={spec.forward_meters:.3f} m")
            continue

        previous = previous_by_sample.get(spec.sample_id)
        if manifest_path.exists() and not args.overwrite:
            record["status"] = "skipped_existing"
            if previous and previous.get("status") == "generated":
                record["status"] = "generated"
            write_jsonl_row(progress_path, record)
            records.append(record)
            print(f"[{offset}/{total}] skip existing {spec.sample_id}")
            continue

        print(
            f"[{offset}/{total}] generate {spec.sample_id} "
            f"d={spec.forward_meters:.3f} m duration={spec.duration:.2f}s "
            f"waypoints={spec.root_waypoints}",
            flush=True,
        )
        if args.generation_mode == "subprocess":
            subprocess.run(cmd, cwd=KIMODO_ROOT, check=True)
        else:
            if persistent_generator is None:
                persistent_generator = PersistentKimodoGenerator(args)
            export_persistent_sample(persistent_generator, spec, args)
        record["status"] = "generated"
        write_jsonl_row(progress_path, record)
        records.append(record)

    return records


def combine_motion_libs(dataset_dir: Path, records: list[dict[str, Any]], dry_run: bool) -> None:
    if dry_run:
        return

    motion_lib_dir = dataset_dir / "motion_lib"
    motion_lib_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, dict[str, Any]] = {"all": {}, "train": {}, "val": {}}
    for record in records:
        if record["status"] not in {"generated", "skipped_existing"}:
            continue
        pkl_path = Path(record["robot_pkl"])
        if not pkl_path.exists():
            raise FileNotFoundError(f"Generated robot PKL not found: {pkl_path}")
        payload = joblib.load(pkl_path)
        grouped["all"].update(payload)
        grouped[record["split"]].update(payload)

    for split, payload in grouped.items():
        if payload:
            joblib.dump(payload, motion_lib_dir / f"{split}.pkl", compress=True)


def build_arg_parser() -> argparse.ArgumentParser:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description="Generate a randomized-distance Kimodo dataset for GEAR-Sonic finetuning."
    )
    parser.add_argument("--dataset_name", type=str, default=f"distance_random_{timestamp}")
    parser.add_argument("--datasets_root", type=str, default=str(DEFAULT_DATASETS_ROOT))
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed for distances/speeds/splits.")
    parser.add_argument("--kimodo_seed_base", type=int, default=10000)

    parser.add_argument("--min_distance", type=float, default=0.1)
    parser.add_argument("--max_distance", type=float, default=10.0)
    parser.add_argument(
        "--sampling",
        choices=["stratified_uniform", "uniform"],
        default="stratified_uniform",
        help="stratified_uniform samples uniformly inside distance bins in round-robin order.",
    )
    parser.add_argument(
        "--distance_bins",
        type=parse_float_list,
        default=parse_float_list("0.1,0.5,2.0,5.0,10.0"),
        help="Comma-separated bin edges used by stratified_uniform.",
    )

    parser.add_argument("--speed_min", type=float, default=0.3)
    parser.add_argument("--speed_max", type=float, default=0.6)
    parser.add_argument("--min_duration", type=float, default=2.5)
    parser.add_argument("--max_duration", type=float, default=30.0)
    parser.add_argument("--waypoint_hz", type=float, default=2.0)
    parser.add_argument("--min_root_waypoints", type=int, default=3)
    parser.add_argument("--max_root_waypoints", type=int, default=61)

    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--target_height", type=float, default=0.78)
    parser.add_argument("--append_target_hold", type=int, default=30)
    parser.add_argument("--snap_to_target_frames", type=int, default=30)
    parser.add_argument("--normalize_root_to_target", type=int, default=1)
    parser.add_argument("--marker_stride", type=int, default=10)

    parser.add_argument("--model", type=str, default="Kimodo-G1-RP-v1")
    parser.add_argument("--diffusion_steps", type=int, default=100)
    parser.add_argument("--num_transition_frames", type=int, default=5)
    parser.add_argument(
        "--generation_mode",
        choices=["persistent", "subprocess"],
        default="persistent",
        help="persistent loads Kimodo once and reuses it; subprocess preserves the older per-sample CLI path.",
    )
    parser.add_argument("--text_encoder_device", type=str, default=None, choices=[None, "cpu", "cuda"])
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="A humanoid robot walks forward {distance:.2f} meters and comes to a stable stop.",
    )
    parser.add_argument("--motion_prefix", type=str, default="kimodo_forward")
    parser.add_argument("--target_prefix", type=str, default="target_forward")

    parser.add_argument("--dry_run", action="store_true", help="Write indexes/config but do not call Kimodo.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate samples even if manifest.json exists.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be positive.")
    if args.min_distance <= 0 or args.max_distance <= args.min_distance:
        raise ValueError("Expected 0 < min_distance < max_distance.")
    if args.speed_min <= 0 or args.speed_max < args.speed_min:
        raise ValueError("Expected 0 < speed_min <= speed_max.")
    if args.sampling == "stratified_uniform":
        if args.distance_bins[0] < args.min_distance or args.distance_bins[-1] > args.max_distance:
            raise ValueError("--distance_bins must stay inside [min_distance, max_distance].")

    dataset_dir = Path(args.datasets_root).expanduser().resolve() / args.dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    config_payload = vars(args).copy()
    config_payload["dataset_dir"] = str(dataset_dir)
    config_payload["generator"] = str(GENERATOR)
    json_dump(dataset_dir / "dataset_config.json", config_payload)

    specs = make_sample_specs(args, dataset_dir)
    spec_rows = [asdict(spec) for spec in specs]
    append_jsonl(dataset_dir / "planned_samples.jsonl", spec_rows)

    records = run_generation(specs, args, dataset_dir)
    records = compact_records(read_jsonl(dataset_dir / "progress.jsonl")) if not args.dry_run else records
    append_jsonl(dataset_dir / "index.jsonl", records)
    append_jsonl(dataset_dir / "train.jsonl", [row for row in records if row["split"] == "train"])
    append_jsonl(dataset_dir / "val.jsonl", [row for row in records if row["split"] == "val"])
    combine_motion_libs(dataset_dir, records, args.dry_run)

    print(f"dataset_dir: {dataset_dir}")
    print(f"index: {dataset_dir / 'index.jsonl'}")
    if not args.dry_run:
        print(f"combined_motion_lib_all: {dataset_dir / 'motion_lib' / 'all.pkl'}")
        print(f"combined_motion_lib_train: {dataset_dir / 'motion_lib' / 'train.pkl'}")
        print(f"combined_motion_lib_val: {dataset_dir / 'motion_lib' / 'val.pkl'}")


if __name__ == "__main__":
    main()
