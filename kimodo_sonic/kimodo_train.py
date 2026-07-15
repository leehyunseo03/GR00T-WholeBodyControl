#!/usr/bin/env python3
"""Kimodo distance-reference finetuning launcher for GEAR-Sonic.

This script is a thin, explicit wrapper around ``gear_sonic/train_agent_trl.py``.
It points SONIC at a Kimodo-generated motion-lib dataset and applies the
position-tracking overrides needed for the "follow Kimodo global path" test.

Default mode injects ``motion_anchor_pos_b`` into the G1 token encoder, so the
actor receives robot-local reference position error.  This changes network input
shape, so the script automatically prepares a warm-start checkpoint with the
incompatible G1 encoder input layer removed.  Use ``--no-position-token`` for a
checkpoint-compatible reward-only baseline.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "sonic_release" / "last.pt"
DEFAULT_BASE_DIR = REPO_ROOT / "sonic_release" / "exported"
DEFAULT_PROJECT_NAME = "kimodo_position_finetune"
DEFAULT_DATASET_DIR = Path("/workspace/kimodo/kimodo_dataset/datasets/distance_random_v1")
KIMODO_UPPER_BODY_REWARD_BODIES = [
    "pelvis",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]
KIMODO_REWARD_POINT_BODIES = [
    "pelvis",
    "torso_link",
    "torso_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]
KIMODO_REWARD_POINT_OFFSETS = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.35],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
]


def str2bool(text: str | bool) -> bool:
    if isinstance(text, bool):
        return text
    lowered = text.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {text!r}")


def shell_join(parts: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def hydra_list(values: Iterable[object]) -> str:
    return "[" + ",".join(_hydra_value(value) for value in values) + "]"


def _hydra_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Iterable):
        return hydra_list(value)
    return str(value)


def resolve_motion_file(args: argparse.Namespace) -> Path:
    if args.motion_file:
        return Path(args.motion_file)
    dataset_dir = Path(args.dataset_dir)
    split_pkl = dataset_dir / "motion_lib" / f"{args.split}.pkl"
    all_pkl = dataset_dir / "motion_lib" / "all.pkl"
    return split_pkl if args.split else all_pkl


def checkpoint_state_dict_key(checkpoint: dict) -> str:
    if "policy_state_dict" in checkpoint:
        return "policy_state_dict"
    if "actor_model_state_dict" in checkpoint:
        return "actor_model_state_dict"
    raise KeyError("Checkpoint has neither 'policy_state_dict' nor 'actor_model_state_dict'.")


def prepare_warmstart_checkpoint(src: Path, dst: Path) -> None:
    """Drop G1 encoder input-layer tensors so position-token finetuning can load.

    ``motion_anchor_pos_b`` increases the G1 encoder input dimension. PyTorch
    still errors on same-name shape mismatches when ``strict=False``, so the
    changed input layer must be absent from the checkpoint.  The matching here
    intentionally targets only the first layer under ``encoders.g1``.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Preparing a warm-start checkpoint requires torch. Run this inside "
            "the Isaac/GEAR-Sonic training environment."
        ) from exc

    # Released SONIC checkpoints may have been pickled with older TRL symbols.
    # Register the moved classes before torch.load unpickles trainer state.
    try:
        from trl.experimental.ppo.ppo_trainer import OnlineTrainerState, exact_div
        import trl.trainer.utils

        trl.trainer.utils.OnlineTrainerState = OnlineTrainerState
        trl.trainer.utils.exact_div = exact_div
        sys.modules["trl.trainer.utils"].OnlineTrainerState = OnlineTrainerState
        sys.modules["trl.trainer.utils"].exact_div = exact_div
    except ImportError:
        pass

    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    state_key = checkpoint_state_dict_key(checkpoint)
    state_dict = checkpoint[state_key]

    drop_patterns = (
        "actor_module.backbone.encoders.g1.module.0",
        "actor_module.backbone.encoders.g1.module.input_layer.0",
        "actor_module.backbone.encoders.g1.input_layer.0",
        "actor_module.backbone.encoders.g1.block.0",
        "backbone.encoders.g1.module.0",
        "backbone.encoders.g1.module.input_layer.0",
        "backbone.encoders.g1.input_layer.0",
        "backbone.encoders.g1.block.0",
        "encoders.g1.module.0",
        "encoders.g1.module.input_layer.0",
        "encoders.g1.input_layer.0",
        "encoders.g1.block.0",
    )

    removed: list[str] = []
    for key in list(state_dict.keys()):
        if any(pattern in key for pattern in drop_patterns):
            removed.append(key)
            state_dict.pop(key)

    if not removed:
        print(
            "WARNING: no G1 encoder input-layer keys matched. If training still "
            "fails with a shape mismatch, inspect the checkpoint keys and extend "
            "drop_patterns in kimodo_train.py.",
            file=sys.stderr,
        )
    else:
        print("Removed warm-start tensors:")
        for key in removed:
            print(f"  {key}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    # The PPO trainer loads "actor_model_state_dict" with strict=True, but
    # "policy_state_dict" with strict=False. Since this warm-start deliberately
    # removes shape-changed tensors, force the non-strict loading path.
    checkpoint["policy_state_dict"] = state_dict
    if "actor_model_state_dict" in checkpoint:
        checkpoint.pop("actor_model_state_dict")
    checkpoint["kimodo_warmstart_position_token"] = True
    torch.save(checkpoint, dst)
    print(f"Wrote warm-start checkpoint: {dst}")


def warmstart_checkpoint_is_current(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        import torch

        try:
            from trl.experimental.ppo.ppo_trainer import OnlineTrainerState, exact_div
            import trl.trainer.utils

            trl.trainer.utils.OnlineTrainerState = OnlineTrainerState
            trl.trainer.utils.exact_div = exact_div
            sys.modules["trl.trainer.utils"].OnlineTrainerState = OnlineTrainerState
            sys.modules["trl.trainer.utils"].exact_div = exact_div
        except ImportError:
            pass

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return bool(checkpoint.get("kimodo_warmstart_position_token", False))


def add_common_overrides(overrides: list[str], args: argparse.Namespace, motion_file: Path) -> None:
    overrides.extend(
        [
            "+exp=manager/universal_token/all_modes/sonic_release",
            f"+checkpoint={args.checkpoint}",
            f"base_dir={args.base_dir}",
            f"project_name={args.project_name}",
            f"experiment_name={args.experiment_name}",
            f"num_envs={args.num_envs}",
            f"headless={str(args.headless)}",
            f"algo.config.num_learning_iterations={args.num_learning_iterations}",
            f"algo.config.save_interval={args.save_interval}",
            f"algo.config.eval_frequency={args.eval_frequency}",
            f"algo.config.actor_learning_rate={args.actor_lr}",
            f"algo.config.critic_learning_rate={args.critic_lr}",
            f"algo.config.num_steps_per_env={args.num_steps_per_env}",
            f"algo.config.num_learning_epochs={args.num_learning_epochs}",
            f"algo.config.num_mini_batches={args.num_mini_batches}",
            f"++manager_env.commands.motion.motion_lib_cfg.motion_file={motion_file}",
            "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy",
            "++manager_env.commands.motion.motion_lib_cfg.multi_thread=False",
            "++manager_env.commands.motion.motion_lib_cfg.num_load_jobs=1",
            "++manager_env.commands.motion.encoder_sample_probs.g1=1.0",
            "++manager_env.commands.motion.encoder_sample_probs.teleop=0.0",
            "++manager_env.commands.motion.encoder_sample_probs.smpl=0.0",
            "++manager_env.commands.motion.cat_upper_body_poses=False",
            "++manager_env.commands.motion.freeze_frame_aug=False",
            "++manager_env.commands.motion.reward_point_body="
            f"{hydra_list(KIMODO_REWARD_POINT_BODIES)}",
            "++manager_env.commands.motion.reward_point_body_offset="
            f"{hydra_list(KIMODO_REWARD_POINT_OFFSETS)}",
            "++manager_env.config.terrain_type=plane",
            "++manager_env.rewards.tracking_anchor_pos.weight=2.0",
            "++manager_env.rewards.tracking_anchor_pos.params.std=0.12",
            "++manager_env.rewards.tracking_anchor_ori.weight=1.0",
            "++manager_env.rewards.tracking_anchor_ori.params.std=0.3",
            "++manager_env.rewards.tracking_relative_body_pos.weight=2.0",
            "++manager_env.rewards.tracking_relative_body_pos.params.std=0.22",
            "++manager_env.rewards.tracking_relative_body_pos.params.body_names="
            f"{hydra_list(KIMODO_UPPER_BODY_REWARD_BODIES)}",
            "++manager_env.rewards.tracking_relative_body_ori.weight=0.75",
            "++manager_env.rewards.tracking_relative_body_ori.params.std=0.45",
            "++manager_env.rewards.tracking_relative_body_ori.params.body_names="
            f"{hydra_list(KIMODO_UPPER_BODY_REWARD_BODIES)}",
            "++manager_env.rewards.tracking_body_linvel.weight=1.5",
            "++manager_env.rewards.tracking_body_linvel.params.std=1.2",
            "++manager_env.rewards.tracking_body_linvel.params.body_names="
            f"{hydra_list(KIMODO_UPPER_BODY_REWARD_BODIES)}",
            "++manager_env.rewards.tracking_body_angvel.weight=0.75",
            "++manager_env.rewards.tracking_body_angvel.params.std=3.5",
            "++manager_env.rewards.tracking_body_angvel.params.body_names="
            f"{hydra_list(KIMODO_UPPER_BODY_REWARD_BODIES)}",
            "++manager_env.rewards.tracking_vr_5point_local.weight=2.5",
            "++manager_env.rewards.tracking_vr_5point_local.params.std=0.16",
            "++manager_env.rewards.action_rate_l2.weight=-0.08",
            "++manager_env.rewards.feet_acc.weight=-1.0e-6",
            "++manager_env.rewards.anti_shake_ang_vel.weight=-0.01",
            "++manager_env.rewards.anti_shake_ang_vel.params.threshold=1.2",
            "++manager_env.rewards.anti_shake_ang_vel.params.body_names="
            f"{hydra_list(KIMODO_UPPER_BODY_REWARD_BODIES[1:])}",
        ]
    )

    if args.max_num_load_motions is not None:
        print(
            "WARNING: --max_num_load_motions is ignored because TrackingCommandCfg "
            "does not expose it as a Hydra field in this checkout.",
            file=sys.stderr,
        )


def add_position_token_overrides(overrides: list[str]) -> None:
    overrides.extend(
        [
            "++manager_env.observations.tokenizer.motion_anchor_pos_b_mf_nonflat._target_=isaaclab.managers.ObservationTermCfg",
            "++manager_env.observations.tokenizer.motion_anchor_pos_b_mf_nonflat.func=gear_sonic.envs.manager_env.mdp:motion_anchor_pos_b_mf",
            "++manager_env.observations.tokenizer.motion_anchor_pos_b_mf_nonflat.params.command_name=motion",
            "++manager_env.observations.tokenizer.motion_anchor_pos_b_mf_nonflat.params.mask_out_z=True",
            "++manager_env.observations.tokenizer.motion_anchor_pos_b_mf_nonflat.params.non_flatten=True",
            "++manager_env.observations.tokenizer.motion_anchor_pos_b_mf_nonflat.noise._target_=isaaclab.utils.noise.UniformNoiseCfg",
            "++manager_env.observations.tokenizer.motion_anchor_pos_b_mf_nonflat.noise.n_min=-0.02",
            "++manager_env.observations.tokenizer.motion_anchor_pos_b_mf_nonflat.noise.n_max=0.02",
            "++algo.config.actor.backbone.encoders.g1.inputs=[command_multi_future_nonflat,motion_anchor_ori_b_mf_nonflat,motion_anchor_pos_b_mf_nonflat]",
        ]
    )


def build_train_command(args: argparse.Namespace) -> list[str]:
    motion_file = resolve_motion_file(args)
    overrides: list[str] = []
    add_common_overrides(overrides, args, motion_file)
    if args.position_token:
        add_position_token_overrides(overrides)
    overrides.extend(args.override)

    train_cmd = ["gear_sonic/train_agent_trl.py", *overrides]
    if args.accelerate:
        return [
            "accelerate",
            "launch",
            f"--num_processes={args.num_processes}",
            *train_cmd,
        ]
    return [sys.executable, *train_cmd]


def build_arg_parser() -> argparse.ArgumentParser:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description="Launch Kimodo position-aware GEAR-Sonic finetuning."
    )
    parser.add_argument("--dataset_dir", type=str, default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "all", ""])
    parser.add_argument("--motion_file", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--project_name", type=str, default=DEFAULT_PROJECT_NAME)
    parser.add_argument("--experiment_name", type=str, default=f"kimodo_pos_track_{timestamp}")

    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--num_learning_iterations", type=int, default=2000)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--eval_frequency", type=int, default=100)
    parser.add_argument("--num_steps_per_env", type=int, default=24)
    parser.add_argument("--num_learning_epochs", type=int, default=5)
    parser.add_argument("--num_mini_batches", type=int, default=4)
    parser.add_argument("--actor_lr", type=float, default=1e-5)
    parser.add_argument("--critic_lr", type=float, default=5e-4)
    parser.add_argument("--max_num_load_motions", type=int, default=None)
    parser.add_argument("--headless", type=str2bool, default=True)

    parser.add_argument(
        "--position-token",
        action="store_true",
        default=True,
        help="Add robot-local reference position error to the G1 token encoder input.",
    )
    parser.add_argument(
        "--no-position-token",
        action="store_false",
        dest="position_token",
        help="Keep the released network shape and use position tracking rewards only.",
    )
    parser.add_argument(
        "--prepare-warmstart-checkpoint",
        action="store_true",
        help="Create a shape-compatible checkpoint for --position-token, then exit unless --run-after-prepare is set.",
    )
    parser.add_argument(
        "--no-auto-warmstart",
        action="store_true",
        help="Do not auto-create the warm-start checkpoint when --position-token is active.",
    )
    parser.add_argument(
        "--warmstart-output",
        type=str,
        default=str(REPO_ROOT / "kimodo_sonic" / "checkpoints" / "sonic_release_position_token_warmstart.pt"),
    )
    parser.add_argument("--run-after-prepare", action="store_true")

    parser.add_argument("--accelerate", action="store_true")
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--subprocess-log",
        type=str,
        default=str(REPO_ROOT / "kimodo_sonic" / "kimodo_train_subprocess.log"),
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Hydra override. Can be passed multiple times.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    os.chdir(REPO_ROOT)

    if args.split == "all" and args.motion_file is None:
        args.split = ""

    should_auto_warmstart = (
        args.position_token
        and not args.no_auto_warmstart
        and not args.prepare_warmstart_checkpoint
        and not args.dry_run
        and Path(args.checkpoint).resolve() == DEFAULT_CHECKPOINT.resolve()
    )
    if should_auto_warmstart:
        dst = Path(args.warmstart_output)
        if not warmstart_checkpoint_is_current(dst):
            prepare_warmstart_checkpoint(DEFAULT_CHECKPOINT, dst)
        args.checkpoint = str(dst)

    if args.prepare_warmstart_checkpoint:
        src = Path(args.checkpoint)
        dst = Path(args.warmstart_output)
        prepare_warmstart_checkpoint(src, dst)
        args.checkpoint = str(dst)
        if not args.run_after_prepare:
            print("Warm-start checkpoint prepared. Re-run with:")
            print(
                shell_join(
                    [
                        sys.executable,
                        str(Path(__file__).relative_to(REPO_ROOT)),
                        "--position-token",
                        f"--checkpoint={dst}",
                        f"--dataset_dir={args.dataset_dir}",
                    ]
                )
            )
            return

    command = build_train_command(args)
    print(shell_join(command), flush=True)
    if args.dry_run:
        return

    log_path = Path(args.subprocess_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(shell_join(command) + "\n\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        result_code = process.wait()
    if result_code != 0:
        print(f"Inner training command failed. Full log: {log_path}", file=sys.stderr)
        print(f"Last 80 lines from {log_path}:", file=sys.stderr)
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in lines[-80:]:
                print(line, file=sys.stderr)
        except OSError:
            pass
        raise subprocess.CalledProcessError(result_code, command)


if __name__ == "__main__":
    main()
