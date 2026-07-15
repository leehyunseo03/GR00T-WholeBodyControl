#!/usr/bin/env bash
set -euo pipefail

# Run this inside the gear-sonic-base container.
REPO_ROOT="${REPO_ROOT:-/workspace/GR00T-WholeBodyControl}"
ISAACLAB_SH="${ISAACLAB_SH:-/workspace/isaaclab/isaaclab.sh}"
# CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/sonic_release/last.pt}"
# CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/sonic_release/exported/kimodo_position_finetune/kimodo_pos_track_continue_4k-20260707_160720/last.pt}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/sonic_release/exported/kimodo_position_finetune/kimodo_pos_track_20260708_142052-20260708_142059/last.pt}"

MOTION_NAME="${MOTION_NAME:-kimodo_to_target_forward_5m_hand_raise}"
MOTION_FILE="${MOTION_FILE:-${REPO_ROOT}/kimodo_sonic/motion/robot_filtered/kimodo_target/${MOTION_NAME}.pkl}"
SAVE_PATH="${SAVE_PATH:-${REPO_ROOT}/metrics/kimodosonic/motion}"

MAX_STEPS="${MAX_STEPS:-1200}"
NUM_ENVS="${NUM_ENVS:-1}"
HEADLESS="${HEADLESS:-False}"
LIVESTREAM="${LIVESTREAM:-2}"

cd "${REPO_ROOT}"

LIVESTREAM="${LIVESTREAM}" "${ISAACLAB_SH}" -p gear_sonic/eval_agent_trl.py \
  "+checkpoint=${CHECKPOINT}" \
  "+headless=${HEADLESS}" \
  "+use_encoder=g1" \
  "+manager_env/terminations=tracking/eval" \
  "++num_envs=${NUM_ENVS}" \
  "++eval_callbacks=[body_tracking]" \
  "++callbacks.body_tracking._target_=metrics.body_tracking_callback.BodyTrackingCallback" \
  "++callbacks.body_tracking.save_path=${SAVE_PATH}" \
  "++callbacks.body_tracking.max_steps=${MAX_STEPS}" \
  "++manager_env.commands.motion.motion_lib_cfg.motion_file=${MOTION_FILE}" \
  "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy" \
  "++manager_env.commands.motion.motion_lib_cfg.multi_thread=False" \
  "++manager_env.commands.motion.visualize_motion_root_trajectory=False"
