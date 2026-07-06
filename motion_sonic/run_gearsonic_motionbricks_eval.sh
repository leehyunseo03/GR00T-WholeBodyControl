#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/GR00T-WholeBodyControl}"
ISAACLAB_SH="${ISAACLAB_SH:-/workspace/isaaclab/isaaclab.sh}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/sonic_release/last.pt}"
MOTION_FILE="${MOTION_FILE:-${REPO_ROOT}/motion_sonic/motion/robot_filtered/motionbricks_target/motionbricks_to_target_forward_5m_target.pkl}"
SAVE_PATH="${SAVE_PATH:-${REPO_ROOT}/metrics/motionsonic/motion}"
MAX_STEPS="${MAX_STEPS:-2000}"
NUM_ENVS="${NUM_ENVS:-1}"
HEADLESS="${HEADLESS:-False}"
LIVESTREAM="${LIVESTREAM:-2}"
FORWARD_METERS="${FORWARD_METERS:-5.0}"
TARGET_VEL="${TARGET_VEL:-0.20}"
LOOKAHEAD_METERS="${LOOKAHEAD_METERS:-0.35}"
REPLAN_INTERVAL_STEPS="${REPLAN_INTERVAL_STEPS:-5}"
SEGMENT_FRAMES="${SEGMENT_FRAMES:-90}"

cd "${REPO_ROOT}"

LIVESTREAM="${LIVESTREAM}" "${ISAACLAB_SH}" -p gear_sonic/eval_agent_trl.py \
  "+checkpoint=${CHECKPOINT}" \
  "+headless=${HEADLESS}" \
  "+use_encoder=g1" \
  "+manager_env/terminations=tracking/eval" \
  "++num_envs=${NUM_ENVS}" \
  "++eval_callbacks=[motionbricks_live,body_tracking]" \
  "++callbacks.motionbricks_live._target_=motion_sonic.live_motionbricks_callback.MotionBricksRecedingHorizonCallback" \
  "++callbacks.motionbricks_live.forward_meters=${FORWARD_METERS}" \
  "++callbacks.motionbricks_live.target_vel=${TARGET_VEL}" \
  "++callbacks.motionbricks_live.lookahead_meters=${LOOKAHEAD_METERS}" \
  "++callbacks.motionbricks_live.replan_interval_steps=${REPLAN_INTERVAL_STEPS}" \
  "++callbacks.motionbricks_live.segment_frames=${SEGMENT_FRAMES}" \
  "++callbacks.body_tracking._target_=metrics.body_tracking_callback.BodyTrackingCallback" \
  "++callbacks.body_tracking.save_path=${SAVE_PATH}" \
  "++callbacks.body_tracking.max_steps=${MAX_STEPS}" \
  "++manager_env.commands.motion.motion_lib_cfg.motion_file=${MOTION_FILE}" \
  "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy" \
  "++manager_env.commands.motion.motion_lib_cfg.multi_thread=False" \
  "++manager_env.commands.motion.visualize_motion_root_trajectory=False"
