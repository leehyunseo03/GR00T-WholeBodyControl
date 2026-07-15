#!/usr/bin/env bash
# Launch the GEAR-SONIC tracker that follows Ardy plans with receding-horizon
# replanning. Run this INSIDE the gear-sonic-base container.
#
#   1) On the HOST:      conda activate ardy && \
#                        python GR00T-WholeBodyControl/ardy_sonic/ardy_planner_server.py --serve
#   2) In the CONTAINER: bash GR00T-WholeBodyControl/ardy_sonic/run_ardy_sonic.sh
#
# The tracker registers ardy_sonic.ardy_replan_callback.ArdyReplanCallback, which
# talks to the host planner over the shared ardy_sonic/runtime/ directory.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/GR00T-WholeBodyControl}"
ISAACLAB_SH="${ISAACLAB_SH:-/workspace/isaaclab/isaaclab.sh}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/sonic_release/last.pt}"
PLACEHOLDER="${PLACEHOLDER:-${REPO_ROOT}/ardy_sonic/runtime/placeholder_motion.pkl}"

# Goal / replanning behaviour (env-overridable).
GOAL_MODE="${GOAL_MODE:-forward}"          # forward | absolute
FORWARD_METERS="${FORWARD_METERS:-5.0}"
ARRIVAL_RADIUS="${ARRIVAL_RADIUS:-0.30}"
HOLD_SECONDS="${HOLD_SECONDS:-3.0}"        # hold at the destination this long after arriving
SHOW_TARGET_MARKERS="${SHOW_TARGET_MARKERS:-True}"   # draw destination pose as blue spheres
MAX_PLAN_DISTANCE="${MAX_PLAN_DISTANCE:-6.0}"
SECONDS_PER_METER="${SECONDS_PER_METER:-2.0}"
TRACK_FRACTION="${TRACK_FRACTION:-0.9}"
MAX_REPLANS="${MAX_REPLANS:-12}"
MAX_STEPS="${MAX_STEPS:-6000}"
PLACEHOLDER_FRAMES="${PLACEHOLDER_FRAMES:-1500}"
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-300}"

NUM_ENVS="${NUM_ENVS:-1}"
HEADLESS="${HEADLESS:-False}"
LIVESTREAM="${LIVESTREAM:-2}"

cd "${REPO_ROOT}"

# --- shared, world-writable IO dir --------------------------------------------
# The container process runs as uid 1000 (ubuntu) while the mounted host files are
# owned by uid 1001 (hslee), so eval cannot create/write dirs under host-owned paths
# (logs_eval/, the checkpoint dir, ...). Redirect every eval output into one shared
# 0777 folder that both host and container can read/write. Override with SHARED_IO.
SHARED_IO="${SHARED_IO:-/workspace/shared_io}"
# umask 000 so every dir/file eval creates under SHARED_IO is 0777 -> the host user
# (uid 1001) and the container user (uid 1000) can both read AND write/delete them.
umask 000
mkdir -p "${SHARED_IO}" 2>/dev/null || true
chmod 777 "${SHARED_IO}" 2>/dev/null || true
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${SHARED_IO}/eval/${RUN_STAMP}"
mkdir -p "${RUN_DIR}" 2>/dev/null || true

# The Ardy<->SONIC request/response dir must live in the shared writable folder too
# (the default ardy_sonic/runtime is host-owned and not writable by the container).
# The HOST planner must point ARDY_SONIC_RUNTIME at the SAME physical dir (its host
# path is <IsaacLab_ws>/shared_io/runtime); run_planner.sh does this by default.
ARDY_SONIC_RUNTIME="${ARDY_SONIC_RUNTIME:-${SHARED_IO}/runtime}"
export ARDY_SONIC_RUNTIME
mkdir -p "${ARDY_SONIC_RUNTIME}/requests" "${ARDY_SONIC_RUNTIME}/responses" "${ARDY_SONIC_RUNTIME}/plans" 2>/dev/null || true

# gear_sonic's eval stack needs several packages that are not in the stock Isaac Sim
# python (joblib for motion_lib, easydict/loguru core deps, vector_quantize_pytorch
# for the FSQ policy, accelerate for the trainer). These are required for ALL SONIC
# eval, not just this bridge. We install into the SONIC python WITHOUT touching
# numpy/scipy: gear_sonic pins numpy==1.26.4 but Isaac Sim ships numpy 2.x, and the
# pin is only declarative -- do NOT downgrade numpy or Isaac Sim breaks.
ensure_dep() {  # $1 = import name, $2 = pip name (default $1)
  local mod="$1"; local pkg="${2:-$1}"
  if ! "${ISAACLAB_SH}" -p -c "import ${mod}" >/dev/null 2>&1; then
    echo "[run_ardy_sonic] '${mod}' missing in the SONIC python; installing ${pkg} ..."
    "${ISAACLAB_SH}" -p -m pip install --quiet "${pkg}" || {
      echo "[run_ardy_sonic] ERROR: could not install ${pkg}. Install it manually:"
      echo "    ${ISAACLAB_SH} -p -m pip install ${pkg}"
      exit 1
    }
  fi
}
ensure_dep joblib
ensure_dep easydict
ensure_dep loguru
ensure_dep vector_quantize_pytorch
ensure_dep accelerate

# gear_sonic needs EXACTLY trl==0.28.0: newer trl (e.g. 1.x) removed
# trl.trainer.ppo_trainer, which gear_sonic.trl.trainer.ppo_trainer extends. A wrong
# version imports fine but fails at eval, so pin the version explicitly.
TRL_VER="$("${ISAACLAB_SH}" -p -c 'import trl,sys;sys.stdout.write(trl.__version__)' 2>/dev/null || true)"
if [ "${TRL_VER}" != "0.28.0" ]; then
  echo "[run_ardy_sonic] trl==0.28.0 required (found '${TRL_VER:-none}'); installing ..."
  "${ISAACLAB_SH}" -p -m pip install --quiet "trl==0.28.0" || {
    echo "[run_ardy_sonic] ERROR: could not install trl==0.28.0. Install it manually:"
    echo "    ${ISAACLAB_SH} -p -m pip install trl==0.28.0"
    exit 1
  }
fi

# Ensure the placeholder motion clip exists (long enough to hold any injected plan).
# If it must be generated and the default (host-owned) path is not writable, fall
# back to the shared writable dir.
if [ ! -f "${PLACEHOLDER}" ]; then
  if [ ! -w "$(dirname "${PLACEHOLDER}")" ]; then
    PLACEHOLDER="${SHARED_IO}/placeholder_motion.pkl"
  fi
  if [ ! -f "${PLACEHOLDER}" ]; then
    echo "[run_ardy_sonic] placeholder motion missing; generating ${PLACEHOLDER}"
    "${ISAACLAB_SH}" -p "${REPO_ROOT}/ardy_sonic/make_placeholder_motion.py" \
        --output "${PLACEHOLDER}" --frames "${PLACEHOLDER_FRAMES}"
  fi
fi

# Stage the checkpoint into the shared writable dir. eval sets experiment_dir =
# checkpoint.parent and also writes a model_step_*.pt next to the checkpoint, so the
# checkpoint's directory must be writable. Symlinks keep this cheap (no big copies).
CKPT_SRC_DIR="$(cd "$(dirname "${CHECKPOINT}")" && pwd)"
CKPT_STAGE="${SHARED_IO}/ckpt"
mkdir -p "${CKPT_STAGE}" 2>/dev/null || true
chmod 777 "${CKPT_STAGE}" 2>/dev/null || true
ln -sfn "${CHECKPOINT}" "${CKPT_STAGE}/$(basename "${CHECKPOINT}")"
[ -f "${CKPT_SRC_DIR}/config.yaml" ] && ln -sfn "${CKPT_SRC_DIR}/config.yaml" "${CKPT_STAGE}/config.yaml"
for f in "${CKPT_SRC_DIR}"/model_step_*.pt; do
  [ -e "$f" ] && ln -sfn "$f" "${CKPT_STAGE}/$(basename "$f")"
done
STAGED_CHECKPOINT="${CKPT_STAGE}/$(basename "${CHECKPOINT}")"

echo "[run_ardy_sonic] checkpoint=${STAGED_CHECKPOINT}  (staged from ${CHECKPOINT})"
echo "[run_ardy_sonic] shared IO=${SHARED_IO}  run dir=${RUN_DIR}"
echo "[run_ardy_sonic] placeholder=${PLACEHOLDER}"
echo "[run_ardy_sonic] goal_mode=${GOAL_MODE} forward_meters=${FORWARD_METERS} arrival_radius=${ARRIVAL_RADIUS}"
echo "[run_ardy_sonic] Make sure the host Ardy planner server is running (--serve)."

# NOTE: eval_agent_trl.py is a @hydra.main script -> Hydra parses argv first and
# rejects non-override flags like `--livestream`. The WebRTC viewer is enabled purely
# by the LIVESTREAM env var (eval_agent_trl reads it via its env fallback) + headless.
HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}" LIVESTREAM="${LIVESTREAM}" \
  "${ISAACLAB_SH}" -p gear_sonic/eval_agent_trl.py \
  "+checkpoint=${STAGED_CHECKPOINT}" \
  "hydra.run.dir=${RUN_DIR}" \
  "++eval_base_dir=${SHARED_IO}/logs_eval" \
  "++base_dir=${SHARED_IO}/logs_rl" \
  "++exported_policy_path=${RUN_DIR}/exported" \
  "++eval_output_dir=${RUN_DIR}/output" \
  "++manager_env.config.save_rendering_dir=${RUN_DIR}/renderings" \
  "+headless=${HEADLESS}" \
  "+use_encoder=g1" \
  "+manager_env/terminations=tracking/eval" \
  "++num_envs=${NUM_ENVS}" \
  "++manager_env.config.episode_length_s=${EPISODE_LENGTH_S}" \
  "++eval_callbacks=[ardy_replan]" \
  "++callbacks.ardy_replan._target_=ardy_sonic.ardy_replan_callback.ArdyReplanCallback" \
  "++callbacks.ardy_replan.goal_mode=${GOAL_MODE}" \
  "++callbacks.ardy_replan.forward_meters=${FORWARD_METERS}" \
  "++callbacks.ardy_replan.arrival_radius=${ARRIVAL_RADIUS}" \
  "++callbacks.ardy_replan.hold_seconds=${HOLD_SECONDS}" \
  "++callbacks.ardy_replan.show_target_markers=${SHOW_TARGET_MARKERS}" \
  "++callbacks.ardy_replan.max_plan_distance=${MAX_PLAN_DISTANCE}" \
  "++callbacks.ardy_replan.seconds_per_meter=${SECONDS_PER_METER}" \
  "++callbacks.ardy_replan.track_fraction=${TRACK_FRACTION}" \
  "++callbacks.ardy_replan.max_replans=${MAX_REPLANS}" \
  "++callbacks.ardy_replan.max_steps=${MAX_STEPS}" \
  "++callbacks.ardy_replan.placeholder_frames=${PLACEHOLDER_FRAMES}" \
  "++callbacks.ardy_replan.runtime_dir=${ARDY_SONIC_RUNTIME}" \
  "++manager_env.terminations.anchor_pos.params.threshold=1000" \
  "++manager_env.terminations.anchor_pos.params.down_threshold=1000" \
  "++manager_env.terminations.ee_body_pos.params.threshold=1000" \
  "++manager_env.terminations.ee_body_pos.params.down_threshold=1000" \
  "++manager_env.terminations.anchor_ori_full.params.threshold=1000" \
  "++manager_env.commands.motion.motion_lib_cfg.motion_file=${PLACEHOLDER}" \
  "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy" \
  "++manager_env.commands.motion.motion_lib_cfg.multi_thread=False" \
  "++manager_env.commands.motion.visualize_motion_root_trajectory=False"
