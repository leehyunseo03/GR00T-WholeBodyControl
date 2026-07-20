#!/usr/bin/env bash
# Run the live-goal Ardy/GEAR-SONIC replanning experiment.
# Start the host planner as usual, then run this inside the gear-sonic-base
# container. A separate terminal can send goals with goal_console.py.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"

export PYTHONPATH="${HERE}:${REPO_ROOT}:${PYTHONPATH:-}"
export ARDY_REPLAN_CALLBACK_TARGET="${ARDY_REPLAN_CALLBACK_TARGET:-live_goal_callback.LiveGoalReplanCallback}"

# Start with the normal forward goal, but any command from goal_console.py
# immediately replaces it.
export GOAL_MODE="${GOAL_MODE:-forward}"
export FORWARD_METERS="${FORWARD_METERS:-5.0}"

# Preserve the useful behavior observed in 2_5_m_left_3m_replan: all non-final
# legs are short local Ardy plans that re-attach to the robot's achieved pose.
export LOCAL_PLAN_DISTANCE="${LOCAL_PLAN_DISTANCE:-1.2}"
export KEEP_ALIVE_AFTER_ARRIVAL="${KEEP_ALIVE_AFTER_ARRIVAL:-True}"
export ACCEPT_EXISTING_GOAL_COMMAND="${ACCEPT_EXISTING_GOAL_COMMAND:-False}"

echo "[run_general_replan] callback=${ARDY_REPLAN_CALLBACK_TARGET}"
echo "[run_general_replan] initial goal: ${FORWARD_METERS}m forward until a command arrives"
echo "[run_general_replan] local plan distance: ${LOCAL_PLAN_DISTANCE}m"
echo "[run_general_replan] command terminal:"
echo "  python3 ardy_sonic/general/goal_console.py"
echo "  # then type absolute env-local/global goals like: 0 3"

exec bash "${REPO_ROOT}/ardy_sonic/run_ardy_sonic.sh" "$@"
