#!/usr/bin/env bash
# Run the 5 m forward plan, but switch to the original goal's left-side target
# once the robot has progressed about 2.5 m along the original 5 m direction.
#
# Run this INSIDE the gear-sonic-base container, after starting the host planner:
#
#   bash ardy_sonic/run_planner.sh
#   bash ardy_sonic/2_5_m_left_3m_replan/run_2_5_m_left_3m_replan.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"

export PYTHONPATH="${HERE}:${REPO_ROOT}:${PYTHONPATH:-}"
export ARDY_REPLAN_CALLBACK_TARGET="${ARDY_REPLAN_CALLBACK_TARGET:-left_replan_callback.TwoPointFiveMeterThenLeftGoalReplanCallback}"

export GOAL_MODE="${GOAL_MODE:-forward}"
export FORWARD_METERS="${FORWARD_METERS:-5.0}"
export SWITCH_PROGRESS_METERS="${SWITCH_PROGRESS_METERS:-2.5}"
export LEFT_METERS="${LEFT_METERS:-3.0}"
export POST_SWITCH_PLAN_DISTANCE="${POST_SWITCH_PLAN_DISTANCE:-1.2}"

echo "[run_2_5_m_left_3m_replan] original goal: ${FORWARD_METERS}m forward"
echo "[run_2_5_m_left_3m_replan] switch: ${SWITCH_PROGRESS_METERS}m progress along original goal direction"
echo "[run_2_5_m_left_3m_replan] new goal: original goal shifted ${LEFT_METERS}m left"
echo "[run_2_5_m_left_3m_replan] post-switch local replan distance: ${POST_SWITCH_PLAN_DISTANCE}m"

exec bash "${REPO_ROOT}/ardy_sonic/run_ardy_sonic.sh" "$@"
