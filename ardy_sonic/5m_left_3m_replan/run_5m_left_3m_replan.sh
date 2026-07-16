#!/usr/bin/env bash
# Run the 5 m forward -> 3 m left Ardy/GEAR-SONIC replanning experiment.
# Run this INSIDE the gear-sonic-base container, after starting the host planner:
#
#   bash ardy_sonic/run_planner.sh
#   bash ardy_sonic/5m_left_3m_replan/run_5m_left_3m_replan.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"

export PYTHONPATH="${HERE}:${REPO_ROOT}:${PYTHONPATH:-}"
export ARDY_REPLAN_CALLBACK_TARGET="${ARDY_REPLAN_CALLBACK_TARGET:-left_replan_callback.FiveMeterThenLeftThreeMeterReplanCallback}"

export GOAL_MODE="${GOAL_MODE:-forward}"
export FORWARD_METERS="${FORWARD_METERS:-5.0}"
export LEFT_SWITCH_RADIUS="${LEFT_SWITCH_RADIUS:-0.2}"
export LEFT_METERS="${LEFT_METERS:-3.0}"

echo "[run_5m_left_3m_replan] first goal: ${FORWARD_METERS}m forward"
echo "[run_5m_left_3m_replan] switch: robot within ${LEFT_SWITCH_RADIUS}m of first goal"
echo "[run_5m_left_3m_replan] second goal: ${LEFT_METERS}m left of the first goal"

exec bash "${REPO_ROOT}/ardy_sonic/run_ardy_sonic.sh" "$@"
