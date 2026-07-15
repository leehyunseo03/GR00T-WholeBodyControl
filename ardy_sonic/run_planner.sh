#!/usr/bin/env bash
# Convenience launcher for the Ardy planner server. Run this on the HOST.
# It activates the `ardy` conda env and serves plan requests from the shared
# runtime/ directory that the container-side tracker writes to.
#
#   bash GR00T-WholeBodyControl/ardy_sonic/run_planner.sh          # serve
#   bash GR00T-WholeBodyControl/ardy_sonic/run_planner.sh --once   # one-shot test
set -euo pipefail

ARDY_ENV="${ARDY_ENV:-ardy}"
CONDA_SH="${CONDA_SH:-/home/hslee/miniforge3/etc/profile.d/conda.sh}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="${ENGINE:-inprocess}"

# Talk to the SAME shared request/response dir the container tracker uses. On the host
# this is <IsaacLab_ws>/shared_io/runtime; inside the container it is the same physical
# dir at /workspace/shared_io/runtime (bind mount). Both sides must agree.
WS_ROOT="$(cd "${HERE}/../.." && pwd)"   # <IsaacLab_ws>
ARDY_SONIC_RUNTIME="${ARDY_SONIC_RUNTIME:-${WS_ROOT}/shared_io/runtime}"
export ARDY_SONIC_RUNTIME
umask 000
mkdir -p "${ARDY_SONIC_RUNTIME}/requests" "${ARDY_SONIC_RUNTIME}/responses" "${ARDY_SONIC_RUNTIME}/plans" 2>/dev/null || true
echo "[run_planner] ARDY_SONIC_RUNTIME=${ARDY_SONIC_RUNTIME}"

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${ARDY_ENV}"

if [ "${1:-}" = "--once" ]; then
  shift
  exec python "${HERE}/ardy_planner_server.py" --once --engine "${ENGINE}" "$@"
fi

exec python "${HERE}/ardy_planner_server.py" --serve --engine "${ENGINE}" "$@"
