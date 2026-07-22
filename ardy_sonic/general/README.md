# General Live Goal Replanning

This experiment keeps the short local replanning behavior from
`2_5_m_left_3m_replan`, but lets you change the goal from a separate terminal.

Run:

```bash
# HOST
cd /home/hslee/IsaacLab_ws/GR00T-WholeBodyControl
bash ardy_sonic/run_planner.sh

# CONTAINER (gear-sonic-base)
cd /workspace/GR00T-WholeBodyControl
#CHECKPOINT=/workspace/GR00T-WholeBodyControl/sonic_release/exported/kimodo_position_finetune/kimodo_pos_track_20260708_142052-20260708_142059/last.pt \
bash ardy_sonic/general/run_general_replan.sh

ESTIMATE_BASE_STATE=True bash ardy_sonic/general/run_general_replan.sh

# HOST or CONTAINER, another terminal
cd /home/hslee/IsaacLab_ws/GR00T-WholeBodyControl
python3 ardy_sonic/general/goal_console.py
```

In the command terminal, type an env-local/global XY goal. This is the default
because commands like `0 3` should now mean "go to absolute x=0, y=3":

```text
goal xy> 0 3
goal xy> 1 -1
```

The callback watches `${ARDY_SONIC_RUNTIME}/general_goal_command.json`. On each
new command it drops the old pending request, reads `goal_xy` as the absolute
env-local/global target, points the terminal heading along the line from the
robot's current root XY to that goal, and continues planning in
`LOCAL_PLAN_DISTANCE=1.2` m chunks.

If you want the old "move by dx dy from the current robot position" behavior,
pass `--relative`:

```bash
python3 ardy_sonic/general/goal_console.py --relative 0 3
```

Useful knobs:

```bash
LOCAL_PLAN_DISTANCE=1.2 bash ardy_sonic/general/run_general_replan.sh
ACCEPT_EXISTING_GOAL_COMMAND=True bash ardy_sonic/general/run_general_replan.sh
python3 ardy_sonic/general/goal_console.py 0 3
python3 ardy_sonic/general/goal_console.py --relative 0 3
```

Base-state estimator experiment:

```bash
ESTIMATE_BASE_STATE=True bash ardy_sonic/general/run_general_replan.sh
ESTIMATE_BASE_STATE=True ESTIMATOR_LOG_INTERVAL=50 bash ardy_sonic/general/run_general_replan.sh
ESTIMATE_BASE_STATE=True ESTIMATOR_INITIAL_XY=1.0,2.0 bash ardy_sonic/general/run_general_replan.sh
ESTIMATE_BASE_STATE=True ESTIMATOR_CONTACT_ENTER_STEPS=2 ESTIMATOR_MAX_ANCHOR_RESIDUAL=0.18 bash ardy_sonic/general/run_general_replan.sh
```

This replaces the replanning callback's root XY/yaw with a strict-signal legged
odometry estimate and, by default, enables `STRICT_NO_PRIVILEGED_STATE=True`.
In that mode the callback does not read simulator `root_pos_w` or `body_pos_w`
for replanning: it builds foot positions from the current joint qpos through
encoder-style FK, uses kinematic lowest-foot contact, filters contact with
short hysteresis, rejects large stance-anchor jumps as likely slip/outliers, and
initializes the odometry frame at `ESTIMATOR_INITIAL_XY` or `[0, 0]`.

Estimator tuning knobs:

```bash
ESTIMATOR_CONTACT_ENTER_STEPS=2          # raw contact frames before accepting stance
ESTIMATOR_CONTACT_EXIT_STEPS=2           # missed frames before dropping stance
ESTIMATOR_XY_CORRECTION_ALPHA=0.75       # blend toward stance-foot XY measurement
ESTIMATOR_MAX_XY_CORRECTION_PER_STEP=0.08
ESTIMATOR_MAX_ANCHOR_RESIDUAL=0.18       # bigger residual is treated as slip/outlier
ESTIMATOR_MAX_YAW_RATE=3.5               # rad/s yaw jump gate for replanning qpos
```

For sim-only debugging, you can opt back into simulator sensor comparisons:

```bash
ESTIMATE_BASE_STATE=True STRICT_NO_PRIVILEGED_STATE=False ESTIMATOR_CONTACT_SOURCE=sim_sensor bash ardy_sonic/general/run_general_replan.sh
ESTIMATE_BASE_STATE=True STRICT_NO_PRIVILEGED_STATE=False ESTIMATOR_LOG_INTERVAL=25 bash ardy_sonic/general/run_general_replan.sh
```

The non-strict estimator log includes sim-only `sim_xy`/`err` and
`sim_yaw`/`yaw_err` comparisons. Treat those as diagnostics only; strict
deployment-style replanning still uses the local foot-odometry estimate rather
than simulator global state.
