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
```

This replaces the replanning callback's root XY with a lightweight foot-odometry
estimate: IMU/base quaternion for heading, ankle-roll foot positions in the base
frame, and foot contact. In Isaac this still uses sim-computed foot link poses as
the FK boundary, then compares the estimated XY against sim root XY in logs such
as `[base_estimator] ... err=...`. On a real G1, replace that FK boundary with
encoder-based kinematics and feed the same estimator contract.
