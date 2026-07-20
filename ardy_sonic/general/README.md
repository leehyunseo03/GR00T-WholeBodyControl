# General Live Goal Replanning

This experiment keeps the short local replanning behavior from
`2_5_m_left_3m_replan`, but lets you change the goal from a separate terminal.

Run:

```bash
# HOST
cd /home/hslee/IsaacLab_ws/GR00T-WholeBodyControl
bash ardy_sonic/run_planner.sh

# CONTAINER
cd /workspace/GR00T-WholeBodyControl
bash ardy_sonic/general/run_general_replan.sh

CHECKPOINT=/workspace/GR00T-WholeBodyControl/sonic_release/exported/kimodo_position_finetune/kimodo_pos_track_20260708_142052-20260708_142059/last.pt \
bash ardy_sonic/general/run_general_replan.sh

# HOST or CONTAINER, another terminal
cd /home/hslee/IsaacLab_ws/GR00T-WholeBodyControl
python3 ardy_sonic/general/goal_console.py
```

In the command terminal, type an absolute env-local XY target:

```text
goal xy> 0 3
goal xy> 4.5 -1
```

The callback watches `${ARDY_SONIC_RUNTIME}/general_goal_command.json`. On each
new command it drops the old pending request, sets the new `goal_xy`, points the
terminal heading along the line from the robot's current root XY to that goal, and
continues planning in `LOCAL_PLAN_DISTANCE=1.2` m chunks.

Useful knobs:

```bash
LOCAL_PLAN_DISTANCE=1.2 bash ardy_sonic/general/run_general_replan.sh
ACCEPT_EXISTING_GOAL_COMMAND=True bash ardy_sonic/general/run_general_replan.sh
python3 ardy_sonic/general/goal_console.py 0 3
```
