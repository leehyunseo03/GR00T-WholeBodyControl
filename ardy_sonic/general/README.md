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

# HOST or CONTAINER, another terminal
cd /home/hslee/IsaacLab_ws/GR00T-WholeBodyControl
python3 ardy_sonic/general/goal_console.py
```

In the command terminal, type an env-local XY delta from the robot's current
position. This is the default because commands like `0 3` should mean "go 3 m
sideways from here", not "go back toward absolute x=0, y=3":

```text
goal delta> 0 3
goal delta> 1 -1
```

The callback watches `${ARDY_SONIC_RUNTIME}/general_goal_command.json`. On each
new command it drops the old pending request, converts `goal_delta_xy` into an
absolute `goal_xy` using the robot's current root XY, points the terminal heading
along the line from the robot's current root XY to that goal, and continues
planning in `LOCAL_PLAN_DISTANCE=1.2` m chunks.

If you really want a literal absolute env-local coordinate, pass `--absolute`:

```bash
python3 ardy_sonic/general/goal_console.py --absolute 0 3
```

Useful knobs:

```bash
LOCAL_PLAN_DISTANCE=1.2 bash ardy_sonic/general/run_general_replan.sh
ACCEPT_EXISTING_GOAL_COMMAND=True bash ardy_sonic/general/run_general_replan.sh
python3 ardy_sonic/general/goal_console.py 0 3
python3 ardy_sonic/general/goal_console.py --absolute 0 3
```
