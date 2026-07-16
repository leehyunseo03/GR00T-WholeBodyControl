# 5 m Forward, Then 3 m Left Replan

This experiment reuses the existing `ardy_sonic/run_ardy_sonic.sh` launcher and
planner server, but swaps in a two-stage callback:

1. Start with the normal `GOAL_MODE=forward`, `FORWARD_METERS=5.0` goal.
2. When the physical robot root XY is within `LEFT_SWITCH_RADIUS=0.2` m of that
   first goal, discard any old pending request and immediately set a new goal
   `LEFT_METERS=3.0` m to the left of the first-goal heading.
3. The normal Ardy planner then replans from the robot's current pose to that
   left-side goal. The final stop/hold behavior is the original callback's
   planner-goal completion behavior, now applied to the second goal.

Run:

```bash
# HOST
cd /home/hslee/IsaacLab_ws/GR00T-WholeBodyControl
bash ardy_sonic/run_planner.sh

# CONTAINER
cd /workspace/GR00T-WholeBodyControl
bash ardy_sonic/5m_left_3m_replan/run_5m_left_3m_replan.sh
```

# finetuned gearsonic
```bash
CHECKPOINT=/workspace/GR00T-WholeBodyControl/sonic_release/exported/kimodo_position_finetune/kimodo_pos_track_20260708_142052-20260708_142059/last.pt \
bash ardy_sonic/5m_left_3m_replan/run_5m_left_3m_replan.sh
```

Useful knobs:

```bash
LEFT_SWITCH_RADIUS=0.2 LEFT_METERS=3.0 bash ardy_sonic/5m_left_3m_replan/run_5m_left_3m_replan.sh
```

For a robot that starts facing +x, the target sequence is approximately:

```text
start -> [5.0, 0.0] -> [5.0, 3.0]
```
