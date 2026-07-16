# 2.5 m Switch, Original Goal Left 3 m Replan

This experiment keeps the normal 5 m forward Ardy plan at the start, but changes
the target mid-course.

Behavior:

1. Start with the normal `GOAL_MODE=forward`, `FORWARD_METERS=5.0` target.
2. Measure the robot's physical root XY progress along the original start-to-goal
   direction.
3. Once progress reaches `SWITCH_PROGRESS_METERS=2.5` m, discard any old pending
   request and set the new target to the original 5 m goal shifted left by
   `LEFT_METERS=3.0` m.
4. Ardy replans from the robot's current pose to that new left-shifted target.

Run:

```bash
# HOST
cd /home/hslee/IsaacLab_ws/GR00T-WholeBodyControl
bash ardy_sonic/run_planner.sh

# CONTAINER
cd /workspace/GR00T-WholeBodyControl
bash ardy_sonic/2_5_m_left_3m_replan/run_2_5_m_left_3m_replan.sh

# finetuned gearsonic
CHECKPOINT=/workspace/GR00T-WholeBodyControl/sonic_release/exported/kimodo_position_finetune/kimodo_pos_track_20260708_142052-20260708_142059/last.pt \
bash ardy_sonic/2_5_m_left_3m_replan/run_2_5_m_left_3m_replan.sh
```

Useful knobs:

```bash
SWITCH_PROGRESS_METERS=2.5 LEFT_METERS=3.0 bash ardy_sonic/2_5_m_left_3m_replan/run_2_5_m_left_3m_replan.sh
```

For a robot that starts facing +x, the target sequence is approximately:

```text
start -> original goal [5.0, 0.0]
at progress 2.5m: new goal becomes [5.0, 3.0]
```
