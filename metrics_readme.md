# 1. Motion Generator
### Walk
```
python3 motion_sonic/generate_to_target_motion_lib.py \
  --reference forward \
  --output_dir motion_sonic/motion/ \
  --motion_name motionbricks_to_target_forward_5m_target \
  --forward_meters 5.0 \
  --forward_target_name forward_5m_target \
  --mode walk \
  --target_vel 0.40 \
  --target_lookahead_meters 0.5 \
  --max_steps 750 \
  --append_target_hold 100
```

# 2. Record
```
LIVESTREAM=2 /workspace/isaaclab/isaaclab.sh -p gear_sonic/eval_agent_trl.py \
  +checkpoint=/workspace/GR00T-WholeBodyControl/sonic_release/last.pt \
  +headless=False \
  +use_encoder=g1 \
  +manager_env/terminations=tracking/eval \
  ++num_envs=1 \
  ++eval_callbacks=body_tracking \
  ++callbacks.body_tracking._target_=metrics.body_tracking_callback.BodyTrackingCallback \
  ++callbacks.body_tracking.save_path=/workspace/GR00T-WholeBodyControl/metrics/motionsonic/motion \
  ++callbacks.body_tracking.max_steps=2000 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/workspace/GR00T-WholeBodyControl/motion_sonic/motion/robot_filtered/motionbricks_target/motionbricks_to_target_forward_5m_target.pkl \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy \
  ++manager_env.commands.motion.motion_lib_cfg.multi_thread=False \
  ++manager_env.commands.motion.visualize_motion_root_trajectory=False
```
# 2.5 MotionBricks + GearSonic 
```
LIVESTREAM=2 /workspace/isaaclab/isaaclab.sh -p gear_sonic/eval_agent_trl.py \
  +checkpoint=/workspace/GR00T-WholeBodyControl/sonic_release/last.pt \
  +headless=False \
  +use_encoder=g1 \
  +motionbricks_metrics_scene=True \
  +manager_env/terminations=tracking/eval \
  ++num_envs=1 \
  ++eval_callbacks=[motionbricks_live,body_tracking] \
  ++callbacks.motionbricks_live._target_=motion_sonic.live_motionbricks_callback.MotionBricksRecedingHorizonCallback \
  ++callbacks.motionbricks_live.forward_meters=5.0 \
  ++callbacks.motionbricks_live.target_vel=0.40 \
  ++callbacks.motionbricks_live.lookahead_meters=0.80 \
  ++callbacks.motionbricks_live.replan_interval_steps=25 \
  ++callbacks.motionbricks_live.segment_frames=72 \
  ++callbacks.body_tracking._target_=metrics.body_tracking_callback.BodyTrackingCallback \
  ++callbacks.body_tracking.save_path=/workspace/GR00T-WholeBodyControl/metrics/motionsonic/motion \
  ++callbacks.body_tracking.max_steps=2000 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/workspace/GR00T-WholeBodyControl/motion_sonic/motion/robot_filtered/motionbricks_target/motionbricks_to_target_forward_5m_target.pkl \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy \
  ++manager_env.commands.motion.motion_lib_cfg.multi_thread=False \
  ++manager_env.commands.motion.visualize_motion_root_trajectory=False
```

## Visualize
```
+motionbricks_metrics_scene=True \
++manager_env.commands.motion.visualize_motion_root_trajectory=True
```

```
LIVESTREAM=2 /workspace/isaaclab/isaaclab.sh -p motion_sonic/motionbricks_metrics.py --livestream 2
```

# 3. Metrics
```
python3 motion_sonic/motionsonic_metrics.py \
  --recording metrics/motionsonic/motion \
  --motionbricks-qpos motion_sonic/motion/qpos/motionbricks_to_target_forward_5m_target.npz \
  --target motion_sonic/motion/target_reference/forward_5m_target.npz \
  --out-dir metrics/plot \
  --unit mm \
  --plot-reference-dof
```

# 4. Ardy + GearSonic Replanning Metrics
Recordings are saved by `ardy_sonic/run_ardy_sonic.sh` into
`ardy_sonic/metrics/recording`. The launcher records until a goal-reaching Ardy
plan has been tracked through its exact landing, prints the robot diagnostic
errors, holds the destination reference for `HOLD_SECONDS` seconds (default `3`),
then exits. `MAX_STEPS` is only a hard safety cap.

```
bash ardy_sonic/run_ardy_sonic.sh
```

For a shorter diagnostic capture:

```
BODY_TRACKING_MAX_STEPS=1000 bash ardy_sonic/run_ardy_sonic.sh
```

```
python3 ardy_sonic/ardysonic_metrics.py \
  --recording ardy_sonic/metrics/recording \
  --out-dir ardy_sonic/metrics/plot \
  --unit mm \
  --plot-reference-dof
```

By default the metrics script uses the newest Ardy response qpos from the active
runtime directory. To pin a specific plan:

```
python3 ardy_sonic/ardysonic_metrics.py \
  --recording ardy_sonic/metrics/recording \
  --ardy-qpos ardy_sonic/runtime/responses/plan_0004.npz \
  --out-dir ardy_sonic/metrics/plot \
  --unit mm \
  --plot-reference-dof
```
