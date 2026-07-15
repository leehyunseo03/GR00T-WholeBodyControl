# MotionBricks Naive Forward + GEAR-Sonic

이 버전은 5m target, 도착 판정, terminal pose를 쓰지 않습니다.

MotionBricks에 계속 같은 명령만 줍니다.

```text
movement_direction = forward
facing_direction = forward
mode = walk
```

GEAR-Sonic 실험에서는 같은 forward 명령으로 MotionBricks segment를 계속 replanning해서 active reference에 덮어씁니다.

## 1. 궤적 생성

```bash
cd /home/hslee/IsaacLab_ws/GR00T-WholeBodyControl

python3 motionbricks_gearsonic/generate_naive_forward_motion_lib.py \
  --output_dir motionbricks_gearsonic/naive_forward \
  --session naive_forward \
  --motion_name motionbricks_naive_forward_walk \
  --mode walk_gun \
  --target_vel 0.40 \
  --max_steps 900 \
  --random_seed 1234
```

출력 motion file:

```text
motionbricks_gearsonic/naive_forward/robot_filtered/naive_forward/motionbricks_naive_forward_walk.pkl
```

`--max_steps`만 길이를 정합니다. 특정 거리 목표는 없습니다.

## 2. GEAR-Sonic 실험

```bash
cd /workspace/GR00T-WholeBodyControl

LIVESTREAM=2 /workspace/isaaclab/isaaclab.sh -p gear_sonic/eval_agent_trl.py \
  +checkpoint=/workspace/GR00T-WholeBodyControl/sonic_release/last.pt \
  +headless=False \
  +use_encoder=g1 \
  +motionbricks_metrics_scene=True \
  +manager_env/terminations=tracking/eval \
  ++num_envs=1 \
  ++eval_callbacks=[naive_forward_live,body_tracking] \
  ++callbacks.naive_forward_live._target_=motionbricks_gearsonic.naive_forward_callback.MotionBricksNaiveForwardCallback \
  ++callbacks.naive_forward_live.mode=walk \
  ++callbacks.naive_forward_live.target_vel=0.40 \
  ++callbacks.naive_forward_live.replan_interval_steps=25 \
  ++callbacks.naive_forward_live.segment_frames=72 \
  ++callbacks.naive_forward_live.debug_print_interval_steps=25 \
  ++callbacks.body_tracking._target_=metrics.body_tracking_callback.BodyTrackingCallback \
  ++callbacks.body_tracking.save_path=/workspace/GR00T-WholeBodyControl/metrics/motionbricks_gearsonic/naive_forward \
  ++callbacks.body_tracking.max_steps=2000 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/workspace/GR00T-WholeBodyControl/motionbricks_gearsonic/naive_forward/robot_filtered/naive_forward/motionbricks_naive_forward_walk.pkl \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy \
  ++manager_env.commands.motion.motion_lib_cfg.multi_thread=False \
  ++manager_env.commands.motion.visualize_motion_root_trajectory=False
```
