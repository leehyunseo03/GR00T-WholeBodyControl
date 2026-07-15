# Kimodo Motion Sonic

Hybrid handoff flow:

1. Run MotionBricks/GEAR-Sonic until the robot is near the final target.
2. Save the actual robot MuJoCo qpos at the stop/settle moment.
3. Generate a short Kimodo residual segment from that qpos to the final qpos.
4. Feed the exported `robot_filtered/kimodo_motion_sonic/*.pkl` to GEAR-Sonic.

This keeps MotionBricks responsible for the long walk and uses Kimodo only for the short terminal correction, where global drift is less harmful.

## One-Shot Residual

From the GR00T repo:

```bash
cd /home/hslee/IsaacLab_ws/GR00T-WholeBodyControl
PYTHONPATH=/home/hslee/IsaacLab_ws/kimodo:$PYTHONPATH \
TEXT_ENCODER_DEVICE=cpu \
python3 kimodo_motion_sonic/residual_kimodo_handoff.py \
  --current_qpos motion_sonic/motion/qpos/motionbricks_to_target_forward_5m_target.npy \
  --current_frame -1 \
  --target_qpos motion_sonic/motion/qpos/motionbricks_to_target_forward_5m_target.npz \
  --target_qpos_key target_qpos \
  --target_frame -1 \
  --output_dir kimodo_motion_sonic/motion \
  --motion_name kimodo_motion_sonic_residual_5m \
  --target_name kimodo_motion_sonic_target_5m \
  --duration 2.0 \
  --root_waypoints 5 \
  --diffusion_steps 50 \
  --prepend_start_hold 8 \
  --snap_to_target_frames 20 \
  --append_target_hold 60
```

If no `--target_qpos` is given, the script uses a fallback zero-DOF target at `--fallback_forward_meters`, `--target_height`, and `--target_yaw`.

The main output is:

```text
kimodo_motion_sonic/motion/robot_filtered/kimodo_motion_sonic/<motion_name>.pkl
```

Use that path as:

```text
++manager_env.commands.motion.motion_lib_cfg.motion_file=/workspace/GR00T-WholeBodyControl/kimodo_motion_sonic/motion/robot_filtered/kimodo_motion_sonic/kimodo_motion_sonic_residual_5m.pkl
++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy
```

## Warm Kimodo Server

For repeated handoffs, keep Kimodo loaded once:

```bash
cd /home/hslee/IsaacLab_ws/GR00T-WholeBodyControl
PYTHONPATH=/home/hslee/IsaacLab_ws/kimodo:$PYTHONPATH \
TEXT_ENCODER_DEVICE=cpu \
python3 kimodo_motion_sonic/residual_kimodo_handoff.py \
  --serve \
  --current_qpos motion_sonic/motion/qpos/motionbricks_to_target_forward_5m_target.npy \
  --target_qpos motion_sonic/motion/qpos/motionbricks_to_target_forward_5m_target.npz \
  --target_qpos_key target_qpos \
  --request_jsonl /tmp/kimodo_motion_sonic_requests.jsonl \
  --response_jsonl /tmp/kimodo_motion_sonic_responses.jsonl
```

Then append JSON requests:

```bash
printf '%s\n' '{"request_id":"handoff_001","current_qpos":"motion_sonic/motion/qpos/motionbricks_to_target_forward_5m_target.npy","current_frame":-1,"target_qpos":"motion_sonic/motion/qpos/motionbricks_to_target_forward_5m_target.npz","target_qpos_key":"target_qpos","target_frame":-1,"output_dir":"kimodo_motion_sonic/motion/handoff_001","motion_name":"handoff_001","target_name":"handoff_001_target","duration":2.0,"root_waypoints":5}' >> /tmp/kimodo_motion_sonic_requests.jsonl
```

Read responses:

```bash
tail -f /tmp/kimodo_motion_sonic_responses.jsonl
```

## Important

- `current_qpos` should be the actual robot qpos at the MotionBricks stop moment, not the ideal MotionBricks reference endpoint.
- For short residual correction, start with `duration=1.5-2.5`, `root_waypoints=4-7`, and `diffusion_steps=50`.
- Handoff distance should usually be about `0.3 m` to `0.8 m`; `0.3 m = 30 cm`, `0.8 m = 80 cm`.
- The script writes a start hold and target hold so GEAR-Sonic has stable frames before and after the correction.
