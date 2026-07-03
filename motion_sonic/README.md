# MotionBricks -> GEAR-SONIC Bridge

This folder contains the first integration path between MotionBricks and
GEAR-SONIC:

```text
MotionBricks pretrained model -> G1 MuJoCo qpos -> SONIC motion_lib PKL
```

It does not modify any running GEAR-SONIC training run. It only creates a new
motion dataset under `motion_sonic/motion/`.

## 1. Generate a SONIC robot motion set

Run from the repo root in the environment where MotionBricks already works:

```bash
cd /home/hslee/IsaacLab_ws/GR00T-WholeBodyControl

python motion_sonic/generate_motionbricks_motion_lib.py \
  --max_steps 300 \
  --num_motions 1 \
  --random_seed 1234
```

Expected output:

```text
motion_sonic/motion/motionbricks_headless/
├── manifest.json
├── qpos/
│   └── motionbricks_g1_seed_00001234.npy
└── robot_filtered/
    └── motionbricks/
        └── motionbricks_g1_seed_00001234.pkl
```

Use this as the SONIC motion file:

```text
motion_sonic/motion/motionbricks_headless/robot_filtered
```

For the first tests, use:

```text
smpl_motion_file=dummy
use_encoder=g1
```

## 2. Evaluate with a GEAR-SONIC checkpoint

Inside the GEAR-SONIC/Isaac Lab container or environment:

```bash
python gear_sonic/eval_agent_trl.py \
  +checkpoint=sonic_release/last.pt \
  +headless=True \
  +use_encoder=g1 \
  +manager_env/terminations=tracking/eval \
  ++num_envs=8 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=motion_sonic/motion/motionbricks_headless/robot_filtered \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy
```

For your finetuned checkpoint, replace `sonic_release/last.pt` with the
finetuned `.pt` path.

## 3. Fine-tune after eval is stable

Only after the eval tracks the MotionBricks-generated reference motion without
early termination, mix it into your stage data with a small ratio first:

- start with existing data 90%, MotionBricks 10%
- then try 80/20
- only move to 70/30 if the first two are stable

Keep all outputs in a new experiment directory, for example:

```bash
python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  +checkpoint=<YOUR_FINETUNED_CHECKPOINT> \
  base_dir=sonic_release/exported \
  project_name=stage2_motion_sonic \
  experiment_name=motionbricks_mix_90_10 \
  num_envs=512 \
  headless=True \
  algo.config.num_learning_iterations=1000 \
  algo.config.save_interval=100 \
  algo.config.eval_frequency=100 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=<MIXED_ROBOT_MOTION_DIR> \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy
```

Do not overwrite `sonic_release/last.pt`.

## Standing Arm-Swing Generation

Default target generation creates an in-place standing arm-swing reference. The
root stays fixed in MuJoCo/G1 coordinates (`x` forward, `z` up), while the
shoulder, elbow, and wrist DOFs oscillate for a clear upper-body motion. Because
this is a full time sequence rather than a single final target frame, the script
exports the generated qpos reference directly as a SONIC motion-lib PKL:

```bash
cd /home/hslee/IsaacLab_ws/GR00T-WholeBodyControl

python motion_sonic/generate_to_target_motion_lib.py \
  --output_dir motion_sonic/motion/standing_arm_swing \
  --arm_swing_seconds 6.0 \
  --arm_swing_frequency 0.5
```

Expected output:

```text
motion_sonic/motion/standing_arm_swing/
├── manifest.json
├── target_reference/
│   ├── standing_arm_swing.npz
│   └── standing_arm_swing.pkl
├── qpos/
│   ├── motionbricks_to_target_standing_arm_swing.npy
│   └── motionbricks_to_target_standing_arm_swing.npz
├── visualization/
│   ├── motionbricks_to_target_standing_arm_swing_trajectory_markers.json
│   └── motionbricks_to_target_standing_arm_swing_trajectory_markers.xml
└── robot_filtered/
    └── motionbricks_target/
        └── motionbricks_to_target_standing_arm_swing.pkl
```

Evaluate the exported standing arm-swing motion with:

```bash
cd /home/hslee/IsaacLab_ws/GR00T-WholeBodyControl

python gear_sonic/eval_agent_trl.py \
  +checkpoint=sonic_release/last.pt \
  +headless=True \
  +use_encoder=g1 \
  +manager_env/terminations=tracking/eval \
  ++num_envs=8 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=motion_sonic/motion/standing_arm_swing/robot_filtered \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy
```

## Forward Target Generation

The old 5 m forward target is still available by selecting `--reference forward`.
This writes the target reference in both `.npz` and `.pkl` formats, asks
MotionBricks to generate a path to that target, and exports the generated path
as a SONIC motion-lib PKL:

```bash
cd /home/hslee/IsaacLab_ws/GR00T-WholeBodyControl

python motion_sonic/generate_to_target_motion_lib.py \
  --reference forward \
  --target_only 1 \
  --output_dir motion_sonic/motion/forward_5m_target \
  --forward_meters 5.0 \
  --forward_target_name forward_5m_target
```

```bash
cd /home/hslee/IsaacLab_ws/GR00T-WholeBodyControl

python motion_sonic/generate_to_target_motion_lib.py \
  --reference forward \
  --output_dir motion_sonic/motion/forward_5m_target \
  --forward_meters 5.0 \
  --forward_target_name forward_5m_target \
  --mode slow_walk \
  --target_vel 0.35 \
  --target_lookahead_meters 0.65 \
  --max_steps 450 \
  --append_target_hold 30
```

Forward target expected output:

```text
motion_sonic/motion/forward_5m_target/
├── manifest.json
├── target_reference/
│   ├── forward_5m_target.npz
│   └── forward_5m_target.pkl
├── qpos/
│   ├── motionbricks_to_target_forward_5m_target.npy
│   └── motionbricks_to_target_forward_5m_target.npz
├── visualization/
│   ├── motionbricks_to_target_forward_5m_target_trajectory_markers.json
│   └── motionbricks_to_target_forward_5m_target_trajectory_markers.xml
└── robot_filtered/
    └── motionbricks_target/
        └── motionbricks_to_target_forward_5m_target.pkl
```

The marker XML shows the generated MotionBricks intermediate root trajectory as
yellow spheres and the final target as a blue sphere. Open it with a MuJoCo
viewer from an environment that can load the G1 XML assets.

If the final reference includes a specific 29-DOF target pose, pass it with
`--target_dof` in MuJoCo order. Use `--target_dof_order isaaclab` when the 29
values are in IsaacLab action/joint order:

```bash
python motion_sonic/generate_to_target_motion_lib.py \
  --output_dir motion_sonic/motion/forward_5m_target \
  --forward_meters 5.0 \
  --target_dof_order isaaclab \
  --target_dof <29_RAD_VALUES> \
  --mode slow_walk \
  --target_vel 0.35 \
  --target_lookahead_meters 0.65 \
  --max_steps 450
```

To generate a trajectory toward the final frame of a teleop/robot file:

```bash
python motion_sonic/generate_to_target_motion_lib.py \
  --target /path/to/teleop_or_robot_motion.pkl \
  --target_frame -1 \
  --mode walk \
  --max_steps 300
```

Supported target inputs:

- `.npy` / `.npz` containing `qpos`, `mujoco_qpos`, `robot_qpos`, or `target_qpos`
  with shape `(36,)` or `(T, 36)`
- `.pkl` containing the same qpos keys
- SONIC-style `.pkl` motion entries with `root_trans_offset`, `root_rot`, and
  `dof`

If your file has several top-level motions or a non-standard qpos key:

```bash
python motion_sonic/generate_to_target_motion_lib.py \
  --target /path/to/teleop.pkl \
  --motion_key my_motion_name \
  --qpos_key mujoco_qpos \
  --target_frame -1 \
  --mode walk
```

The script uses the target frame's root position and yaw as MotionBricks
conditions. It also appends the exact target qpos for `--append_target_hold`
frames, so the exported SONIC reference motion includes the final full-body
target pose.

Evaluate the exported 5 m target motion with the single generated PKL. Pointing
at the exact file avoids accidentally loading older PKLs left in the same
`robot_filtered` directory:

```bash
python gear_sonic/eval_agent_trl.py \
  +checkpoint=sonic_release/last.pt \
  +headless=True \
  +use_encoder=g1 \
  +manager_env/terminations=tracking/eval \
  ++num_envs=8 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=motion_sonic/motion/forward_5m_target/robot_filtered/motionbricks_target/motionbricks_to_target_forward_5m_target.pkl \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy
```

For WebRTC livestream, enable the trajectory overlay only when you want those
extra yellow/blue debug markers:

```bash
LIVESTREAM=2 /workspace/isaaclab/isaaclab.sh -p gear_sonic/eval_agent_trl.py \
  +checkpoint=sonic_release/last.pt \
  +headless=False \
  +use_encoder=g1 \
  +manager_env/terminations=tracking/eval \
  ++num_envs=1 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=motion_sonic/motion/forward_5m_target/robot_filtered/motionbricks_target/motionbricks_to_target_forward_5m_target.pkl \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy \
  ++manager_env.commands.motion.visualize_motion_root_trajectory=True
```
