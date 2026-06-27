# Stage 1 GEAR-SONIC Finetuning

This runbook keeps the original release checkpoint untouched and writes all
stage-1 outputs under:

```text
sonic_release/exported/stage1_gear_sonic/
```

Run every command inside the container.

```bash
docker exec -it gear-sonic-base bash
cd /workspace/GR00T-WholeBodyControl
```

## 1. Optional: Rebuild Weighted Motion Sets

The prepared sets are symlink-only mixtures. Source PKLs are not modified.

```bash
python gear_sonic/scripts/prepare_stage1_motions.py \
  --output stage1_motions/mix_70_30 \
  --total-slots 20 \
  --group sample:sample_data/robot_filtered:sample_data/smpl_filtered:0.7 \
  --group custom:custom_motions/robot_arms_up_goal:custom_motions/smpl_arms_up_goal:0.3

python gear_sonic/scripts/prepare_stage1_motions.py \
  --output stage1_motions/mix_40_60 \
  --total-slots 20 \
  --group sample:sample_data/robot_filtered:sample_data/smpl_filtered:0.4 \
  --group custom:custom_motions/robot_arms_up_goal:custom_motions/smpl_arms_up_goal:0.6
```

Use a new output path, such as `stage1_motions/mix_40_60_v2`, when rebuilding
so previous training inputs remain reproducible.

## 2. Baseline Eval

```bash
python gear_sonic/eval_agent_trl.py \
  +checkpoint=sonic_release/last.pt \
  +headless=True \
  +use_encoder=g1 \
  +manager_env/terminations=tracking/eval \
  ++num_envs=8 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=stage1_motions/mix_70_30/robot_filtered \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=stage1_motions/mix_70_30/smpl_filtered
```

## 3. Smoke Train

```bash
python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  +checkpoint=sonic_release/last.pt \
  base_dir=sonic_release/exported \
  project_name=stage1_gear_sonic \
  experiment_name=stage1_smoke_mix_70_30 \
  num_envs=16 \
  headless=True \
  algo.config.num_learning_iterations=10 \
  algo.config.save_interval=5 \
  algo.config.eval_frequency=5 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=stage1_motions/mix_70_30/robot_filtered \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=stage1_motions/mix_70_30/smpl_filtered
```

Expected output directory:

```text
sonic_release/exported/stage1_gear_sonic/stage1_smoke_mix_70_30-YYYYMMDD_HHMMSS/
```

## 4. Conservative Finetune

```bash
python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  +checkpoint=sonic_release/last.pt \
  base_dir=sonic_release/exported \
  project_name=stage1_gear_sonic \
  experiment_name=stage1_mix_70_30 \
  num_envs=512 \
  headless=True \
  algo.config.num_learning_iterations=2000 \
  algo.config.save_interval=100 \
  algo.config.eval_frequency=100 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=stage1_motions/mix_70_30/robot_filtered \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=stage1_motions/mix_70_30/smpl_filtered
```

## 5. More Custom-Heavy Finetune

Start this from the best conservative checkpoint, not from the original release.
Replace `<BEST_70_30_CHECKPOINT>` with `.../last.pt` or a saved model file.

```bash
python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  +checkpoint=<BEST_70_30_CHECKPOINT> \
  base_dir=sonic_release/exported \
  project_name=stage1_gear_sonic \
  
  experiment_name=stage1_mix_40_60 \
  num_envs=512 \
  headless=True \
  algo.config.num_learning_iterations=2000 \
  algo.config.save_interval=100 \
  algo.config.eval_frequency=100 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=stage1_motions/mix_40_60/robot_filtered \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=stage1_motions/mix_40_60/smpl_filtered
```

## 6. Eval A Finetuned Checkpoint

```bash
python gear_sonic/eval_agent_trl.py \
  +checkpoint=<FINETUNED_CHECKPOINT> \
  +headless=True \
  +use_encoder=g1 \
  +manager_env/terminations=tracking/eval \
  ++num_envs=8 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=stage1_motions/mix_40_60/robot_filtered \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=stage1_motions/mix_40_60/smpl_filtered
```

## Notes

- Do not overwrite `sonic_release/last.pt`.
- Use `resume=True experiment_dir=<existing-run-dir>` only when continuing the
  same run directory.
- Raise `num_envs` from `512` to `1024`, `2048`, then `4096` only after the
  smoke and first finetune are stable on the available GPU memory.
