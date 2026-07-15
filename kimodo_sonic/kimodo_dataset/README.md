# Kimodo Distance Dataset

Generate randomized straight-walk Kimodo references for GEAR-Sonic finetuning.
Outputs are saved under:

```text
kimodo_dataset/datasets/<dataset_name>/
```

Each sample has the same layout as `kimodo_sonic/generate_to_target_motion_lib.py`,
and the script also writes combined motion-lib files:

```text
motion_lib/all.pkl
motion_lib/train.pkl
motion_lib/val.pkl
```

By default the generator runs in `--generation_mode persistent`: the Kimodo model
is loaded once, then reused for every sample. This is much faster than launching
a new Python process per motion.

Generation is resumable. If the process is stopped, rerun the same command with
the same `--dataset_name`; samples with an existing `manifest.json` are skipped,
and unfinished samples are regenerated. Progress is appended to `progress.jsonl`
after each completed sample.

## Smoke Test

Run from `/home/hslee/IsaacLab_ws/kimodo` in the Kimodo environment:

```bash
python3 kimodo_dataset/generate_distance_dataset.py \
  --dataset_name distance_smoke \
  --num_samples 4 \
  --diffusion_steps 20
```

## Main Dataset

This samples distances between `0.1 m` and `10.0 m`, balanced across
`[0.1, 0.5]`, `[0.5, 2.0]`, `[2.0, 5.0]`, and `[5.0, 10.0]`.

```bash
python3 kimodo_dataset/generate_distance_dataset.py \
  --dataset_name distance_random_v1 \
  --num_samples 200 \
  --val_fraction 0.1 \
  --min_distance 0.1 \
  --max_distance 10.0 \
  --distance_bins 0.1,0.5,2.0,5.0,10.0 \
  --speed_min 0.3 \
  --speed_max 0.6 \
  --append_target_hold 30
```

If you need the older one-process-per-sample behavior for debugging:

```bash
python3 kimodo_dataset/generate_distance_dataset.py \
  --dataset_name distance_random_v1_debug \
  --num_samples 8 \
  --generation_mode subprocess
```

For a quick command preview without running Kimodo:

```bash
python3 kimodo_dataset/generate_distance_dataset.py \
  --dataset_name distance_plan_only \
  --num_samples 8 \
  --dry_run
```

## GEAR-Sonic Motion File

After generation, use the combined training PKL as the motion file:

```text
/home/hslee/IsaacLab_ws/kimodo/kimodo_dataset/datasets/distance_random_v1/motion_lib/train.pkl
```

Inside the container, use the corresponding mounted path for
`manager_env.commands.motion.motion_lib_cfg.motion_file`.
