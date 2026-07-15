# Smoke Test
```
python kimodo_sonic/kimodo_train.py \
  --dataset_dir /workspace/GR00T-WholeBodyControl/kimodo_sonic/kimodo_dataset/datasets/distance_random_v1 \
  --num_envs 16 \
  --num_learning_iterations 10 \
  --save_interval 5 \
  --eval_frequency 5 \
  --max_num_load_motions 16
```

# Train
```
python kimodo_sonic/kimodo_train.py \
  --dataset_dir /workspace/GR00T-WholeBodyControl/kimodo_sonic/kimodo_dataset/datasets/distance_random_v1 \
  --num_envs 512 \
  --num_learning_iterations 10000 \
  --save_interval 100 \
  --eval_frequency 100
```

```
python kimodo_sonic/kimodo_train_foot.py \
  --dataset_dir /workspace/GR00T-WholeBodyControl/kimodo_sonic/kimodo_dataset/datasets/distance_random_v1 \
  --num_envs 512 \
  --num_learning_iterations 10000 \
  --save_interval 100 \
  --eval_frequency 100
```

```
python kimodo_sonic/kimodo_train.py \
  --dataset_dir /workspace/GR00T-WholeBodyControl/kimodo_sonic/kimodo_dataset/datasets/distance_random_v1 \
  --num_envs 512 \
  --num_learning_iterations 10000 \
  --save_interval 100 \
  --eval_frequency 100
```