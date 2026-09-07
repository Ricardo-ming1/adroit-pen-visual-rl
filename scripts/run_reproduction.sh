#!/usr/bin/env bash
set -euo pipefail

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MUJOCO_GL=egl

python -m scripts.build_visual_dataset --config configs/data.yaml
python -m scripts.train_oracle --config configs/oracle_full.yaml --seed 0 --resume

for seed in 101 102 103 104 105; do
  python -m scripts.train_vision --config configs/vision_frozen.yaml --seed "$seed" --resume
done

for seed in 101 102 103; do
  source_dir="runs/frozen/vision/full_seed_${seed}"
  target_dir="runs/frozen/vision/full_no_anchor_seed_${seed}"
  mkdir -p "$target_dir"
  cp --update=none "$source_dir/best_bc_validation.pt" "$target_dir/"
  cp --update=none "$source_dir/last_critic_warmup.pt" "$target_dir/"
  cp --update=none "$source_dir/best_offline_awac_validation.pt" "$target_dir/"
  python -m scripts.train_vision --config configs/vision_frozen.yaml --seed "$seed" --stage ppo --no-anchor --resume
done

for seed in 101 102 103; do
  run_dir="runs/frozen/vision/full_seed_${seed}"
  python -m scripts.evaluate --run-dir "$run_dir" --stage bc --split test
  python -m scripts.evaluate --run-dir "$run_dir" --stage awac --split test
  python -m scripts.evaluate --run-dir "runs/frozen/vision/full_no_anchor_seed_${seed}" --stage ppo --split test
done

for seed in 101 102 103 104 105; do
  run_dir="runs/frozen/vision/full_seed_${seed}"
  for intervention in normal black episode_shuffle temporal_shuffle target_occlusion; do
    python -m scripts.evaluate --run-dir "$run_dir" --stage ppo --split test --intervention "$intervention"
  done
done

python -m scripts.summarize_results
