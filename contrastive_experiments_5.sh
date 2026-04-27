#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/tinybert_contrastive_best.yaml"

run_experiment() {
  local run_name="$1"
  shift

  echo "Running ${run_name}"

  python train.py fit \
    --config="${CONFIG}" \
    --trainer.logger.init_args.name="${run_name}" \
    "$@"
}

bs=1024
lr=4e-4
temperatures=("$@")

for temp in "${temperatures[@]}"; do
  run_experiment "long-contrastive-bs=$bs-lr=$lr-temp=$temp" \
    --model.init_args.loss_dict.loss_config.temperature="$temp"
done

echo "Done."
