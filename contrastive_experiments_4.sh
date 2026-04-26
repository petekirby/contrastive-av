#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/tinybert_contrastive.yaml"

run_experiment() {
  local run_name="$1"
  shift

  echo "Running ${run_name}"

  python train.py fit \
    --config="${CONFIG}" \
    --trainer.logger.init_args.name="${run_name}" \
    "$@"
}

bs="$1"
shift
learning_rates=("$@")


for lr in "${learning_rates[@]}"; do
  run_experiment "contrastive-bs=$bs-lr=$lr" \
    --trainer.max_epochs=10 \
    --model.init_args.microbatch_size=256 \
    --data.batch_size=$bs \
    --model.init_args.lr="$lr"
  run_experiment "long-contrastive-bs=$bs-lr=$lr" \
    --trainer.max_epochs=40 \
    --model.init_args.microbatch_size=256 \
    --data.batch_size=$bs \
    --model.init_args.lr="$lr"
done

echo "Done."
