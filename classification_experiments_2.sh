#!/usr/bin/env bash
set -euo pipefail

cuda_device="$1"
shift
batch_sizes=("$@")
learning_rates=(8e-5 4e-5 2e-5)

for lr in "${learning_rates[@]}"; do
  for bs in "${batch_sizes[@]}"; do
    run_name=$(printf "classification-bs=%d-lr=%s" "$bs" "$lr")
    echo "Running ${run_name} on CUDA device ${cuda_device}"

    CUDA_VISIBLE_DEVICES="$cuda_device" python train.py fit \
      --config="configs/tinybert_classification.yaml" \
      --trainer.accelerator=cuda \
      --trainer.devices=1 \
      --data.batch_size="$bs" \
      --optimizer.init_args.lr="$lr" \
      --trainer.logger.init_args.name="$run_name"
  done
done

echo "Done."
