#!/usr/bin/env bash
set -euo pipefail

cuda_device="0"
learning_rates=(2e-4 8e-4 2e-3)
batch_sizes=(8 16 32 64 128)

for lr in "${learning_rates[@]}"; do
  for bs in "${batch_sizes[@]}"; do
    run_name=$(printf "contrastive-bs=%d-lr=%s" "$bs" "$lr")
    echo "Running ${run_name} on CUDA device ${cuda_device}"

    CUDA_VISIBLE_DEVICES="$cuda_device" python train.py fit \
      --config="configs/tinybert_contrastive.yaml" \
      --trainer.accelerator=cuda \
      --trainer.devices=1 \
      --data.batch_size="$bs" \
      --model.lr="$lr" \
      --trainer.logger.init_args.name="$run_name"
  done
done

echo "Done."
