#!/usr/bin/env bash
set -euo pipefail

cuda_device="$1"
shift
learning_rate="$1"
shift
batch_sizes=("$@")

for bs in "${batch_sizes[@]}"; do
  run_name=$(printf "contrastive-bs=%d-lr=%s" "$bs" "$lr")
  echo "Running ${run_name} on CUDA device ${cuda_device}"

  CUDA_VISIBLE_DEVICES="$cuda_device" python train.py fit \
    --config="configs/tinybert_contrastive.yaml" \
    --trainer.accelerator=cuda \
    --trainer.devices=1 \
    --data.batch_size="$bs" \
    --model.lr="$learning_rate" \
    --trainer.logger.init_args.name="$run_name"
done

echo "Done."
