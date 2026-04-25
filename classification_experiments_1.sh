#!/usr/bin/env bash
set -euo pipefail

cuda_device="$1"
shift
batch_sizes=("$@")
negatives_per_positive=(1 2 4)

for bs in "${batch_sizes[@]}"; do
  for npp in "${negatives_per_positive[@]}"; do
    run_name=$(printf "classification-bs=%d-npp=%d-focal" "$bs" "$npp")
    echo "Running ${run_name} on CUDA device ${cuda_device}"

    CUDA_VISIBLE_DEVICES="$cuda_device" python train.py fit \
      --config="configs/tinybert_classification_focal.yaml" \
      --trainer.accelerator=cuda \
      --trainer.devices=1 \
      --data.batch_size="$bs" \
      --model.init_args.negatives_per_positive="$npp" \
      --trainer.logger.init_args.name="$run_name"
  done
done
echo "Done."
