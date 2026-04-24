#!/usr/bin/env bash
set -euo pipefail

batch_sizes=(8 16 32 64 128 256 512 1024)
negatives_per_positive=(1 2 4)

for bs in "${batch_sizes[@]}"; do
  for npp in "${negatives_per_positive[@]}"; do
    run_name=$(printf "classification: bs=%d, npp=%d" "$bs" "$npp")
    echo "Running ${run_name}"

    python train.py fit \
      --config="configs/tinybert_classification.yaml" \
      --data.batch_size="$bs" \
      --model.init_args.negatives_per_positive="$npp" \
      --trainer.logger.init_args.name="$run_name"
  done
done
echo "Done."
