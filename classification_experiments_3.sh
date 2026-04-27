#!/usr/bin/env bash
set -euo pipefail

for suffix in focal bce; do
  for npp in 4 2 1; do
    run_name="classification-${suffix}-npp=${npp}"
    echo "Running ${run_name}"

    python train.py fit \
      --config="configs/tinybert_classification_${suffix}.yaml" \
      --trainer.accelerator=cuda \
      --trainer.devices=1 \
      --model.negatives_per_positive="$npp" \
      --trainer.logger.init_args.name="$run_name"
  done
done

echo "Done."
