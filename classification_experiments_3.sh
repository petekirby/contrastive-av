#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  "configs/tinybert_classification.yaml"
  "configs/tinybert_classification_focal.yaml"
)

for config in "${CONFIGS[@]}"; do
  for npp in 1 2 4; do
    echo "Running $config with negatives_per_positive=$npp"

    python train.py \
      --config "$config" \
      --negatives_per_positive "$npp"
  done
done
