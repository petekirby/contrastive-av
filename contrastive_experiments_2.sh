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

run_experiment "contrastive-bs=512-lr=4e-5" \
  --trainer.max_epochs=10 \
  --model.init_args.microbatch_size=256 \
  --data.batch_size=512 \
  --model.init_args.lr=4e-5

run_experiment "contrastive-bs=1024-lr=4e-5" \
  --trainer.max_epochs=10 \
  --model.init_args.microbatch_size=256 \
  --data.batch_size=1024 \
  --model.init_args.lr=4e-5

# Long Baseline: batch size 256
run_experiment "long-contrastive-bs=256-lr=4e-5" \
  --trainer.max_epochs=40 \
  --data.batch_size=256 \
  --model.init_args.lr=4e-5

# GradCache
run_experiment "long-contrastive-bs=1024-lr=4e-5" \
  --trainer.max_epochs=40 \
  --model.init_args.microbatch_size=256 \
  --data.batch_size=1024 \
  --model.init_args.lr=4e-5

# Gradient accumulation
run_experiment "long-contrastive-accum=1024-lr=4e-5" \
  --trainer.max_epochs=40 \
  --data.batch_size=256 \
  --trainer.accumulate_grad_batches=4 \
  --model.init_args.lr=4e-5

# Cross-Batch Memory
run_experiment "long-contrastive-xbm=1024-lr=4e-5" \
  --trainer.max_epochs=40 \
  --model.init_args.cross_batch_memory_size=1024 \
  --data.batch_size=256 \
  --model.init_args.lr=4e-5

echo "Done."
