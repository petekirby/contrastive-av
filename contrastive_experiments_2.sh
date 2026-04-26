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

lr="$1"

# Short Baseline: batch size 256
run_experiment "contrastive-bs=256-lr=$lr" \
  --trainer.max_epochs=10 \
  --data.batch_size=256 \
  --model.init_args.lr="$lr"

run_experiment "contrastive-bs=512-lr=$lr" \
  --trainer.max_epochs=10 \
  --model.init_args.microbatch_size=256 \
  --data.batch_size=512 \
  --model.init_args.lr="$lr"

run_experiment "contrastive-bs=1024-lr=$lr" \
  --trainer.max_epochs=10 \
  --model.init_args.microbatch_size=256 \
  --data.batch_size=1024 \
  --model.init_args.lr="$lr"

# Long Baseline: batch size 256
run_experiment "long-contrastive-bs=256-lr=$lr" \
  --trainer.max_epochs=40 \
  --data.batch_size=256 \
  --model.init_args.lr="$lr"

# GradCache
run_experiment "long-contrastive-bs=1024-lr=$lr" \
  --trainer.max_epochs=40 \
  --model.init_args.microbatch_size=256 \
  --data.batch_size=1024 \
  --model.init_args.lr="$lr"

# Gradient accumulation
run_experiment "long-contrastive-accum=1024-lr=$lr" \
  --trainer.max_epochs=40 \
  --data.batch_size=256 \
  --trainer.accumulate_grad_batches=4 \
  --model.init_args.lr="$lr"

# Cross-Batch Memory
run_experiment "long-contrastive-xbm=1024-lr=$lr" \
  --trainer.max_epochs=40 \
  --model.init_args.cross_batch_memory_size=1024 \
  --data.batch_size=256 \
  --model.init_args.lr="$lr"

echo "Done."
