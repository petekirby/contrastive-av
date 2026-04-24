#!/usr/bin/env bash
set -euo pipefail

poolings=(mean max cls mean_first_last mean_first_last_concat cls_concat)
heads=(none simcse ln_gelu_residual two_linear_layer)
losses=(supcon contrastive multisimilarity circle generalized_lifted proxyanchor softtriple)
batch_sizes=(8 16 32 64 128 256)

for pooling in "${poolings[@]}"; do
  run_name=$(printf "contrastive: pooling=%s" "$pooling")
  echo "Running ${run_name}"

  python train.py fit \
    --config="configs/tinybert_contrastive.yaml" \
    --model.init_args.model_config.pooling="$pooling" \
    --trainer.logger.init_args.name="$run_name"
done

for head in "${heads[@]}"; do
  run_name=$(printf "contrastive: head=%s" "$head")
  echo "Running ${run_name}"

  python train.py fit \
    --config="configs/tinybert_contrastive.yaml" \
    --model.init_args.model_config.head_type="$head" \
    --trainer.logger.init_args.name="$run_name"
done

for loss in "${losses[@]}"; do
  run_name=$(printf "contrastive: loss=%s" "$loss")
  echo "Running ${run_name}"

  python train.py fit \
    --config="configs/tinybert_contrastive.yaml" \
    --model.init_args.loss_dict.name="$loss" \
    --trainer.logger.init_args.name="$run_name"
done

for bs in "${batch_sizes[@]}"; do
  run_name=$(printf "contrastive: bs=%d" "$bs")
  echo "Running ${run_name}"

  python train.py fit \
    --config="configs/tinybert_contrastive.yaml" \
    --data.batch_size="$bs" \
    --trainer.logger.init_args.name="$run_name"
done
echo "Done."
