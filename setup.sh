#!/usr/bin/env bash
set -euo pipefail

conda env create -f environment.yml || conda env update -f environment.yml
conda run -n contrastive-av python -m pip install \
  "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu128torch2.10-cp312-cp312-linux_x86_64.whl"

