#!/usr/bin/env bash
set -e

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p /root/miniconda3

source /root/miniconda3/etc/profile.d/conda.sh

echo "alias miniconda='source /root/miniconda3/etc/profile.d/conda.sh && conda activate base'" >> /root/.bashrc

conda env create -f environment.yml || conda env update -f environment.yml

conda run -n contrastive-av python -m pip install \
  "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu128torch2.10-cp312-cp312-linux_x86_64.whl"
