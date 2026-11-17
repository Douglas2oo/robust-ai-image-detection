#!/bin/bash
set -euo pipefail
module load Anaconda3/2023.09-0
python -m venv .venv
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu124 'torch==2.6.0' 'torchvision==0.21.0'
.venv/bin/python -m pip install numpy pandas Pillow PyYAML scikit-learn tqdm
export TORCH_HOME="$PWD/weights"
.venv/bin/python -c 'from torchvision.models import ViT_B_16_Weights; ViT_B_16_Weights.IMAGENET1K_V1.get_state_dict(check_hash=True)'
