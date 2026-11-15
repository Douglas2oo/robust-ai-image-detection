#!/bin/bash
# 在 SuperPOD 登录节点上一次性配置环境（需要外网；只安装，不做计算）。
#   bash setup/setup_env.sh
set -euo pipefail
source "$(dirname "$0")/../slurm/config.sh"
mkdir -p "$MY_PROJ"

# 1. Miniforge（已有 conda 则跳过）
if [ ! -d "$HOME/miniforge3" ]; then
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/mf.sh
  bash /tmp/mf.sh -b -p "$HOME/miniforge3"
fi
source "$HOME/miniforge3/etc/profile.d/conda.sh"

# 2. 环境（H800：使用 CUDA 12.x 的 PyTorch）
conda create -y -n "$CONDA_ENV" python=3.11 p7zip rclone unzip zip -c conda-forge
set +u; conda activate "$CONDA_ENV"; set -u
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r "$CODE/requirements.txt" kaggle gdown

# 3. 预先下载 CLIP ViT-L/14 权重（计算节点可能没有外网）
mkdir -p "$HF_HOME"
python - <<'PY'
from transformers import CLIPVisionModelWithProjection
m = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
print("CLIP ok, projection dim =", m.config.projection_dim)
PY

mkdir -p "$SPLITS" "$FEATS" "$RUNS" "$RESULTS" "$CODE/logs"
echo "Environment ready."
