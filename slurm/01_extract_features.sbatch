#!/bin/bash
#SBATCH --job-name=aigi_feats
#SBATCH --account=mscaispod2026
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# 冻结 CLIP ViT-L/14 抽特征。1 张 H800，Slurm 自动分配 28 线程（不要加 --mem / CPU 参数）。
# 已存在的缓存会跳过，超时后直接重新提交即可。
set -euo pipefail
source slurm/config.sh
activate_env
export HF_HUB_OFFLINE=1          # 权重已在登录节点缓存
nvidia-smi --query-gpu=name,memory.total --format=csv

# 无增强：train_clean.pt + val_eval.pt + test/test_*.pt（8 种扰动）
python -m clip_detector.cache_features --config clip_detector/configs/baseline.yaml $PATH_ARGS
# 有增强：train_aug.pt（验证集和测试集缓存与上面共用，会自动跳过）
python -m clip_detector.cache_features --config clip_detector/configs/augmented.yaml $PATH_ARGS
ls -lh "$FEATS" "$FEATS/test"
