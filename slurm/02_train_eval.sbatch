#!/bin/bash
#SBATCH --job-name=aigi_train
#SBATCH --account=mscaispod2026
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# 在缓存特征上训练线性头：2 组配置 × 3 个种子，只用验证集选参、校准、定阈值；
# 然后在所有测试集 × 扰动上评测。已训练过的种子会跳过。
set -euo pipefail
source slurm/config.sh
activate_env
python -m clip_detector.train --config clip_detector/configs/baseline.yaml  $PATH_ARGS
python -m clip_detector.train --config clip_detector/configs/augmented.yaml $PATH_ARGS
python -m scripts.evaluate --runs "$RUNS/clip_noaug" "$RUNS/clip_aug" --feat_dir "$FEATS/test" \
  --out_dir "$RESULTS" --save_preds
