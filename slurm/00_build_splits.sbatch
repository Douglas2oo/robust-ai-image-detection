#!/bin/bash
#SBATCH --job-name=aigi_splits
#SBATCH --account=mscaispod2026
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# 生成固定的数据清单（pHash 去重 + 审计）。全队只运行一次；cpu 分区每个作业最多 8 核。
set -euo pipefail
source slurm/config.sh
activate_env
EXT=()   # 外部测试集只有在已下载时才加入
[ -d "$DATA/$EXT_PROGAN" ] && EXT+=(--ext progan="$EXT_PROGAN")
[ -d "$DATA/${EXT_LDM200%%:*}" ] && EXT+=(--ext ldm200="$EXT_LDM200")

python -m scripts.prepare_data --data_root "$DATA" \
  --gen sd14="$GEN_SD14" --gen sd15="$GEN_SD15" --gen adm="$GEN_ADM" \
  --gen midjourney="$GEN_MJ" --gen biggan="$GEN_BIGGAN" "${EXT[@]}" \
  --num_classes 100 --n_train 20000 --n_val 2000 --n_test 1000 --n_ext 1000 \
  --workers "$SLURM_CPUS_PER_TASK" --out_dir "$SPLITS"
