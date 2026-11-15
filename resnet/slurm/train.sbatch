#!/bin/bash
#SBATCH --account=mscaispod2026
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --job-name=resnet-28400
#SBATCH --output=logs/train-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
"$HOME/.venvs/resnet5201/bin/python" -c "import json; a=json.load(open('outputs/preflight_train.json')); assert a['ok'] and a['unique_images']==32400, 'Run the data preflight successfully first'"
nvidia-smi
"$HOME/.venvs/resnet5201/bin/python" -u -m resnet.train --config resnet/config.yaml "$@"
