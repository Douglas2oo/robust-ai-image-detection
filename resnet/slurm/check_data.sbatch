#!/bin/bash
#SBATCH --account=mscaispod2026
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --job-name=rn28400-check
#SBATCH --output=logs/check-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
"$HOME/.venvs/resnet5201/bin/python" -u -m resnet.check_shared_data --stage train --report outputs/preflight_train.json
