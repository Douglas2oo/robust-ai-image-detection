#!/bin/bash
#SBATCH --account=mscaispod2026
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --job-name=rn28400-checktest
#SBATCH --output=logs/checktest-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
"$HOME/.venvs/resnet5201/bin/python" -u -m resnet.check_shared_data --stage test --report outputs/preflight_test.json
