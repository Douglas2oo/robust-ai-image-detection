#!/bin/bash
#SBATCH --account=mscaispod2026
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --job-name=rn28400-test
#SBATCH --output=logs/test-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4
# Explicitly fail if SD1.5 or another required test image is inaccessible. Never silently omit a domain.
"$HOME/.venvs/resnet5201/bin/python" -u -m resnet.check_shared_data --stage test --report outputs/preflight_test.json
"$HOME/.venvs/resnet5201/bin/python" -u -m resnet.evaluate_local \
  --checkpoint outputs/resnet_shared28400/seed_0/best.pt \
  --manifest-dir manifests/clip_shared_v1 --output outputs/resnet_shared28400/test_seed0
