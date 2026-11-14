#!/bin/bash
# Submit the whole pipeline with dependencies (run from the repo root on the login node).
#   bash slurm/submit_all.sh            # features + training/eval (splits already built)
#   bash slurm/submit_all.sh --splits   # also build the splits first
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
dep=""
if [ "${1:-}" = "--splits" ]; then
  j0=$(sbatch --parsable slurm/00_build_splits.sbatch); echo "splits: $j0"; dep="--dependency=afterok:$j0"
fi
j1=$(sbatch --parsable $dep slurm/01_extract_features.sbatch); echo "features: $j1"
j2=$(sbatch --parsable --dependency=afterok:$j1 slurm/02_train_eval.sbatch); echo "train+eval: $j2"
echo "monitor: squeue -u \$USER ; tail -f logs/*.out"
