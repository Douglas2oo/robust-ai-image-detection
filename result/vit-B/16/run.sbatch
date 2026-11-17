#!/bin/bash
#SBATCH --job-name=aigi_vit_b16
#SBATCH --account=mscaispod2026
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
export PATH="$PWD/.venv/bin:$PATH"
DATA=/project/mscaispod2026/group11
ROOT="$PWD/full"
export TORCH_HOME="$PWD/weights"
export OMP_NUM_THREADS=1
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
CONFIG=clip_detector/configs/vit_b16.yaml
MANIFESTS=manifests/splits_v1
if [[ "${1:-}" == smoke ]]; then
    ROOT="$PWD/smoke"
    mkdir -p "$ROOT/manifests"
    python - "$ROOT" <<'PYSMOKE'
import sys
from pathlib import Path
import pandas as pd
import yaml

root = Path(sys.argv[1])
for manifest in Path("manifests/splits_v1").glob("*.csv"):
    if not (manifest.stem in {"train", "val"} or manifest.stem.startswith("test_")):
        continue
    data = pd.read_csv(manifest)
    data.groupby("label").head(8).to_csv(root / "manifests" / manifest.name, index=False)
config = yaml.safe_load(Path("clip_detector/configs/vit_b16.yaml").read_text())
config["training"].update(seeds=[0], learning_rates=[0.001], weight_decays=[0.0], epochs=2)
config["features"].update(batch_size=4, num_workers=2)
(root / "config.yaml").write_text(yaml.safe_dump(config))
PYSMOKE
    CONFIG="$ROOT/config.yaml"
    MANIFESTS="$ROOT/manifests"
fi
ARGS=(--config "$CONFIG" --data_root "$DATA" --manifest_dir "$MANIFESTS"
      --cache_dir "$ROOT/features" --checkpoint_dir "$ROOT/runs" --output_dir "$ROOT/results")

# Fail before training if any of the shared images is missing or unreadable.
for split in train val test; do
    python -u -m clip_detector.cache_features "${ARGS[@]}" --only "$split"
done
python - "$ROOT/features" <<'PY'
from pathlib import Path
import sys
import torch

root = Path(sys.argv[1])
paths = [root / "train_clean.pt", root / "val_eval.pt"]
paths += [root / "test" / f"test_{name}.pt"
          for name in ("sd14", "sd15", "adm", "midjourney", "biggan")]
for path in paths:
    cache = torch.load(path, map_location="cpu", weights_only=True)
    failed = [p for p, ok in zip(cache["relpath"], cache["ok"]) if not ok]
    if failed:
        raise RuntimeError(f"{path}: {len(failed)} unreadable images, e.g. {failed[:5]}")
PY
python -u -m clip_detector.train "${ARGS[@]}"
python -u -m scripts.evaluate --runs "$ROOT/runs/vit_b16_noaug" \
  --feat_dir "$ROOT/features/test" --out_dir "$ROOT/results" --save_preds
