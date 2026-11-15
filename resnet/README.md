# ResNet-50 AI image detector

This directory implements the ResNet baseline trained on HKUST SuperPOD. Changes are confined to `resnet/`; the team's `common/`, CLIP implementation and frozen source manifests are preserved.

## Experiment protocol

- Labels: real = 0, generated = 1.
- Frozen input: `manifests/splits_v1/`. Training: 28,400 images (14,200 real + 14,200 SD1.4); validation: 4,000 images (2,000 per label).
- Five clean test pairs: each contains the same 1,000 real images and 1,000 generated images from SD1.4, SD1.5, ADM, BigGAN or Midjourney. There are 6,000 unique test images, not 10,000 independent images.
- This uses every row of the team's training manifest, **not the entire GenImage dataset**. Only SD1.4 generated images are used for fitting; other generators are held out.
- ResNet-50 ImageNet-1K V2 initialization; replace classifier with one logit; fine-tune all layers with BCEWithLogitsLoss, AdamW, learning rate 1e-4, weight decay 1e-4, cosine schedule, batch 32, maximum 20 epochs, patience 5, seed 0. Baseline augmentation is disabled.
- Preprocessing: V2 resize 232, center crop 224, RGB and ImageNet normalization. CUDA training uses AMP; evaluation uses FP32.
- Select checkpoint by minimum validation loss. Fit temperature and threshold on validation only; test data is never used for selection. Report macro averages across the correlated generator pairs.

## Installation

Run from repository root with Python 3.10 or later. For the verified SuperPOD environment, use the existing `$HOME/.venvs/resnet5201` environment. A fresh environment can be installed as follows (match CUDA wheels to the cluster driver):

```bash
python3 -m venv "$HOME/.venvs/resnet5201"
source "$HOME/.venvs/resnet5201/bin/activate"
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r resnet/requirements.txt
```

The first training run downloads the official torchvision V2 initialization. Completed custom checkpoints do not require that download for inference.

## Preserve and adapt the frozen manifests

```bash
python -m resnet.import_clip_splits --source manifests/splits_v1
mkdir -p logs
```

This converts columns into `manifests/clip_shared_v1/` without sampling or re-splitting. It validates expected counts, labels, disjoint fitting/test paths, identical supplied pHashes across partitions, and the shared real test set. It refuses to overwrite existing output. Existing converted manifests from the trained experiment can instead be copied unchanged to this location.

`resnet/manifest.py` preserves this experiment's list-of-dictionaries manifest interface because the team's current `common.data.read_manifest` returns a DataFrame with different columns. Numerical preprocessing and model architecture remain the same. The supplied pHash is not an independently computed image-content hash.

The configured dataset root is `/project/mscaispod2026/group11`; CSV paths already start with `genimage/`. Edit `resnet/config.yaml` for another installation before starting a new run.

## SuperPOD jobs

Run submission commands from repository root. Do not train on login nodes; the CPU partition uses batch jobs.

```bash
sbatch resnet/slurm/check_data.sbatch
# Wait for PREFLIGHT_OK, then:
sbatch resnet/slurm/train.sbatch
# Once training finishes:
sbatch resnet/slurm/evaluate.sbatch
```

Check status with `squeue -u "$USER"`; follow the actual job ID in `logs/check-JOBID.log`, `logs/train-JOBID.log` or `logs/test-JOBID.log`. Batch jobs continue after the local laptop disconnects. `Ctrl+C` in `tail -F` only stops watching the log. The preflight checks existence, read permission and nonempty files; it does not decode every image. Training/evaluation fail on image read errors rather than silently dropping them.

For standalone inference:

```bash
python -m resnet.predict_local example.jpg --checkpoint outputs/resnet_shared28400/seed_0/best.pt
```

For CPU integration tests using temporary synthetic images (not performance measurements):

```bash
python -m unittest discover -s resnet/tests -v
```

## Existing trained model and results

The completed run is stored on HPC at `~/resnet_clip_full28400/outputs/resnet_shared28400/seed_0/`. Its best checkpoint is epoch 19, with validation accuracy 99.80%. Reported clean test results from the completed job 782354 are:

| Generator | Accuracy / balanced accuracy | AUROC (rounded from log) |
| --- | ---: | ---: |
| ADM | 51.15% | 0.7142 |
| BigGAN | 49.70% | 0.6568 |
| Midjourney | 64.75% | 0.8636 |
| SD1.4 | 99.05% | 0.9994 |
| SD1.5 | 99.10% | 0.9995 |
| Macro | 72.75% | 0.8467218 |

These are one seed's user-provided HPC log results; the checkpoint and raw output JSON files have not been independently retrieved into this repository. The high SD accuracies do not establish general detection accuracy on other generators. Keep the complete raw metrics and hashes with the model archive.

`best.pt` contains calibrated inference weights. `last.pt` also stores optimizer, scheduler, scaler and RNG state for resuming. Only load trusted checkpoints: optimizer/RNG checkpoints use Python pickle through torch.load. Weights, images and personal HPC logs are excluded from Git.

To use the completed run in a new checkout on the same HPC, copy its original manifests and outputs unchanged:

```bash
cp -a ~/resnet_clip_full28400/manifests/clip_shared_v1 manifests/
mkdir -p outputs/resnet_shared28400
cp -a ~/resnet_clip_full28400/outputs/resnet_shared28400/seed_0 outputs/resnet_shared28400/
```

Do not rerun training into this completed output directory. Use a new run directory for a new experiment, or `--resume` with the original config and manifests to recover an interrupted run. Existing seed-0 training completed all configured epochs. For a separate evaluation without overwriting the original test results:

```bash
python -m resnet.evaluate_local --checkpoint outputs/resnet_shared28400/seed_0/best.pt --manifest-dir manifests/clip_shared_v1 --output outputs/resnet_shared28400/test_seed0_recheck
```

Store the roughly 90 MB best checkpoint and 270 MB resume checkpoint in the HPC backup or a separate model asset. Upload source code through this directory, not the entire training folder.
