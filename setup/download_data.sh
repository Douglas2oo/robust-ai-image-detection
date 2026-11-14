#!/bin/bash
# Data download helpers. Only ONE group member needs to run these: everything lands in the
# shared /project/mscaispod2026/data and is reused by the CLIP and ResNet runs.
#
# Run downloads on the login node inside tmux (network-bound, light CPU):
#     tmux new -s dl
#     bash setup/download_data.sh <command>
# Run heavy unzipping on a CPU compute node instead (see `extract` below).
#
# Commands
#   kaggle_sd14            GenImage SD1.4 (train + val)   — Kaggle mirror of GenImage
#   kaggle_val             GenImage val sets of all generators (SD1.5/ADM/MJ/BigGAN tests)
#   official <GDRIVE_DIR>  Official GenImage Google Drive folder via rclone, e.g. "stable_diffusion_v_1_4"
#   ufd_diffusion          UniversalFakeDetect diffusion_datasets (contains ldm_200 + real sets)
#   extract <dir> <dest>   Unpack every archive in <dir> into <dest> (handles .z01 splits, .001 volumes, .zip)
#   inspect <dir>          Show folder tree (depth 5) and image counts to set GEN_* in slurm/config.sh
set -euo pipefail
source "$(dirname "$0")/../slurm/config.sh"
activate_env
RAW=$DATA/raw
mkdir -p "$RAW"

GENIMAGE_GDRIVE_ROOT=1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS   # official folder (GenImage README)
UFD_DIFFUSION_ID=1FXlGIRh_Ud3cScMgSVDbEWmPDmjcrm1t       # UniversalFakeDetect README

cmd=${1:-help}; shift || true
case "$cmd" in
  kaggle_sd14)
    # needs ~/.kaggle/kaggle.json (kaggle.com -> Settings -> API -> Create New Token), chmod 600
    kaggle datasets download -d vtphatt2/GenImage-stable-diffusion-v1-4 -p "$RAW/kaggle_sd14"
    ;;
  kaggle_val)
    kaggle datasets download -d vtphatt2/GenImage-Validation -p "$RAW/kaggle_val"
    ;;
  official)
    # One-time: on your laptop run `rclone authorize "drive"`, then on the server
    # `rclone config` -> new remote "gdrive", type drive, scope drive.readonly, paste the token.
    dir=${1:?give a folder name, e.g. stable_diffusion_v_1_4}
    rclone lsd gdrive: --drive-root-folder-id "$GENIMAGE_GDRIVE_ROOT"
    rclone copy "gdrive:$dir" "$RAW/official_$dir" --drive-root-folder-id "$GENIMAGE_GDRIVE_ROOT" \
      -P --transfers 4 --checkers 8 --drive-acknowledge-abuse
    ;;
  ufd_diffusion)
    mkdir -p "$RAW/ufd" && cd "$RAW/ufd"
    gdown "$UFD_DIFFUSION_ID" -O diffusion_datasets.zip
    ;;
  extract)
    src=${1:?archive dir}; dest=${2:?destination dir}; mkdir -p "$dest"
    shopt -s nullglob
    for z in "$src"/*.zip; do
      base=${z%.zip}
      if compgen -G "$base.z[0-9]*" >/dev/null; then        # split zip: x.z01 x.z02 ... x.zip
        echo "[merge] $z"
        zip -s 0 "$z" --out "$base.full.zip" > "$base.merge.log" 2>&1 || { echo "[FAIL] merge failed: $z"; tail -3 "$base.merge.log"; exit 1; }
        echo "[unzip] $base.full.zip"
        unzip -q -n "$base.full.zip" -d "$dest" || { echo "[FAIL] unzip failed: $z"; exit 1; }
        rm -f "$base.full.zip" "$base.merge.log"
      elif [[ "$z" != *.full.zip ]]; then
        echo "[unzip] $z"; unzip -q -n "$z" -d "$dest" || { echo "[FAIL] unzip failed: $z"; exit 1; }
      fi
    done
    for v in "$src"/*.001; do echo "[7z] $v"; 7z x -y -o"$dest" "$v" >/dev/null; done
    for t in "$src"/*.tar "$src"/*.tar.gz "$src"/*.tgz; do echo "[tar] $t"; tar -xf "$t" -C "$dest"; done
    echo "done -> $dest"; du -sh "$dest"
    ;;
  inspect)
    d=${1:?dir}
    find "$d" -maxdepth 5 -type d | head -80
    for s in train val; do for l in ai nature; do
      echo "$s/$l: $(find "$d" -type f -path "*/$s/$l/*" | wc -l) files"
    done; done
    echo "0_real: $(find "$d" -type f -path '*/0_real/*' | wc -l)   1_fake: $(find "$d" -type f -path '*/1_fake/*' | wc -l)"
    ;;
  *) sed -n '2,20p' "$0" ;;
esac
