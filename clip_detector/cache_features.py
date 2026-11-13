"""Extract and cache frozen CLIP features for one configured experiment."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset

from common.data import ManifestImageDataset, read_manifest
from common.preprocessing import ClipImagePreprocessor
from clip_detector.augmentations import ClipAugmenter
from clip_detector.config import (
    cache_signature,
    clip_checkpoint_identity,
    file_sha256,
    load_config,
    resolve_project_path,
)
from clip_detector.model import clip_feature_dim, encode_image_features, load_clip_encoder


def project_root_for_config(config_path: Path) -> Path:
    resolved = Path(config_path).resolve()
    if len(resolved.parents) < 3:
        raise ValueError(f"Expected config under clip_detector/configs: {config_path}")
    return resolved.parents[2]


def resolve_device(requested: Optional[str] = None) -> torch.device:
    if requested:
        device = torch.device(requested)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def cache_file_path(cache_dir: Path, split: str, signature: str) -> Path:
    return Path(cache_dir) / f"{split}_{signature[:16]}.pt"


def _torch_load(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was introduced.
        return torch.load(path, map_location="cpu")


def _atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _check_split(records: List[Dict[str, str]], expected: str, path: Path) -> None:
    wrong = sorted({record["split"] for record in records if record["split"] != expected})
    if wrong:
        raise ValueError(
            f"Manifest {path} is configured as {expected} but contains split values: {wrong}"
        )


def _validate_manifest_pair(train_records, val_records, train_path: Path, val_path: Path) -> None:
    train_ids = {row["image_id"] for row in train_records}
    val_ids = {row["image_id"] for row in val_records}
    overlap = train_ids & val_ids
    if overlap:
        raise ValueError(f"Train/validation manifests share image IDs: {sorted(overlap)[:5]}")
    train_groups = {row["duplicate_group"] for row in train_records if row["duplicate_group"]}
    val_groups = {row["duplicate_group"] for row in val_records if row["duplicate_group"]}
    group_overlap = train_groups & val_groups
    if group_overlap:
        raise ValueError(
            "Train/validation manifests share duplicate groups: "
            f"{sorted(group_overlap)[:5]} ({train_path}, {val_path})"
        )


def _chunk_file_path(parts_dir: Path, chunk_index: int, start: int, end: int) -> Path:
    return parts_dir / f"chunk_{chunk_index:06d}_{start:09d}_{end:09d}.pt"


def _load_valid_chunk(
    path: Path,
    *,
    signature: str,
    split: str,
    chunk_index: int,
    start: int,
    end: int,
    views_per_image: int,
    feature_dim: int,
    records: List[Dict[str, str]],
) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        payload = _torch_load(path)
        expected_ids = [row["image_id"] for row in records[start:end]]
        expected_labels = torch.tensor(
            [int(row["label"]) for row in records[start:end]], dtype=torch.long
        )
        features = payload.get("features")
        labels = payload.get("labels")
        valid = (
            payload.get("format_version") == 2
            and payload.get("signature") == signature
            and payload.get("split") == split
            and payload.get("chunk_index") == chunk_index
            and payload.get("start") == start
            and payload.get("end") == end
            and isinstance(features, torch.Tensor)
            and tuple(features.shape) == (end - start, views_per_image, feature_dim)
            and torch.isfinite(features).all().item()
            and isinstance(labels, torch.Tensor)
            and torch.equal(labels.long(), expected_labels)
            and payload.get("image_ids") == expected_ids
        )
        return payload if valid else None
    except (
        OSError, EOFError, KeyError, TypeError, ValueError, AttributeError,
        RuntimeError, pickle.UnpicklingError,
    ):
        return None


def _extract_chunk(
    *,
    dataset: ManifestImageDataset,
    records: List[Dict[str, str]],
    start: int,
    end: int,
    chunk_index: int,
    signature: str,
    split: str,
    views_per_image: int,
    feature_dim: int,
    batch_size: int,
    num_workers: int,
    encoder,
    device: torch.device,
    normalize_features: bool,
) -> Dict[str, Any]:
    flat_start = start * views_per_image
    flat_end = end * views_per_image
    loader = DataLoader(
        Subset(dataset, range(flat_start, flat_end)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    feature_batches = []
    label_batches = []
    flat_image_ids: List[str] = []
    flat_view_indices: List[int] = []
    for images, labels, batch_ids, view_indices in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = encode_image_features(encoder, images, normalize_features)
            else:
                features = encode_image_features(encoder, images, normalize_features)
        feature_batches.append(features.cpu())
        label_batches.append(labels.to(dtype=torch.long).cpu())
        flat_image_ids.extend(list(batch_ids))
        flat_view_indices.extend(int(value) for value in view_indices.tolist())

    flat_features = torch.cat(feature_batches, dim=0)
    flat_labels = torch.cat(label_batches, dim=0)
    source_records = records[start:end]
    expected_flat_ids = [
        row["image_id"] for row in source_records for _view in range(views_per_image)
    ]
    expected_view_indices = [
        view for _row in source_records for view in range(views_per_image)
    ]
    expected_count = (end - start) * views_per_image
    if tuple(flat_features.shape) != (expected_count, feature_dim):
        raise RuntimeError(
            f"Chunk {chunk_index} expected features [{expected_count},{feature_dim}], "
            f"got {tuple(flat_features.shape)}"
        )
    if flat_image_ids != expected_flat_ids or flat_view_indices != expected_view_indices:
        raise RuntimeError(f"Chunk {chunk_index} data order does not match the manifest")
    features = flat_features.reshape(end - start, views_per_image, feature_dim).contiguous()
    labels_by_view = flat_labels.reshape(end - start, views_per_image)
    if not torch.all(labels_by_view == labels_by_view[:, :1]):
        raise RuntimeError(f"Chunk {chunk_index} has inconsistent labels across views")
    labels = labels_by_view[:, 0].contiguous()
    return {
        "format_version": 2,
        "signature": signature,
        "split": split,
        "chunk_index": chunk_index,
        "start": start,
        "end": end,
        "features": features,
        "labels": labels,
        "image_ids": [row["image_id"] for row in source_records],
        "views_per_image": views_per_image,
        "feature_dim": feature_dim,
    }


def extract_split(
    config: Dict[str, Any],
    project_root: Path,
    split: str,
    encoder,
    preprocess,
    device: torch.device,
) -> Tuple[Path, Dict[str, Any]]:
    manifest_key = "train_manifest" if split == "train" else "val_manifest"
    manifest_path = resolve_project_path(project_root, config["data"][manifest_key])
    data_root = resolve_project_path(project_root, config["data"]["root"])
    records = read_manifest(manifest_path)
    _check_split(records, split, manifest_path)
    signature = cache_signature(config, manifest_path, split)
    cache_dir = resolve_project_path(project_root, config["features"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_file_path(cache_dir, split, signature)
    metadata_path = cache_path.with_suffix(".json")

    if cache_path.is_file() and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            payload = _torch_load(cache_path)
            expected_views = int(config["features"]["views_per_image"]) if split == "train" and config["augmentation"]["enabled"] else 1
            expected_feature_dim = int(config["model"]["feature_dim"])
            cache_is_valid = (
                metadata.get("signature") == signature
                and metadata.get("manifest_sha256") == file_sha256(manifest_path)
                and metadata.get("count") == len(records)
                and payload.get("signature") == signature
                and tuple(payload["features"].shape) == (len(records), expected_views, expected_feature_dim)
                and payload["labels"].shape[0] == len(records)
                and payload.get("image_ids") == [row["image_id"] for row in records]
            )
            if cache_is_valid:
                return cache_path, metadata
        except (
            OSError, EOFError, KeyError, TypeError, ValueError, AttributeError,
            RuntimeError, pickle.UnpicklingError,
        ):
            pass

    augment_enabled = split == "train" and bool(config["augmentation"]["enabled"])
    views_per_image = int(config["features"]["views_per_image"]) if split == "train" else 1
    augmenter = ClipAugmenter(config["augmentation"]) if augment_enabled else None
    dataset = ManifestImageDataset(
        records=records,
        data_root=data_root,
        preprocess=ClipImagePreprocessor(preprocess),
        views_per_image=views_per_image,
        base_seed=int(config["augmentation"].get("seed", config["seed"])),
        image_transform=augmenter,
    )
    encoder.eval()
    feature_dim = int(config["model"]["feature_dim"])
    batch_size = int(config["features"]["batch_size"])
    num_workers = int(config["features"]["num_workers"])
    chunk_size = int(config["features"].get("cache_chunk_size", 512))
    total_chunks = (len(records) + chunk_size - 1) // chunk_size
    parts_dir = cache_dir / f"{cache_path.stem}.parts_chunk_{chunk_size}"
    parts_dir.mkdir(parents=True, exist_ok=True)
    progress_path = parts_dir / "progress.json"
    chunk_specs: List[Tuple[Path, int, int, int]] = []
    for chunk_index in range(total_chunks):
        start = chunk_index * chunk_size
        end = min(start + chunk_size, len(records))
        chunk_path = _chunk_file_path(parts_dir, chunk_index, start, end)
        chunk = _load_valid_chunk(
            chunk_path,
            signature=signature,
            split=split,
            chunk_index=chunk_index,
            start=start,
            end=end,
            views_per_image=views_per_image,
            feature_dim=feature_dim,
            records=records,
        )
        if chunk is None:
            chunk = _extract_chunk(
                dataset=dataset,
                records=records,
                start=start,
                end=end,
                chunk_index=chunk_index,
                signature=signature,
                split=split,
                views_per_image=views_per_image,
                feature_dim=feature_dim,
                batch_size=batch_size,
                num_workers=num_workers,
                encoder=encoder,
                device=device,
                normalize_features=bool(config["model"]["normalize_features"]),
            )
            _atomic_torch_save(chunk, chunk_path)
        chunk_specs.append((chunk_path, chunk_index, start, end))
        _atomic_write_json(
            progress_path,
            {
                "format_version": 2,
                "signature": signature,
                "split": split,
                "chunk_size": chunk_size,
                "completed_chunks": chunk_index + 1,
                "total_chunks": total_chunks,
                "completed_images": end,
                "total_images": len(records),
            },
        )
        print(
            f"{split}: cached chunk {chunk_index + 1}/{total_chunks} "
            f"({end}/{len(records)} images)"
        )

    features = torch.empty((len(records), views_per_image, feature_dim), dtype=torch.float32)
    labels = torch.empty(len(records), dtype=torch.long)
    image_ids: List[str] = []
    for chunk_path, chunk_index, start, end in chunk_specs:
        chunk = _load_valid_chunk(
            chunk_path,
            signature=signature,
            split=split,
            chunk_index=chunk_index,
            start=start,
            end=end,
            views_per_image=views_per_image,
            feature_dim=feature_dim,
            records=records,
        )
        if chunk is None:
            raise RuntimeError(f"Completed cache chunk failed validation: {chunk_path}")
        features[start:end].copy_(chunk["features"].float())
        labels[start:end].copy_(chunk["labels"].long())
        image_ids.extend(chunk["image_ids"])
    if tuple(features.shape) != (len(records), views_per_image, feature_dim):
        raise RuntimeError(f"Assembled {split} cache has an unexpected feature shape")
    if labels.shape[0] != len(records) or image_ids != [row["image_id"] for row in records]:
        raise RuntimeError(f"Assembled {split} cache does not match the manifest")

    payload = {
        "format_version": 2,
        "signature": signature,
        "split": split,
        "features": features.contiguous(),
        "labels": labels.contiguous(),
        "image_ids": image_ids,
        "views_per_image": views_per_image,
        "feature_dim": feature_dim,
    }
    _atomic_torch_save(payload, cache_path)
    metadata = {
        "format_version": 2,
        "signature": signature,
        "manifest": str(manifest_path.relative_to(project_root)),
        "manifest_sha256": file_sha256(manifest_path),
        "model": config["model"]["name"],
        "provider": config["model"]["provider"],
        "weights_identity": clip_checkpoint_identity(config["model"]["name"]),
        "feature_dim": feature_dim,
        "normalize_features": bool(config["model"]["normalize_features"]),
        "preprocess": "openai_clip_preprocess_v1",
        "augmentation": config["augmentation"] if augment_enabled else {"enabled": False},
        "augmentation_seed": int(config["augmentation"].get("seed", config["seed"])) if augment_enabled else None,
        "split": split,
        "views_per_image": views_per_image,
        "count": len(records),
        "cache_chunk_size": chunk_size,
        "total_chunks": total_chunks,
        "parts_dir": parts_dir.name,
        "device": str(device),
    }
    _atomic_write_json(metadata_path, metadata)
    return cache_path, metadata


def build_caches(config_path: Path, requested_device: Optional[str] = None) -> List[Path]:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    project_root = project_root_for_config(config_path)
    train_path = resolve_project_path(project_root, config["data"]["train_manifest"])
    val_path = resolve_project_path(project_root, config["data"]["val_manifest"])
    train_records = read_manifest(train_path)
    val_records = read_manifest(val_path)
    _check_split(train_records, "train", train_path)
    _check_split(val_records, "val", val_path)
    _validate_manifest_pair(train_records, val_records, train_path, val_path)

    device = resolve_device(requested_device)
    encoder, preprocess = load_clip_encoder(
        config["model"]["name"], device=str(device),
        download_root=str(resolve_project_path(project_root, "cache/openai_clip")),
    )
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    expected_dim = int(config["model"].get("feature_dim", clip_feature_dim(encoder)))
    if expected_dim != clip_feature_dim(encoder):
        raise ValueError(
            f"Configured feature_dim={expected_dim} does not match CLIP output "
            f"{clip_feature_dim(encoder)}"
        )
    paths = []
    for split in ("train", "val"):
        path, metadata = extract_split(config, project_root, split, encoder, preprocess, device)
        print(f"{split}: {metadata['count']} images -> {path}")
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="CLIP YAML configuration")
    parser.add_argument("--device", default=None, help="Override auto-selected device, e.g. cuda:0")
    args = parser.parse_args()
    if not args.config.is_file():
        parser.error(f"Configuration file does not exist: {args.config}")
    try:
        build_caches(args.config, args.device)
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        parser.exit(status=2, message=f"Feature caching failed: {exc}\n")


if __name__ == "__main__":
    main()
