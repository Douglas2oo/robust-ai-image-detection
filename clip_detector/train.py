"""Train the linear binary head on cached frozen CLIP features."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from clip_detector.cache_features import cache_file_path, project_root_for_config
from clip_detector.config import cache_signature, load_config, resolve_project_path, seed_everything
from clip_detector.model import LinearBinaryHead


def _atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _torch_load(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was introduced.
        return torch.load(path, map_location="cpu")


def _run_signature(config: Dict[str, Any], seed: int) -> str:
    identity = copy.deepcopy(config)
    identity["seed"] = int(seed)
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _capture_rng_state(train_generator: torch.Generator) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "train_loader_generator": train_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Dict[str, Any], train_generator: torch.Generator) -> None:
    required = {"python", "torch", "train_loader_generator"}
    if not isinstance(state, dict) or not required.issubset(state):
        raise ValueError("Resume checkpoint is missing reproducibility state")
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    train_generator.set_state(state["train_loader_generator"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _load_cache(
    path: Path,
    expected_signature: str,
    expected_views: int,
    expected_feature_dim: int,
) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Feature cache is missing: {path}. Run clip_detector.cache_features first."
        )
    payload = _torch_load(path)
    if payload.get("signature") != expected_signature:
        raise ValueError(
            f"Feature cache signature mismatch for {path}; rebuild it with the current config."
        )
    features = payload.get("features")
    labels = payload.get("labels")
    if not isinstance(features, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise ValueError(f"Invalid feature cache tensors: {path}")
    if features.ndim != 3 or features.shape[1] < 1 or labels.ndim != 1:
        raise ValueError(f"Expected features [N,V,D] and labels [N] in {path}")
    if features.shape[0] != labels.shape[0]:
        raise ValueError(f"Feature/label count mismatch in {path}")
    if features.shape[1:] != (expected_views, expected_feature_dim):
        raise ValueError(
            f"Expected cache features [N,{expected_views},{expected_feature_dim}], "
            f"got {tuple(features.shape)} in {path}"
        )
    if len(payload.get("image_ids", [])) != features.shape[0]:
        raise ValueError(f"Feature/image ID count mismatch in {path}")
    if not torch.isfinite(features).all():
        raise ValueError(f"Feature cache contains non-finite values: {path}")
    return payload


def _flatten_features(payload: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    features = payload["features"].float()
    labels = payload["labels"].long()
    image_ids = list(payload["image_ids"])
    return features, labels, image_ids


def _binary_metrics(labels: torch.Tensor, logits: torch.Tensor) -> Dict[str, float]:
    labels_np = labels.cpu().numpy()
    logits_np = logits.cpu().numpy()
    metrics: Dict[str, float] = {
        "loss": float(nn.functional.binary_cross_entropy_with_logits(logits, labels.float()).item()),
    }
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        if len(set(labels_np.tolist())) == 2:
            metrics["auroc"] = float(roc_auc_score(labels_np, logits_np))
            metrics["average_precision"] = float(average_precision_score(labels_np, logits_np))
    except ImportError:
        pass
    return metrics


def _write_predictions(path: Path, image_ids: List[str], labels: torch.Tensor, logits: torch.Tensor) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "label", "logit"])
        writer.writeheader()
        for image_id, label, logit in zip(image_ids, labels.tolist(), logits.tolist()):
            writer.writerow({"image_id": image_id, "label": int(label), "logit": f"{float(logit):.9g}"})


def _evaluate(
    head: LinearBinaryHead,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor]:
    head.eval()
    all_logits = []
    all_labels = []
    with torch.inference_mode():
        for features, labels in loader:
            logits = head(features.to(device))
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    return _binary_metrics(labels, logits), labels, logits


def _best_checkpoint_payload(
    head_state: Dict[str, torch.Tensor],
    feature_dim: int,
    seed: int,
    config: Dict[str, Any],
    run_signature: str,
    best_epoch: int,
    best_val_loss: float,
    train_signature: str,
    val_signature: str,
) -> Dict[str, Any]:
    return {
        "format_version": 2,
        "checkpoint_type": "best_inference",
        "model_type": "frozen_clip_vit_l14_linear_head",
        "head_state_dict": head_state,
        "feature_dim": int(feature_dim),
        "seed": int(seed),
        "experiment": config["experiment"],
        "run_signature": run_signature,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "train_cache_signature": train_signature,
        "val_cache_signature": val_signature,
        "config": config,
    }


def _validate_resume_checkpoint(
    checkpoint: Dict[str, Any],
    *,
    run_signature: str,
    seed: int,
    experiment: str,
    feature_dim: int,
    train_signature: str,
    val_signature: str,
    max_epochs: int,
) -> None:
    expected = {
        "checkpoint_type": "training_state",
        "model_type": "frozen_clip_vit_l14_linear_head",
        "run_signature": run_signature,
        "seed": int(seed),
        "experiment": experiment,
        "feature_dim": int(feature_dim),
        "train_cache_signature": train_signature,
        "val_cache_signature": val_signature,
    }
    mismatches = [
        key for key, value in expected.items() if checkpoint.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "Resume checkpoint does not match the current run: " + ", ".join(mismatches)
        )
    required = {
        "head_state_dict", "optimizer_state_dict", "best_head_state_dict",
        "completed_epoch", "best_epoch", "best_val_loss", "patience_counter",
        "history", "rng_state", "status",
    }
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError("Resume checkpoint is missing fields: " + ", ".join(missing))
    completed_epoch = int(checkpoint["completed_epoch"])
    history = checkpoint["history"]
    if completed_epoch < 1 or completed_epoch > max_epochs:
        raise ValueError("Resume checkpoint has an invalid completed_epoch")
    if not isinstance(history, list) or len(history) != completed_epoch:
        raise ValueError("Resume checkpoint history does not match completed_epoch")
    if checkpoint["status"] not in {"running", "complete"}:
        raise ValueError("Resume checkpoint has an invalid status")


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def train(
    config_path: Path,
    requested_seed: Optional[int] = None,
    requested_device: Optional[str] = None,
    requested_resume: Optional[Union[str, Path]] = None,
) -> Path:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    project_root = project_root_for_config(config_path)
    seed = int(config["seed"] if requested_seed is None else requested_seed)
    seed_everything(seed)
    device = torch.device(
        requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    train_manifest = resolve_project_path(project_root, config["data"]["train_manifest"])
    val_manifest = resolve_project_path(project_root, config["data"]["val_manifest"])
    train_signature = cache_signature(config, train_manifest, "train")
    val_signature = cache_signature(config, val_manifest, "val")
    cache_dir = resolve_project_path(project_root, config["features"]["cache_dir"])
    train_views = int(config["features"]["views_per_image"])
    feature_dim = int(config["model"]["feature_dim"])
    train_cache = _load_cache(
        cache_file_path(cache_dir, "train", train_signature), train_signature,
        train_views, feature_dim,
    )
    val_cache = _load_cache(
        cache_file_path(cache_dir, "val", val_signature), val_signature, 1, feature_dim
    )
    train_features, train_labels, _train_ids = _flatten_features(train_cache)
    val_features, val_labels, val_ids = _flatten_features(val_cache)
    if train_features.shape[-1] != val_features.shape[-1]:
        raise ValueError("Train and validation feature dimensions differ")
    if val_features.shape[1] != 1:
        raise ValueError("Validation features must contain only the original image view")
    if set(train_labels.tolist()) != {0, 1}:
        raise ValueError("Training manifest must contain both Real=0 and AI=1 classes")
    if set(val_labels.tolist()) != {0, 1}:
        raise ValueError("Validation manifest must contain both Real=0 and AI=1 classes")

    train_dataset = TensorDataset(train_features, train_labels.float())
    val_dataset = TensorDataset(val_features[:, 0], val_labels.float())
    train_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        generator=train_generator,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    head = LinearBinaryHead(train_features.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    checkpoint_dir = resolve_project_path(project_root, config["paths"]["checkpoint_dir"]) / f"seed_{seed}"
    output_dir = resolve_project_path(project_root, config["paths"]["output_dir"]) / f"seed_{seed}"
    effective_config = copy.deepcopy(config)
    effective_config["seed"] = seed
    effective_config["runtime"] = {"device": str(device), "project_root": str(project_root)}
    run_signature = _run_signature(config, seed)
    max_epochs = int(config["training"]["epochs"])
    patience = int(config["training"]["patience"])
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"

    resume_path: Optional[Path] = None
    resume_checkpoint: Optional[Dict[str, Any]] = None
    if requested_resume is not None:
        if str(requested_resume) == "auto":
            resume_path = last_path
        else:
            resume_path = Path(requested_resume).expanduser().resolve()
            if resume_path.is_dir():
                resume_path = resume_path / "last.pt"
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint is missing: {resume_path}")
        resume_checkpoint = _torch_load(resume_path)
        _validate_resume_checkpoint(
            resume_checkpoint,
            run_signature=run_signature,
            seed=seed,
            experiment=config["experiment"],
            feature_dim=int(train_features.shape[-1]),
            train_signature=train_signature,
            val_signature=val_signature,
            max_epochs=max_epochs,
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        if checkpoint_dir.exists() or output_dir.exists():
            raise FileExistsError(
                f"Run directory already exists for seed {seed}: {checkpoint_dir} or {output_dir}. "
                "Use --resume to continue it, or archive the old run explicitly."
            )
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        output_dir.mkdir(parents=True, exist_ok=False)

    config_yaml = yaml.safe_dump(effective_config, sort_keys=False)
    _atomic_write_text(output_dir / "config.yaml", config_yaml)
    _atomic_write_text(checkpoint_dir / "config.yaml", config_yaml)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience_counter = 0
    history: List[Dict[str, Any]] = []
    start_epoch = 1
    if resume_checkpoint is not None:
        head.load_state_dict(resume_checkpoint["head_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        _move_optimizer_state(optimizer, device)
        best_state = resume_checkpoint["best_head_state_dict"]
        best_epoch = int(resume_checkpoint["best_epoch"])
        best_val_loss = float(resume_checkpoint["best_val_loss"])
        patience_counter = int(resume_checkpoint["patience_counter"])
        history = list(resume_checkpoint["history"])
        start_epoch = int(resume_checkpoint["completed_epoch"]) + 1
        if resume_checkpoint["status"] == "complete":
            start_epoch = max_epochs + 1
        _restore_rng_state(resume_checkpoint["rng_state"], train_generator)
        print(f"Resuming {config['experiment']} seed={seed} from epoch {start_epoch}")

    for epoch in range(start_epoch, max_epochs + 1):
        head.train()
        train_losses = []
        for features, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            features = features.to(device)
            labels = labels.to(device)
            batch_size, views, feature_dim = features.shape
            logits = head(features.reshape(batch_size * views, feature_dim)).reshape(batch_size, views)
            view_labels = labels[:, None].expand_as(logits)
            per_image_loss = nn.functional.binary_cross_entropy_with_logits(
                logits, view_labels, reduction="none"
            ).mean(dim=1)
            loss = per_image_loss.mean()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))
        val_metrics, _labels, _logits = _evaluate(head, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": sum(train_losses) / len(train_losses),
            "val_loss": val_metrics["loss"],
            **{key: value for key, value in val_metrics.items() if key != "loss"},
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        improved = val_metrics["loss"] < best_val_loss
        if improved:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in head.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
        if best_state is None:
            raise RuntimeError("Training produced no best checkpoint state")
        if improved:
            _atomic_torch_save(
                _best_checkpoint_payload(
                    best_state, int(train_features.shape[-1]), seed, effective_config,
                    run_signature, best_epoch, best_val_loss, train_signature, val_signature,
                ),
                best_path,
            )
        complete = patience_counter >= patience or epoch >= max_epochs
        last_checkpoint = {
            "format_version": 2,
            "checkpoint_type": "training_state",
            "model_type": "frozen_clip_vit_l14_linear_head",
            "status": "complete" if complete else "running",
            "head_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in head.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "best_head_state_dict": best_state,
            "feature_dim": int(train_features.shape[-1]),
            "seed": seed,
            "experiment": config["experiment"],
            "run_signature": run_signature,
            "completed_epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            "history": history,
            "rng_state": _capture_rng_state(train_generator),
            "train_cache_signature": train_signature,
            "val_cache_signature": val_signature,
            "config": effective_config,
        }
        _atomic_torch_save(last_checkpoint, last_path)
        _atomic_write_text(output_dir / "history.json", json.dumps(history, indent=2))
        if complete:
            break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    head.load_state_dict(best_state)
    _atomic_torch_save(
        _best_checkpoint_payload(
            best_state, int(train_features.shape[-1]), seed, effective_config,
            run_signature, best_epoch, best_val_loss, train_signature, val_signature,
        ),
        best_path,
    )
    final_metrics, val_labels_final, val_logits_final = _evaluate(head, val_loader, device)
    _atomic_write_text(output_dir / "history.json", json.dumps(history, indent=2))
    _atomic_write_text(output_dir / "metrics.json", json.dumps(final_metrics, indent=2))
    _write_predictions(output_dir / "val_predictions.csv", val_ids, val_labels_final, val_logits_final)
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="CLIP YAML configuration")
    parser.add_argument("--seed", type=int, default=None, help="Override the training seed")
    parser.add_argument("--device", default=None, help="Override auto-selected device, e.g. cuda:0")
    parser.add_argument(
        "--resume", nargs="?", const="auto", default=None, metavar="LAST_PT",
        help="Resume from LAST_PT, or from this seed's last.pt when no path is given",
    )
    args = parser.parse_args()
    if not args.config.is_file():
        parser.error(f"Configuration file does not exist: {args.config}")
    try:
        checkpoint = train(args.config, args.seed, args.device, args.resume)
    except (FileNotFoundError, FileExistsError, ImportError, RuntimeError, ValueError) as exc:
        parser.exit(status=2, message=f"CLIP training failed: {exc}\n")
    print(f"Saved checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
