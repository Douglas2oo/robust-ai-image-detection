"""Focused tests for manifest, augmentation, model, cache, and training contracts."""

from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image

from common.data import ManifestImageDataset, read_manifest
from clip_detector.augmentations import ClipAugmenter


HAS_TORCH_STACK = all(importlib.util.find_spec(name) for name in ("torch", "yaml"))


def _write_manifest(path: Path, rows) -> None:
    columns = [
        "image_id", "relative_path", "label", "generator", "semantic_class", "split", "duplicate_group"
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _make_rows(data_root: Path, split: str, prefix: str, count: int):
    rows = []
    for index in range(count):
        image_id = f"{prefix}_{index}"
        relative_path = f"{split}/{image_id}.png"
        image_path = data_root / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new(
            "RGB",
            (21 + index, 18 + index),
            (40 + (index % 2) * 180 if index % 3 == 0 else 0,
             40 + (index % 2) * 180 if index % 3 == 1 else 0,
             40 + (index % 2) * 180 if index % 3 == 2 else 0),
        )
        image.save(image_path)
        rows.append(
            {
                "image_id": image_id,
                "relative_path": relative_path,
                "label": index % 2,
                "generator": "real" if index % 2 == 0 else "sd14",
                "semantic_class": "toy",
                "split": split,
                "duplicate_group": f"{prefix}_group_{index}",
            }
        )
    return rows


class ManifestAndAugmentationTests(unittest.TestCase):
    def test_manifest_dataset_reads_rgb_and_keeps_view_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = _make_rows(root, "train", "train", 2)
            manifest = root / "train.csv"
            _write_manifest(manifest, rows)
            dataset = ManifestImageDataset(
                read_manifest(manifest), root, preprocess=lambda image: image.size,
                views_per_image=2, base_seed=9,
            )
            tensor_like, label, image_id, view_index = dataset[2]
            with Image.open(root / rows[1]["relative_path"]) as expected_image:
                self.assertEqual(tensor_like, expected_image.size)
            self.assertEqual(label, 1)
            self.assertEqual(image_id, "train_1")
            self.assertEqual(view_index, 0)

    def test_manifest_rejects_duplicate_ids_and_invalid_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.csv"
            row = {
                "image_id": "same", "relative_path": "a.png", "label": 0,
                "generator": "real", "semantic_class": "", "split": "train", "duplicate_group": "",
            }
            _write_manifest(path, [row, row])
            with self.assertRaisesRegex(ValueError, "repeats image_id"):
                read_manifest(path)
            row["image_id"] = "other"
            row["label"] = 3
            _write_manifest(path, [row])
            with self.assertRaisesRegex(ValueError, "label must be 0 or 1"):
                read_manifest(path)

    def test_augmentation_is_reproducible_and_returns_rgb(self):
        config = {
            "order": ["jpeg", "resize", "gaussian_blur"],
            "jpeg": {"probability": 1.0, "quality_range": [70, 70]},
            "resize": {"probability": 1.0, "scale_range": [0.6, 0.6]},
            "gaussian_blur": {"probability": 1.0, "sigma_range": [0.5, 0.5]},
        }
        augmenter = ClipAugmenter(config)
        image = Image.new("RGBA", (32, 24), (200, 40, 10, 100))
        first = augmenter(image, 123)
        second = augmenter(image, 123)
        self.assertEqual(first.mode, "RGB")
        self.assertEqual(first.size, image.size)
        self.assertEqual(first.tobytes(), second.tobytes())


@unittest.skipUnless(HAS_TORCH_STACK, "PyTorch and PyYAML are required")
class ClipModelAndPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from torch import nn

        cls.torch = torch

        class TinyEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.visual = SimpleNamespace(output_dim=6)
                self.projection = nn.Linear(3, 6)

            def encode_image(self, images):
                return self.projection(images.mean(dim=(2, 3)))

        cls.TinyEncoder = TinyEncoder

    def test_frozen_encoder_normalized_features_and_single_logit(self):
        from clip_detector.model import FrozenCLIPClassifier

        torch = self.torch
        model = FrozenCLIPClassifier(self.TinyEncoder(), normalize_features=True)
        model.train()
        self.assertFalse(model.encoder.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.encoder.parameters()))
        images = torch.randn(5, 3, 8, 8)
        features = model.encode_features(images)
        logits = model(images)
        self.assertEqual(tuple(features.shape), (5, 6))
        self.assertTrue(torch.allclose(features.norm(dim=-1), torch.ones(5), atol=1e-6))
        self.assertEqual(tuple(logits.shape), (5,))
        self.assertTrue(all(parameter.grad is None for parameter in model.encoder.parameters()))

    def test_both_experiments_cache_train_and_reload_checkpoint(self):
        import yaml
        from clip_detector import cache_features
        from clip_detector.model import load_linear_head_checkpoint
        from clip_detector.train import train

        torch = self.torch
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_dir = root / "clip_detector" / "configs"
            config_dir.mkdir(parents=True)
            data_root = root / "data"
            train_rows = _make_rows(data_root, "train", "tr", 8)
            val_rows = _make_rows(data_root, "val", "va", 4)
            train_manifest = root / "manifests" / "train.csv"
            val_manifest = root / "manifests" / "val.csv"
            train_manifest.parent.mkdir()
            _write_manifest(train_manifest, train_rows)
            _write_manifest(val_manifest, val_rows)

            def preprocess(image):
                resized = image.resize((8, 8))
                array = torch.tensor(list(resized.getdata()), dtype=torch.float32)
                return (array.reshape(8, 8, 3).permute(2, 0, 1) / 255.0).contiguous()

            def loader(_name, device="cpu", download_root=None):
                return self.TinyEncoder().to(device), preprocess

            configs = [
                {
                    "experiment": "clip_baseline", "seed": 42,
                    "model": {"name": "ViT-L/14", "provider": "openai_clip", "freeze_encoder": True,
                              "output_dim": 1, "feature_dim": 6, "normalize_features": True},
                    "data": {"root": "data", "train_manifest": "manifests/train.csv",
                             "val_manifest": "manifests/val.csv", "image_size": 224,
                             "color_mode": "RGB", "normalization": "pretrained_weights"},
                    "training": {"loss": "binary_cross_entropy_with_logits", "batch_size": 4,
                                 "epochs": 3, "learning_rate": 0.001, "weight_decay": 0.0001,
                                 "patience": 2},
                    "augmentation": {"enabled": False},
                    "features": {"mode": "cached_original", "views_per_image": 1,
                                 "batch_size": 4, "num_workers": 0, "cache_dir": "cache/test_baseline"},
                    "paths": {"checkpoint_dir": "checkpoints/test_baseline",
                              "output_dir": "outputs/test_baseline"},
                },
                {
                    "experiment": "clip_augmented", "seed": 42,
                    "model": {"name": "ViT-L/14", "provider": "openai_clip", "freeze_encoder": True,
                              "output_dim": 1, "feature_dim": 6, "normalize_features": True},
                    "data": {"root": "data", "train_manifest": "manifests/train.csv",
                             "val_manifest": "manifests/val.csv", "image_size": 224,
                             "color_mode": "RGB", "normalization": "pretrained_weights"},
                    "training": {"loss": "binary_cross_entropy_with_logits", "batch_size": 4,
                                 "epochs": 3, "learning_rate": 0.001, "weight_decay": 0.0001,
                                 "patience": 2},
                    "augmentation": {"enabled": True, "seed": 42,
                                     "order": ["jpeg", "resize", "gaussian_blur"],
                                     "jpeg": {"probability": 0.5, "quality_range": [50, 100]},
                                     "resize": {"probability": 0.5, "scale_range": [0.5, 1.0]},
                                     "gaussian_blur": {"probability": 0.5, "sigma_range": [0.1, 1.0]}},
                    "features": {"mode": "cached_augmented_views", "views_per_image": 2,
                                 "batch_size": 4, "num_workers": 0, "cache_dir": "cache/test_augmented"},
                    "paths": {"checkpoint_dir": "checkpoints/test_augmented",
                              "output_dir": "outputs/test_augmented"},
                },
            ]
            for config in configs:
                config_path = config_dir / f"{config['experiment']}.yaml"
                config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
                with mock.patch.object(cache_features, "load_clip_encoder", side_effect=loader):
                    cache_features.build_caches(config_path, "cpu")
                checkpoint_path = train(config_path, requested_seed=42, requested_device="cpu")
                head, metadata = load_linear_head_checkpoint(checkpoint_path)
                self.assertGreaterEqual(metadata["best_epoch"], 1)
                self.assertLessEqual(metadata["best_epoch"], config["training"]["epochs"])
                last_path = checkpoint_path.with_name("last.pt")
                self.assertTrue(last_path.is_file())
                last_checkpoint = torch.load(last_path, map_location="cpu")
                self.assertEqual(last_checkpoint["checkpoint_type"], "training_state")
                self.assertEqual(last_checkpoint["status"], "complete")
                self.assertEqual(
                    len(last_checkpoint["history"]), last_checkpoint["completed_epoch"]
                )
                predictions_path = (
                    root / config["paths"]["output_dir"] / "seed_42" / "val_predictions.csv"
                )
                with predictions_path.open("r", encoding="utf-8", newline="") as handle:
                    predictions = list(csv.DictReader(handle))
                self.assertEqual(len(predictions), len(val_rows))
                cache_dir = root / config["features"]["cache_dir"]
                train_cache_path = next(cache_dir.glob("train_*.pt"))
                train_cache = torch.load(train_cache_path, map_location="cpu")
                expected_views = 2 if config["augmentation"]["enabled"] else 1
                self.assertEqual(tuple(train_cache["features"].shape[:2]), (len(train_rows), expected_views))
                val_cache_path = next(cache_dir.glob("val_*.pt"))
                val_cache = torch.load(val_cache_path, map_location="cpu")
                self.assertEqual(tuple(val_cache["features"].shape[:2]), (len(val_rows), 1))
                with torch.inference_mode():
                    restored_logits = head(val_cache["features"][:, 0])
                self.assertEqual(tuple(restored_logits.shape), (len(val_rows),))
                self.assertTrue(torch.isfinite(restored_logits).all())

    def test_feature_cache_resumes_from_completed_chunks(self):
        import yaml
        from clip_detector import cache_features

        torch = self.torch
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_dir = root / "clip_detector" / "configs"
            config_dir.mkdir(parents=True)
            data_root = root / "data"
            train_rows = _make_rows(data_root, "train", "tr", 6)
            val_rows = _make_rows(data_root, "val", "va", 4)
            manifest_dir = root / "manifests"
            manifest_dir.mkdir()
            _write_manifest(manifest_dir / "train.csv", train_rows)
            _write_manifest(manifest_dir / "val.csv", val_rows)
            config = {
                "experiment": "cache_resume", "seed": 42,
                "model": {"name": "ViT-L/14", "provider": "openai_clip", "freeze_encoder": True,
                          "output_dim": 1, "feature_dim": 6, "normalize_features": True},
                "data": {"root": "data", "train_manifest": "manifests/train.csv",
                         "val_manifest": "manifests/val.csv", "image_size": 224,
                         "color_mode": "RGB", "normalization": "pretrained_weights"},
                "training": {"loss": "binary_cross_entropy_with_logits", "batch_size": 4,
                             "epochs": 3, "learning_rate": 0.001, "weight_decay": 0.0001,
                             "patience": 2},
                "augmentation": {"enabled": False},
                "features": {"mode": "cached_original", "views_per_image": 1,
                             "batch_size": 2, "cache_chunk_size": 2, "num_workers": 0,
                             "cache_dir": "cache/resume"},
                "paths": {"checkpoint_dir": "checkpoints/resume",
                          "output_dir": "outputs/resume"},
            }
            config_path = config_dir / "cache_resume.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

            def preprocess(image):
                resized = image.resize((8, 8))
                array = torch.tensor(list(resized.getdata()), dtype=torch.float32)
                return (array.reshape(8, 8, 3).permute(2, 0, 1) / 255.0).contiguous()

            def loader(_name, device="cpu", download_root=None):
                return self.TinyEncoder().to(device), preprocess

            original_encode = cache_features.encode_image_features
            interrupted_calls = 0

            def interrupted_encode(*args, **kwargs):
                nonlocal interrupted_calls
                interrupted_calls += 1
                if interrupted_calls == 2:
                    raise RuntimeError("simulated cache interruption")
                return original_encode(*args, **kwargs)

            with mock.patch.object(cache_features, "load_clip_encoder", side_effect=loader), \
                    mock.patch.object(cache_features, "encode_image_features", side_effect=interrupted_encode):
                with self.assertRaisesRegex(RuntimeError, "simulated cache interruption"):
                    cache_features.build_caches(config_path, "cpu")

            cache_dir = root / "cache" / "resume"
            parts_dir = next(cache_dir.glob("train_*.parts_chunk_2"))
            self.assertEqual(len(list(parts_dir.glob("chunk_*.pt"))), 1)
            (parts_dir / ".chunk_ignored.tmp").write_bytes(b"incomplete")

            resumed_calls = 0

            def counted_encode(*args, **kwargs):
                nonlocal resumed_calls
                resumed_calls += 1
                return original_encode(*args, **kwargs)

            with mock.patch.object(cache_features, "load_clip_encoder", side_effect=loader), \
                    mock.patch.object(cache_features, "encode_image_features", side_effect=counted_encode):
                cache_features.build_caches(config_path, "cpu")

            self.assertEqual(resumed_calls, 4)
            train_cache = torch.load(next(cache_dir.glob("train_*.pt")), map_location="cpu")
            self.assertEqual(train_cache["image_ids"], [row["image_id"] for row in train_rows])
            self.assertEqual(tuple(train_cache["features"].shape), (6, 1, 6))
            progress = (parts_dir / "progress.json").read_text(encoding="utf-8")
            self.assertIn('"completed_chunks": 3', progress)

    def test_training_resumes_after_epoch_and_rejects_mismatch(self):
        import yaml
        from clip_detector import cache_features, train as train_module

        torch = self.torch
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_dir = root / "clip_detector" / "configs"
            config_dir.mkdir(parents=True)
            data_root = root / "data"
            train_rows = _make_rows(data_root, "train", "tr", 8)
            val_rows = _make_rows(data_root, "val", "va", 4)
            manifest_dir = root / "manifests"
            manifest_dir.mkdir()
            _write_manifest(manifest_dir / "train.csv", train_rows)
            _write_manifest(manifest_dir / "val.csv", val_rows)
            config = {
                "experiment": "train_resume", "seed": 42,
                "model": {"name": "ViT-L/14", "provider": "openai_clip", "freeze_encoder": True,
                          "output_dim": 1, "feature_dim": 6, "normalize_features": True},
                "data": {"root": "data", "train_manifest": "manifests/train.csv",
                         "val_manifest": "manifests/val.csv", "image_size": 224,
                         "color_mode": "RGB", "normalization": "pretrained_weights"},
                "training": {"loss": "binary_cross_entropy_with_logits", "batch_size": 4,
                             "epochs": 4, "learning_rate": 0.001, "weight_decay": 0.0001,
                             "patience": 10},
                "augmentation": {"enabled": False},
                "features": {"mode": "cached_original", "views_per_image": 1,
                             "batch_size": 4, "cache_chunk_size": 4, "num_workers": 0,
                             "cache_dir": "cache/train_resume"},
                "paths": {"checkpoint_dir": "checkpoints/train_resume",
                          "output_dir": "outputs/train_resume"},
            }
            config_path = config_dir / "train_resume.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

            def preprocess(image):
                resized = image.resize((8, 8))
                array = torch.tensor(list(resized.getdata()), dtype=torch.float32)
                return (array.reshape(8, 8, 3).permute(2, 0, 1) / 255.0).contiguous()

            def loader(_name, device="cpu", download_root=None):
                return self.TinyEncoder().to(device), preprocess

            with mock.patch.object(cache_features, "load_clip_encoder", side_effect=loader):
                cache_features.build_caches(config_path, "cpu")

            original_evaluate = train_module._evaluate
            evaluate_calls = 0

            def interrupted_evaluate(*args, **kwargs):
                nonlocal evaluate_calls
                evaluate_calls += 1
                if evaluate_calls == 2:
                    raise RuntimeError("simulated training interruption")
                return original_evaluate(*args, **kwargs)

            with mock.patch.object(train_module, "_evaluate", side_effect=interrupted_evaluate):
                with self.assertRaisesRegex(RuntimeError, "simulated training interruption"):
                    train_module.train(config_path, requested_seed=42, requested_device="cpu")

            last_path = root / "checkpoints" / "train_resume" / "seed_42" / "last.pt"
            interrupted = torch.load(last_path, map_location="cpu")
            self.assertEqual(interrupted["completed_epoch"], 1)
            self.assertEqual(interrupted["status"], "running")

            best_path = train_module.train(
                config_path, requested_seed=42, requested_device="cpu", requested_resume="auto"
            )
            resumed = torch.load(last_path, map_location="cpu")
            self.assertEqual(resumed["completed_epoch"], 4)
            self.assertEqual(resumed["status"], "complete")
            self.assertEqual([row["epoch"] for row in resumed["history"]], [1, 2, 3, 4])
            self.assertTrue(best_path.is_file())

            uninterrupted_config = yaml.safe_load(yaml.safe_dump(config))
            uninterrupted_config["paths"] = {
                "checkpoint_dir": "checkpoints/train_uninterrupted",
                "output_dir": "outputs/train_uninterrupted",
            }
            uninterrupted_path = config_dir / "train_uninterrupted.yaml"
            uninterrupted_path.write_text(
                yaml.safe_dump(uninterrupted_config, sort_keys=False), encoding="utf-8"
            )
            uninterrupted_best_path = train_module.train(
                uninterrupted_path, requested_seed=42, requested_device="cpu"
            )
            uninterrupted_last = torch.load(
                uninterrupted_best_path.with_name("last.pt"), map_location="cpu"
            )
            self.assertEqual(resumed["history"], uninterrupted_last["history"])
            resumed_best = torch.load(best_path, map_location="cpu")
            uninterrupted_best = torch.load(uninterrupted_best_path, map_location="cpu")
            for key, value in resumed_best["head_state_dict"].items():
                self.assertTrue(torch.equal(value, uninterrupted_best["head_state_dict"][key]))

            mismatched = dict(config)
            mismatched["training"] = dict(config["training"])
            mismatched["training"]["learning_rate"] = 0.002
            config_path.write_text(yaml.safe_dump(mismatched, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "run_signature"):
                train_module.train(
                    config_path, requested_seed=42, requested_device="cpu",
                    requested_resume=last_path,
                )


if __name__ == "__main__":
    unittest.main()
